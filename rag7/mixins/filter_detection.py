from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import HumanMessage

from .. import _cache, prompts
from ..helpers.disk_cache import field_rank_cache_save
from ..models import (
    CollectionIntent,
    FieldPriority,
    FilterIntent,
)

if TYPE_CHECKING:
    from langchain_core.messages import SystemMessage

    from ..backend import IndexConfig, SearchBackend


class FilterDetectionMixin:
    """LLM-driven filter / collection / field-priority detection.

    Methods here read schema metadata and route the question through
    structured-output LLM chains. Pulled out of AgenticRAG so the
    detection layer can evolve independently of retrieval and graph
    wiring.
    """

    backend: SearchBackend
    collections: dict[str, Any] | None
    collection_descriptions: dict[str, str]
    _index_config: IndexConfig
    _enable_filter_intent: bool
    _filter_intent_words: frozenset[str]
    _filter_values: dict[str, list]
    _field_values_cache: dict[tuple[str, ...], dict[str, list]]
    _filter_intent_chain: Any
    _field_priority_chain: Any
    _select_collection_chain: Any
    _custom_instructions: str
    _field_priority: dict[str, int]
    _field_priority_unranked: list[str]
    _field_priority_sample: list[dict]

    if TYPE_CHECKING:

        def _sys(self, prompt: str) -> SystemMessage: ...

        def _schema_signature(self, sample: list[dict]) -> str: ...

    def _sample_field_values(
        self, fields: list[str], limit: int = 100
    ) -> dict[str, list]:
        cache_key = tuple(fields)
        cached = self._field_values_cache.get(cache_key)
        if cached is not None:
            return cached
        sample_limit = max(limit, 500)
        field_values: dict[str, list] = {}
        for field in fields:
            samples = self.backend.sample_documents(
                limit=sample_limit, attributes_to_retrieve=[field]
            )
            seen: set[str] = set()
            values: list = []
            for doc in samples:
                v = doc.get(field)
                if v is None:
                    continue
                for item in v if isinstance(v, list) else [v]:
                    k = str(item)
                    if k not in seen:
                        seen.add(k)
                        values.append(item)
            if values:
                field_values[field] = values[:limit]
        self._field_values_cache[cache_key] = field_values
        return field_values

    async def _adetect_filter_intent(self, question: str) -> FilterIntent:
        if not self._enable_filter_intent:
            return FilterIntent(field=None, value="", operator="")
        config = self._index_config
        filterable = config.filterable_attributes
        if not filterable:
            return FilterIntent(field=None, value="", operator="")

        _GENERIC = {"id", "_id", "content", "text", "body", "document", "doc"}
        if all(f.lower() in _GENERIC for f in filterable):
            return FilterIntent(field=None, value="", operator="")

        words = question.strip().split()

        _is_product_code = re.compile(r"^[a-z]\d{2,}$")
        has_strong_signal = any(
            (w and w[0].isdigit())
            or (len(w) >= 3 and w.isupper())
            or bool(_is_product_code.match(w))
            for w in words
        )
        has_weak_capital_signal = any(
            i > 0 and w and w[0].isupper() and not w.isupper()
            for i, w in enumerate(words)
        )
        has_filter_word = any(w.lower() in self._filter_intent_words for w in words)

        if len(words) <= 3:
            if not has_filter_word:
                return FilterIntent(field=None, value="", operator="")
        elif not (has_strong_signal or has_weak_capital_signal or has_filter_word):
            return FilterIntent(field=None, value="", operator="")

        cached = _cache.load("filter-intent-v12", tuple(filterable), question)
        if cached:
            try:
                return FilterIntent.model_validate(cached)
            except Exception:
                pass

        if self._filter_values:
            field_values: dict[str, list] = {
                field: list(self._filter_values.get(field, []))
                for field in filterable
                if self._filter_values.get(field)
            }
        else:
            loop = asyncio.get_running_loop()
            field_values = await loop.run_in_executor(
                None, self._sample_field_values, filterable
            )

        values_block = "\n".join(
            f"  {field}: {vals[:10]}" for field, vals in field_values.items()
        )

        try:
            result = cast(
                FilterIntent,
                await self._filter_intent_chain.ainvoke(
                    [
                        self._sys(
                            prompts.filter_intent(
                                filterable,
                                values_block,
                                self._custom_instructions,
                            )
                        ),
                        HumanMessage(question),
                    ]
                ),
            )
            _cache.save(
                "filter-intent-v12",
                tuple(filterable),
                question,
                value=result.model_dump(),
            )
            return result
        except Exception:
            return FilterIntent(field=None, value="", operator="")

    async def _aselect_collections(self, question: str) -> list[str]:
        if not self.collections:
            return []
        all_names = list(self.collections.keys())
        if len(all_names) == 1:
            return all_names

        descriptions = "\n".join(
            f"- {name}: {self.collection_descriptions.get(name, name)}"
            for name in all_names
        )
        cached = _cache.load("select-collection-v1", tuple(all_names), question)
        if cached:
            try:
                filtered = [n for n in cached if n in self.collections]
                if filtered:
                    return filtered
            except Exception:
                pass
        try:
            result = cast(
                CollectionIntent,
                await self._select_collection_chain.ainvoke(
                    [
                        self._sys(prompts.collection_select(descriptions)),
                        HumanMessage(question),
                    ]
                ),
            )
            filtered = [n for n in result.collections if n in self.collections]
            if filtered:
                _cache.save(
                    "select-collection-v1", tuple(all_names), question, value=filtered
                )
                return filtered
            return all_names
        except Exception:
            return all_names

    async def _aensure_field_priority(self) -> None:
        unranked = getattr(self, "_field_priority_unranked", None)
        if not unranked:
            return
        sample = getattr(self, "_field_priority_sample", [])
        try:
            schema_sig = self._schema_signature(sample) if sample else None
        except Exception:
            schema_sig = None
        fields_for_llm = list(unranked)
        self._field_priority_unranked = []
        try:
            blocks = []
            for f in fields_for_llm:
                vals = []
                for d in sample:
                    v = d.get(f)
                    if v is None or v == "":
                        continue
                    s = str(v)
                    vals.append(s[:120])
                    if len(vals) >= 3:
                        break
                blocks.append(f"  {f}: {vals}")
            fields_block = "\n".join(blocks)
            result = cast(
                FieldPriority,
                await self._field_priority_chain.ainvoke(
                    [
                        self._sys(
                            prompts.field_priority(
                                fields_block, self._custom_instructions
                            )
                        ),
                        HumanMessage(f"Rank these {len(fields_for_llm)} fields."),
                    ]
                ),
            )
            ranks = {
                fr.name: int(fr.rank)
                for fr in result.ranks
                if fr.name in fields_for_llm
            }
            offset = len(self._index_config.searchable_attributes)
            for f, r in ranks.items():
                self._field_priority[f] = offset + max(0, min(9, r))
            if schema_sig:
                field_rank_cache_save(schema_sig, ranks)
        except Exception:
            pass
