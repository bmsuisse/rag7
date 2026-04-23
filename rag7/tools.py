from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool

from .backend import SearchRequest
from .utils import _SKIP_FIELDS

if TYPE_CHECKING:
    from .core import AgenticRAG


class RAGToolset:
    def __init__(self, rag: AgenticRAG) -> None:
        self._rag = rag

    def get_index_settings(self, index_name: str = "") -> dict:
        try:
            config = self._rag.backend.get_index_config()
            samples = self._rag.backend.sample_documents(limit=20)

            _ID_SUFFIXES = ("_id", "_code", "_key", "_num")
            sortable = config.sortable_attributes
            bool_signals: list[str] = []
            num_signals: list[str] = []

            if sortable and samples:
                field_sample: dict[str, Any] = {}
                for doc in samples:
                    for f in sortable:
                        if f not in field_sample and doc.get(f) is not None:
                            field_sample[f] = doc[f]
                for f in sortable:
                    v = field_sample.get(f)
                    if isinstance(v, bool):
                        bool_signals.append(f)
                    elif (
                        isinstance(v, (int, float))
                        and v >= 0
                        and not any(f.endswith(s) for s in _ID_SUFFIXES)
                    ):
                        num_signals.append(f)

            return {
                "filterable": config.filterable_attributes,
                "searchable": config.searchable_attributes,
                "sortable": config.sortable_attributes,
                "ranking_rules": config.ranking_rules,
                "embedders": config.embedders,
                "boost_fields": bool_signals + num_signals,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def search_bm25(
        self, query: str, filter_expr: str = "", top_k: int = 20
    ) -> list[dict]:
        limit = top_k * self._rag.retrieval_factor
        hits = self._rag.backend.search(
            SearchRequest(
                query=query,
                limit=limit,
                filter_expr=filter_expr or None,
                index_uid=self._rag.index,
            )
        )
        return hits[:top_k]

    def search_hybrid(
        self,
        query: str,
        filter_expr: str = "",
        semantic_ratio: float = 0.7,
        top_k: int = 20,
        sort_fields: list[str] | None = None,
    ) -> list[dict]:
        limit = top_k * self._rag.retrieval_factor

        vector: list[float] | None = None
        if self._rag.embed_fn:
            try:
                vector = self._rag.embed_fn(query)
            except Exception:
                vector = None

        hits = self._rag.backend.search(
            SearchRequest(
                query=query,
                limit=limit,
                vector=vector,
                semantic_ratio=semantic_ratio,
                filter_expr=filter_expr or None,
                sort_fields=sort_fields,
                embedder_name=self._rag.embedder_name,
                index_uid=self._rag.index,
            )
        )
        return hits[:top_k]

    def get_filter_values(self, field: str, sample_limit: int = 30) -> dict:
        samples = self._rag.backend.sample_documents(
            limit=sample_limit,
            attributes_to_retrieve=[field],
        )

        values: list = []
        seen: set = set()
        for h in samples:
            v = h.get(field)
            if v is None:
                continue
            for item in v if isinstance(v, list) else [v]:
                k = str(item)
                if k not in seen:
                    seen.add(k)
                    values.append(item)

        return {
            "field": field,
            "values": values[:sample_limit],
            "contains_supported": True,
            "filter_hint": (
                f'exact:    {field} = "value"  |  '
                f'partial:  {field} CONTAINS "partial"  |  '
                f'multiple: {field} IN ["a", "b"]'
            ),
        }

    def rerank_results(
        self, query: str, hits: list[dict], top_n: int = 10
    ) -> list[dict]:
        if not hits:
            return []
        docs = [
            " | ".join(
                f"{k}: {v}"
                for k, v in h.items()
                if isinstance(v, (str, int, float)) and v and k not in _SKIP_FIELDS
            )
            for h in hits
        ]
        results = self._rag._reranker.rerank(
            query=query,
            documents=docs,
            top_n=top_n,
        )
        return [hits[r.index] for r in results]

    def as_tools(self) -> list[StructuredTool]:
        tool_methods = [
            self.get_index_settings,
            self.search_bm25,
            self.search_hybrid,
            self.get_filter_values,
            self.rerank_results,
        ]
        return [
            StructuredTool.from_function(
                func=method,
                name=method.__name__,
                description=method.__doc__ or "",
            )
            for method in tool_methods
        ]
