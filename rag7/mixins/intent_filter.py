from __future__ import annotations

import re as _re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from ..models import FilterIntent, RAGState


class IntentFilterMixin:
    """Document-level checks against a FilterIntent: matching docs to
    intent fields, applying NOT_CONTAINS post-filters in Python, and
    deciding whether the reasoning step is needed.
    """

    rerank_top_n: int

    @staticmethod
    def _doc_matches_intent(doc: Document, intent: FilterIntent) -> bool:
        field, value, op = intent.field, intent.value, intent.operator
        if not field:
            return True
        actual = doc.metadata.get(field)
        if actual is None:
            return op in ("NOT_CONTAINS", "!=")
        actual_str = str(actual).lower()
        value_str = str(value).lower()
        if op == "=":
            return actual_str == value_str
        if op == "CONTAINS":
            return value_str in actual_str
        if op == "NOT_CONTAINS":
            all_values = [value_str] + [v.lower() for v in intent.extra_excludes]
            for val in all_values:
                if val in actual_str:
                    return False
                spaced = _re.sub(r"(?<=[a-z])(?=[A-Z])", " ", val).lower()
                if spaced != val and spaced in actual_str:
                    return False
                ws = val.split()
                if len(ws) >= 3 and " ".join(ws[:2]) in actual_str:
                    return False
                sw = spaced.split()
                if len(sw) >= 3:
                    sp2 = " ".join(sw[:2])
                    if sp2 in actual_str:
                        return False
            return True
        if op == "!=":
            return actual_str != value_str
        return True

    @staticmethod
    def _content_contains_exclusion(text: str, value: str) -> bool:
        low = text.lower()
        val = value.lower()
        if val in low:
            return True
        spaced = _re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value).lower()
        if spaced != val and spaced in low:
            return True
        words = val.split()
        if len(words) >= 3:
            prefix2 = " ".join(words[:2])
            if prefix2 in low:
                return True
            spaced_words = spaced.split()
            if len(spaced_words) >= 3:
                sp2 = " ".join(spaced_words[:2])
                if sp2 != prefix2 and sp2 in low:
                    return True
        return False

    def _apply_intent_post_filter(
        self, docs: list[Document], intent: FilterIntent | None
    ) -> list[Document]:
        if not intent or not docs:
            return docs
        primary_negates = (
            not intent.field
            and intent.operator in ("NOT_CONTAINS", "!=")
            and intent.value
        )
        if primary_negates:
            exclude_vals = [str(intent.value)] + list(intent.extra_excludes)
            docs = [
                d
                for d in docs
                if not any(
                    self._content_contains_exclusion(d.page_content, v)
                    for v in exclude_vals
                )
            ]
        for af in intent.and_filters:
            if af.operator in ("NOT_CONTAINS", "!=") and af.value:
                af_vals = [str(af.value)] + list(af.extra_excludes)
                docs = [
                    d
                    for d in docs
                    if not any(
                        self._content_contains_exclusion(d.page_content, v)
                        for v in af_vals
                    )
                ]
        return docs

    def _needs_reasoning(self, state: RAGState) -> bool:
        if not state.documents:
            return False
        intent = state.filter_intent
        if intent and intent.operator in ("NOT_CONTAINS", "!="):
            exclude_vals = [str(intent.value)] + list(intent.extra_excludes)
            for d in state.documents[: self.rerank_top_n]:
                if any(
                    self._content_contains_exclusion(d.page_content, v)
                    for v in exclude_vals
                ):
                    return True
        if intent:
            for af in intent.and_filters:
                if af.operator in ("NOT_CONTAINS", "!="):
                    af_vals = [str(af.value)] + list(af.extra_excludes)
                    for d in state.documents[: self.rerank_top_n]:
                        if any(
                            self._content_contains_exclusion(d.page_content, v)
                            for v in af_vals
                        ):
                            return True
        if state.iterations > 1:
            return True
        return False
