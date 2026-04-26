from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.documents import Document

_HTML_TAG_RE = re.compile(r"<[^>]+>")

_MAX_LIST_ITEMS = 8
"""Cap bulky list values (stock_data per-warehouse, etc.) at this many
items when rendering for the reranker. Spec lists like akeneo_values
rarely exceed this count; warehouse/location arrays can be 20+ and
would otherwise push downstream fields past the rerank_chars cutoff."""


def _clean_string(s: str) -> str:
    return re.sub(r"\s+", " ", _HTML_TAG_RE.sub(" ", s)).strip()


def _doc_to_grader_text(doc: Document) -> str:
    return doc.page_content


def _render_value(v: Any, indent: int = 0) -> str:
    if isinstance(v, str):
        return _clean_string(v)
    if isinstance(v, bool) or isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, dict):
        lines: list[str] = []
        for k, val in v.items():
            rendered = _render_value(val, indent + 1)
            if not rendered:
                continue
            if "\n" in rendered:
                lines.append(f"- **{k}**:\n{rendered}")
            else:
                lines.append(f"- **{k}**: {rendered}")
        return "\n".join(lines)
    if isinstance(v, (list, tuple)):
        lines = []
        items = list(v)
        truncated = len(items) > _MAX_LIST_ITEMS
        for item in items[:_MAX_LIST_ITEMS]:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[0], str)
            ):
                rendered = _render_value(item[1], indent + 1)
                if not rendered:
                    continue
                if "\n" in rendered:
                    lines.append(f"- **{item[0]}**:\n{rendered}")
                else:
                    lines.append(f"- **{item[0]}**: {rendered}")
            else:
                rendered = _render_value(item, indent + 1)
                if rendered:
                    lines.append(f"- {rendered}")
        if truncated:
            lines.append(f"- (…{len(items) - _MAX_LIST_ITEMS} more)")
        return "\n".join(lines)
    return _clean_string(str(v))
