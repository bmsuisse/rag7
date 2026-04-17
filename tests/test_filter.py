"""Tests for ``filter`` init param and deprecated ``base_filter`` back-compat.

Covers:
- ``filter`` is threaded into every search request (BM25 + hybrid)
- ``filter`` AND-joins with per-call filters (intent, language, etc.)
- ``base_filter`` kwarg still works but emits ``DeprecationWarning``
- ``rag.base_filter`` read/write both warn and forward to ``rag.filter``
- Passing both ``filter`` and ``base_filter`` → ``filter`` wins
"""

from __future__ import annotations

import asyncio
import os
import warnings
from typing import Any

os.environ.setdefault("OPENAI_API_KEY", "test-key-unused")

from rag7 import AgenticRAG, InMemoryBackend  # noqa: E402
from rag7.backend import IndexConfig, SearchRequest  # noqa: E402


class _RecordingBackend(InMemoryBackend):
    """Capture every SearchRequest the agent emits, without hitting a real index."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[SearchRequest] = []

    def batch_search(self, requests):  # type: ignore[override]
        self.requests.extend(requests)
        return [[{"id": "1", "_rankingScore": 0.9, "content": "doc"}] for _ in requests]

    def search(self, req):  # type: ignore[override]
        self.requests.append(req)
        return [{"id": "1", "_rankingScore": 0.9, "content": "doc"}]

    def get_index_config(self) -> IndexConfig:
        return IndexConfig()


def _agent(backend: Any, **kw: Any) -> AgenticRAG:
    return AgenticRAG("test", backend=backend, auto_strategy=False, **kw)


# ── filter is applied to every search ────────────────────────────────────────


def test_filter_threaded_into_request() -> None:
    backend = _RecordingBackend()
    rag = _agent(backend, filter="brand = 'Bosch'")
    asyncio.run(rag._afast_keyword_retrieve("drill", limit=5))
    assert backend.requests, "no request recorded"
    assert backend.requests[0].filter_expr == "brand = 'Bosch'"


def test_filter_none_means_no_filter() -> None:
    backend = _RecordingBackend()
    rag = _agent(backend)
    asyncio.run(rag._afast_keyword_retrieve("drill", limit=5))
    assert backend.requests[0].filter_expr is None


def test_filter_and_joins_with_per_call_filter() -> None:
    backend = _RecordingBackend()
    rag = _agent(backend, filter="brand = 'Bosch'")
    req = rag._make_search_request("x", 5, filter_expr="in_stock = true")
    assert req.filter_expr == "brand = 'Bosch' AND in_stock = true"


def test_filter_accessible_as_attribute() -> None:
    rag = _agent(_RecordingBackend(), filter="category = 'tools'")
    assert rag.filter == "category = 'tools'"


# ── base_filter back-compat ──────────────────────────────────────────────────


def test_base_filter_kwarg_forwards_to_filter() -> None:
    backend = _RecordingBackend()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rag = _agent(backend, base_filter="brand = 'Bosch'")
    assert rag.filter == "brand = 'Bosch'"
    assert any(
        issubclass(w.category, DeprecationWarning) and "base_filter" in str(w.message)
        for w in caught
    ), "expected DeprecationWarning mentioning base_filter"


def test_base_filter_property_read_warns() -> None:
    rag = _agent(_RecordingBackend(), filter="brand = 'Bosch'")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = rag.base_filter
    assert value == "brand = 'Bosch'"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_base_filter_property_write_warns_and_updates_filter() -> None:
    rag = _agent(_RecordingBackend(), filter="brand = 'Bosch'")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rag.base_filter = "brand = 'Makita'"
    assert rag.filter == "brand = 'Makita'"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_filter_wins_when_both_passed() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        rag = _agent(
            _RecordingBackend(),
            filter="brand = 'Bosch'",
            base_filter="brand = 'Makita'",
        )
    assert rag.filter == "brand = 'Bosch'"


def test_no_warning_when_only_new_api_used() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _agent(_RecordingBackend(), filter="brand = 'Bosch'")
    base_filter_warnings = [w for w in caught if "base_filter" in str(w.message)]
    assert not base_filter_warnings, (
        f"new API should not emit base_filter warning, got: {base_filter_warnings}"
    )
