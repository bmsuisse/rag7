"""Regression test: fast keyword path must be pure BM25 (semantic_ratio=0).

Bug: with an embedder configured and default `semantic_ratio>0`, Meilisearch
hybrid mode would return 0 hits for keyword-only queries like "fixit 516"
because the vector+bm25 fusion pushed relevant docs out of the top-k window.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-unused")

from rag7 import AgenticRAG, InMemoryBackend  # noqa: E402
from rag7.backend import IndexConfig, SearchRequest  # noqa: E402


class _RecordingBackend(InMemoryBackend):
    """InMemoryBackend that records the SearchRequest objects it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[SearchRequest] = []

    def batch_search(self, requests):  # type: ignore[override]
        self.requests.extend(requests)
        # Return a single canned hit per request so fast path sees docs
        return [
            [{"id": "1", "_rankingScore": 0.95, "content": "Fixit 516"}]
            for _ in requests
        ]

    def get_index_config(self) -> IndexConfig:
        return IndexConfig()


def _make_agent(backend: Any) -> AgenticRAG:
    return AgenticRAG(
        "test",
        backend=backend,
        embed_fn=lambda _text: [0.0] * 8,
        embedder_name="fake",
        semantic_ratio=0.5,
        auto_strategy=False,
    )


def test_fast_keyword_forces_pure_bm25() -> None:
    """`_afast_keyword_retrieve` must request semantic_ratio=0 regardless of
    the agent's default hybrid ratio."""
    backend = _RecordingBackend()
    rag = _make_agent(backend)

    docs = asyncio.run(rag._afast_keyword_retrieve("fixit 516", limit=20))
    assert docs, "fast keyword path returned no docs"
    assert backend.requests, "backend received no request"
    req = backend.requests[0]
    assert req.semantic_ratio == 0.0, (
        f"expected pure BM25 (semantic_ratio=0), got {req.semantic_ratio}"
    )
    assert req.show_ranking_score is True
    assert req.query == "fixit 516"


def test_fast_keyword_ranking_score_attached() -> None:
    backend = _RecordingBackend()
    rag = _make_agent(backend)
    docs = asyncio.run(rag._afast_keyword_retrieve("fixit 516", limit=10))
    assert docs[0].metadata.get("_rankingScore") == pytest.approx(0.95)
