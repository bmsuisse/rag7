"""Regression: single-token queries that miss exact BM25 must not
return random high-signal products. Low BM25 confidence should trigger
the swarm-retrieve path (LLM-rewritten query variants) so synonym /
compound-word matches surface.

Real incident: the retriever for "bieröffner" on a catalog that
spells it "Flaschenöffner" returned a random fitting at top-1
because BM25 had no match and the reranker picked a high-sales
unrelated doc. Fix: low-confidence fallback to swarm retrieval.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from rag7 import AgenticRAG, InMemoryBackend
from rag7.models import MultiQuery


def _hash_embed(text: str) -> list[float]:
    """Tiny deterministic embed for testing only.

    Hash each token into a fixed bucket. Two texts sharing tokens get
    overlap; the swarm's LLM-rewritten variants are what lets the
    ``bieröffner`` query reach the ``flaschenöffner`` doc.
    """
    import hashlib

    dim = 16
    v = [0.0] * dim
    for tok in text.lower().split():
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


@pytest.fixture
def catalog() -> InMemoryBackend:
    """A tiny catalog where the synonym spelling doesn't match the query."""
    backend = InMemoryBackend(embed_fn=_hash_embed)
    backend.add_documents(
        [
            {
                "id": "1",
                "article_name": "Flaschenöffner Edelstahl Wandmontage",
                "content": "flaschenöffner edelstahl wandmontage",
            },
            {
                "id": "2",
                "article_name": "Einschraubteil 770.370.106 schwarz",
                "content": "einschraubteil fitting fischer rohr",
            },
            {
                "id": "3",
                "article_name": "Trockenbeton 0-16 mm Sack 25kg",
                "content": "trockenbeton mörtel sack zement",
            },
        ]
    )
    return backend


def _make_llm_stub(multi_query_response: list[str]) -> Any:
    """Build a LangChain-like chat model stub that returns pre-canned outputs."""

    class _StubLLM:
        def invoke(self, *args: Any, **kwargs: Any) -> Any:
            return MagicMock(content="")

        async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
            return MagicMock(content="")

        def with_structured_output(self, schema: Any, method: str = "") -> Any:
            parent = self

            class _Structured:
                async def ainvoke(self, *a: Any, **kw: Any) -> Any:
                    # MultiQuery gets the synonym variants
                    if schema is MultiQuery or schema.__name__ == "MultiQuery":
                        return MultiQuery(queries=multi_query_response)
                    return schema(**{})

                def invoke(self, *a: Any, **kw: Any) -> Any:
                    if schema.__name__ == "MultiQuery":
                        return MultiQuery(queries=multi_query_response)
                    return schema(**{})

            return _Structured()

        def __getattr__(self, name: str) -> Any:
            return MagicMock()

    return _StubLLM()


def test_low_confidence_query_triggers_swarm_path(catalog: InMemoryBackend) -> None:
    """A 'bieröffner'-style single-token query with no exact BM25 match
    should surface the synonym doc via the swarm-rewrite fallback."""
    stub_llm = _make_llm_stub(
        multi_query_response=[
            "flaschenöffner",
            "flaschenöffner edelstahl",
            "bar opener",
        ]
    )

    rag = AgenticRAG(
        index="catalog",
        backend=catalog,
        embed_fn=_hash_embed,
        llm=stub_llm,
        gen_llm=stub_llm,
        auto_strategy=False,
    )

    _, docs = asyncio.run(rag._aretrieve_documents("bieröffner", top_k=3))

    assert docs, "expected non-empty retrieval"
    top_ids = [d.metadata.get("id") for d in docs]
    assert "1" in top_ids, (
        f"expected Flaschenöffner (id=1) in top-3 via swarm fallback; got {top_ids}"
    )


def test_high_confidence_query_does_not_need_swarm(catalog: InMemoryBackend) -> None:
    """Exact BM25 match → skip the swarm call (we check swarm wasn't called
    by asserting we didn't need the stub's rewrite variants)."""
    stub_llm = _make_llm_stub(multi_query_response=["SHOULD NOT BE USED"])

    rag = AgenticRAG(
        index="catalog",
        backend=catalog,
        embed_fn=_hash_embed,
        llm=stub_llm,
        gen_llm=stub_llm,
        auto_strategy=False,
    )

    # "einschraubteil" is a direct hit in content for doc 2
    _, docs = asyncio.run(rag._aretrieve_documents("einschraubteil", top_k=3))

    assert docs
    assert docs[0].metadata.get("id") == "2", (
        f"expected direct BM25 hit for 'einschraubteil'; got {docs[0].metadata}"
    )
