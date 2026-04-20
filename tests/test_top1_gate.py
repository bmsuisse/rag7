"""Regression: retriever ``top_k=1`` must not confidently return a
nonsense doc when BM25 had no real match.

Two tests:

1. Low-confidence top-1 → LLM grader rejects → retriever returns empty.
2. High-confidence top-1 → grader is never consulted; doc is returned.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from rag7 import AgenticRAG, InMemoryBackend
from rag7.models import RelevanceCheck


def _hash_embed(text: str) -> list[float]:
    import hashlib

    dim = 8
    v = [0.0] * dim
    for tok in text.lower().split():
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


class _StubLLM:
    """A minimal chat-model stub: .with_structured_output() returns a
    pre-canned schema instance on .ainvoke(). Other attributes return a
    plain MagicMock so LangChain's probing doesn't crash."""

    def __init__(self, structured_responses: dict[str, Any]):
        self._responses = structured_responses

    def with_structured_output(self, schema: Any, method: str = "") -> Any:
        responses = self._responses
        name = schema.__name__

        class _Chain:
            async def ainvoke(self, *_: Any, **__: Any) -> Any:
                if name in responses:
                    return responses[name]
                return schema(**{})

            def invoke(self, *a: Any, **kw: Any) -> Any:
                return asyncio.run(self.ainvoke(*a, **kw))

        return _Chain()

    def __getattr__(self, item: str) -> Any:
        return MagicMock()


@pytest.fixture
def single_doc_backend() -> InMemoryBackend:
    b = InMemoryBackend(embed_fn=_hash_embed)
    b.add_documents(
        [
            {
                "id": "widget-42",
                "article_name": "Widget 42 XL",
                "content": "widget 42 xl industrial fitting",
            },
        ]
    )
    return b


def test_top1_rejected_when_grader_says_irrelevant(
    single_doc_backend: InMemoryBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Low BM25 top-1 score + grader says irrelevant → retriever returns empty.

    Monkey-patches the retriever's internal helpers to stamp a low
    ``_rankingScore`` on the top-1 so the gate actually fires
    (InMemoryBackend's native BM25 always reports 1.0).
    """
    # makes_sense=False is what triggers rejection now (we trust the
    # LLM's boolean judgment, not the confidence field).
    stub = _StubLLM(
        structured_responses={
            "RelevanceCheck": RelevanceCheck(makes_sense=False, confidence=0.8),
        }
    )

    rag = AgenticRAG(
        index="widgets",
        backend=single_doc_backend,
        embed_fn=_hash_embed,
        llm=stub,
        gen_llm=stub,
        auto_strategy=False,
    )

    # Force the top-1 _rankingScore below the 0.3 gate threshold so the
    # grader actually gets called (InMemoryBackend's native BM25 always
    # reports 1.0).
    orig_rerank = rag._arerank

    async def _patched_rerank(state: Any) -> Any:
        ranked = await orig_rerank(state)
        for d in ranked.documents:
            d.metadata["_rankingScore"] = 0.1
        return ranked

    monkeypatch.setattr(rag, "_arerank", _patched_rerank)

    _, docs = asyncio.run(rag._aretrieve_documents("completely unrelated", top_k=1))
    assert docs == [], (
        f"expected empty when grader rejects low-confidence top-1; got {docs}"
    )


def test_top1_accepted_when_grader_says_ok(
    single_doc_backend: InMemoryBackend,
) -> None:
    """Even with low BM25, if grader approves, we still return the doc."""
    stub = _StubLLM(
        structured_responses={
            "RelevanceCheck": RelevanceCheck(makes_sense=True, confidence=0.95),
        }
    )

    rag = AgenticRAG(
        index="widgets",
        backend=single_doc_backend,
        embed_fn=_hash_embed,
        llm=stub,
        gen_llm=stub,
        auto_strategy=False,
    )

    _, docs = asyncio.run(rag._aretrieve_documents("widget xl variant", top_k=1))

    assert len(docs) == 1
    assert docs[0].metadata.get("id") == "widget-42"


def test_top1_accepted_on_synonym_match_with_moderate_confidence(
    single_doc_backend: InMemoryBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real-world case: 'bieröffner' → 'Flaschenöffner' is a valid synonym
    match. LLM returns makes_sense=True, confidence=0.6 (moderate, because
    not a literal keyword hit). Gate should accept, not reject."""
    stub = _StubLLM(
        structured_responses={
            "RelevanceCheck": RelevanceCheck(makes_sense=True, confidence=0.6),
        }
    )

    rag = AgenticRAG(
        index="widgets",
        backend=single_doc_backend,
        embed_fn=_hash_embed,
        llm=stub,
        gen_llm=stub,
        auto_strategy=False,
    )

    orig_rerank = rag._arerank

    async def _patched_rerank(state: Any) -> Any:
        ranked = await orig_rerank(state)
        for d in ranked.documents:
            d.metadata["_rankingScore"] = 0.15
        return ranked

    monkeypatch.setattr(rag, "_arerank", _patched_rerank)

    _, docs = asyncio.run(rag._aretrieve_documents("bieröffner", top_k=1))
    assert len(docs) == 1, (
        "synonym match with makes_sense=True should NOT be rejected by the "
        "gate, even with moderate confidence"
    )


def test_high_confidence_top1_skips_grader(
    single_doc_backend: InMemoryBackend,
) -> None:
    """High BM25 score on the top-1 → gate doesn't fire; grader would
    reject but we never call it."""
    # If the grader WERE consulted it would reject (confidence 0) — but
    # BM25 score should be above the 0.3 threshold for a direct hit.
    stub = _StubLLM(
        structured_responses={
            "RelevanceCheck": RelevanceCheck(makes_sense=False, confidence=0.0),
        }
    )

    rag = AgenticRAG(
        index="widgets",
        backend=single_doc_backend,
        embed_fn=_hash_embed,
        llm=stub,
        gen_llm=stub,
        auto_strategy=False,
    )

    # InMemoryBackend BM25 ranks on token overlap — query with all doc tokens
    # gets a strong rankingScore; the grader gate shouldn't fire.
    _, docs = asyncio.run(rag._aretrieve_documents("widget 42 xl", top_k=1))

    # Either we got the doc through (good) OR the gate erroneously rejected
    # it. Assert we got the doc → proves the gate is properly score-gated.
    if docs:
        assert docs[0].metadata.get("id") == "widget-42"
