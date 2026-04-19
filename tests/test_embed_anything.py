"""Tests for EmbedAnything integration (local embeddings + reranking, no API keys).

Verifies that embed-anything can be used as a drop-in embed_fn and reranker
for AgenticRAG, running fully offline with small HuggingFace models.

Run: uv run --with embed-anything pytest tests/test_embed_anything.py -v
"""

from __future__ import annotations

import os

import pytest

ea = pytest.importorskip("embed_anything", reason="embed-anything not installed")

os.environ.setdefault("OPENAI_API_KEY", "test-key-unused")

from rag7 import AgenticRAG, InMemoryBackend  # noqa: E402
from rag7.backend import SearchRequest  # noqa: E402
from rag7.embedder import EmbedAnythingEmbedder  # noqa: E402
from rag7.rerankers import EmbedAnythingReranker  # noqa: E402
from rag7.models import RerankResult  # noqa: E402

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_ID = "jinaai/jina-reranker-v1-turbo-en"


@pytest.fixture(scope="module")
def embedder():
    return EmbedAnythingEmbedder(MODEL_ID)


@pytest.fixture(scope="module")
def embed_fn(embedder):
    return embedder


# ── Embedding tests ──────────────────────────────────────────────────────────


class TestEmbedAnythingEmbeddings:
    def test_embed_returns_vector(self, embed_fn):
        vec = embed_fn("Python is a programming language")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_embed_deterministic(self, embed_fn):
        v1 = embed_fn("test query")
        v2 = embed_fn("test query")
        assert v1 == pytest.approx(v2, abs=1e-6)

    def test_embed_different_texts_differ(self, embed_fn):
        v1 = embed_fn("Python programming")
        v2 = embed_fn("Cats and dogs")
        assert v1 != pytest.approx(v2, abs=0.1)

    def test_embed_dimension_consistent(self, embed_fn):
        v1 = embed_fn("short")
        v2 = embed_fn("a much longer sentence about many different topics")
        assert len(v1) == len(v2)

    def test_embed_method_matches_call(self, embedder):
        v1 = embedder.embed("hello world")
        v2 = embedder("hello world")
        assert v1 == pytest.approx(v2, abs=1e-6)

    def test_repr(self, embedder):
        assert MODEL_ID in repr(embedder)


# ── AgenticRAG integration ───────────────────────────────────────────────────


class TestEmbedAnythingWithAgenticRAG:
    def test_agent_creation_with_embed_fn(self, embed_fn):
        backend = InMemoryBackend(
            documents=[
                {"id": "1", "content": "Python is a programming language"},
                {"id": "2", "content": "Cats sit on mats"},
            ]
        )
        rag = AgenticRAG(
            "test",
            backend=backend,
            embed_fn=embed_fn,
            embedder_name="embed-anything",
            auto_strategy=False,
        )
        assert rag.embed_fn is embed_fn

    def test_search_request_with_vector(self, embed_fn):
        vec = embed_fn("programming")
        assert len(vec) > 0
        req = SearchRequest(query="programming", limit=5, vector=vec)
        assert req.vector == vec


# ── Reranker tests ───────────────────────────────────────────────────────────


class TestEmbedAnythingReranker:
    @pytest.fixture(scope="class")
    def reranker(self):
        try:
            return EmbedAnythingReranker(RERANKER_ID)
        except Exception:
            pytest.skip("Reranker model not available (needs ONNX weights)")

    def test_rerank_returns_results(self, reranker):
        results = reranker.rerank(
            "programming language",
            [
                "Python is a programming language",
                "Cats sit on mats",
                "Java is used for enterprise software",
            ],
            3,
        )
        assert len(results) > 0
        assert all(isinstance(r, RerankResult) for r in results)
        assert all(isinstance(r.relevance_score, float) for r in results)
        assert all(isinstance(r.index, int) for r in results)

    def test_rerank_ordering(self, reranker):
        documents = [
            "Cats sit on mats",
            "Python is a programming language",
            "Dogs are loyal companions",
        ]
        results = reranker.rerank("programming language", documents, 3)
        assert results[0].index == 1, "Python doc should rank first"

    def test_rerank_empty_documents(self, reranker):
        results = reranker.rerank("test", [], 5)
        assert results == []

    def test_rerank_top_n_limits(self, reranker):
        docs = ["doc a", "doc b", "doc c", "doc d"]
        results = reranker.rerank("test", docs, 2)
        assert len(results) <= 2

    def test_repr(self, reranker):
        assert RERANKER_ID in repr(reranker)


# ── Mix-and-match integration ────────────────────────────────────────────────


class TestMixAndMatch:
    """Verify embed-anything embedder + reranker work together with AgenticRAG."""

    def test_embedder_plus_reranker_with_agent(self, embed_fn):
        try:
            reranker = EmbedAnythingReranker(RERANKER_ID)
        except Exception:
            pytest.skip("Reranker model not available")

        backend = InMemoryBackend(
            documents=[
                {"id": "1", "content": "Python is a programming language"},
                {"id": "2", "content": "Cats sit on mats"},
                {"id": "3", "content": "Java is used for enterprise"},
            ]
        )
        rag = AgenticRAG(
            "test",
            backend=backend,
            embed_fn=embed_fn,
            embedder_name="embed-anything",
            reranker=reranker,
            auto_strategy=False,
        )
        assert rag.embed_fn is embed_fn
        assert rag._reranker is reranker

    def test_embedder_callable_protocol(self, embed_fn):
        assert callable(embed_fn)
        vec = embed_fn("test")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)
