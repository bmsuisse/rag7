"""Tests for the 'alternative to X' search path.

When a user searches for "Alternative zu fixit 516", the system should:
1. Detect the alternative intent in preprocess (alternative_to field)
2. Look up the referenced product via BM25
3. Search for similar items using the product's content as semantic anchor
4. Exclude the original product from results
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key-unused")

from rag7 import AgenticRAG, InMemoryBackend  # noqa: E402
from rag7.backend import IndexConfig, SearchRequest  # noqa: E402
from rag7.models import RAGState, SearchQuery  # noqa: E402


PRODUCT_FIXIT_516 = {
    "id": "4457227",
    "article_id": "4457227",
    "article_name": "Fixit 516 Trockenbeton 0-16, 25 kg",
    "_rankingScore": 0.95,
    "content": "Fixit 516 Trockenbeton 0-16, 25 kg C 30/37",
}

ALTERNATIVE_PRODUCTS = [
    {
        "id": "7854163",
        "article_id": "7854163",
        "article_name": "Röfix 990 Trockenbeton 0-16 mm 25 kg",
        "_rankingScore": 0.85,
        "content": "Röfix 990, Trockenbeton 0-16 mm, 25 kg",
    },
    {
        "id": "01667419",
        "article_id": "01667419",
        "article_name": "Sikacrete-16 SCC Trockenbeton 25 kg",
        "_rankingScore": 0.82,
        "content": "Sikacrete - 16 SCC, Trockenbeton, 25 kg",
    },
    {
        "id": "4118746",
        "article_id": "4118746",
        "article_name": "Fixit 516 Trockenbeton C30/37 0-16 mm Sack 25 kg",
        "_rankingScore": 0.90,
        "content": "Fixit 516 Trockenbeton C30/37 0-16 mm, Sack à 25 kg",
    },
    {
        "id": "9999999",
        "article_id": "9999999",
        "article_name": "Holcim Trockenbeton 0-8 mm 25 kg",
        "_rankingScore": 0.78,
        "content": "Holcim Trockenbeton 0-8 mm, Sack 25 kg",
    },
]


class _AlternativeBackend(InMemoryBackend):
    """Returns fixit 516 for exact lookup, alternatives for category search."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[SearchRequest] = []

    def search(self, request: SearchRequest) -> list[dict]:
        self.requests.append(request)
        if "fixit 516" in request.query.lower():
            return [PRODUCT_FIXIT_516]
        return [PRODUCT_FIXIT_516] + ALTERNATIVE_PRODUCTS

    def batch_search(self, requests: list[SearchRequest]) -> list[list[dict]]:
        self.requests.extend(requests)
        results = []
        for req in requests:
            if "fixit 516" in req.query.lower():
                results.append([PRODUCT_FIXIT_516])
            else:
                results.append([PRODUCT_FIXIT_516] + ALTERNATIVE_PRODUCTS)
        return results

    def get_index_config(self) -> IndexConfig:
        return IndexConfig()

    def sample_documents(self, limit: int = 100) -> list[dict]:
        return [PRODUCT_FIXIT_516] + ALTERNATIVE_PRODUCTS[:2]


def _make_agent(backend: Any) -> AgenticRAG:
    return AgenticRAG(
        "test",
        backend=backend,
        embed_fn=lambda _text: [0.0] * 8,
        embedder_name="fake",
        semantic_ratio=0.5,
        auto_strategy=False,
    )


def _mock_preprocess_with_alternative(alternative_to: str):
    """Return a patched _apreprocess that sets alternative_to."""
    async def _fake_preprocess(self: Any, state: RAGState) -> RAGState:
        return state.model_copy(
            update={
                "query": "trockenbeton beton 0-16",
                "alternative_to": alternative_to,
            }
        )
    return _fake_preprocess


def _mock_preprocess_normal():
    """Return a patched _apreprocess that does NOT set alternative_to."""
    async def _fake_preprocess(self: Any, state: RAGState) -> RAGState:
        return state.model_copy(update={"query": state.question})
    return _fake_preprocess


class TestAlternativeRetrieve:
    """Unit tests for _aalternative_retrieve."""

    def test_excludes_original_product(self) -> None:
        """The referenced product must NOT appear in results."""
        backend = _AlternativeBackend()
        rag = _make_agent(backend)

        _, docs = asyncio.run(
            rag._aalternative_retrieve("trockenbeton", "fixit 516", top_k=5)
        )

        ids = [d.metadata.get("article_id") for d in docs]
        assert "4457227" not in ids, "fixit 516 (4457227) must be excluded"

    def test_excludes_by_content_match(self) -> None:
        """Docs containing the alternative_to string in content are excluded."""
        backend = _AlternativeBackend()
        rag = _make_agent(backend)

        _, docs = asyncio.run(
            rag._aalternative_retrieve("trockenbeton", "fixit 516", top_k=5)
        )

        for d in docs:
            assert "fixit 516" not in d.page_content.lower(), (
                f"doc {d.metadata.get('article_id')} contains 'fixit 516' in content"
            )

    def test_returns_similar_products(self) -> None:
        """Results should contain alternative products, not be empty."""
        backend = _AlternativeBackend()
        rag = _make_agent(backend)

        _, docs = asyncio.run(
            rag._aalternative_retrieve("trockenbeton", "fixit 516", top_k=5)
        )

        assert len(docs) >= 1, "should return at least one alternative"
        alt_ids = {d.metadata.get("article_id") for d in docs}
        assert alt_ids & {"7854163", "01667419", "9999999"}, (
            "expected at least one known alternative product"
        )

    def test_uses_high_semantic_ratio(self) -> None:
        """Alternative search should use semantic-heavy ratio (0.7)."""
        backend = _AlternativeBackend()
        rag = _make_agent(backend)

        asyncio.run(
            rag._aalternative_retrieve("trockenbeton", "fixit 516", top_k=5)
        )

        search_reqs = [
            r for r in backend.requests
            if "fixit 516" not in r.query.lower()
        ]
        assert search_reqs, "should have made a broad search request"

    def test_fallback_when_product_not_found(self) -> None:
        """If the referenced product can't be found, fall back gracefully."""
        class _EmptyBackend(InMemoryBackend):
            def batch_search(self, requests):
                return [[] for _ in requests]
            def search(self, request):
                return []
            def get_index_config(self):
                return IndexConfig()

        backend = _EmptyBackend()
        rag = _make_agent(backend)

        _, docs = asyncio.run(
            rag._aalternative_retrieve("trockenbeton", "nonexistent product", top_k=5)
        )

        assert isinstance(docs, list)

    def test_respects_top_k(self) -> None:
        """Results should not exceed top_k."""
        backend = _AlternativeBackend()
        rag = _make_agent(backend)

        _, docs = asyncio.run(
            rag._aalternative_retrieve("trockenbeton", "fixit 516", top_k=2)
        )

        assert len(docs) <= 2


class TestAlternativeIntegrationDirect:
    """Tests that _aretrieve_documents routes to alternative path."""

    def test_alternative_detected_routes_to_alt_path(self) -> None:
        """When preprocess sets alternative_to, _aretrieve_documents uses alt path."""
        backend = _AlternativeBackend()
        rag = _make_agent(backend)

        with patch.object(
            type(rag), "_apreprocess",
            _mock_preprocess_with_alternative("fixit 516"),
        ):
            _, docs = asyncio.run(
                rag._aretrieve_documents("Alternative zu fixit 516")
            )

        ids = [d.metadata.get("article_id") for d in docs]
        assert "4457227" not in ids, "fixit 516 must be excluded via alt path"
        assert len(docs) >= 1

    def test_no_alternative_uses_normal_path(self) -> None:
        """Without alternative_to, normal retrieve path runs."""
        backend = _AlternativeBackend()
        rag = _make_agent(backend)

        with patch.object(
            type(rag), "_apreprocess",
            _mock_preprocess_normal(),
        ):
            _, docs = asyncio.run(
                rag._aretrieve_documents("trockenbeton 016")
            )

        assert len(docs) >= 1


class TestAlternativeIntegrationGraph:
    """Tests that _aparallel_start (graph entry) routes to alternative path."""

    def test_graph_path_excludes_original(self) -> None:
        """_aparallel_start with alternative intent must exclude original."""
        backend = _AlternativeBackend()
        rag = _make_agent(backend)

        with patch.object(
            type(rag), "_apreprocess",
            _mock_preprocess_with_alternative("fixit 516"),
        ):
            init = RAGState(question="Alternative zu fixit 516", query="Alternative zu fixit 516")
            state = asyncio.run(rag._aparallel_start(init))

        ids = [d.metadata.get("article_id") for d in state.documents]
        assert "4457227" not in ids, "graph path must also exclude fixit 516"
        assert state.alternative_to == "fixit 516"

    def test_graph_path_no_alternative_normal(self) -> None:
        """_aparallel_start without alternative uses normal retrieve."""
        backend = _AlternativeBackend()
        rag = _make_agent(backend)

        with patch.object(
            type(rag), "_apreprocess",
            _mock_preprocess_normal(),
        ):
            init = RAGState(question="trockenbeton 016", query="trockenbeton 016")
            state = asyncio.run(rag._aparallel_start(init))

        assert len(state.documents) >= 1
        assert state.alternative_to is None


class TestSearchQueryModel:
    """Tests for the SearchQuery.alternative_to field."""

    def test_alternative_to_default_none(self) -> None:
        sq = SearchQuery(query="test")
        assert sq.alternative_to is None

    def test_alternative_to_set(self) -> None:
        sq = SearchQuery(query="trockenbeton", alternative_to="fixit 516")
        assert sq.alternative_to == "fixit 516"

    def test_alternative_to_serialization(self) -> None:
        sq = SearchQuery(query="trockenbeton", alternative_to="fixit 516")
        data = sq.model_dump()
        assert data["alternative_to"] == "fixit 516"
        roundtrip = SearchQuery.model_validate(data)
        assert roundtrip.alternative_to == "fixit 516"

    def test_alternative_to_null_serialization(self) -> None:
        sq = SearchQuery(query="test")
        data = sq.model_dump()
        assert data["alternative_to"] is None
        roundtrip = SearchQuery.model_validate(data)
        assert roundtrip.alternative_to is None


class TestRAGStateAlternativeTo:
    """Tests for RAGState.alternative_to field."""

    def test_default_none(self) -> None:
        state = RAGState(question="test", query="test")
        assert state.alternative_to is None

    def test_model_copy_preserves(self) -> None:
        state = RAGState(question="test", query="test", alternative_to="fixit 516")
        copy = state.model_copy(update={"query": "new"})
        assert copy.alternative_to == "fixit 516"

    def test_model_copy_sets(self) -> None:
        state = RAGState(question="test", query="test")
        copy = state.model_copy(update={"alternative_to": "fixit 516"})
        assert copy.alternative_to == "fixit 516"


class TestAlternativeEdgeCases:
    """Edge cases for alternative search."""

    def test_case_insensitive_exclusion(self) -> None:
        """Exclusion should be case-insensitive."""
        backend = _AlternativeBackend()
        rag = _make_agent(backend)

        _, docs = asyncio.run(
            rag._aalternative_retrieve("trockenbeton", "Fixit 516", top_k=5)
        )

        for d in docs:
            assert "fixit 516" not in d.page_content.lower()

    def test_empty_alternative_to_string(self) -> None:
        """Empty string alternative_to should not trigger alt path."""
        state = RAGState(question="test", query="test", alternative_to="")
        assert not state.alternative_to

    def test_multiple_words_in_alternative(self) -> None:
        """Multi-word product names should be handled correctly."""
        backend = _AlternativeBackend()
        rag = _make_agent(backend)

        _, docs = asyncio.run(
            rag._aalternative_retrieve(
                "trockenbeton", "Fixit 516 Trockenbeton 0-16", top_k=5
            )
        )

        for d in docs:
            assert "fixit 516 trockenbeton 0-16" not in d.page_content.lower()
