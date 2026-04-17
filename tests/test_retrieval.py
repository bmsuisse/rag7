"""Retrieval quality tests against populated benchmark backends.

Requires: uv run python scripts/populate_backends.py --max-docs 200

Tests verify that relevant documents are returned for known queries
across all backends. Backends skip gracefully if not populated.

Run: uv run pytest tests/test_retrieval.py -v
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from rag7.backend import SearchRequest

# ── Shared config ─────────────────────────────────────────────────────────────

INDEX_NAME = "benchmark-docs"
TABLE_NAME = "benchmark_docs"  # pgvector/duckdb use underscores
EMBED_DIM = 1536
PGVECTOR_DSN = "postgresql://test:test@localhost:5433/testdb"


def _make_embed_fn():
    """Reuse the same Azure OpenAI embedder as populate script."""
    from rag7.utils import _make_azure_embed_fn

    fn = _make_azure_embed_fn()
    if fn is None:
        pytest.skip("No AZURE_OPENAI_ENDPOINT/KEY -- cannot embed queries")
    return fn


# ── Known queries with expected content ───────────────────────────────────────

RETRIEVAL_CASES = [
    {
        "query": "Apple Macintosh operating system",
        "expect_any": ["mac", "apple", "os", "macintosh"],
        "field": "content",
        "limit": 5,
    },
    {
        "query": "Supreme Court India legal judgment",
        "expect_any": ["court", "india", "judgment", "appeal"],
        "field": "content",
        "limit": 5,
    },
    {
        "query": "Osaka Japan city",
        "expect_any": ["osaka", "japan"],
        "field": "content",
        "limit": 5,
    },
]


def _check_retrieval(hits: list[dict], case: dict) -> None:
    """Assert that at least one hit contains expected content."""
    assert len(hits) > 0, f"No results for query: {case['query']}"
    field = case["field"]
    all_text = " ".join(h.get(field, "") for h in hits).lower()
    matched = any(term in all_text for term in case["expect_any"])
    assert matched, (
        f"Query '{case['query']}' returned {len(hits)} hits but none contain "
        f"expected terms {case['expect_any']}.\n"
        f"First hit: {hits[0].get(field, '')[:200]}"
    )


# ── pgvector ──────────────────────────────────────────────────────────────────


def _pgvector_populated() -> bool:
    try:
        import psycopg

        conn = psycopg.connect(PGVECTOR_DSN, connect_timeout=2)
        cur = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        count = cur.fetchone()[0]  # type: ignore[index]
        conn.close()
        return count > 0
    except Exception:
        return False


@pytest.mark.skipif(not _pgvector_populated(), reason="pgvector not populated")
class TestPgvectorRetrieval:
    @pytest.fixture(autouse=True)
    def setup_backend(self):
        from rag7.backend import PgvectorBackend

        self.embed_fn = _make_embed_fn()
        self.backend = PgvectorBackend(
            table=TABLE_NAME,
            dsn=PGVECTOR_DSN,
            embed_fn=self.embed_fn,
            vector_column="embedding",
            content_column="content",
        )

    @pytest.mark.parametrize(
        "case", RETRIEVAL_CASES, ids=[c["query"][:40] for c in RETRIEVAL_CASES]
    )
    def test_vector_search(self, case: dict) -> None:
        req = SearchRequest(query=case["query"], limit=case["limit"])
        hits = self.backend.search(req)
        _check_retrieval(hits, case)

    def test_filter_eq(self) -> None:
        """SQL = operator: exact match."""
        req = SearchRequest(query="data", limit=10, filter_expr="language = 'de'")
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("language") == "de" for h in hits)

    def test_filter_ne(self) -> None:
        """SQL != operator: exclude a value."""
        req = SearchRequest(query="data", limit=10, filter_expr="language != 'de'")
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("language") != "de" for h in hits)

    def test_filter_and(self) -> None:
        """SQL AND: combine two conditions."""
        req = SearchRequest(
            query="data",
            limit=10,
            filter_expr="language = 'en' AND source = 'rteb/mteb/nq'",
        )
        hits = self.backend.search(req)
        if hits:
            assert all(h.get("language") == "en" for h in hits)
            assert all(h.get("source") == "rteb/mteb/nq" for h in hits)

    def test_filter_or(self) -> None:
        """SQL OR: match either condition."""
        req = SearchRequest(
            query="data",
            limit=10,
            filter_expr="language = 'fr' OR language = 'it'",
        )
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("language") in ("fr", "it") for h in hits)

    def test_filter_in(self) -> None:
        """SQL IN: match against a list of values."""
        req = SearchRequest(
            query="data",
            limit=10,
            filter_expr="language IN ('de', 'fr')",
        )
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("language") in ("de", "fr") for h in hits)

    def test_filter_like(self) -> None:
        """SQL LIKE: pattern match on source field (via sample, no vector)."""
        docs = self.backend.sample_documents(
            limit=10, filter_expr="source LIKE 'rteb%'"
        )
        assert len(docs) > 0
        assert all(d.get("source", "").startswith("rteb") for d in docs)

    def test_filter_with_vector(self) -> None:
        """Filter combined with vector search."""
        req = SearchRequest(
            query="court judgment", limit=5, filter_expr="language = 'en'"
        )
        hits = self.backend.search(req)
        if hits:
            assert all(h.get("language") == "en" for h in hits)
            assert "_rankingScore" in hits[0]

    def test_sample_with_filter(self) -> None:
        """sample_documents respects SQL filter."""
        docs = self.backend.sample_documents(limit=5, filter_expr="language = 'de'")
        assert len(docs) > 0
        assert all(d.get("language") == "de" for d in docs)

    def test_ranking_score_present(self) -> None:
        req = SearchRequest(query="chemistry element", limit=3)
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert "_rankingScore" in hits[0]
        assert 0.0 <= hits[0]["_rankingScore"] <= 1.0


# ── Qdrant ────────────────────────────────────────────────────────────────────

QDRANT_PATH = "data/qdrant_benchmark"


def _qdrant_populated() -> bool:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(path=QDRANT_PATH)
        info = client.get_collection(INDEX_NAME)
        client.close()
        return info.points_count > 0
    except Exception:
        return False


@pytest.mark.skipif(not _qdrant_populated(), reason="Qdrant not populated")
class TestQdrantRetrieval:
    @pytest.fixture(autouse=True)
    def setup_backend(self):
        from qdrant_client import QdrantClient

        from rag7.backend import QdrantBackend

        self.embed_fn = _make_embed_fn()
        client = QdrantClient(path=QDRANT_PATH)
        b = QdrantBackend(collection=INDEX_NAME, embed_fn=self.embed_fn)
        b._client = client
        self.backend = b
        yield
        client.close()

    @pytest.mark.parametrize(
        "case", RETRIEVAL_CASES, ids=[c["query"][:40] for c in RETRIEVAL_CASES]
    )
    def test_vector_search(self, case: dict) -> None:
        req = SearchRequest(query=case["query"], limit=case["limit"])
        hits = self.backend.search(req)
        _check_retrieval(hits, case)

    def test_ranking_score_present(self) -> None:
        req = SearchRequest(query="chemistry element", limit=3)
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert "_rankingScore" in hits[0]

    def test_filter_must(self) -> None:
        """Qdrant must filter: exact match on language."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        filt = Filter(
            must=[FieldCondition(key="language", match=MatchValue(value="de"))]
        )
        req = SearchRequest(query="Osaka", limit=10, filter_expr=filt)
        hits = self.backend.search(req)
        if hits:
            assert all(h.get("language") == "de" for h in hits)

    def test_filter_must_not(self) -> None:
        """Qdrant must_not filter: exclude a language."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        filt = Filter(
            must_not=[FieldCondition(key="language", match=MatchValue(value="de"))]
        )
        req = SearchRequest(query="data", limit=10, filter_expr=filt)
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("language") != "de" for h in hits)

    def test_filter_multi_must(self) -> None:
        """Qdrant multiple must conditions (AND)."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        filt = Filter(
            must=[
                FieldCondition(key="language", match=MatchValue(value="en")),
                FieldCondition(key="source", match=MatchValue(value="rteb/mteb/nq")),
            ]
        )
        req = SearchRequest(query="data", limit=10, filter_expr=filt)
        hits = self.backend.search(req)
        if hits:
            assert all(h.get("language") == "en" for h in hits)
            assert all(h.get("source") == "rteb/mteb/nq" for h in hits)

    def test_filter_should(self) -> None:
        """Qdrant should filter (OR-like)."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        filt = Filter(
            should=[
                FieldCondition(key="language", match=MatchValue(value="fr")),
                FieldCondition(key="language", match=MatchValue(value="it")),
            ]
        )
        req = SearchRequest(query="data", limit=10, filter_expr=filt)
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("language") in ("fr", "it") for h in hits)

    def test_filter_with_vector(self) -> None:
        """Filter combined with vector search."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        filt = Filter(
            must=[FieldCondition(key="language", match=MatchValue(value="en"))]
        )
        req = SearchRequest(query="court judgment", limit=5, filter_expr=filt)
        hits = self.backend.search(req)
        if hits:
            assert all(h.get("language") == "en" for h in hits)
            assert "_rankingScore" in hits[0]


# ── LanceDB ───────────────────────────────────────────────────────────────────

LANCEDB_PATH = "data/lancedb_benchmark"


def _lancedb_populated() -> bool:
    try:
        import lancedb

        db = lancedb.connect(LANCEDB_PATH)
        result = db.list_tables()
        names = result.tables if hasattr(result, "tables") else list(result)
        return TABLE_NAME in names
    except Exception:
        return False


@pytest.mark.skipif(not _lancedb_populated(), reason="LanceDB not populated")
class TestLanceDBRetrieval:
    @pytest.fixture(autouse=True)
    def setup_backend(self):
        from rag7.backend import LanceDBBackend

        self.embed_fn = _make_embed_fn()
        self.backend = LanceDBBackend(
            table=TABLE_NAME,
            db_uri=LANCEDB_PATH,
            embed_fn=self.embed_fn,
            vector_column="vector",
            text_column="content",
        )

    @pytest.mark.parametrize(
        "case", RETRIEVAL_CASES, ids=[c["query"][:40] for c in RETRIEVAL_CASES]
    )
    def test_vector_search(self, case: dict) -> None:
        req = SearchRequest(query=case["query"], limit=case["limit"])
        hits = self.backend.search(req)
        _check_retrieval(hits, case)

    def test_sample_documents(self) -> None:
        docs = self.backend.sample_documents(limit=5)
        assert len(docs) > 0
        assert "content" in docs[0]

    def test_filter_eq(self) -> None:
        """LanceDB filter: exact match via pandas query syntax."""
        req = SearchRequest(query="court", limit=10, filter_expr="language = 'en'")
        hits = self.backend.search(req)
        if hits:
            assert all(h.get("language") == "en" for h in hits)

    def test_filter_ne(self) -> None:
        """LanceDB filter: not-equal."""
        req = SearchRequest(query="data", limit=10, filter_expr="language != 'de'")
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("language") != "de" for h in hits)

    def test_filter_in(self) -> None:
        """LanceDB filter: IN list."""
        req = SearchRequest(
            query="data",
            limit=10,
            filter_expr="language IN ('fr', 'it')",
        )
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("language") in ("fr", "it") for h in hits)

    def test_filter_and(self) -> None:
        """LanceDB filter: AND combination."""
        req = SearchRequest(
            query="data",
            limit=10,
            filter_expr="language = 'en' AND source = 'rteb/mteb/nq'",
        )
        hits = self.backend.search(req)
        if hits:
            assert all(h.get("language") == "en" for h in hits)
            assert all(h.get("source") == "rteb/mteb/nq" for h in hits)

    def test_filter_with_vector(self) -> None:
        """Filter combined with vector search."""
        req = SearchRequest(
            query="court judgment", limit=5, filter_expr="language = 'en'"
        )
        hits = self.backend.search(req)
        if hits:
            assert all(h.get("language") == "en" for h in hits)

    def test_sample_with_filter(self) -> None:
        """sample_documents with filter."""
        docs = self.backend.sample_documents(limit=5, filter_expr="language = 'de'")
        if docs:
            assert all(d.get("language") == "de" for d in docs)


# ── DuckDB ────────────────────────────────────────────────────────────────────

DUCKDB_PATH = "data/duckdb_benchmark.db"


def _duckdb_populated() -> bool:
    try:
        import duckdb

        conn = duckdb.connect(DUCKDB_PATH, read_only=True)
        result = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()
        count = result[0] if result else 0  # type: ignore[index]
        conn.close()
        return count > 0
    except Exception:
        return False


@pytest.mark.skipif(not _duckdb_populated(), reason="DuckDB not populated")
class TestDuckDBRetrieval:
    @pytest.fixture(autouse=True)
    def setup_backend(self):
        from rag7.backend import DuckDBBackend

        self.embed_fn = _make_embed_fn()
        self.backend = DuckDBBackend(
            table=TABLE_NAME,
            db_path=DUCKDB_PATH,
            embed_fn=self.embed_fn,
            vector_column="embedding",
            content_column="content",
        )

    @pytest.mark.parametrize(
        "case", RETRIEVAL_CASES, ids=[c["query"][:40] for c in RETRIEVAL_CASES]
    )
    def test_vector_search(self, case: dict) -> None:
        req = SearchRequest(query=case["query"], limit=case["limit"])
        hits = self.backend.search(req)
        _check_retrieval(hits, case)

    def test_filter_eq(self) -> None:
        """SQL = operator."""
        req = SearchRequest(
            query="element", limit=10, filter_expr="source = 'wikipedia'"
        )
        hits = self.backend.search(req)
        if hits:
            assert all(h.get("source") == "wikipedia" for h in hits)

    def test_filter_ne(self) -> None:
        """SQL != operator."""
        req = SearchRequest(query="data", limit=10, filter_expr="language != 'de'")
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("language") != "de" for h in hits)

    def test_filter_and(self) -> None:
        """SQL AND: combine two conditions."""
        req = SearchRequest(
            query="data",
            limit=10,
            filter_expr="language = 'en' AND source = 'rteb/mteb/nq'",
        )
        hits = self.backend.search(req)
        if hits:
            assert all(h.get("language") == "en" for h in hits)
            assert all(h.get("source") == "rteb/mteb/nq" for h in hits)

    def test_filter_or(self) -> None:
        """SQL OR: match either condition."""
        req = SearchRequest(
            query="data",
            limit=10,
            filter_expr="language = 'fr' OR language = 'it'",
        )
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("language") in ("fr", "it") for h in hits)

    def test_filter_in(self) -> None:
        """SQL IN: match against list."""
        req = SearchRequest(
            query="data",
            limit=10,
            filter_expr="language IN ('de', 'fr')",
        )
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("language") in ("de", "fr") for h in hits)

    def test_filter_like(self) -> None:
        """SQL LIKE: pattern match."""
        req = SearchRequest(
            query="data",
            limit=10,
            filter_expr="source LIKE 'rteb%'",
        )
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("source", "").startswith("rteb") for h in hits)

    def test_filter_with_vector(self) -> None:
        """Filter + vector search combined."""
        req = SearchRequest(
            query="court judgment", limit=5, filter_expr="language = 'en'"
        )
        hits = self.backend.search(req)
        if hits:
            assert all(h.get("language") == "en" for h in hits)
            assert "_rankingScore" in hits[0]

    def test_sample_with_filter(self) -> None:
        """sample_documents respects SQL filter."""
        docs = self.backend.sample_documents(limit=5, filter_expr="language = 'de'")
        assert len(docs) > 0
        assert all(d.get("language") == "de" for d in docs)

    def test_ranking_score_present(self) -> None:
        req = SearchRequest(query="chemistry", limit=3)
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert "_rankingScore" in hits[0]


# ── Azure AI Search ───────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.getenv("AZURE_SEARCH_API_KEY"), reason="AZURE_SEARCH_API_KEY not set"
)
class TestAzureRetrieval:
    @pytest.fixture(autouse=True)
    def setup_backend(self):
        pytest.importorskip("azure.search.documents")
        from rag7.backend import AzureAISearchBackend

        self.backend = AzureAISearchBackend(
            index=INDEX_NAME,
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", ""),
        )

    @pytest.mark.parametrize(
        "case", RETRIEVAL_CASES, ids=[c["query"][:40] for c in RETRIEVAL_CASES]
    )
    def test_text_search(self, case: dict) -> None:
        req = SearchRequest(query=case["query"], limit=case["limit"])
        hits = self.backend.search(req)
        _check_retrieval(hits, case)

    def test_filter_eq(self) -> None:
        """OData eq operator: exact match on string field."""
        req = SearchRequest(query="*", limit=10, filter_expr="language eq 'de'")
        hits = self.backend.search(req)
        assert len(hits) > 0, "No German docs found"
        assert all(h.get("language") == "de" for h in hits)

    def test_filter_ne(self) -> None:
        """OData ne operator: exclude a value."""
        req = SearchRequest(query="*", limit=10, filter_expr="language ne 'de'")
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("language") != "de" for h in hits)

    def test_filter_and(self) -> None:
        """OData and: combine two conditions."""
        req = SearchRequest(
            query="*",
            limit=10,
            filter_expr="language eq 'en' and source eq 'rteb/mteb/nq'",
        )
        hits = self.backend.search(req)
        if hits:
            assert all(h.get("language") == "en" for h in hits)
            assert all(h.get("source") == "rteb/mteb/nq" for h in hits)

    def test_filter_or(self) -> None:
        """OData or: match either condition."""
        req = SearchRequest(
            query="*",
            limit=10,
            filter_expr="language eq 'fr' or language eq 'it'",
        )
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("language") in ("fr", "it") for h in hits)

    def test_filter_search_in(self) -> None:
        """OData search.in: match against a list of values."""
        req = SearchRequest(
            query="*",
            limit=10,
            filter_expr="search.in(language, 'de,fr', ',')",
        )
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("language") in ("de", "fr") for h in hits)

    def test_filter_with_query(self) -> None:
        """Filter + text query combined."""
        req = SearchRequest(
            query="court judgment",
            limit=5,
            filter_expr="language eq 'en'",
        )
        hits = self.backend.search(req)
        if hits:
            assert all(h.get("language") == "en" for h in hits)

    def test_sample_with_filter(self) -> None:
        """sample_documents respects OData filter."""
        docs = self.backend.sample_documents(limit=5, filter_expr="language eq 'de'")
        assert len(docs) > 0
        assert all(d.get("language") == "de" for d in docs)

    def test_sample_documents(self) -> None:
        docs = self.backend.sample_documents(limit=5)
        assert len(docs) > 0
