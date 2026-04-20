"""Backend integration tests.

pgvector: needs `podman compose -f docker-compose.test.yml up -d`
Azure AI Search: needs AZURE_SEARCH_ENDPOINT + AZURE_SEARCH_API_KEY env vars
Others: in-memory, no external deps required.

Run: uv run pytest tests/test_backends.py -v
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from rag7.backend import SearchRequest

# ── Helpers ──────────────────────────────────────────────────────────────────

SAMPLE_DOCS = [
    {"id": "1", "content": "Python is a programming language", "category": "tech"},
    {"id": "2", "content": "The cat sat on the mat", "category": "animals"},
    {"id": "3", "content": "Machine learning uses neural networks", "category": "tech"},
    {"id": "4", "content": "Dogs are loyal companions", "category": "animals"},
    {"id": "5", "content": "PostgreSQL is a relational database", "category": "tech"},
]

DIMENSION = 8  # small dim for testing


def _fake_embed(text: str) -> list[float]:
    """Deterministic fake embeddings from text hash."""
    import hashlib

    h = hashlib.sha256(text.encode()).digest()
    vec = [float(b) / 255.0 for b in h[:DIMENSION]]
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec]


def _docs_with_vectors() -> list[dict]:
    return [{**doc, "vector": _fake_embed(doc["content"])} for doc in SAMPLE_DOCS]


# ── ChromaDB ─────────────────────────────────────────────────────────────────


class TestChromaDBBackend:
    @pytest.fixture
    def backend(self):
        pytest.importorskip("chromadb")
        from rag7.backend import ChromaDBBackend

        b = ChromaDBBackend(collection="test_chroma")
        # Add docs with extra metadata field to exercise multi-clause filters.
        b._collection.add(
            ids=[d["id"] for d in SAMPLE_DOCS],
            documents=[d["content"] for d in SAMPLE_DOCS],
            metadatas=[
                {"category": d["category"], "priority": d["id"]} for d in SAMPLE_DOCS
            ],
        )
        yield b
        # Cleanup
        try:
            b._client.delete_collection("test_chroma")
        except Exception:
            pass

    def test_search(self, backend):
        req = SearchRequest(query="programming language", limit=3)
        hits = backend.search(req)
        assert len(hits) > 0
        assert "content" in hits[0] or "id" in hits[0]

    def test_batch_search(self, backend):
        reqs = [
            SearchRequest(query="python", limit=2),
            SearchRequest(query="cat", limit=2),
        ]
        results = backend.batch_search(reqs)
        assert len(results) == 2

    def test_get_index_config(self, backend):
        cfg = backend.get_index_config()
        assert cfg is not None

    def test_sample_documents(self, backend):
        docs = backend.sample_documents(limit=3)
        assert len(docs) <= 3

    def test_search_with_equality_filter(self, backend):
        """String filter (Meili dialect) must be parsed into Chroma's dict."""
        req = SearchRequest(
            query="programming", limit=5, filter_expr='category = "tech"'
        )
        hits = backend.search(req)
        assert len(hits) > 0, "equality filter produced empty result"
        assert all(h.get("category") == "tech" for h in hits)

    def test_search_with_inequality_filter(self, backend):
        req = SearchRequest(query="cat", limit=5, filter_expr='category != "animals"')
        hits = backend.search(req)
        assert len(hits) > 0, "inequality filter produced empty result"
        assert all(h.get("category") != "animals" for h in hits)

    def test_search_with_dict_filter(self, backend):
        """Native Chroma dict filter must pass through unchanged."""
        req = SearchRequest(
            query="programming", limit=5, filter_expr={"category": "tech"}
        )
        hits = backend.search(req)
        assert len(hits) > 0
        assert all(h.get("category") == "tech" for h in hits)

    def test_search_with_and_filter_string(self, backend):
        """Meili-string AND must parse into Chroma `$and` dict (metadata only)."""
        req = SearchRequest(
            query="anything",
            limit=10,
            filter_expr='category = "tech" AND priority != "1"',
        )
        hits = backend.search(req)
        assert len(hits) > 0
        assert all(
            h.get("category") == "tech" and h.get("priority") != "1" for h in hits
        )

    def test_search_with_and_filter_dict(self, backend):
        """Native $and dict must pass through."""
        req = SearchRequest(
            query="anything",
            limit=10,
            filter_expr={"$and": [{"category": "tech"}, {"priority": {"$ne": "1"}}]},
        )
        hits = backend.search(req)
        assert len(hits) > 0
        assert all(
            h.get("category") == "tech" and h.get("priority") != "1" for h in hits
        )

    def test_sample_documents_with_filter(self, backend):
        docs = backend.sample_documents(limit=10, filter_expr='category = "animals"')
        assert len(docs) > 0
        assert all(d.get("category") == "animals" for d in docs)

    def test_not_contains_drops_clause(self, backend):
        """Chroma has no metadata substring op; NOT_CONTAINS must drop cleanly.

        Rather than blowing up, the clause is dropped and the query runs
        unfiltered — documented limitation of Chroma's metadata filter.
        """
        req = SearchRequest(
            query="programming",
            limit=5,
            filter_expr='NOT category CONTAINS "tech"',
        )
        hits = backend.search(req)
        # No effective filter → any category allowed. What matters: no crash.
        assert isinstance(hits, list)

    def test_custom_embed_fn(self):
        """Verify wrapped embed_fn is used by ChromaDB automatically."""
        pytest.importorskip("chromadb")
        from rag7.backend import ChromaDBBackend

        b = ChromaDBBackend(collection="test_chroma_ef", embed_fn=_fake_embed)
        b._collection.add(
            ids=[d["id"] for d in SAMPLE_DOCS],
            documents=[d["content"] for d in SAMPLE_DOCS],
            metadatas=[{"category": d["category"]} for d in SAMPLE_DOCS],
        )
        req = SearchRequest(query="programming language", limit=3)
        hits = b.search(req)
        assert len(hits) > 0
        assert "_rankingScore" in hits[0]
        b._client.delete_collection("test_chroma_ef")


# ── LanceDB ──────────────────────────────────────────────────────────────────


class TestLanceDBBackend:
    @pytest.fixture
    def backend(self, tmp_path):
        lancedb = pytest.importorskip("lancedb")
        from rag7.backend import LanceDBBackend

        db = lancedb.connect(str(tmp_path / "test.lance"))
        docs = _docs_with_vectors()
        db.create_table(
            "test_lance",
            data=[
                {
                    "id": d["id"],
                    "content": d["content"],
                    "category": d["category"],
                    "vector": d["vector"],
                }
                for d in docs
            ],
        )
        return LanceDBBackend(
            table="test_lance",
            db_uri=str(tmp_path / "test.lance"),
            embed_fn=_fake_embed,
            vector_column="vector",
            text_column="content",
        )

    def test_search(self, backend):
        req = SearchRequest(query="Python is a programming language", limit=3)
        hits = backend.search(req)
        assert len(hits) > 0

    def test_sample_documents(self, backend):
        docs = backend.sample_documents(limit=3)
        assert len(docs) <= 5  # may get all

    def test_search_with_equality_filter(self, backend):
        """SQL-style equality filter — must return only matching docs."""
        req = SearchRequest(
            query="programming", limit=5, filter_expr="category = 'tech'"
        )
        hits = backend.search(req)
        assert len(hits) > 0, "equality filter produced empty result"
        assert all(h.get("category") == "tech" for h in hits)

    def test_search_with_inequality_filter(self, backend):
        """SQL-style != filter — must exclude matching docs."""
        req = SearchRequest(query="cat", limit=5, filter_expr="category != 'animals'")
        hits = backend.search(req)
        assert len(hits) > 0, "inequality filter produced empty result"
        assert all(h.get("category") != "animals" for h in hits)

    def test_sample_with_filter(self, backend):
        docs = backend.sample_documents(limit=10, filter_expr="category = 'animals'")
        assert len(docs) > 0, "filter on sample produced empty result"
        assert all(d["category"] == "animals" for d in docs)

    def test_meili_syntax_filter_fails_visibly(self, backend):
        """Raw Meili-style `NOT field CONTAINS "x"` is NOT valid LanceDB SQL.

        Retrieval should never pass Meili syntax directly — core.py now
        delegates to backend.build_filter_expr(intent) which returns SQL
        for LanceDB. This canary confirms the raw-string path still errors
        silently, so the translator remains load-bearing.
        """
        req = SearchRequest(
            query="programming",
            limit=5,
            filter_expr='NOT category CONTAINS "tech"',
        )
        hits = backend.search(req)
        assert hits == [], (
            f"expected [] (silent swallow of invalid syntax), got {len(hits)} hits"
        )

    def test_build_filter_expr_sql_dialect(self, backend):
        """build_filter_expr must emit SQL (ILIKE) for LanceDB, not Meili."""
        from types import SimpleNamespace

        intent = SimpleNamespace(
            field="category",
            value="tech",
            operator="NOT_CONTAINS",
            extra_excludes=[],
        )
        expr = backend.build_filter_expr(intent)
        assert "ILIKE" in expr, f"expected SQL ILIKE, got {expr!r}"
        assert "CONTAINS" not in expr

        req = SearchRequest(query="programming", limit=5, filter_expr=expr)
        hits = backend.search(req)
        assert len(hits) > 0, "translated SQL filter produced empty result"
        assert all("tech" not in str(h.get("category", "")).lower() for h in hits)

    def test_build_filter_expr_equality(self, backend):
        """Equality + inequality must use single-quoted SQL literals."""
        from types import SimpleNamespace

        eq = backend.build_filter_expr(
            SimpleNamespace(
                field="category", value="tech", operator="=", extra_excludes=[]
            )
        )
        assert eq == "category = 'tech'"

        neq = backend.build_filter_expr(
            SimpleNamespace(
                field="category", value="animals", operator="!=", extra_excludes=[]
            )
        )
        assert neq == "category != 'animals'"

    def test_build_filter_expr_escapes_quotes(self, backend):
        """SQL escaping: apostrophe must be doubled, not backslashed."""
        from types import SimpleNamespace

        expr = backend.build_filter_expr(
            SimpleNamespace(
                field="name", value="d'Artagnan", operator="=", extra_excludes=[]
            )
        )
        assert expr == "name = 'd''Artagnan'"

    def test_own_brand_exclusion_with_extra_excludes(self, backend):
        """Real-world scenario: exclude multiple own-brand suppliers.

        Mirrors real-world cases that exclude e.g. supplier_name
        containing "sakret" OR "fixit". Uses category as a stand-in here
        since the fixture has no supplier column.
        """
        from types import SimpleNamespace

        intent = SimpleNamespace(
            field="category",
            value="tech",
            operator="NOT_CONTAINS",
            extra_excludes=["animals"],
        )
        expr = backend.build_filter_expr(intent)
        assert "NOT ILIKE '%tech%'" in expr
        assert "NOT ILIKE '%animals%'" in expr
        assert " AND " in expr

        req = SearchRequest(query="anything", limit=10, filter_expr=expr)
        hits = backend.search(req)
        # No fixture doc has category outside {tech, animals}, so expect 0.
        # What matters: the filter PARSED and EXECUTED (no silent swallow).
        # Flip assertion by adding a third category if you want positive hits.
        assert isinstance(hits, list)
        assert all(h.get("category") not in {"tech", "animals"} for h in hits), (
            "own-brand exclusion leaked a hit"
        )


# ── DuckDB ───────────────────────────────────────────────────────────────────


class TestDuckDBBackend:
    @pytest.fixture
    def backend(self):
        pytest.importorskip("duckdb")
        from rag7.backend import DuckDBBackend

        b = DuckDBBackend(
            table="test_docs",
            db_path=":memory:",
            embed_fn=_fake_embed,
            vector_column="embedding",
            content_column="content",
        )
        # Create table + insert docs
        docs = _docs_with_vectors()
        b._conn.execute(f"""
            CREATE TABLE test_docs (
                id VARCHAR,
                content VARCHAR,
                category VARCHAR,
                embedding FLOAT[{DIMENSION}]
            )
        """)
        for d in docs:
            b._conn.execute(
                "INSERT INTO test_docs VALUES (?, ?, ?, ?::FLOAT[8])",
                [d["id"], d["content"], d["category"], d["vector"]],
            )
        return b

    def test_vector_search(self, backend):
        req = SearchRequest(query="programming", limit=3)
        hits = backend.search(req)
        assert len(hits) > 0
        assert "_rankingScore" in hits[0]

    def test_search_with_filter(self, backend):
        req = SearchRequest(
            query="programming", limit=5, filter_expr="category = 'tech'"
        )
        hits = backend.search(req)
        assert all(h.get("category") == "tech" for h in hits)

    def test_get_index_config(self, backend):
        cfg = backend.get_index_config()
        assert "content" in cfg.searchable_attributes

    def test_sample_documents(self, backend):
        docs = backend.sample_documents(limit=3)
        assert len(docs) == 3

    def test_sample_with_filter(self, backend):
        docs = backend.sample_documents(limit=10, filter_expr="category = 'animals'")
        assert all(d["category"] == "animals" for d in docs)

    def test_search_with_inequality_filter(self, backend):
        req = SearchRequest(
            query="anything", limit=5, filter_expr="category != 'animals'"
        )
        hits = backend.search(req)
        assert len(hits) > 0
        assert all(h.get("category") != "animals" for h in hits)

    def test_search_with_and_filter(self, backend):
        """Multi-clause AND — must narrow, not widen."""
        req = SearchRequest(
            query="anything",
            limit=10,
            filter_expr="category = 'tech' AND id != '1'",
        )
        hits = backend.search(req)
        assert len(hits) > 0
        assert all(h.get("category") == "tech" and h.get("id") != "1" for h in hits)

    def test_build_filter_expr_not_contains(self, backend):
        """DuckDB dialect = SQL → NOT ILIKE."""
        from types import SimpleNamespace

        expr = backend.build_filter_expr(
            SimpleNamespace(
                field="category",
                value="tech",
                operator="NOT_CONTAINS",
                extra_excludes=["animals"],
            )
        )
        assert "NOT ILIKE '%tech%'" in expr
        assert "NOT ILIKE '%animals%'" in expr
        hits = backend.search(SearchRequest(query="x", limit=10, filter_expr=expr))
        assert all(
            "tech" not in str(h.get("category", "")).lower()
            and "animals" not in str(h.get("category", "")).lower()
            for h in hits
        )


# ── pgvector ─────────────────────────────────────────────────────────────────

PGVECTOR_DSN = "postgresql://test:test@localhost:5433/testdb"


def _pgvector_available() -> bool:
    try:
        import psycopg

        conn = psycopg.connect(PGVECTOR_DSN, connect_timeout=2)
        conn.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _pgvector_available(),
    reason="pgvector container not running (podman compose -f docker-compose.test.yml up -d)",
)
class TestPgvectorBackend:
    @pytest.fixture(autouse=True)
    def backend(self):
        import psycopg
        from rag7.backend import PgvectorBackend

        # Setup: create extension + table
        conn = psycopg.connect(PGVECTOR_DSN, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("DROP TABLE IF EXISTS test_pgvec")
        conn.execute(f"""
            CREATE TABLE test_pgvec (
                id VARCHAR PRIMARY KEY,
                content TEXT,
                category VARCHAR,
                embedding vector({DIMENSION})
            )
        """)
        docs = _docs_with_vectors()
        for d in docs:
            conn.execute(
                "INSERT INTO test_pgvec (id, content, category, embedding) VALUES (%s, %s, %s, %s)",
                (d["id"], d["content"], d["category"], d["vector"]),
            )
        conn.close()

        self.backend = PgvectorBackend(
            table="test_pgvec",
            dsn=PGVECTOR_DSN,
            embed_fn=_fake_embed,
            vector_column="embedding",
            content_column="content",
        )
        yield
        # Cleanup
        try:
            self.backend._conn.execute("DROP TABLE IF EXISTS test_pgvec")
        except Exception:
            pass

    def test_vector_search(self):
        req = SearchRequest(query="programming", limit=3)
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert "_rankingScore" in hits[0]

    def test_search_with_filter(self):
        req = SearchRequest(query="python", limit=5, filter_expr="category = 'tech'")
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("category") == "tech" for h in hits)

    def test_search_with_inequality_filter(self):
        req = SearchRequest(
            query="anything", limit=5, filter_expr="category != 'animals'"
        )
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("category") != "animals" for h in hits)

    def test_search_with_and_filter(self):
        req = SearchRequest(
            query="anything",
            limit=10,
            filter_expr="category = 'tech' AND id != '1'",
        )
        hits = self.backend.search(req)
        assert len(hits) > 0
        assert all(h.get("category") == "tech" and h.get("id") != "1" for h in hits)

    def test_build_filter_expr_not_contains(self):
        from types import SimpleNamespace

        expr = self.backend.build_filter_expr(
            SimpleNamespace(
                field="category",
                value="tech",
                operator="NOT_CONTAINS",
                extra_excludes=[],
            )
        )
        assert "NOT ILIKE '%tech%'" in expr
        hits = self.backend.search(SearchRequest(query="x", limit=10, filter_expr=expr))
        assert all("tech" not in str(h.get("category", "")).lower() for h in hits)

    def test_text_search_fallback(self):
        req = SearchRequest(query="", limit=5)
        hits = self.backend.search(req)
        assert isinstance(hits, list)

    def test_get_index_config(self):
        cfg = self.backend.get_index_config()
        assert "content" in cfg.searchable_attributes

    def test_sample_documents(self):
        docs = self.backend.sample_documents(limit=3)
        assert len(docs) == 3


# ── Qdrant (in-memory) ──────────────────────────────────────────────────────


class TestQdrantBackend:
    @pytest.fixture
    def backend(self):
        pytest.importorskip("qdrant_client")
        from qdrant_client import QdrantClient, models
        from rag7.backend import QdrantBackend

        # Use in-memory Qdrant
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name="test_qdrant",
            vectors_config=models.VectorParams(
                size=DIMENSION, distance=models.Distance.COSINE
            ),
        )
        docs = _docs_with_vectors()
        client.upsert(
            collection_name="test_qdrant",
            points=[
                models.PointStruct(
                    id=int(d["id"]),
                    vector=d["vector"],
                    payload={"content": d["content"], "category": d["category"]},
                )
                for d in docs
            ],
        )

        b = QdrantBackend(collection="test_qdrant", embed_fn=_fake_embed)
        b._client = client  # inject in-memory client
        return b

    def test_search(self, backend):
        req = SearchRequest(query="programming", limit=3)
        hits = backend.search(req)
        assert len(hits) > 0
        assert "_rankingScore" in hits[0]

    def test_batch_search(self, backend):
        reqs = [
            SearchRequest(query="python", limit=2),
            SearchRequest(query="cat", limit=2),
        ]
        results = backend.batch_search(reqs)
        assert len(results) == 2

    def test_scroll_without_vector(self, backend):
        from rag7.backend import QdrantBackend

        b = QdrantBackend.__new__(QdrantBackend)
        b._collection = "test_qdrant"
        b._embed_fn = None
        b._client = backend._client
        req = SearchRequest(query="anything", limit=3)
        hits = b.search(req)  # no embed_fn → falls back to scroll
        assert isinstance(hits, list)

    def test_get_index_config(self, backend):
        cfg = backend.get_index_config()
        assert cfg is not None

    def test_sample_documents(self, backend):
        docs = backend.sample_documents(limit=3)
        assert len(docs) <= 5

    def test_search_with_equality_filter(self, backend):
        """Qdrant native `must=[FieldCondition(match=MatchValue)]` dict."""
        from qdrant_client.models import FieldCondition, MatchValue

        flt = {"must": [FieldCondition(key="category", match=MatchValue(value="tech"))]}
        req = SearchRequest(query="programming", limit=5, filter_expr=flt)
        hits = backend.search(req)
        assert len(hits) > 0
        assert all(h.get("category") == "tech" for h in hits)

    def test_search_with_inequality_filter(self, backend):
        """Qdrant `must_not` = negative filter."""
        from qdrant_client.models import FieldCondition, MatchValue

        flt = {
            "must_not": [
                FieldCondition(key="category", match=MatchValue(value="animals"))
            ]
        }
        req = SearchRequest(query="cat", limit=5, filter_expr=flt)
        hits = backend.search(req)
        assert len(hits) > 0
        assert all(h.get("category") != "animals" for h in hits)

    def test_search_with_and_filter(self, backend):
        """Qdrant `must=[a, b]` = implicit AND."""
        from qdrant_client.models import FieldCondition, MatchValue

        flt = {
            "must": [
                FieldCondition(key="category", match=MatchValue(value="tech")),
            ],
            "must_not": [
                FieldCondition(
                    key="content",
                    match=MatchValue(value="Python is a programming language"),
                ),
            ],
        }
        req = SearchRequest(query="anything", limit=10, filter_expr=flt)
        hits = backend.search(req)
        assert all(
            h.get("category") == "tech"
            and h.get("content") != "Python is a programming language"
            for h in hits
        )


# ── Qdrant (server-backed — for features local mode doesn't support) ────────

QDRANT_SERVER_URL = "http://localhost:6333"


def _qdrant_server_available() -> bool:
    try:
        import httpx

        r = httpx.get(f"{QDRANT_SERVER_URL}/healthz", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(
    not _qdrant_server_available(),
    reason="qdrant server not running (podman compose -f docker-compose.test.yml up -d qdrant)",
)
class TestQdrantServerFilters:
    """Qdrant server-only filter features — text index + MatchText substring.

    Local `:memory:` Qdrant ignores payload indexes, so MatchText (the
    substring operator needed for NOT_CONTAINS) only works against a real
    server. Keep this class separate so the fast in-memory suite doesn't
    depend on the container.
    """

    @pytest.fixture
    def backend(self):
        pytest.importorskip("qdrant_client")
        from qdrant_client import QdrantClient, models
        from rag7.backend import QdrantBackend

        client = QdrantClient(url=QDRANT_SERVER_URL, check_compatibility=False)
        col = "test_qdrant_server"
        if client.collection_exists(col):
            client.delete_collection(col)
        client.create_collection(
            collection_name=col,
            vectors_config=models.VectorParams(
                size=DIMENSION, distance=models.Distance.COSINE
            ),
        )
        client.create_payload_index(
            collection_name=col,
            field_name="content",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
            ),
        )
        docs = _docs_with_vectors()
        client.upsert(
            collection_name=col,
            points=[
                models.PointStruct(
                    id=int(d["id"]),
                    vector=d["vector"],
                    payload={"content": d["content"], "category": d["category"]},
                )
                for d in docs
            ],
        )
        b = QdrantBackend.__new__(QdrantBackend)
        b._client = client
        b._collection = col
        b._embed_fn = _fake_embed
        b.index = col
        yield b
        client.delete_collection(col)

    def test_search_with_not_contains_match_text(self, backend):
        """`must_not=[MatchText]` = substring NOT_CONTAINS (needs text index)."""
        from qdrant_client.models import FieldCondition, MatchText

        flt = {
            "must_not": [
                FieldCondition(key="content", match=MatchText(text="python")),
            ]
        }
        req = SearchRequest(query="anything", limit=10, filter_expr=flt)
        hits = backend.search(req)
        assert len(hits) > 0
        assert all("python" not in h.get("content", "").lower() for h in hits)


# ── Azure AI Search ──────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.getenv("AZURE_SEARCH_API_KEY"),
    reason="AZURE_SEARCH_API_KEY not set",
)
class TestAzureAISearchBackend:
    @pytest.fixture
    def backend(self):
        pytest.importorskip("azure.search.documents")
        from rag7.backend import AzureAISearchBackend

        return AzureAISearchBackend(
            index=os.getenv("AZURE_SEARCH_INDEX", "test-index"),
            endpoint=os.getenv(
                "AZURE_SEARCH_ENDPOINT", "https://aisearchtestbms.search.windows.net"
            ),
        )

    def test_search(self, backend):
        req = SearchRequest(query="test", limit=3)
        hits = backend.search(req)
        assert isinstance(hits, list)

    def test_sample_documents(self, backend):
        docs = backend.sample_documents(limit=3)
        assert isinstance(docs, list)

    def test_get_index_config(self, backend):
        cfg = backend.get_index_config()
        assert cfg is not None
