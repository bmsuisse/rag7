"""MTEB Retrieval Benchmark — pytest, multi-backend, Ollama embeddings.

Session-scoped fixtures handle the expensive work (HF download, Ollama embedding,
backend ingestion) once.  Parametrized tests evaluate AgenticRAG across every
(backend x dataset) combination.

Backends:
  lancedb   — local file-based (always available)
  chromadb  — local persistent
  qdrant    — local file-based (qdrant-client, no server)
  pgvector  — PostgreSQL via podman (docker-compose.test.yml)

Run:
  uv run pytest tests/test_mteb_benchmark.py -v -s          # all local backends
  uv run pytest tests/test_mteb_benchmark.py -k lancedb -s  # single backend
  uv run pytest tests/test_mteb_benchmark.py -k scifact -s  # single dataset
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import requests as _requests
from dotenv import load_dotenv

pytest.importorskip("datasets")

load_dotenv()

# -- Config --------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / ".mteb_data"
OLLAMA_MODEL = "embeddinggemma"
OLLAMA_URL = "http://localhost:11434/api/embed"
PGVECTOR_DSN = "postgresql://test:test@localhost:5433/testdb"
VECTOR_DIM = 768
MAX_QUERIES = 50
CONCURRENCY = 10


# -- Dataset specs -------------------------------------------------------------


@dataclass
class DatasetSpec:
    name: str
    table: str
    hf_dataset: str
    corpus_config: str
    queries_config: str
    qrels_config: str
    lang: str
    domain: str
    text_field: str = "text"
    title_field: str | None = "title"
    id_field: str = "_id"


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        "Mintaka DE",
        "mintaka_de",
        "mteb/MintakaRetrieval",
        "de-corpus",
        "de-queries",
        "de-qrels",
        "de",
        "Encyclopedic",
        id_field="id",
        title_field="title",
    ),
    DatasetSpec(
        "Mintaka FR",
        "mintaka_fr",
        "mteb/MintakaRetrieval",
        "fr-corpus",
        "fr-queries",
        "fr-qrels",
        "fr",
        "Encyclopedic",
        id_field="id",
        title_field="title",
    ),
    DatasetSpec(
        "Mintaka IT",
        "mintaka_it",
        "mteb/MintakaRetrieval",
        "it-corpus",
        "it-queries",
        "it-qrels",
        "it",
        "Encyclopedic",
        id_field="id",
        title_field="title",
    ),
    DatasetSpec(
        "Mintaka ES",
        "mintaka_es",
        "mteb/MintakaRetrieval",
        "es-corpus",
        "es-queries",
        "es-qrels",
        "es",
        "Encyclopedic",
        id_field="id",
        title_field="title",
    ),
    DatasetSpec(
        "GermanQuAD",
        "germanquad",
        "mteb/germanquad-retrieval",
        "corpus",
        "queries",
        "mteb/germanquad-retrieval-qrels",
        "de",
        "General QA",
        title_field=None,
    ),
    DatasetSpec(
        "NFCorpus",
        "nfcorpus",
        "mteb/nfcorpus",
        "corpus",
        "queries",
        "default",
        "en",
        "Health/Nutrition",
    ),
    DatasetSpec(
        "SciFact",
        "scifact",
        "mteb/scifact",
        "corpus",
        "queries",
        "default",
        "en",
        "Science",
    ),
    DatasetSpec(
        "FiQA", "fiqa", "mteb/fiqa", "corpus", "queries", "default", "en", "Finance"
    ),
]

DATASET_BY_TABLE = {s.table: s for s in DATASETS}

BACKEND_NAMES = ["lancedb", "chromadb", "qdrant", "pgvector"]


# -- Ollama embedding ----------------------------------------------------------

_session = _requests.Session()


def ollama_embed(text: str) -> list[float]:
    for attempt in range(3):
        try:
            resp = _session.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "input": text[:8000] or "empty"},
                timeout=60,
            )
            resp.raise_for_status()
            embeddings = resp.json()["embeddings"]
            if embeddings:
                return embeddings[0]
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return [0.0] * VECTOR_DIM


def ollama_embed_batch(texts: list[str]) -> list[list[float]]:
    from tqdm import tqdm

    return [ollama_embed(t) for t in tqdm(texts, desc="  embedding", unit="doc")]


# -- HuggingFace data loading --------------------------------------------------


def _first_split(ds: Any) -> Any:
    return ds[list(ds.keys())[0]]


def _load_qrels(spec: DatasetSpec) -> dict[str, list[str]]:
    from datasets import load_dataset

    if "/" in spec.qrels_config:
        ds = load_dataset(spec.qrels_config)
    else:
        ds = load_dataset(spec.hf_dataset, spec.qrels_config)
    split_name = "test" if "test" in ds else list(ds.keys())[0]
    qrels_ds = ds[split_name]
    relevant: dict[str, list[str]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        score = row.get("score", 1)
        if score > 0:
            relevant.setdefault(qid, []).append(cid)
    return relevant


def _load_queries(spec: DatasetSpec) -> dict[str, str]:
    from datasets import load_dataset

    ds = load_dataset(spec.hf_dataset, spec.queries_config)
    queries_ds = _first_split(ds)
    return {str(row[spec.id_field]): row[spec.text_field] for row in queries_ds}


def _load_corpus(spec: DatasetSpec) -> list[dict[str, Any]]:
    from datasets import load_dataset

    relevant = _load_qrels(spec)
    needed_ids = set()
    for ids in relevant.values():
        needed_ids.update(ids)

    ds = load_dataset(spec.hf_dataset, spec.corpus_config)
    corpus_ds = _first_split(ds)

    docs: list[dict[str, Any]] = []
    for row in corpus_ds:
        doc_id = str(row[spec.id_field])
        text = row.get(spec.text_field) or ""
        title = (row.get(spec.title_field) or "") if spec.title_field else ""
        content = f"{title}\n{text}".strip() if title else text
        docs.append(
            {
                "corpus_id": doc_id,
                "content": content[:8000],
                "title": title,
                "_needed": doc_id in needed_ids,
            }
        )

    needed = [d for d in docs if d["_needed"]]
    others = [d for d in docs if not d["_needed"]]
    others.sort(key=lambda d: hashlib.md5(d["corpus_id"].encode()).hexdigest())
    noise_cap = max(100, len(needed))
    final = needed + others[:noise_cap]
    for d in final:
        d.pop("_needed", None)

    print(
        f"    {len(needed)} relevant + {min(len(others), noise_cap)} noise = {len(final)} docs"
    )
    return final


# -- Metrics -------------------------------------------------------------------


def hit_rate_at_k(
    retrieved_ids: list[str], relevant_ids: list[str], k: int = 5
) -> float:
    return 1.0 if set(retrieved_ids[:k]) & set(relevant_ids) else 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 10) -> float:
    rel_set = set(relevant_ids)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, did in enumerate(retrieved_ids[:k])
        if did in rel_set
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# -- Backend ingestion ---------------------------------------------------------


def _ingest_lancedb(spec: DatasetSpec, docs: list[dict[str, Any]]) -> None:
    import lancedb

    db_path = str(DATA_DIR / "lancedb")
    db = lancedb.connect(db_path)
    if spec.table in db.table_names():
        print(f"    [{spec.table}] already exists, skipping")
        return
    db.create_table(spec.table, docs)
    print(f"    [{spec.table}] created ({len(docs)} docs)")


def _ingest_chromadb(spec: DatasetSpec, docs: list[dict[str, Any]]) -> None:
    import chromadb

    db_path = str(DATA_DIR / "chromadb")
    client = chromadb.PersistentClient(path=db_path)
    try:
        col = client.get_collection(spec.table)
        print(f"    [{spec.table}] already exists ({col.count()} docs), skipping")
        return
    except Exception:
        pass
    col = client.create_collection(spec.table)
    batch_size = 5000
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        col.add(
            ids=[d["corpus_id"] for d in batch],
            documents=[d["content"] for d in batch],
            embeddings=[d["vector"] for d in batch],
            metadatas=[
                {"corpus_id": d["corpus_id"], "title": d.get("title", "")}
                for d in batch
            ],
        )
    print(f"    [{spec.table}] created ({len(docs)} docs)")


def _ingest_qdrant(spec: DatasetSpec, docs: list[dict[str, Any]]) -> None:
    from qdrant_client import QdrantClient, models

    db_path = str(DATA_DIR / "qdrant")
    client = QdrantClient(path=db_path)
    existing = [c.name for c in client.get_collections().collections]
    if spec.table in existing:
        info = client.get_collection(spec.table)
        print(f"    [{spec.table}] already exists ({info.points_count} docs), skipping")
        client.close()
        return
    client.create_collection(
        collection_name=spec.table,
        vectors_config=models.VectorParams(
            size=VECTOR_DIM, distance=models.Distance.COSINE
        ),
    )
    batch_size = 500
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        client.upsert(
            collection_name=spec.table,
            points=[
                models.PointStruct(
                    id=idx + i,
                    vector=d["vector"],
                    payload={
                        "corpus_id": d["corpus_id"],
                        "content": d["content"],
                        "title": d.get("title", ""),
                    },
                )
                for idx, d in enumerate(batch)
            ],
        )
    print(f"    [{spec.table}] created ({len(docs)} docs)")
    client.close()


def _ingest_pgvector(spec: DatasetSpec, docs: list[dict[str, Any]]) -> None:
    import psycopg

    conn = psycopg.connect(PGVECTOR_DSN, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur = conn.execute(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
        (spec.table,),
    )
    if cur.fetchone()[0]:  # type: ignore[index]
        cur2 = conn.execute(f"SELECT count(*) FROM {spec.table}")  # noqa: S608
        count = cur2.fetchone()[0]  # type: ignore[index]
        print(f"    [{spec.table}] already exists ({count} docs), skipping")
        conn.close()
        return
    conn.execute(f"""
        CREATE TABLE {spec.table} (
            id SERIAL PRIMARY KEY,
            corpus_id VARCHAR,
            content TEXT,
            title TEXT,
            embedding vector({VECTOR_DIM})
        )
    """)
    for d in docs:
        conn.execute(
            f"INSERT INTO {spec.table} (corpus_id, content, title, embedding) VALUES (%s, %s, %s, %s)",
            (d["corpus_id"], d["content"], d.get("title", ""), d["vector"]),
        )
    print(f"    [{spec.table}] created ({len(docs)} docs)")
    conn.close()


_INGEST_FNS = {
    "lancedb": _ingest_lancedb,
    "chromadb": _ingest_chromadb,
    "qdrant": _ingest_qdrant,
    "pgvector": _ingest_pgvector,
}


# -- Backend factory -----------------------------------------------------------


def _make_backend(backend_name: str, spec: DatasetSpec):
    if backend_name == "lancedb":
        from rag7 import LanceDBBackend

        return LanceDBBackend(
            table=spec.table,
            db_uri=str(DATA_DIR / "lancedb"),
            embed_fn=ollama_embed,
            text_column="content",
            vector_column="vector",
        )
    elif backend_name == "chromadb":
        from rag7 import ChromaDBBackend

        return ChromaDBBackend(
            collection=spec.table,
            embed_fn=ollama_embed,
            path=str(DATA_DIR / "chromadb"),
        )
    elif backend_name == "qdrant":
        from rag7 import QdrantBackend
        from qdrant_client import QdrantClient

        client = QdrantClient(path=str(DATA_DIR / "qdrant"))
        b = QdrantBackend(collection=spec.table, embed_fn=ollama_embed)
        b._client = client
        return b
    elif backend_name == "pgvector":
        from rag7 import PgvectorBackend

        return PgvectorBackend(
            table=spec.table,
            dsn=PGVECTOR_DSN,
            embed_fn=ollama_embed,
            vector_column="embedding",
            content_column="content",
        )
    raise ValueError(f"Unknown backend: {backend_name}")


# -- Availability checks ------------------------------------------------------


def _ollama_available() -> bool:
    try:
        resp = _requests.get("http://localhost:11434/api/tags", timeout=3)
        return resp.ok
    except Exception:
        return False


def _pgvector_available() -> bool:
    try:
        import psycopg

        conn = psycopg.connect(PGVECTOR_DSN, connect_timeout=2)
        conn.close()
        return True
    except Exception:
        return False


_OLLAMA_UP = _ollama_available()
_PGVECTOR_UP = _pgvector_available()

pytestmark = pytest.mark.skipif(
    not _OLLAMA_UP, reason="Ollama not running at localhost:11434"
)


# -- Result container ----------------------------------------------------------


@dataclass
class DatasetResult:
    name: str
    lang: str
    domain: str
    backend: str = ""
    queries_total: int = 0
    hit_rate_5: float = 0.0
    ndcg_10: float = 0.0
    avg_time_ms: float = 0.0
    hits: int = 0
    misses: int = 0
    errors: list[str] = field(default_factory=list)


# -- Session fixtures ----------------------------------------------------------


@pytest.fixture(scope="session")
def embedded_corpora() -> dict[str, list[dict[str, Any]]]:
    """Load + embed all MTEB corpora once per session. Returns {table: docs_with_vectors}."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpora: dict[str, list[dict[str, Any]]] = {}
    for spec in DATASETS:
        print(f"\n  [LOAD] {spec.name}")
        docs = _load_corpus(spec)
        print(f"  [EMBED] {len(docs)} docs via Ollama {OLLAMA_MODEL}...")
        texts = [d["content"] for d in docs]
        vectors = ollama_embed_batch(texts)
        for doc, vec in zip(docs, vectors):
            doc["vector"] = vec
        corpora[spec.table] = docs
    return corpora


@pytest.fixture(scope="session")
def ingested_lancedb(embedded_corpora: dict[str, list[dict[str, Any]]]) -> set[str]:
    """Ingest all corpora into LanceDB. Returns set of ingested table names."""
    tables: set[str] = set()
    for spec in DATASETS:
        try:
            _ingest_lancedb(spec, embedded_corpora[spec.table])
            tables.add(spec.table)
        except Exception as e:
            print(f"  [SKIP] lancedb/{spec.table}: {e}")
    return tables


@pytest.fixture(scope="session")
def ingested_chromadb(embedded_corpora: dict[str, list[dict[str, Any]]]) -> set[str]:
    tables: set[str] = set()
    for spec in DATASETS:
        try:
            _ingest_chromadb(spec, embedded_corpora[spec.table])
            tables.add(spec.table)
        except Exception as e:
            print(f"  [SKIP] chromadb/{spec.table}: {e}")
    return tables


@pytest.fixture(scope="session")
def ingested_qdrant(embedded_corpora: dict[str, list[dict[str, Any]]]) -> set[str]:
    tables: set[str] = set()
    for spec in DATASETS:
        try:
            _ingest_qdrant(spec, embedded_corpora[spec.table])
            tables.add(spec.table)
        except Exception as e:
            print(f"  [SKIP] qdrant/{spec.table}: {e}")
    return tables


@pytest.fixture(scope="session")
def ingested_pgvector(embedded_corpora: dict[str, list[dict[str, Any]]]) -> set[str]:
    if not _PGVECTOR_UP:
        return set()
    tables: set[str] = set()
    for spec in DATASETS:
        try:
            _ingest_pgvector(spec, embedded_corpora[spec.table])
            tables.add(spec.table)
        except Exception as e:
            print(f"  [SKIP] pgvector/{spec.table}: {e}")
    return tables


# -- Retrieval runner ----------------------------------------------------------


async def _run_retrieval(spec: DatasetSpec, backend_name: str) -> DatasetResult:
    from rag7 import AgenticRAG

    result = DatasetResult(
        name=spec.name,
        lang=spec.lang,
        domain=spec.domain,
        backend=backend_name,
    )

    queries = _load_queries(spec)
    relevant = _load_qrels(spec)

    run_qids = [qid for qid in queries if qid in relevant]
    if MAX_QUERIES and len(run_qids) > MAX_QUERIES:
        run_qids.sort(key=lambda q: hashlib.md5(q.encode()).hexdigest())
        run_qids = run_qids[:MAX_QUERIES]

    result.queries_total = len(run_qids)

    backend = _make_backend(backend_name, spec)
    rag = AgenticRAG(
        spec.table,
        backend=backend,
        embed_fn=ollama_embed,
        embedder_name="ollama",
    )

    sem = asyncio.Semaphore(CONCURRENCY)
    hit_rates: list[float] = []
    ndcgs: list[float] = []
    times: list[float] = []

    async def _run_one(qid: str) -> None:
        query_text = queries[qid]
        rel_ids = relevant[qid]
        try:
            async with sem:
                t0 = time.monotonic()
                _, docs = await rag._aretrieve_documents(query_text)
                elapsed = (time.monotonic() - t0) * 1000

            retrieved_ids = [str(d.metadata.get("corpus_id", "")) for d in docs]
            hr = hit_rate_at_k(retrieved_ids, rel_ids, k=5)
            nd = ndcg_at_k(retrieved_ids, rel_ids, k=10)
            hit_rates.append(hr)
            ndcgs.append(nd)
            times.append(elapsed)

            status = "HIT " if hr > 0 else "MISS"
            print(f"    {status} | '{query_text[:50]:50s}' top5={retrieved_ids[:3]}")
            if hr > 0:
                result.hits += 1
            else:
                result.misses += 1
        except Exception as e:
            result.errors.append(f"{qid}: {e}")
            print(f"    ERR  | '{query_text[:50]}' -> {e}")

    await asyncio.gather(*[_run_one(qid) for qid in run_qids])

    if hit_rates:
        result.hit_rate_5 = sum(hit_rates) / len(hit_rates)
    if ndcgs:
        result.ndcg_10 = sum(ndcgs) / len(ndcgs)
    if times:
        result.avg_time_ms = sum(times) / len(times)

    return result


# -- Test classes per backend --------------------------------------------------

_dataset_tables = [s.table for s in DATASETS]


class TestLanceDBMTEB:
    """MTEB retrieval benchmark on LanceDB backend."""

    @pytest.mark.parametrize("table", _dataset_tables)
    def test_retrieval(self, table: str, ingested_lancedb: set[str]) -> None:
        spec = DATASET_BY_TABLE[table]
        if table not in ingested_lancedb:
            pytest.skip(f"lancedb/{table} not ingested")

        result = asyncio.run(_run_retrieval(spec, "lancedb"))

        print(
            f"\n  [{spec.name}] lancedb: HR@5={result.hit_rate_5:.3f}  "
            f"nDCG@10={result.ndcg_10:.3f}  {result.avg_time_ms:.0f}ms/q  "
            f"({result.hits}/{result.queries_total} hits)"
        )

        assert result.queries_total > 0, "No queries found"
        assert result.hits > 0, f"Zero hits on {spec.name} — retrieval broken"
        assert not result.errors, f"Errors: {result.errors[:3]}"


class TestChromaDBMTEB:
    """MTEB retrieval benchmark on ChromaDB backend."""

    @pytest.mark.parametrize("table", _dataset_tables)
    def test_retrieval(self, table: str, ingested_chromadb: set[str]) -> None:
        spec = DATASET_BY_TABLE[table]
        if table not in ingested_chromadb:
            pytest.skip(f"chromadb/{table} not ingested")

        result = asyncio.run(_run_retrieval(spec, "chromadb"))

        print(
            f"\n  [{spec.name}] chromadb: HR@5={result.hit_rate_5:.3f}  "
            f"nDCG@10={result.ndcg_10:.3f}  {result.avg_time_ms:.0f}ms/q  "
            f"({result.hits}/{result.queries_total} hits)"
        )

        assert result.queries_total > 0
        assert result.hits > 0, f"Zero hits on {spec.name}"
        assert not result.errors, f"Errors: {result.errors[:3]}"


class TestQdrantMTEB:
    """MTEB retrieval benchmark on Qdrant backend."""

    @pytest.mark.parametrize("table", _dataset_tables)
    def test_retrieval(self, table: str, ingested_qdrant: set[str]) -> None:
        spec = DATASET_BY_TABLE[table]
        if table not in ingested_qdrant:
            pytest.skip(f"qdrant/{table} not ingested")

        result = asyncio.run(_run_retrieval(spec, "qdrant"))

        print(
            f"\n  [{spec.name}] qdrant: HR@5={result.hit_rate_5:.3f}  "
            f"nDCG@10={result.ndcg_10:.3f}  {result.avg_time_ms:.0f}ms/q  "
            f"({result.hits}/{result.queries_total} hits)"
        )

        assert result.queries_total > 0
        assert result.hits > 0, f"Zero hits on {spec.name}"
        assert not result.errors, f"Errors: {result.errors[:3]}"


@pytest.mark.skipif(
    not _PGVECTOR_UP,
    reason="pgvector not running (podman compose -f docker-compose.test.yml up -d)",
)
class TestPgvectorMTEB:
    """MTEB retrieval benchmark on pgvector backend."""

    @pytest.mark.parametrize("table", _dataset_tables)
    def test_retrieval(self, table: str, ingested_pgvector: set[str]) -> None:
        spec = DATASET_BY_TABLE[table]
        if table not in ingested_pgvector:
            pytest.skip(f"pgvector/{table} not ingested")

        result = asyncio.run(_run_retrieval(spec, "pgvector"))

        print(
            f"\n  [{spec.name}] pgvector: HR@5={result.hit_rate_5:.3f}  "
            f"nDCG@10={result.ndcg_10:.3f}  {result.avg_time_ms:.0f}ms/q  "
            f"({result.hits}/{result.queries_total} hits)"
        )

        assert result.queries_total > 0
        assert result.hits > 0, f"Zero hits on {spec.name}"
        assert not result.errors, f"Errors: {result.errors[:3]}"


# -- Terminal summary ----------------------------------------------------------


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print summary hint after all tests complete."""
    terminalreporter.section("MTEB Benchmark")
    terminalreporter.line(
        "Run with -s flag to see per-query HIT/MISS details and metric scores."
    )
