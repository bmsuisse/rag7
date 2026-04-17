"""IR Benchmark tests — NQ, MS MARCO, HotpotQA slices with real embeddings.

Loads small slices from standard IR datasets via HuggingFace, embeds with
Azure OpenAI (or Ollama fallback), ingests into multiple backends, and
measures hit_rate@5 and nDCG@10.

Requires:
  - Azure OpenAI env vars OR Ollama running at localhost:11434
  - Docker services for pgvector, Qdrant (optional — skips if unavailable)
  - DuckDB + ChromaDB always available (in-memory)

Run:
  uv run pytest tests/test_ir_benchmarks.py -v -s
  uv run pytest tests/test_ir_benchmarks.py -k nq -v -s          # single dataset
  uv run pytest tests/test_ir_benchmarks.py -k duckdb -v -s      # single backend
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
from dotenv import load_dotenv

load_dotenv()

from rag7.backend import SearchRequest

# ── Config ──────────────────────────────────────────────────────────────────

PGVECTOR_DSN = "postgresql://test:test@localhost:5433/testdb"
MAX_CORPUS_DOCS = 500
MAX_QUERIES = 30
VECTOR_DIM: int | None = None


# ── Dataset specs ───────────────────────────────────────────────────────────


@dataclass
class IRDatasetSpec:
    name: str
    hf_dataset: str
    corpus_config: str
    queries_config: str
    qrels_config: str
    domain: str
    text_field: str = "text"
    title_field: str | None = "title"
    id_field: str = "_id"
    qrels_hf_dataset: str | None = None


IR_DATASETS: list[IRDatasetSpec] = [
    IRDatasetSpec(
        name="NQ",
        hf_dataset="mteb/nq",
        corpus_config="corpus",
        queries_config="queries",
        qrels_config="default",
        domain="Wikipedia QA",
    ),
    IRDatasetSpec(
        name="MS MARCO",
        hf_dataset="mteb/msmarco-v2",
        corpus_config="corpus",
        queries_config="queries",
        qrels_config="default",
        domain="Web Search",
    ),
    IRDatasetSpec(
        name="HotpotQA",
        hf_dataset="mteb/hotpotqa",
        corpus_config="corpus",
        queries_config="queries",
        qrels_config="default",
        domain="Multi-hop QA",
    ),
]

IR_DATASET_BY_NAME = {s.name: s for s in IR_DATASETS}


# ── Embedding ───────────────────────────────────────────────────────────────


def _make_embed_fn():
    """Try Azure OpenAI first, fall back to Ollama."""
    from rag7.utils import _make_azure_embed_fn

    fn = _make_azure_embed_fn()
    if fn is not None:
        return fn, "azure"

    import requests

    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.ok:
            session = requests.Session()

            def _ollama_embed(text: str) -> list[float]:
                r = session.post(
                    "http://localhost:11434/api/embed",
                    json={"model": "embeddinggemma", "input": text[:8000] or "empty"},
                    timeout=60,
                )
                r.raise_for_status()
                return r.json()["embeddings"][0]

            return _ollama_embed, "ollama"
    except Exception:
        pass

    return None, None


_embed_fn, _embed_source = _make_embed_fn()

pytestmark = pytest.mark.skipif(
    _embed_fn is None,
    reason="No embedding source (need AZURE_OPENAI_ENDPOINT or Ollama)",
)


# ── HF data loading ────────────────────────────────────────────────────────


def _first_split(ds: Any) -> Any:
    return ds[list(ds.keys())[0]]


def _load_qrels(spec: IRDatasetSpec) -> dict[str, list[str]]:
    from datasets import load_dataset

    hf_id = spec.qrels_hf_dataset or spec.hf_dataset
    ds = load_dataset(hf_id, spec.qrels_config)
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


def _load_queries(spec: IRDatasetSpec) -> dict[str, str]:
    from datasets import load_dataset

    ds = load_dataset(spec.hf_dataset, spec.queries_config)
    queries_ds = _first_split(ds)
    return {str(row[spec.id_field]): row[spec.text_field] for row in queries_ds}


def _load_corpus(spec: IRDatasetSpec) -> tuple[list[dict], dict[str, list[str]]]:
    from datasets import load_dataset

    relevant = _load_qrels(spec)
    needed_ids = set()
    for ids in relevant.values():
        needed_ids.update(ids)

    ds = load_dataset(spec.hf_dataset, spec.corpus_config)
    corpus_ds = _first_split(ds)

    docs: list[dict] = []
    for row in corpus_ds:
        doc_id = str(row[spec.id_field])
        text = row.get(spec.text_field) or ""
        title = (row.get(spec.title_field) or "") if spec.title_field else ""
        content = f"{title}\n{text}".strip() if title else text
        docs.append(
            {
                "corpus_id": doc_id,
                "content": content[:4000],
                "title": title,
                "_needed": doc_id in needed_ids,
            }
        )

    needed = [d for d in docs if d["_needed"]]
    others = [d for d in docs if not d["_needed"]]
    others.sort(key=lambda d: hashlib.md5(d["corpus_id"].encode()).hexdigest())
    noise_cap = max(50, MAX_CORPUS_DOCS - len(needed))
    final = needed + others[:noise_cap]
    for d in final:
        d.pop("_needed", None)

    print(
        f"    {spec.name}: {len(needed)} relevant + {min(len(others), noise_cap)} noise = {len(final)} docs"
    )
    return final, relevant


# ── Embed corpus ────────────────────────────────────────────────────────────


def _embed_docs(docs: list[dict]) -> list[dict]:
    global VECTOR_DIM
    from tqdm import tqdm

    for d in tqdm(docs, desc="  embedding corpus", unit="doc"):
        vec = _embed_fn(d["content"])
        d["vector"] = vec
        if VECTOR_DIM is None:
            VECTOR_DIM = len(vec)
    return docs


# ── Metrics ─────────────────────────────────────────────────────────────────


def hit_rate_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int = 10) -> float:
    rel_set = set(relevant)
    dcg = sum(
        1.0 / math.log2(i + 2) for i, did in enumerate(retrieved[:k]) if did in rel_set
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class BenchmarkResult:
    dataset: str
    backend: str
    domain: str
    n_queries: int = 0
    hit_rate_5: float = 0.0
    ndcg_10: float = 0.0
    avg_latency_ms: float = 0.0
    hits: int = 0
    misses: int = 0
    errors: list[str] = field(default_factory=list)


# ── Session fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def ir_corpora() -> dict[str, tuple[list[dict], dict[str, list[str]], dict[str, str]]]:
    """Load + embed all IR corpora once per session."""
    corpora = {}
    for spec in IR_DATASETS:
        print(f"\n  [LOAD] {spec.name}")
        try:
            docs, relevant = _load_corpus(spec)
            queries = _load_queries(spec)
            print(f"  [EMBED] {len(docs)} docs via {_embed_source}...")
            docs = _embed_docs(docs)
            corpora[spec.name] = (docs, relevant, queries)
        except Exception as e:
            print(f"  [SKIP] {spec.name}: {e}")
    return corpora


# ── Benchmark runner ────────────────────────────────────────────────────────


def _run_benchmark(
    backend,
    queries: dict[str, str],
    relevant: dict[str, list[str]],
    dataset_name: str,
    backend_name: str,
    domain: str,
) -> BenchmarkResult:
    result = BenchmarkResult(dataset=dataset_name, backend=backend_name, domain=domain)

    run_qids = [qid for qid in queries if qid in relevant]
    run_qids.sort(key=lambda q: hashlib.md5(q.encode()).hexdigest())
    run_qids = run_qids[:MAX_QUERIES]
    result.n_queries = len(run_qids)

    hit_rates = []
    ndcgs = []
    latencies = []

    for qid in run_qids:
        query_text = queries[qid]
        rel_ids = relevant[qid]

        try:
            t0 = time.monotonic()
            hits = backend.search(SearchRequest(query=query_text, limit=20))
            elapsed = (time.monotonic() - t0) * 1000

            retrieved_ids = [str(h.get("corpus_id", "")) for h in hits]
            hr = hit_rate_at_k(retrieved_ids, rel_ids, k=5)
            nd = ndcg_at_k(retrieved_ids, rel_ids, k=10)

            hit_rates.append(hr)
            ndcgs.append(nd)
            latencies.append(elapsed)

            status = "HIT " if hr > 0 else "MISS"
            result.hits += 1 if hr > 0 else 0
            result.misses += 0 if hr > 0 else 1
            print(f"    {status} [{elapsed:6.0f}ms] '{query_text[:60]}'")
        except Exception as e:
            result.errors.append(f"{qid}: {e}")

    if hit_rates:
        result.hit_rate_5 = sum(hit_rates) / len(hit_rates)
    if ndcgs:
        result.ndcg_10 = sum(ndcgs) / len(ndcgs)
    if latencies:
        result.avg_latency_ms = sum(latencies) / len(latencies)

    return result


def _print_result(result: BenchmarkResult) -> None:
    print(
        f"\n  [{result.dataset}] {result.backend}: "
        f"HR@5={result.hit_rate_5:.3f}  nDCG@10={result.ndcg_10:.3f}  "
        f"{result.avg_latency_ms:.0f}ms/q  "
        f"({result.hits}/{result.n_queries} hits)"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  DuckDB (in-memory)
# ═══════════════════════════════════════════════════════════════════════════


_ir_names = [s.name for s in IR_DATASETS]


class TestDuckDBIR:
    @pytest.mark.parametrize("ds_name", _ir_names)
    def test_retrieval(self, ds_name: str, ir_corpora) -> None:
        if ds_name not in ir_corpora:
            pytest.skip(f"{ds_name} not loaded")

        duckdb = pytest.importorskip("duckdb")
        from rag7.backend import DuckDBBackend

        docs, relevant, queries = ir_corpora[ds_name]
        dim = len(docs[0]["vector"])

        conn = duckdb.connect(":memory:")
        conn.execute(f"""
            CREATE TABLE ir_docs (
                corpus_id VARCHAR,
                content VARCHAR,
                title VARCHAR,
                embedding FLOAT[{dim}]
            )
        """)
        for d in docs:
            conn.execute(
                f"INSERT INTO ir_docs VALUES (?,?,?,?::FLOAT[{dim}])",
                [d["corpus_id"], d["content"], d.get("title", ""), d["vector"]],
            )

        backend = DuckDBBackend(
            table="ir_docs",
            db_path=":memory:",
            embed_fn=_embed_fn,
            vector_column="embedding",
            content_column="content",
        )
        backend._conn = conn

        result = _run_benchmark(
            backend,
            queries,
            relevant,
            ds_name,
            "duckdb",
            IR_DATASET_BY_NAME[ds_name].domain,
        )
        _print_result(result)

        assert result.n_queries > 0
        assert result.hit_rate_5 > 0.1, f"Hit rate too low: {result.hit_rate_5:.3f}"


# ═══════════════════════════════════════════════════════════════════════════
#  ChromaDB (in-memory)
# ═══════════════════════════════════════════════════════════════════════════


class TestChromaDBIR:
    @pytest.mark.parametrize("ds_name", _ir_names)
    def test_retrieval(self, ds_name: str, ir_corpora) -> None:
        if ds_name not in ir_corpora:
            pytest.skip(f"{ds_name} not loaded")

        pytest.importorskip("chromadb")
        from rag7.backend import ChromaDBBackend

        docs, relevant, queries = ir_corpora[ds_name]
        table = ds_name.lower().replace(" ", "_")

        backend = ChromaDBBackend(collection=f"ir_{table}", embed_fn=_embed_fn)
        batch_size = 5000
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            backend._collection.add(
                ids=[d["corpus_id"] for d in batch],
                documents=[d["content"] for d in batch],
                embeddings=[d["vector"] for d in batch],
                metadatas=[
                    {"corpus_id": d["corpus_id"], "title": d.get("title", "")}
                    for d in batch
                ],
            )

        result = _run_benchmark(
            backend,
            queries,
            relevant,
            ds_name,
            "chromadb",
            IR_DATASET_BY_NAME[ds_name].domain,
        )
        _print_result(result)

        try:
            backend._client.delete_collection(f"ir_{table}")
        except Exception:
            pass

        assert result.n_queries > 0
        assert result.hit_rate_5 > 0.1, f"Hit rate too low: {result.hit_rate_5:.3f}"


# ═══════════════════════════════════════════════════════════════════════════
#  Qdrant (docker)
# ═══════════════════════════════════════════════════════════════════════════


def _qdrant_available() -> bool:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(host="localhost", port=6333, timeout=2)
        client.get_collections()
        client.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _qdrant_available(), reason="Qdrant not running")
class TestQdrantIR:
    @pytest.mark.parametrize("ds_name", _ir_names)
    def test_retrieval(self, ds_name: str, ir_corpora) -> None:
        if ds_name not in ir_corpora:
            pytest.skip(f"{ds_name} not loaded")

        from qdrant_client import QdrantClient, models
        from rag7.backend import QdrantBackend

        docs, relevant, queries = ir_corpora[ds_name]
        dim = len(docs[0]["vector"])
        col_name = f"ir_{ds_name.lower().replace(' ', '_')}"

        client = QdrantClient(host="localhost", port=6333)
        try:
            client.delete_collection(col_name)
        except Exception:
            pass

        client.create_collection(
            collection_name=col_name,
            vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE
            ),
        )

        batch_size = 500
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            client.upsert(
                collection_name=col_name,
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

        backend = QdrantBackend(collection=col_name, embed_fn=_embed_fn)
        backend._client = client

        result = _run_benchmark(
            backend,
            queries,
            relevant,
            ds_name,
            "qdrant",
            IR_DATASET_BY_NAME[ds_name].domain,
        )
        _print_result(result)

        try:
            client.delete_collection(col_name)
        except Exception:
            pass

        assert result.n_queries > 0
        assert result.hit_rate_5 > 0.1, f"Hit rate too low: {result.hit_rate_5:.3f}"


# ═══════════════════════════════════════════════════════════════════════════
#  pgvector (docker)
# ═══════════════════════════════════════════════════════════════════════════


def _pgvector_available() -> bool:
    try:
        import psycopg

        conn = psycopg.connect(PGVECTOR_DSN, connect_timeout=2)
        conn.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _pgvector_available(), reason="pgvector not running")
class TestPgvectorIR:
    @pytest.mark.parametrize("ds_name", _ir_names)
    def test_retrieval(self, ds_name: str, ir_corpora) -> None:
        if ds_name not in ir_corpora:
            pytest.skip(f"{ds_name} not loaded")

        import psycopg
        from rag7.backend import PgvectorBackend

        docs, relevant, queries = ir_corpora[ds_name]
        dim = len(docs[0]["vector"])
        table = f"ir_{ds_name.lower().replace(' ', '_')}"

        conn = psycopg.connect(PGVECTOR_DSN, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(f"""
            CREATE TABLE {table} (
                id SERIAL PRIMARY KEY,
                corpus_id VARCHAR,
                content TEXT,
                title TEXT,
                embedding vector({dim})
            )
        """)
        for d in docs:
            conn.execute(
                f"INSERT INTO {table} (corpus_id, content, title, embedding) VALUES (%s, %s, %s, %s)",
                (d["corpus_id"], d["content"], d.get("title", ""), d["vector"]),
            )
        conn.close()

        backend = PgvectorBackend(
            table=table,
            dsn=PGVECTOR_DSN,
            embed_fn=_embed_fn,
            vector_column="embedding",
            content_column="content",
        )

        result = _run_benchmark(
            backend,
            queries,
            relevant,
            ds_name,
            "pgvector",
            IR_DATASET_BY_NAME[ds_name].domain,
        )
        _print_result(result)

        try:
            conn2 = psycopg.connect(PGVECTOR_DSN, autocommit=True)
            conn2.execute(f"DROP TABLE IF EXISTS {table}")
            conn2.close()
        except Exception:
            pass

        assert result.n_queries > 0
        assert result.hit_rate_5 > 0.1, f"Hit rate too low: {result.hit_rate_5:.3f}"
