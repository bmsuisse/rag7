"""Populate all search backends with benchmark data from local parquet files.

Loads documents, generates embeddings via Azure OpenAI, and pushes to:
  - pgvector (PostgreSQL)
  - Qdrant (local or in-memory)
  - LanceDB (local file)
  - Azure AI Search
  - DuckDB (local file)

Usage:
    uv run python scripts/populate_backends.py [--max-docs 200] [--backends all]

Requires: .env with AZURE_OPENAI_* keys for embeddings.
pgvector requires: podman compose -f docker-compose.test.yml up -d
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
EMBED_DIM = 1536  # text-embedding-3-small
BATCH_SIZE = 100  # docs per embedding API call
ALLOWED_FIELDS = {"id", "title", "content", "language", "source", "url"}
INDEX_NAME = "benchmark-docs"

PGVECTOR_DSN = "postgresql://test:test@localhost:5433/testdb"

ALL_BACKENDS = ["pgvector", "qdrant", "lancedb", "azure", "duckdb"]


# ── Embedding ─────────────────────────────────────────────────────────────────


def make_embed_fn() -> Any:
    """Create batch embedding function using Azure OpenAI."""
    import requests as _requests

    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or ""
    deploy = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    api_ver = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not endpoint or not api_key:
        print(
            "WARNING: No AZURE_OPENAI_ENDPOINT/KEY — using deterministic fake embeddings"
        )
        return None

    url = f"{endpoint}/openai/deployments/{deploy}/embeddings?api-version={api_ver}"
    session = _requests.Session()
    session.headers.update({"api-key": api_key, "Content-Type": "application/json"})

    def _embed_batch(texts: list[str]) -> list[list[float]]:
        # Truncate to ~8000 tokens per text to stay within limits
        truncated = [t[:8000] for t in texts]
        resp = session.post(url, json={"input": truncated}, timeout=60)
        resp.raise_for_status()
        data = resp.json()["data"]
        return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]

    return _embed_batch


def fake_embed_batch(texts: list[str]) -> list[list[float]]:
    """Deterministic fake embeddings from text hash — for offline testing."""
    results = []
    for text in texts:
        h = hashlib.sha256(text.encode()).digest() * (EMBED_DIM // 32 + 1)
        vec = [float(b) / 255.0 for b in h[:EMBED_DIM]]
        norm = sum(v * v for v in vec) ** 0.5
        results.append([v / norm for v in vec])
    return results


# ── Data loading ──────────────────────────────────────────────────────────────


def load_docs(max_docs: int) -> pd.DataFrame:
    """Load parquet files, sample evenly across sources, cap at max_docs."""
    frames = []
    for parquet_file in sorted(DATA_DIR.glob("**/*.parquet")):
        df = pd.read_parquet(parquet_file)
        if df.empty:
            continue
        # Normalize columns to ALLOWED_FIELDS
        df = df[[c for c in df.columns if c in ALLOWED_FIELDS]]
        for col in ALLOWED_FIELDS:
            if col not in df.columns:
                df[col] = ""
        frames.append(df)

    if not frames:
        print("ERROR: No parquet files found in data/")
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["id"])
    merged = merged[merged["content"].str.len() > 50]  # skip empty/tiny docs

    # Sample evenly across source+language combos for diversity
    groups = merged.groupby(["source", "language"])
    per_group = max(1, max_docs // groups.ngroups)
    parts = []
    for _, group_df in groups:
        parts.append(group_df.sample(n=min(len(group_df), per_group), random_state=42))
    sampled = pd.concat(parts, ignore_index=True).head(max_docs)
    sources = sampled.groupby("source").size().to_dict()
    print(f"Loaded {len(sampled)} docs from {len(frames)} files — {sources}")
    return sampled


def embed_docs(df: pd.DataFrame, embed_batch_fn: Any) -> list[list[float]]:
    """Generate embeddings for all docs in batches."""
    all_vectors: list[list[float]] = []
    contents = df["content"].tolist()
    total = len(contents)

    for i in range(0, total, BATCH_SIZE):
        batch = contents[i : i + BATCH_SIZE]
        vectors = embed_batch_fn(batch)
        all_vectors.extend(vectors)
        done = min(i + BATCH_SIZE, total)
        print(f"  Embedded {done}/{total}", end="\r")
        if embed_batch_fn is not fake_embed_batch:
            time.sleep(0.2)  # rate-limit courtesy

    print(f"  Embedded {total}/{total} docs")
    return all_vectors


# ── Backend populators ────────────────────────────────────────────────────────


def populate_pgvector(docs: list[dict], vectors: list[list[float]]) -> None:
    """Populate pgvector table with documents and embeddings."""
    import psycopg
    from pgvector.psycopg import register_vector

    print("\n[pgvector] Connecting...")
    conn = psycopg.connect(PGVECTOR_DSN, autocommit=True)
    register_vector(conn)

    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(f"DROP TABLE IF EXISTS {INDEX_NAME.replace('-', '_')}")
    table = INDEX_NAME.replace("-", "_")
    conn.execute(f"""
        CREATE TABLE {table} (
            id VARCHAR PRIMARY KEY,
            title TEXT,
            content TEXT,
            language VARCHAR(10),
            source VARCHAR(200),
            url TEXT,
            embedding vector({EMBED_DIM})
        )
    """)

    for i, (doc, vec) in enumerate(zip(docs, vectors)):
        conn.execute(
            f"INSERT INTO {table} (id, title, content, language, source, url, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
            (
                doc["id"],
                doc["title"],
                doc["content"][:32000],
                doc["language"],
                doc["source"],
                doc["url"],
                vec,
            ),
        )
        if (i + 1) % 100 == 0:
            print(f"  Inserted {i + 1}/{len(docs)}", end="\r")

    # Create HNSW index for fast search
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table}_embedding
        ON {table} USING hnsw (embedding vector_cosine_ops)
    """)
    conn.close()
    print(f"  [pgvector] {len(docs)} docs → table '{table}'")


def populate_qdrant(docs: list[dict], vectors: list[list[float]]) -> None:
    """Populate Qdrant collection with documents and embeddings."""
    from qdrant_client import QdrantClient, models

    print("\n[qdrant] Creating in-memory collection...")
    path = "data/qdrant_benchmark"
    client = QdrantClient(path=path)

    # Recreate collection
    try:
        client.delete_collection(INDEX_NAME)
    except Exception:
        pass

    client.create_collection(
        collection_name=INDEX_NAME,
        vectors_config=models.VectorParams(
            size=EMBED_DIM, distance=models.Distance.COSINE
        ),
    )

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        batch_vecs = vectors[i : i + batch_size]
        points = [
            models.PointStruct(
                id=idx + i,
                vector=vec,
                payload={
                    "id": d["id"],
                    "title": d["title"],
                    "content": d["content"][:32000],
                    "language": d["language"],
                    "source": d["source"],
                    "url": d["url"],
                },
            )
            for idx, (d, vec) in enumerate(zip(batch_docs, batch_vecs))
        ]
        client.upsert(collection_name=INDEX_NAME, points=points)
        print(f"  Upserted {min(i + batch_size, len(docs))}/{len(docs)}", end="\r")

    client.close()
    print(f"  [qdrant] {len(docs)} docs → collection '{INDEX_NAME}' at {path}")


def populate_lancedb(docs: list[dict], vectors: list[list[float]]) -> None:
    """Populate LanceDB table with documents and embeddings (polars/arrow)."""
    import lancedb
    import polars as pl

    db_path = "data/lancedb_benchmark"
    print(f"\n[lancedb] Writing to {db_path}...")
    db = lancedb.connect(db_path)

    # Drop if exists
    try:
        db.drop_table(INDEX_NAME.replace("-", "_"))
    except Exception:
        pass

    table_name = INDEX_NAME.replace("-", "_")
    df = pl.DataFrame(
        {
            "id": [d["id"] for d in docs],
            "title": [d["title"] for d in docs],
            "content": [d["content"][:32000] for d in docs],
            "language": [d["language"] for d in docs],
            "source": [d["source"] for d in docs],
            "url": [d["url"] for d in docs],
            "vector": vectors,
        }
    )

    db.create_table(table_name, data=df.to_arrow())
    print(f"  [lancedb] {len(docs)} docs → table '{table_name}' at {db_path}")


def populate_azure(docs: list[dict]) -> None:
    """Populate Azure AI Search index (text-only, Azure handles embeddings)."""
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        SearchableField,
        SearchFieldDataType,
        SearchIndex,
        SimpleField,
    )

    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    api_key = os.getenv("AZURE_SEARCH_API_KEY", "")
    if not endpoint or not api_key:
        print("\n[azure] Skipped — AZURE_SEARCH_ENDPOINT/KEY not set")
        return

    print("\n[azure] Creating index...")
    cred = AzureKeyCredential(api_key)

    # Create index
    index_client = SearchIndexClient(endpoint=endpoint, credential=cred)
    fields = [
        SimpleField(
            name="id", type=SearchFieldDataType.String, key=True, filterable=True
        ),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(
            name="language",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(name="url", type=SearchFieldDataType.String),
    ]
    index = SearchIndex(name=INDEX_NAME, fields=fields)
    try:
        index_client.delete_index(INDEX_NAME)
    except Exception:
        pass
    index_client.create_index(index)

    # Upload docs in batches
    search_client = SearchClient(
        endpoint=endpoint, index_name=INDEX_NAME, credential=cred
    )
    batch_size = 100
    total = 0
    for i in range(0, len(docs), batch_size):
        batch = [
            {
                "id": d["id"],
                "title": str(d.get("title", "")),
                "content": str(d["content"])[:32000],
                "language": str(d.get("language", "")),
                "source": str(d.get("source", "")),
                "url": str(d.get("url", "")),
            }
            for d in docs[i : i + batch_size]
        ]
        result = search_client.upload_documents(batch)
        succeeded = sum(1 for r in result if r.succeeded)
        total += succeeded
        print(f"  Uploaded {total}/{len(docs)}", end="\r")

    print(f"  [azure] {total} docs → index '{INDEX_NAME}'")


def populate_duckdb(docs: list[dict], vectors: list[list[float]]) -> None:
    """Populate DuckDB table with documents and embeddings."""
    import duckdb

    db_path = "data/duckdb_benchmark.db"
    print(f"\n[duckdb] Writing to {db_path}...")
    conn = duckdb.connect(db_path)
    table = INDEX_NAME.replace("-", "_")

    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(f"""
        CREATE TABLE {table} (
            id VARCHAR PRIMARY KEY,
            title VARCHAR,
            content VARCHAR,
            language VARCHAR,
            source VARCHAR,
            url VARCHAR,
            embedding FLOAT[{EMBED_DIM}]
        )
    """)

    for i, (doc, vec) in enumerate(zip(docs, vectors)):
        conn.execute(
            f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?::FLOAT[{EMBED_DIM}])",
            [
                doc["id"],
                doc["title"],
                doc["content"][:32000],
                doc["language"],
                doc["source"],
                doc["url"],
                vec,
            ],
        )
        if (i + 1) % 100 == 0:
            print(f"  Inserted {i + 1}/{len(docs)}", end="\r")

    conn.close()
    print(f"  [duckdb] {len(docs)} docs → table '{table}' at {db_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

POPULATORS: dict[str, Any] = {
    "pgvector": lambda docs, vecs: populate_pgvector(docs, vecs),
    "qdrant": lambda docs, vecs: populate_qdrant(docs, vecs),
    "lancedb": lambda docs, vecs: populate_lancedb(docs, vecs),
    "azure": lambda docs, vecs: populate_azure(docs),
    "duckdb": lambda docs, vecs: populate_duckdb(docs, vecs),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate search backends with benchmark data"
    )
    parser.add_argument(
        "--max-docs", type=int, default=200, help="Max docs to load (default: 200)"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=ALL_BACKENDS,
        choices=ALL_BACKENDS + ["all"],
        help="Backends to populate (default: all)",
    )
    args = parser.parse_args()

    backends = ALL_BACKENDS if "all" in args.backends else args.backends

    # Load data
    df = load_docs(args.max_docs)
    docs = df.to_dict("records")

    # Generate embeddings (skip for azure — text-only search)
    needs_vectors = [b for b in backends if b != "azure"]
    vectors: list[list[float]] = []
    if needs_vectors:
        embed_fn = make_embed_fn()
        batch_fn = embed_fn if embed_fn else fake_embed_batch
        if embed_fn is None:
            print("Using fake embeddings (no Azure OpenAI key)")
        print(f"\nGenerating {len(docs)} embeddings...")
        vectors = embed_docs(df, batch_fn)

    # Populate each backend
    for name in backends:
        try:
            POPULATORS[name](docs, vectors)
        except Exception as e:
            print(f"\n[{name}] FAILED: {e}")

    print(f"\nDone. Populated {len(backends)} backends with {len(docs)} docs.")


if __name__ == "__main__":
    main()
