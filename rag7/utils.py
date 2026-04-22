from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Coroutine

import requests as _requests
from stop_words import get_stop_words as _get_stop_words


def _run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine from sync code, tolerating an already-running loop.

    Databricks notebooks, Jupyter, FastAPI handlers, etc. already have a
    loop running, so asyncio.run() raises RuntimeError. In that case run
    the coroutine in a worker thread with its own loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


_DEFAULT_STOP_LANG = os.getenv("RAG_STOP_WORDS_LANG", "de")
_STOP_WORDS: frozenset[str] = frozenset(
    w.lower() for w in _get_stop_words(_DEFAULT_STOP_LANG)
)

_SKIP_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "url",
        "web_image_url",
        "update_date",
        "corpus_id",
        "language",
        "_id",
        "score",
        "_rankingScore",
    }
)


def _strip_stop_words(text: str) -> str:
    return " ".join(w for w in text.split() if w.lower() not in _STOP_WORDS)


def _doc_id(meta: dict) -> str:
    return str(meta.get("id") or meta.get("article_id") or meta.get("corpus_id", ""))


# ── Embedding cache ───────────────────────────────────────────────────────────
# Enabled by default (embedding is deterministic for a given model + text).
# Opt-out with RAG7_EMBED_CACHE=0.
#
# Backend picked by RAG7_EMBED_CACHE_URL:
#   unset / "local" / path      → single-file JSON at ~/.cache/rag7/embeddings/
#   "redis://host:port/db"      → Redis, one key per (model, sha256(text))
#   "postgres://user@host/db"   → Postgres table (reuses ``_cache.py`` schema)
# Any backend that can't load its driver (``redis`` / ``psycopg``) silently
# falls back to the local disk cache.

_EMBED_CACHE_MAX_BYTES = 50 * 1024 * 1024  # 50 MB
_EMBED_CACHE_LOCK = threading.Lock()
_EMBED_CACHE_MEM: dict[str, dict[str, list[float]]] = {}
_EMBED_CACHE_DIRTY: set[str] = set()
_EMBED_REDIS_CLIENT: Any = None
_EMBED_PG_POOL: Any = None
_EMBED_BACKEND_READY = False
_EMBED_BACKEND: str = "local"  # "local" | "redis" | "pg"


def _embed_cache_enabled() -> bool:
    return os.environ.get("RAG7_EMBED_CACHE", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _embed_cache_backend() -> str:
    """Resolve + lazy-init the active cache backend from ``RAG7_EMBED_CACHE_URL``.

    Returns 'local' / 'redis' / 'pg'. Caches the resolution after first call.
    """
    global _EMBED_BACKEND_READY, _EMBED_BACKEND, _EMBED_REDIS_CLIENT, _EMBED_PG_POOL
    if _EMBED_BACKEND_READY:
        return _EMBED_BACKEND
    _EMBED_BACKEND_READY = True
    url = os.environ.get("RAG7_EMBED_CACHE_URL", "").strip()
    if url.startswith("redis://") or url.startswith("rediss://"):
        try:
            import redis  # type: ignore[import-not-found]

            _EMBED_REDIS_CLIENT = redis.Redis.from_url(url, decode_responses=False)
            _EMBED_REDIS_CLIENT.ping()
            _EMBED_BACKEND = "redis"
        except Exception:
            _EMBED_BACKEND = "local"
    elif url.startswith("postgres://") or url.startswith("postgresql://"):
        try:
            import logging as _logging

            # Silence the retry barrage on invalid connection strings.
            _logging.getLogger("psycopg.pool").setLevel(_logging.CRITICAL)
            from psycopg_pool import ConnectionPool  # type: ignore[import-not-found]

            _EMBED_PG_POOL = ConnectionPool(
                url, min_size=1, max_size=4, open=False, timeout=2.0
            )
            _EMBED_PG_POOL.open(wait=True, timeout=2.0)
            with _EMBED_PG_POOL.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS rag7_embed_cache ("
                    "  model TEXT NOT NULL,"
                    "  key TEXT NOT NULL,"
                    "  vec JSONB NOT NULL,"
                    "  created_at TIMESTAMPTZ DEFAULT NOW(),"
                    "  PRIMARY KEY (model, key)"
                    ")"
                )
                conn.commit()
            _EMBED_BACKEND = "pg"
        except Exception:
            _EMBED_BACKEND = "local"
    else:
        _EMBED_BACKEND = "local"
    return _EMBED_BACKEND


def _embed_cache_dir() -> Path:
    return Path.home() / ".cache" / "rag7" / "embeddings"


def _embed_cache_path(model: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model) or "model"
    return _embed_cache_dir() / f"{safe}.json"


def _embed_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed_cache_load(model: str) -> dict[str, list[float]]:
    if model in _EMBED_CACHE_MEM:
        return _EMBED_CACHE_MEM[model]
    path = _embed_cache_path(model)
    data: dict[str, list[float]] = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, list):
                        data[k] = [float(x) for x in v]
        except Exception:
            data = {}
    _EMBED_CACHE_MEM[model] = data
    return data


def _embed_cache_flush(model: str) -> None:
    if model not in _EMBED_CACHE_DIRTY:
        return
    data = _EMBED_CACHE_MEM.get(model)
    if data is None:
        return
    path = _embed_cache_path(model)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)
        _EMBED_CACHE_DIRTY.discard(model)
    except Exception:
        pass


def _embed_cache_trim(data: dict[str, list[float]]) -> None:
    # Approx JSON size: each entry is "hash": [...floats...] ~ 65 + ~18*dim bytes.
    # Drop oldest (FIFO by dict insertion order) until estimate fits.
    if not data:
        return
    any_vec = next(iter(data.values()))
    per_entry = 70 + 18 * len(any_vec)
    max_entries = max(1, _EMBED_CACHE_MAX_BYTES // per_entry)
    while len(data) > max_entries:
        oldest = next(iter(data))
        del data[oldest]


def _embed_cache_get(model: str, text: str) -> list[float] | None:
    if not _embed_cache_enabled():
        return None
    key = _embed_text_hash(text)
    backend = _embed_cache_backend()
    if backend == "redis":
        try:
            raw = _EMBED_REDIS_CLIENT.get(f"rag7:embed:{model}:{key}")
            return [float(x) for x in json.loads(raw)] if raw else None
        except Exception:
            return None
    if backend == "pg":
        try:
            with _EMBED_PG_POOL.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT vec FROM rag7_embed_cache WHERE model=%s AND key=%s",
                    (model, key),
                )
                row = cur.fetchone()
                return [float(x) for x in row[0]] if row else None
        except Exception:
            return None
    with _EMBED_CACHE_LOCK:
        data = _embed_cache_load(model)
        vec = data.get(key)
        return list(vec) if vec is not None else None


def _embed_cache_put(model: str, text: str, vec: list[float]) -> None:
    if not _embed_cache_enabled():
        return
    key = _embed_text_hash(text)
    backend = _embed_cache_backend()
    if backend == "redis":
        try:
            _EMBED_REDIS_CLIENT.set(f"rag7:embed:{model}:{key}", json.dumps(vec))
        except Exception:
            pass
        return
    if backend == "pg":
        try:
            with _EMBED_PG_POOL.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO rag7_embed_cache (model, key, vec) "
                    "VALUES (%s, %s, %s::jsonb) "
                    "ON CONFLICT (model, key) DO NOTHING",
                    (model, key, json.dumps(vec)),
                )
                conn.commit()
        except Exception:
            pass
        return
    with _EMBED_CACHE_LOCK:
        data = _embed_cache_load(model)
        if key in data:
            return
        data[key] = list(vec)
        _embed_cache_trim(data)
        _EMBED_CACHE_DIRTY.add(model)
        _embed_cache_flush(model)


def _make_azure_embed_fn(
    dimensions: int | None = None,
) -> Callable[[str], list[float]] | None:
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or ""
    deploy = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    api_ver = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    # Env override so CLI/test users can pin a server-side dim without code edits.
    env_dim = os.getenv("AZURE_OPENAI_EMBEDDING_DIM") or os.getenv("RAG7_EMBED_DIM")
    if dimensions is None and env_dim:
        try:
            dimensions = int(env_dim)
        except ValueError:
            dimensions = None

    if not endpoint or not api_key:
        return None

    url = f"{endpoint}/openai/deployments/{deploy}/embeddings?api-version={api_ver}"
    session = _requests.Session()
    session.headers.update({"api-key": api_key, "Content-Type": "application/json"})

    from . import _cache as _c

    # Namespace the cache by dim so 1536-d cached vectors don't shadow a
    # 512-d configuration on the same deployment.
    cache_ns = f"{deploy}@{dimensions}" if dimensions else deploy
    body_extra = {"dimensions": dimensions} if dimensions else {}

    def _embed(text: str) -> list[float]:
        hit = _embed_cache_get(cache_ns, text)
        if hit is not None:
            return hit
        cached = _c.load("embed-v1", cache_ns, text)
        if cached is not None:
            vec = list(cached)
            _embed_cache_put(cache_ns, text, vec)
            return vec
        resp = session.post(url, json={"input": [text], **body_extra}, timeout=15)
        resp.raise_for_status()
        vec = resp.json()["data"][0]["embedding"]
        _c.save("embed-v1", cache_ns, text, value=vec)
        _embed_cache_put(cache_ns, text, vec)
        return vec

    # Mark the function so _align_embed_fn_with_backend can rebuild it with
    # the index's native dim (server-side Matryoshka) instead of slicing.
    _embed._rag7_azure_rebuild = lambda dim: _make_azure_embed_fn(dimensions=dim)  # type: ignore[attr-defined]
    return _embed


def _adapt_embed_fn_to_dim(
    embed_fn: Callable[[str], list[float]],
    target_dim: int,
) -> Callable[[str], list[float]]:
    """Wrap ``embed_fn`` so outputs are sliced + L2-renormalized to ``target_dim``.

    Matryoshka-trained models (text-embedding-3-*, BGE-M3, Nomic, Jina-v3) keep
    meaning in the leading dims, so slice-then-renormalize is a valid projection.
    For non-Matryoshka models the result is degraded but still better than
    dim-mismatched zero-hit search.
    """
    import math

    def _wrapped(text: str) -> list[float]:
        vec = embed_fn(text)
        if len(vec) == target_dim:
            return vec
        sliced = vec[:target_dim]
        norm = math.sqrt(sum(x * x for x in sliced)) or 1.0
        return [x / norm for x in sliced]

    return _wrapped


def _rrf_fuse(
    results: list[list[dict]],
    k: int = 60,
    ranking_score_weight: float = 0.0,
) -> list[dict]:
    scores: dict[str, float] = {}
    seen: dict[str, dict] = {}
    for ranked in results:
        for rank, doc in enumerate(ranked):
            doc_id = doc.get("id") or doc.get("corpus_id") or str(doc)
            rrf = 1.0 / (k + rank + 1)
            if ranking_score_weight > 0.0:
                rrf *= 1.0 + ranking_score_weight * doc.get("_rankingScore", 0.5)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf
            seen.setdefault(doc_id, doc)
    return [seen[did] for did in sorted(scores, key=scores.__getitem__, reverse=True)]


def _dbsf_fuse(
    results: list[list[dict]],
    score_field: str = "_rankingScore",
) -> list[dict]:
    """Distribution-Based Score Fusion — normalize scores per result set, then sum."""
    scores: dict[str, float] = {}
    seen: dict[str, dict] = {}
    for ranked in results:
        if not ranked:
            continue
        raw = [doc.get(score_field, 0.0) for doc in ranked]
        lo, hi = min(raw), max(raw)
        span = hi - lo if hi > lo else 1.0
        for doc, raw_score in zip(ranked, raw):
            doc_id = doc.get("id") or doc.get("corpus_id") or str(doc)
            normalized = (raw_score - lo) / span
            scores[doc_id] = scores.get(doc_id, 0.0) + normalized
            seen.setdefault(doc_id, doc)
    return [seen[did] for did in sorted(scores, key=scores.__getitem__, reverse=True)]


class _RedisByteStore:
    """Minimal LangChain-compatible ByteStore backed by Redis."""

    def __init__(self, url: str) -> None:
        import redis as _redis  # type: ignore[import-not-found]

        self._client = _redis.Redis.from_url(url, decode_responses=False)

    def mget(self, keys: list[str]) -> list[bytes | None]:
        return self._client.mget(keys) if keys else []  # type: ignore[return-value]

    def mset(self, key_value_pairs: list[tuple[str, bytes]]) -> None:
        if key_value_pairs:
            self._client.mset(dict(key_value_pairs))

    def mdelete(self, keys: list[str]) -> None:
        if keys:
            self._client.delete(*keys)

    def yield_keys(self, *, prefix: str | None = None):  # type: ignore[return]
        pattern = f"{prefix}*" if prefix else "*"
        for k in self._client.scan_iter(pattern):
            yield k.decode() if isinstance(k, bytes) else k


class _PgByteStore:
    """Minimal LangChain-compatible ByteStore backed by Postgres (psycopg)."""

    def __init__(self, dsn: str, table: str = "rag7_lc_embed_cache") -> None:
        import psycopg  # type: ignore[import-not-found]

        self._dsn = dsn
        self._table = table
        with psycopg.connect(dsn) as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} "  # noqa: S608
                "(key TEXT PRIMARY KEY, value BYTEA NOT NULL)"
            )
            conn.commit()

    def mget(self, keys: list[str]) -> list[bytes | None]:
        import psycopg  # type: ignore[import-not-found]

        if not keys:
            return []
        with psycopg.connect(self._dsn) as conn:
            return [
                (lambda row: bytes(row[0]) if row else None)(
                    conn.execute(
                        f"SELECT value FROM {self._table} WHERE key = %s", (k,)  # noqa: S608
                    ).fetchone()
                )
                for k in keys
            ]

    def mset(self, key_value_pairs: list[tuple[str, bytes]]) -> None:
        import psycopg  # type: ignore[import-not-found]

        if not key_value_pairs:
            return
        with psycopg.connect(self._dsn) as conn:
            for key, value in key_value_pairs:
                conn.execute(
                    f"INSERT INTO {self._table}(key, value) VALUES (%s, %s) "  # noqa: S608
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    (key, value),
                )
            conn.commit()

    def mdelete(self, keys: list[str]) -> None:
        import psycopg  # type: ignore[import-not-found]

        if not keys:
            return
        with psycopg.connect(self._dsn) as conn:
            for key in keys:
                conn.execute(
                    f"DELETE FROM {self._table} WHERE key = %s", (key,)  # noqa: S608
                )
            conn.commit()

    def yield_keys(self, *, prefix: str | None = None):  # type: ignore[return]
        import psycopg  # type: ignore[import-not-found]

        with psycopg.connect(self._dsn) as conn:
            if prefix:
                rows = conn.execute(
                    f"SELECT key FROM {self._table} WHERE key LIKE %s",  # noqa: S608
                    (prefix + "%",),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT key FROM {self._table}"  # noqa: S608
                ).fetchall()
        yield from (row[0] for row in rows)


def _make_embed_byte_store(uri: str) -> Any:
    """Return a LangChain-compatible ByteStore from a URI or filesystem path.

    - ``redis://`` / ``rediss://``  → ``_RedisByteStore`` (requires ``redis``)
    - ``postgres://`` / ``postgresql://`` → ``_PgByteStore`` (requires ``psycopg``)
    - anything else → ``LocalFileStore`` (treated as directory path, ``~`` expanded)
    """
    if uri.startswith(("redis://", "rediss://")):
        return _RedisByteStore(uri)
    if uri.startswith(("postgres://", "postgresql://")):
        return _PgByteStore(uri)
    from langchain.storage import LocalFileStore  # type: ignore[import-not-found]

    return LocalFileStore(str(Path(uri).expanduser()))


async def _embed_all_async(
    embed_fn: Callable[[str], list[float]],
    texts: list[str],
) -> list[list[float] | None]:
    loop = asyncio.get_running_loop()

    async def _one(text: str) -> list[float] | None:
        try:
            return await loop.run_in_executor(None, embed_fn, text)
        except Exception:
            return None

    return await asyncio.gather(*[_one(t) for t in texts])


def embed_fn_from_langchain(
    embeddings: Any,
    *,
    prefer_query: bool = True,
    cache: str | Path | None = None,
    namespace: str = "embeddings",
) -> Callable[[str], list[float]]:
    """Wrap any LangChain ``Embeddings`` instance as an rag7 ``embed_fn``.

    Caching is auto-enabled when ``RAG7_EMBED_CACHE_URL`` is set, or pass
    ``cache`` explicitly.  Both accept a filesystem path, Redis URL, or
    Postgres DSN:

    .. code-block:: python

        from langchain_openai import AzureOpenAIEmbeddings
        from rag7.utils import embed_fn_from_langchain

        # disk cache
        embed_fn_from_langchain(emb, cache="~/.cache/rag7/lc", namespace="text-embedding-3-small")
        # Redis
        embed_fn_from_langchain(emb, cache="redis://localhost:6379/0", namespace="te3s")
        # Postgres
        embed_fn_from_langchain(emb, cache="postgresql://user:pw@host/db", namespace="te3s")
        # env-driven (set RAG7_EMBED_CACHE_URL in environment)
        embed_fn_from_langchain(emb, namespace="text-embedding-3-small")

    Or bring your own ``CacheBackedEmbeddings`` — the cache bypass bug
    (``embed_query`` skips the cache) is fixed automatically.

    Parameters
    ----------
    embeddings:
        Any object exposing ``.embed_query(text) -> list[float]`` and/or
        ``.embed_documents([text]) -> list[list[float]]``.
    prefer_query:
        Call ``.embed_query()`` when available (default True). Ignored when
        caching is active or a ``CacheBackedEmbeddings`` instance is detected.
    cache:
        Filesystem path, ``redis://`` URL, or ``postgres://`` DSN. Falls back
        to ``RAG7_EMBED_CACHE_URL`` env var when omitted.
    namespace:
        Cache key prefix — use the model name to avoid collisions between
        different embedding deployments sharing the same store.
    """
    cache_uri = str(cache) if cache is not None else os.environ.get("RAG7_EMBED_CACHE_URL", "").strip()
    if cache_uri:
        from langchain.embeddings import CacheBackedEmbeddings  # type: ignore[import-not-found]

        store = _make_embed_byte_store(cache_uri)
        embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings,
            store,
            namespace=namespace,
            query_embedding_store=store,
        )
        prefer_query = False

    # CacheBackedEmbeddings caches embed_documents, not embed_query (unless
    # query_embedding_store is set). Route through embed_documents to hit cache.
    if prefer_query:
        try:
            from langchain.embeddings import CacheBackedEmbeddings

            if isinstance(embeddings, CacheBackedEmbeddings):
                prefer_query = False
        except ImportError:
            pass

    if prefer_query and hasattr(embeddings, "embed_query"):
        return embeddings.embed_query  # type: ignore[no-any-return]
    if hasattr(embeddings, "embed_documents"):

        def _embed(text: str) -> list[float]:
            return embeddings.embed_documents([text])[0]

        return _embed
    raise TypeError(
        f"{type(embeddings).__name__} has neither .embed_query nor "
        ".embed_documents — not a LangChain Embeddings-compatible object."
    )
