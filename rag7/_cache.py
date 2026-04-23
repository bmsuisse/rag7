from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

_ENABLED = os.environ.get("RAG7_CACHE", "0").lower() in ("1", "true", "yes", "on")
_PG_URL = os.environ.get("RAG7_CACHE_PG_URL") or os.environ.get("DATABASE_URL")
_DIR = os.environ.get("RAG7_CACHE_DIR")

_mem: dict[str, Any] = {}
_pg_pool: Any = None
_pg_ready = False
_disk_ready = False


def _try_pg() -> Any:
    global _pg_pool, _pg_ready
    if _pg_ready:
        return _pg_pool
    _pg_ready = True
    if not (_ENABLED and _PG_URL):
        return None
    try:
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(_PG_URL, min_size=1, max_size=4, open=True)
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS rag7_cache "
                "(key text PRIMARY KEY, value jsonb NOT NULL, "
                "created_at timestamptz DEFAULT now())"
            )
            conn.commit()
        _pg_pool = pool
        return pool
    except Exception:
        return None


def _disk_dir() -> Path | None:
    global _disk_ready
    if not (_ENABLED and _DIR):
        return None
    p = Path(_DIR)
    if not _disk_ready:
        try:
            p.mkdir(parents=True, exist_ok=True)
            _disk_ready = True
        except Exception:
            return None
    return p


def _hash(ns: str, *parts: Any) -> str:
    h = hashlib.sha256(ns.encode())
    for p in parts:
        h.update(b"\x00")
        h.update(repr(p).encode())
    return f"{ns}_{h.hexdigest()[:24]}"


def load(ns: str, *parts: Any) -> Any:
    if not _ENABLED:
        return None
    key = _hash(ns, *parts)
    pool = _try_pg()
    if pool is not None:
        try:
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT value FROM rag7_cache WHERE key=%s", (key,))
                row = cur.fetchone()
                return row[0] if row else None
        except Exception:
            return None
    d = _disk_dir()
    if d is not None:
        f = d / f"{key}.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                return None
        return None
    return _mem.get(key)


def save(ns: str, *parts: Any, value: Any) -> None:
    if not _ENABLED:
        return
    key = _hash(ns, *parts)
    pool = _try_pg()
    if pool is not None:
        try:
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO rag7_cache(key, value) VALUES (%s, %s) "
                    "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
                    (key, json.dumps(value)),
                )
                conn.commit()
        except Exception:
            pass
        return
    d = _disk_dir()
    if d is not None:
        try:
            (d / f"{key}.json").write_text(json.dumps(value))
        except Exception:
            pass
        return
    _mem[key] = value
