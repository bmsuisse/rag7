from __future__ import annotations

import asyncio
import concurrent.futures
import os
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


def _make_azure_embed_fn() -> Callable[[str], list[float]] | None:
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or ""
    deploy = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    api_ver = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not endpoint or not api_key:
        return None

    url = f"{endpoint}/openai/deployments/{deploy}/embeddings?api-version={api_ver}"
    session = _requests.Session()
    session.headers.update({"api-key": api_key, "Content-Type": "application/json"})

    from . import _cache as _c

    def _embed(text: str) -> list[float]:
        cached = _c.load("embed-v1", deploy, text)
        if cached is not None:
            return list(cached)
        resp = session.post(url, json={"input": [text]}, timeout=15)
        resp.raise_for_status()
        vec = resp.json()["data"][0]["embedding"]
        _c.save("embed-v1", deploy, text, value=vec)
        return vec

    return _embed


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
