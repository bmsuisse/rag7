from __future__ import annotations

import asyncio
import os
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .models import RerankResult
from .utils import _run_sync


def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate" in msg


class CohereReranker:
    """Cohere reranker — default when cohere is installed.

    Works with both Cohere API and Azure Cohere deployments.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        try:
            import cohere as _cohere
        except ImportError as e:
            raise ImportError(
                "cohere is required for CohereReranker. "
                "Install it with: pip install agenticrag[cohere]"
            ) from e

        endpoint = base_url or os.getenv("AZURE_COHERE_ENDPOINT", "")
        _base = (
            endpoint.removesuffix("/v2/rerank").removesuffix("/rerank")
            if endpoint
            else ""
        )
        key = (
            api_key or os.getenv("AZURE_COHERE_API_KEY") or os.getenv("COHERE_API_KEY")
        )

        self._client = _cohere.AsyncClientV2(
            api_key=key,
            base_url=_base or None,
        )
        self._model = model or os.getenv("AZURE_COHERE_DEPLOYMENT", "rerank-v3.5")

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        async def _do() -> list[RerankResult]:
            response = await self._client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=top_n,
            )
            return [
                RerankResult(index=r.index, relevance_score=r.relevance_score)
                for r in response.results
            ]

        return _run_sync(_do())

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        response = await self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_n,
        )
        return [
            RerankResult(index=r.index, relevance_score=r.relevance_score)
            for r in response.results
        ]


class RerankersReranker:
    """Bridge to the `rerankers` library by answer.ai.

    Single interface for cross-encoders, ColBERT, RankGPT, Flashrank, and API models.
    See: https://github.com/AnswerDotAI/rerankers

    Requires: pip install rerankers  (or rerankers[transformers], rerankers[flashrank], etc.)

    Examples:
        RerankersReranker("cross-encoder/ms-marco-MiniLM-L-6-v2", model_type="cross-encoder")
        RerankersReranker("colbert-ir/colbertv2.0", model_type="colbert")
        RerankersReranker("flashrank", model_type="flashrank")
        RerankersReranker("gpt-5.4-mini", model_type="rankgpt", api_key="...")
    """

    def __init__(self, model: str, model_type: str | None = None, **kwargs: object):
        try:
            from rerankers import Reranker  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "rerankers is required for RerankersReranker. "
                "Install it with: pip install rerankers"
            ) from e

        self._ranker = Reranker(model, model_type=model_type, **kwargs)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        if not documents:
            return []
        results = self._ranker.rank(query=query, docs=documents)
        top = results.top_k(top_n)
        return [
            RerankResult(index=r.doc_id, relevance_score=float(r.score)) for r in top
        ]

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.rerank, query, documents, top_n)


class JinaReranker:
    """Jina AI reranker — REST API, strong multilingual support.

    Models: jina-reranker-v2-base-multilingual (default), jina-reranker-v1-base-en
    API key: https://jina.ai (free tier available)

    Requires: pip install httpx
    """

    _API_URL = "https://api.jina.ai/v1/rerank"

    def __init__(
        self,
        model: str = "jina-reranker-v2-base-multilingual",
        api_key: str | None = None,
    ):
        try:
            import httpx as _httpx  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "httpx is required for JinaReranker. Install it with: pip install httpx"
            ) from e

        self._model = model
        self._api_key = api_key or os.getenv("JINA_API_KEY", "")

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        import httpx

        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        response = httpx.post(self._API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        results = response.json()["results"]
        return [
            RerankResult(index=r["index"], relevance_score=r["relevance_score"])
            for r in results
        ]

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        import httpx

        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(self._API_URL, json=payload, headers=headers)
            response.raise_for_status()
        results = response.json()["results"]
        return [
            RerankResult(index=r["index"], relevance_score=r["relevance_score"])
            for r in results
        ]


class HuggingFaceReranker:
    """Cross-encoder reranker using a local HuggingFace model.

    Runs entirely locally — no API key, no network calls after model download.
    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality).
    For multilingual: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

    Requires: pip install sentence-transformers
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        max_length: int = 512,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for HuggingFaceReranker. "
                "Install it with: pip install sentence-transformers"
            ) from e

        self._model = CrossEncoder(model, device=device, max_length=max_length)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs).tolist()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
        return [RerankResult(index=i, relevance_score=float(s)) for i, s in ranked]

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.rerank, query, documents, top_n)


class LLMReranker:
    """LLM reranker — parallel per-doc scoring via asyncio.gather.

    Use as an expert fallback on hard questions. Costs one LLM call per doc —
    keep doc count small (top 10–15) and cap input size.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        doc_chars: int = 3000,
        max_parallel: int = 6,
    ):
        self._llm = llm
        self._doc_chars = doc_chars
        self._sem = asyncio.Semaphore(max_parallel)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        return cast(
            "list[RerankResult]",
            _run_sync(self.arerank(query, documents, top_n)),
        )

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        if not self._llm or not documents:
            return [
                RerankResult(index=i, relevance_score=1.0 / (i + 1))
                for i in range(min(top_n, len(documents)))
            ]
        scored = await asyncio.gather(
            *(self._score(query, d, i) for i, d in enumerate(documents))
        )
        scored.sort(key=lambda x: x[1], reverse=True)
        return [RerankResult(index=i, relevance_score=s) for i, s in scored[:top_n]]

    async def _score(self, query: str, doc: str, idx: int) -> tuple[int, float]:
        import re

        if self._llm is None:
            return idx, 0.0
        prompt = (
            f"Rate the relevance of this document to the query on a scale 0.0-1.0.\n"
            f"Query: {query}\n\nDocument: {doc[: self._doc_chars]}\n\n"
            f"Return ONLY a single float between 0.0 and 1.0."
        )
        async with self._sem:
            try:
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception(_is_rate_limit),
                    wait=wait_exponential_jitter(initial=1, max=16),
                    stop=stop_after_attempt(4),
                    reraise=True,
                ):
                    with attempt:
                        resp = await self._llm.ainvoke([HumanMessage(prompt)])
                        m = re.search(r"\d+\.?\d*", str(resp.content).strip())
                        return idx, float(m.group()) if m else 0.0
            except Exception:
                return idx, 0.0
        return idx, 0.0


class EmbedAnythingReranker:
    """Local reranker via embed-anything ONNX models (no API key).

    Implements the ``Reranker`` protocol — drop-in replacement for
    ``CohereReranker`` and friends.

    Examples::

        reranker = EmbedAnythingReranker("jinaai/jina-reranker-v1-turbo-en")
        rag = AgenticRAG("docs", backend=backend, reranker=reranker)

    Requires: ``pip install embed-anything``
    """

    def __init__(
        self,
        model_id: str = "jinaai/jina-reranker-v1-turbo-en",
        dtype: object | None = None,
        path_in_repo: str | None = "onnx",
    ):
        try:
            import embed_anything as ea
        except ImportError as e:
            raise ImportError(
                "embed-anything is required for EmbedAnythingReranker. "
                "Install it with: pip install embed-anything"
            ) from e

        if dtype is None:
            dtype = ea.Dtype.F32
        self._model = ea.Reranker.from_pretrained(
            model_id, dtype=dtype, path_in_repo=path_in_repo
        )
        self._model_id = model_id

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        if not documents:
            return []
        results = self._model.rerank([query], documents, top_n)
        doc_to_idx = {doc: i for i, doc in enumerate(documents)}
        ranked: list[RerankResult] = []
        for res in results:
            for doc_rank in res.documents:
                idx = doc_to_idx.get(doc_rank.document, doc_rank.rank - 1)
                ranked.append(
                    RerankResult(index=idx, relevance_score=doc_rank.relevance_score)
                )
        ranked.sort(key=lambda r: r.relevance_score, reverse=True)
        return ranked[:top_n]

    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        import asyncio as _asyncio

        loop = _asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.rerank, query, documents, top_n)

    def __repr__(self) -> str:
        return f"EmbedAnythingReranker({self._model_id!r})"
