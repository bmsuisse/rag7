from __future__ import annotations

import asyncio

from .models import RerankResult


class EmbedAnythingEmbedder:
    """Local embeddings via embed-anything (Rust-accelerated, no API key).

    Works as a drop-in ``embed_fn`` — call the instance directly or use ``.embed()``.

    Examples::

        embedder = EmbedAnythingEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        rag = AgenticRAG("docs", backend=backend, embed_fn=embedder)

        # Or with ONNX for faster inference:
        from embed_anything import WhichModel
        embedder = EmbedAnythingEmbedder.from_onnx(WhichModel.Bert)

    Requires: ``pip install embed-anything``
    """

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            import embed_anything as ea
        except ImportError as e:
            raise ImportError(
                "embed-anything is required for EmbedAnythingEmbedder. "
                "Install it with: pip install embed-anything"
            ) from e

        self._ea = ea
        self._model = ea.EmbeddingModel.from_pretrained_hf(model_id)
        self._model_id = model_id

    @classmethod
    def from_onnx(cls, which_model: object, **kwargs: object) -> EmbedAnythingEmbedder:
        try:
            import embed_anything as ea
        except ImportError as e:
            raise ImportError(
                "embed-anything is required for EmbedAnythingEmbedder. "
                "Install it with: pip install embed-anything"
            ) from e

        instance = cls.__new__(cls)
        instance._ea = ea
        instance._model = ea.EmbeddingModel.from_pretrained_onnx(which_model, **kwargs)
        instance._model_id = str(which_model)
        return instance

    def embed(self, text: str) -> list[float]:
        data = self._ea.embed_query([text], embedder=self._model)
        return data[0].embedding

    def __call__(self, text: str) -> list[float]:
        return self.embed(text)

    def __repr__(self) -> str:
        return f"EmbedAnythingEmbedder({self._model_id!r})"


class EmbedAnythingReranker:
    """Local reranker via embed-anything ONNX models (no API key).

    Implements the ``Reranker`` protocol — drop-in replacement for CohereReranker etc.

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
        ranked = []
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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.rerank, query, documents, top_n)

    def __repr__(self) -> str:
        return f"EmbedAnythingReranker({self._model_id!r})"
