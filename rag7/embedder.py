"""Local embedder implementations.

Currently ships ``EmbedAnythingEmbedder`` — Rust-backed embeddings via
the `embed-anything` package. Drop-in replacement for remote embed_fns
(Azure, Cohere, Voyage) when you want fully-local inference.
"""

from __future__ import annotations


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
