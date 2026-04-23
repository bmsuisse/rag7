from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .core import AgenticRAG

_BACKEND_ALIASES: dict[str, str] = {
    "memory": "InMemoryBackend",
    "in_memory": "InMemoryBackend",
    "inmemory": "InMemoryBackend",
    "chroma": "ChromaDBBackend",
    "chromadb": "ChromaDBBackend",
    "qdrant": "QdrantBackend",
    "lance": "LanceDBBackend",
    "lancedb": "LanceDBBackend",
    "duckdb": "DuckDBBackend",
    "duck": "DuckDBBackend",
    "pgvector": "PgvectorBackend",
    "postgres": "PgvectorBackend",
    "pg": "PgvectorBackend",
    "postgres_fts": "PostgresFTSBackend",
    "pg_fts": "PostgresFTSBackend",
    "sqlite": "SQLiteFTSBackend",
    "sqlite_fts": "SQLiteFTSBackend",
    "meilisearch": "MeilisearchBackend",
    "meili": "MeilisearchBackend",
    "azure": "AzureAISearchBackend",
    "azure_ai_search": "AzureAISearchBackend",
    "azureaisearch": "AzureAISearchBackend",
}

_RERANKER_ALIASES: dict[str, str] = {
    "cohere": "CohereReranker",
    "huggingface": "HuggingFaceReranker",
    "hf": "HuggingFaceReranker",
    "jina": "JinaReranker",
    "llm": "LLMReranker",
    "rerankers": "RerankersReranker",
}


def _build_backend(
    backend_str: str,
    index: str,
    url: str | None,
    embed_fn: Callable[[str], list[float]] | None,
    backend_kwargs: dict[str, Any],
) -> Any:
    cls_name = _BACKEND_ALIASES.get(backend_str.lower())
    if cls_name is None:
        raise ValueError(
            f"Unknown backend {backend_str!r}. "
            f"Choose from: {', '.join(sorted(_BACKEND_ALIASES))}."
        )

    from . import backend as _backend_mod

    cls = getattr(_backend_mod, cls_name)

    import inspect

    sig = inspect.signature(cls.__init__)
    params = set(sig.parameters)

    kw: dict[str, Any] = {}
    if url is not None and "url" in params:
        kw["url"] = url
    if url is not None and "endpoint" in params:
        kw["endpoint"] = url
    if embed_fn is not None and "embed_fn" in params:
        kw["embed_fn"] = embed_fn
    kw.update(backend_kwargs)

    _takes_index = {
        "MeilisearchBackend",
        "ChromaDBBackend",
        "LanceDBBackend",
        "QdrantBackend",
        "DuckDBBackend",
        "PgvectorBackend",
        "PostgresFTSBackend",
        "SQLiteFTSBackend",
        "AzureAISearchBackend",
    }
    if cls_name in _takes_index:
        return cls(index, **kw)
    return cls(**kw)


def _build_collections_map(
    collections: list[str] | dict[str, str],
    *,
    backend_str: str,
    url: str | None,
    embed_fn: Callable[[str], list[float]] | None,
    backend_kwargs: dict[str, Any],
) -> dict[str, Any]:
    names = list(collections.keys() if isinstance(collections, dict) else collections)
    if not names:
        raise ValueError("collections must contain at least one name")
    return {
        name: _build_backend(backend_str, name, url, embed_fn, backend_kwargs)
        for name in names
    }


def _build_reranker(
    reranker_str: str,
    reranker_model: str | None,
    reranker_kwargs: dict[str, Any],
) -> Any:
    cls_name = _RERANKER_ALIASES.get(reranker_str.lower())
    if cls_name is None:
        raise ValueError(
            f"Unknown reranker {reranker_str!r}. "
            f"Choose from: {', '.join(sorted(_RERANKER_ALIASES))}."
        )

    from . import rerankers as _rerankers_mod

    cls = getattr(_rerankers_mod, cls_name)

    import inspect

    sig = inspect.signature(cls.__init__)
    params = set(sig.parameters)

    kw: dict[str, Any] = {}
    if reranker_model is not None and "model" in params:
        kw["model"] = reranker_model
    kw.update(reranker_kwargs)
    return cls(**kw)


def init_agent(
    index: str = "",
    *,
    collections: list[str] | dict[str, str] | None = None,
    model: str | None = None,
    gen_model: str | None = None,
    backend: str | Any | None = None,
    backend_url: str | None = None,
    backend_kwargs: dict[str, Any] | None = None,
    reranker: str | Any | None = None,
    reranker_model: str | None = None,
    reranker_kwargs: dict[str, Any] | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
    **kwargs: Any,
) -> "AgenticRAG":
    from .core import AgenticRAG

    resolved_backend: Any = None
    resolved_collections: dict[str, Any] | None = None
    collection_descriptions: dict[str, str] | None = None

    if collections is not None:
        if isinstance(collections, dict):
            collection_descriptions = dict(collections)
        resolved_collections = _build_collections_map(
            collections,
            backend_str=backend if isinstance(backend, str) else "memory",
            url=backend_url,
            embed_fn=embed_fn,
            backend_kwargs=backend_kwargs or {},
        )
    elif backend is None or isinstance(backend, str):
        if not index:
            raise ValueError("init_agent requires `index` or `collections`")
        resolved_backend = _build_backend(
            backend or "memory",
            index,
            backend_url,
            embed_fn,
            backend_kwargs or {},
        )
    else:
        resolved_backend = backend

    resolved_reranker: Any | None = None
    if reranker is not None:
        if isinstance(reranker, str):
            resolved_reranker = _build_reranker(
                reranker, reranker_model, reranker_kwargs or {}
            )
        else:
            resolved_reranker = reranker

    if model is not None:
        from langchain.chat_models import (
            init_chat_model,  # type: ignore[import-untyped]
        )

        llm = init_chat_model(model, temperature=0)
        gen_llm = init_chat_model(gen_model, temperature=0) if gen_model else llm
        kwargs["llm"] = llm
        kwargs["gen_llm"] = gen_llm
    elif gen_model is not None:
        from langchain.chat_models import (
            init_chat_model,  # type: ignore[import-untyped]
        )

        kwargs["gen_llm"] = init_chat_model(gen_model, temperature=0)

    if embed_fn is not None:
        kwargs.setdefault("embed_fn", embed_fn)

    if resolved_reranker is not None:
        kwargs["reranker"] = resolved_reranker

    if "config" not in kwargs:
        from .config import RAGConfig

        kwargs["config"] = RAGConfig.auto()

    if resolved_collections is not None:
        return AgenticRAG(
            index=index or ",".join(resolved_collections),
            collections=resolved_collections,
            collection_descriptions=collection_descriptions,
            **kwargs,
        )
    return AgenticRAG(index=index, backend=resolved_backend, **kwargs)
