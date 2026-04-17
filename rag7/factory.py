"""Convenience factory for one-line Agent initialisation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .core import AgenticRAG

# ── backend aliases ────────────────────────────────────────────────────────────

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
    "meilisearch": "MeilisearchBackend",
    "meili": "MeilisearchBackend",
    "azure": "AzureAISearchBackend",
    "azure_ai_search": "AzureAISearchBackend",
    "azureaisearch": "AzureAISearchBackend",
}

# ── reranker aliases ───────────────────────────────────────────────────────────

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

    # merge url / embed_fn into backend_kwargs only if the class accepts them
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

    # backends that take the index / collection name as first positional arg
    _takes_index = {
        "MeilisearchBackend",
        "ChromaDBBackend",
        "LanceDBBackend",
        "QdrantBackend",
        "DuckDBBackend",
        "PgvectorBackend",
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
    """Build one backend per collection name, sharing backend type/url.

    Accepts either a list of names or a {name: description} dict — the
    descriptions are consumed by AgenticRAG, not here.
    """
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
    """Create a fully configured ``AgenticRAG`` in one call.

    Parameters
    ----------
    index:
        Collection / index name.
    model:
        Provider-prefixed model string passed to ``init_chat_model``,
        e.g. ``"openai:gpt-5.4"``, ``"anthropic:claude-sonnet-4-6"``,
        ``"ollama:llama3"``.  If omitted, the default LLM from environment
        variables is used (same behaviour as constructing ``Agent`` directly).
    gen_model:
        Separate generation model.  Defaults to ``model``.
    backend:
        Either a backend instance **or** a string alias:

        =============================  ================================
        String                         Class
        =============================  ================================
        ``"memory"`` / ``"in_memory"`` ``InMemoryBackend`` (default)
        ``"chroma"`` / ``"chromadb"``  ``ChromaDBBackend``
        ``"qdrant"``                   ``QdrantBackend``
        ``"lancedb"`` / ``"lance"``    ``LanceDBBackend``
        ``"duckdb"``                   ``DuckDBBackend``
        ``"pgvector"`` / ``"pg"``      ``PgvectorBackend``
        ``"meilisearch"``              ``MeilisearchBackend``
        ``"azure"``                    ``AzureAISearchBackend``
        =============================  ================================

    backend_url:
        URL / endpoint forwarded to the backend constructor (``url`` or
        ``endpoint`` parameter, depending on the backend).
    backend_kwargs:
        Extra keyword arguments forwarded verbatim to the backend constructor.
    reranker:
        Either a reranker instance **or** a string alias:

        =====================  ========================
        String                 Class
        =====================  ========================
        ``"cohere"``           ``CohereReranker``
        ``"huggingface"``      ``HuggingFaceReranker``
        ``"jina"``             ``JinaReranker``
        ``"llm"``              ``LLMReranker``
        ``"rerankers"``        ``RerankersReranker``
        =====================  ========================

    reranker_model:
        Model name forwarded to the reranker constructor (where applicable).
    reranker_kwargs:
        Extra keyword arguments forwarded verbatim to the reranker constructor.
    embed_fn:
        Embedding callable ``(str) -> list[float]``.  Forwarded to both the
        agent and the backend constructor (when the backend accepts it).
    **kwargs:
        Any remaining ``AgenticRAG.__init__`` keyword arguments (``top_k``,
        ``instructions``, ``auto_strategy``, ``filter``, …).

    Examples
    --------
    Minimal — in-memory, env-var LLM:

    >>> from rag7 import init_agent
    >>> rag = init_agent("docs")

    OpenAI + Qdrant + Cohere reranker:

    >>> rag = init_agent(
    ...     "my-collection",
    ...     model="openai:gpt-5.4",
    ...     backend="qdrant",
    ...     backend_url="http://localhost:6333",
    ...     reranker="cohere",
    ... )

    Anthropic + Azure AI Search (native vectorisation):

    >>> rag = init_agent(
    ...     "my-index",
    ...     model="anthropic:claude-sonnet-4-6",
    ...     gen_model="anthropic:claude-opus-4-6",
    ...     backend="azure",
    ...     backend_url="https://my-search.search.windows.net",
    ...     reranker="huggingface",
    ...     auto_strategy=True,
    ... )

    Local Ollama + ChromaDB + local HuggingFace reranker:

    >>> rag = init_agent(
    ...     "docs",
    ...     model="ollama:llama3",
    ...     backend="chroma",
    ...     reranker="huggingface",
    ...     reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ... )
    """
    from .core import AgenticRAG

    # ── backend(s) ────────────────────────────────────────────────────────────
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

    # ── reranker ──────────────────────────────────────────────────────────────
    resolved_reranker: Any | None = None
    if reranker is not None:
        if isinstance(reranker, str):
            resolved_reranker = _build_reranker(
                reranker, reranker_model, reranker_kwargs or {}
            )
        else:
            resolved_reranker = reranker

    # ── LLM ───────────────────────────────────────────────────────────────────
    if model is not None:
        from langchain.chat_models import init_chat_model  # type: ignore[import-untyped]

        llm = init_chat_model(model, temperature=0)
        gen_llm = init_chat_model(gen_model, temperature=0) if gen_model else llm
        kwargs["llm"] = llm
        kwargs["gen_llm"] = gen_llm
    elif gen_model is not None:
        from langchain.chat_models import init_chat_model  # type: ignore[import-untyped]

        kwargs["gen_llm"] = init_chat_model(gen_model, temperature=0)

    if embed_fn is not None:
        kwargs.setdefault("embed_fn", embed_fn)

    if resolved_reranker is not None:
        kwargs["reranker"] = resolved_reranker

    if resolved_collections is not None:
        return AgenticRAG(
            index=index or ",".join(resolved_collections),
            collections=resolved_collections,
            collection_descriptions=collection_descriptions,
            **kwargs,
        )
    return AgenticRAG(index=index, backend=resolved_backend, **kwargs)
