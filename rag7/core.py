from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import time
import warnings
from typing import Any, Callable, Literal, Mapping, cast

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from tenacity import retry, stop_after_attempt, wait_exponential

from . import _cache
from .backend import (
    _ACTIVE_COLLECTIONS,
    InMemoryBackend,
    SearchBackend,
    SearchRequest,
    _MultiBackend,
)
from .models import (
    CollectionIntent,
    ConversationTurn,
    FilterIntent,
    MultiQuery,
    QualityAssessment,
    RAGState,
    ReasoningVerdict,
    RelevanceCheck,
    Reranker,
    SearchQuery,
)
from .rerankers import CohereReranker, LLMReranker
from .tools import RAGToolset
from .utils import (
    _SKIP_FIELDS,
    _dbsf_fuse,
    _doc_id,
    _embed_all_async,
    _rrf_fuse,
    _run_sync,
    _strip_stop_words,
)

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
    module="pydantic",
)
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality",
    category=UserWarning,
    module="langchain_core",
)

_ID_SUFFIXES = ("_id", "_code", "_key", "_num")

_llm_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


_CATEGORY_FIELD_PATTERN = re.compile(
    r"(group|category|kategorie|categorie|categoria|class|klass|type|typ|section|department|rubrik|family)",
    re.IGNORECASE,
)


def _detect_category_fields(backend: SearchBackend) -> list[str]:
    """Heuristic: find metadata fields likely to encode a category/group.

    Scans sample docs for keys matching category-like substrings with string
    values. Returns specific (e.g. ``*_l3``) before general (``category``).
    """
    try:
        samples = backend.sample_documents(limit=20)
    except Exception:
        return []
    if not samples:
        return []
    seen: dict[str, int] = {}
    for doc in samples:
        for k, v in doc.items():
            if not isinstance(v, str) or not v:
                continue
            if _CATEGORY_FIELD_PATTERN.search(k):
                seen[k] = seen.get(k, 0) + 1
    if not seen:
        return []

    # Rank: specific suffixes (_l3 > _l2 > _l1) first, then higher coverage.
    def rank(k: str) -> tuple[int, int]:
        depth = 0
        for suffix, d in (("_l3", 3), ("_l2", 2), ("_l1", 1)):
            if k.lower().endswith(suffix):
                depth = d
                break
        return (-depth, -seen[k])

    return sorted(seen, key=rank)


def _detect_index_signals(
    backend: SearchBackend,
) -> tuple[list[str] | None, Callable[[dict], float] | None, list[str]]:
    config = backend.get_index_config()
    sortable = config.sortable_attributes
    if not sortable:
        return None, None, []

    samples = backend.sample_documents(limit=20)
    if not samples:
        return None, None, []

    field_sample: dict[str, Any] = {}
    for doc in samples:
        for f in sortable:
            if f not in field_sample and doc.get(f) is not None:
                field_sample[f] = doc[f]
        if len(field_sample) == len(sortable):
            break

    bool_fields: list[str] = []
    num_fields: list[str] = []
    for f in sortable:
        value = field_sample.get(f)
        if isinstance(value, bool):
            bool_fields.append(f)
        elif (
            isinstance(value, (int, float))
            and value >= 0
            and not any(f.endswith(s) for s in _ID_SUFFIXES)
        ):
            num_fields.append(f)

    if not bool_fields and not num_fields:
        return None, None, []

    sort_fields = [f"{f}:desc" for f in bool_fields + num_fields]

    def boost_fn(meta: dict) -> float:
        binary_boost = 1.2 if any(meta.get(f) for f in bool_fields) else 1.0
        top_num = max((float(meta.get(f, 0) or 0) for f in num_fields), default=0.0)
        if top_num < 150_000:
            top_num = 0.0
        return binary_boost * (1.0 + math.log1p(top_num) / 50)

    return sort_fields, boost_fn, num_fields


class AgenticRAG:
    """Autonomous retrieval-augmented generation agent built on LangGraph.

    rag7 is not a search pipeline — it is an agent. Given a question it
    thinks, searches, judges the results, rewrites the query if needed, and
    keeps going until it is confident enough to generate a grounded, cited
    answer. All of this happens autonomously, without any orchestration code
    on your side.

    Two operating modes
    -------------------
    **Graph mode** (``chat`` / ``invoke``)
        A LangGraph state machine runs the full pipeline:

        1. *Preprocess* — extract keywords, detect semantic/BM25 ratio, pick
           RRF or DBSF fusion strategy.
        2. *HyDE* — generate a hypothetical document to improve vector recall
           on vague queries (runs in parallel with preprocessing).
        3. *Hybrid search* — multi-query BM25 + vector search across N query
           variants, fused with Reciprocal Rank Fusion or Distribution-Based
           Score Fusion.
        4. *Rerank* — Cohere, HuggingFace cross-encoder, Jina, or any custom
           reranker surfaces the most relevant hits.
        5. *Quality gate* — an LLM judges whether the retrieved documents
           actually answer the question.
        6. *Generate or rewrite* — if quality is sufficient, generate a cited
           answer; otherwise rewrite the query and loop (up to ``max_iter``
           times). If all attempts fail, swarm retrieval fans out to parallel
           strategies before giving up.

    **Tool-calling agent mode** (``invoke_agent``)
        The LLM receives a toolset and reasons step-by-step, calling tools in
        whatever order makes sense. No fixed pipeline — the agent inspects the
        index schema, samples real field values, builds precise filter
        expressions on the fly, and decides whether to boost by business
        signals such as popularity or recency.

        Available tools: ``get_index_settings``, ``get_filter_values``,
        ``search_hybrid``, ``search_bm25``, ``rerank_results``.

    Parameters
    ----------
    index:
        Collection / index name in the backend.
    backend:
        A ``SearchBackend`` instance. Defaults to ``InMemoryBackend``
        (zero dependencies, useful for testing and quick prototyping).
    llm:
        Fast LLM used for query preprocessing, rewriting, and the quality
        gate. Defaults to Azure OpenAI if ``AZURE_OPENAI_ENDPOINT`` is set,
        otherwise OpenAI.
    gen_llm:
        Generation LLM used for the final answer. Defaults to ``llm`` with a
        longer timeout. Useful to separate a cheap routing model from a
        powerful generation model.
    reranker:
        Reranker instance. Defaults to ``CohereReranker`` if the ``cohere``
        package is installed, otherwise ``LLMReranker``.
    top_k:
        Number of documents returned after reranking. Default: 10
        (``RAG_TOP_K``).
    rerank_top_n:
        Number of candidates passed to the reranker. Default: 5
        (``RAG_RERANK_TOP_N``).
    retrieval_factor:
        Over-retrieval multiplier — ``top_k * retrieval_factor`` documents
        are fetched before reranking. Default: 4 (``RAG_RETRIEVAL_FACTOR``).
    max_iter:
        Maximum retrieve-rewrite cycles before giving up. Default: 20
        (``RAG_MAX_ITER``).
    semantic_ratio:
        Hybrid search semantic weight (0.0 = pure BM25, 1.0 = pure vector).
        Overridden per query by the preprocessor when ``auto_strategy=True``.
        Default: 0.5 (``RAG_SEMANTIC_RATIO``).
    fusion:
        Score fusion strategy: ``"rrf"`` (Reciprocal Rank Fusion, rank-based,
        stable) or ``"dbsf"`` (Distribution-Based Score Fusion,
        score-normalised). Default: ``"rrf"`` (``RAG_FUSION``).
    instructions:
        Extra text appended to the system prompt for answer generation.
        Use to inject domain-specific constraints or tone requirements.
    embed_fn:
        Callable ``(str) -> list[float]`` used to embed queries for vector
        search and HyDE. If ``None`` and the backend supports server-side
        vectorisation (e.g. Azure AI Search integrated vectorizer), no
        client-side embedding is performed.
    boost_fn:
        Callable ``(doc_dict) -> float`` applied after reranking to boost
        documents by business signals (e.g. in-stock flag, sales rank).
    filter:
        Meilisearch-style filter expression (e.g. ``"brand = 'Bosch'"``,
        ``"in_stock = true AND price < 100"``) applied to every search
        request — BM25, vector, and hybrid. Useful for tenant isolation,
        always-on brand/category scoping, or multi-occupant indexes.
        Per-query filters (intent-detected) are AND-joined with this.
    base_filter:
        Deprecated alias for ``filter``. Will be removed in a future release.
    hyde_min_words:
        Minimum query word count to trigger HyDE expansion. Default: 8
        (``RAG_HYDE_MIN_WORDS``).
    hyde_style_hint:
        Short phrase injected into the HyDE prompt to match the document
        style, e.g. ``"German product spec sheet"``.
    collections:
        Dict mapping collection name → ``SearchBackend``. Enables
        multi-collection mode: the agent selects which collections to query
        per request using an LLM routing step. Mutually exclusive with
        ``backend``.
    collection_descriptions:
        Human-readable description for each collection in ``collections``.
        Fed to the routing LLM so it can make informed decisions.
        Defaults to the collection name when not provided.
    auto_strategy:
        If ``True`` (default), sample documents at init time and let an LLM
        detect ``hyde_style_hint``, ``hyde_min_words``, and a
        ``domain_hint`` that guides query preprocessing — no hardcoded
        domain logic required. Set ``False`` only for testing or when you
        want full manual control.
    group_field:
        Metadata field used to detect query-group mismatches during
        boost-aware reranking.
    name_field:
        Metadata field used as the document name for token-overlap boosting.
    checkpointer:
        LangGraph checkpointer for persistent memory across calls. Pass a
        ``MemorySaver`` for in-process memory, or ``SqliteSaver`` /
        ``PostgresSaver`` for durable persistence. When set, pass
        ``config={"configurable": {"thread_id": "..."}}`` to ``invoke`` /
        ``chat`` to scope memory to a conversation thread.
    memory_store:
        LangGraph ``BaseStore`` for long-term cross-thread memory. The agent
        reads relevant past exchanges before retrieval and writes a summary
        after each answer. Pass ``config={"configurable": {"user_id": "..."}}``
        to scope memories per user. Use ``InMemoryStore`` for development or
        ``AsyncPostgresStore`` / ``AsyncSqliteStore`` for production.

    Examples
    --------
    Minimal — in-memory backend, LLM from env vars, auto-strategy on by default:

    >>> from rag7 import Agent, InMemoryBackend
    >>> rag = Agent(index="docs", backend=InMemoryBackend(embed_fn=my_embed))
    >>> print(rag.query("What is hybrid search?"))

    Multi-collection routing — agent picks which collections to search:

    >>> from rag7 import Agent, MeilisearchBackend
    >>> backends = {
    ...     "products": MeilisearchBackend("products"),
    ...     "manuals": MeilisearchBackend("manuals"),
    ... }
    >>> rag = Agent(
    ...     index="catalog",
    ...     collections=backends,
    ...     collection_descriptions={
    ...         "products": "Product listings with prices and specs",
    ...         "manuals": "Installation and user guides",
    ...     },
    ... )
    >>> state = rag.chat("How do I install product X?", history=[])

    Always-on filter — restrict every search to a subset of documents:

    >>> rag = Agent(
    ...     index="products",
    ...     backend=MeilisearchBackend("products"),
    ...     filter="in_stock = true AND brand != 'HouseBrand'",
    ... )
    >>> state = rag.chat("brake cleaner 500ml", history=[])

    Tool-calling agent mode for dynamic, schema-aware filtering:

    >>> result = rag.invoke_agent("Show me the top rated products from Bosch")
    """

    @staticmethod
    def _default_llm(
        *,
        timeout: int = 30,
    ) -> BaseChatModel:
        """Create default LLM from environment variables.

        Tries Azure OpenAI first (if AZURE_OPENAI_ENDPOINT set),
        falls back to OpenAI (if OPENAI_API_KEY set).
        """
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            from langchain_openai import AzureChatOpenAI

            deploy = os.getenv("AZURE_OPENAI_FAST_DEPLOYMENT") or os.getenv(
                "AZURE_OPENAI_DEPLOYMENT"
            )
            return AzureChatOpenAI(  # type: ignore[call-arg]
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=deploy,  # ty: ignore[unknown-argument]
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),  # ty: ignore[unknown-argument]
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),  # ty: ignore[unknown-argument]
                temperature=0,
                request_timeout=timeout,  # type: ignore[call-arg]
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(  # type: ignore[call-arg]
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),  # ty: ignore[unknown-argument]
            api_key=os.getenv("OPENAI_API_KEY"),  # ty: ignore[unknown-argument]
            temperature=0,
            request_timeout=timeout,  # type: ignore[call-arg]
        )

    @staticmethod
    def _default_gen_llm() -> BaseChatModel:
        """Create default generation LLM (higher timeout)."""
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            from langchain_openai import AzureChatOpenAI

            deploy = os.getenv("AZURE_OPENAI_GENERATION_DEPLOYMENT") or os.getenv(
                "AZURE_OPENAI_DEPLOYMENT"
            )
            return AzureChatOpenAI(  # type: ignore[call-arg]
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=deploy,  # ty: ignore[unknown-argument]
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),  # ty: ignore[unknown-argument]
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),  # ty: ignore[unknown-argument]
                temperature=0,
                request_timeout=60,  # type: ignore[call-arg]
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(  # type: ignore[call-arg]
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),  # ty: ignore[unknown-argument]
            api_key=os.getenv("OPENAI_API_KEY"),  # ty: ignore[unknown-argument]
            temperature=0,
            request_timeout=60,  # type: ignore[call-arg]
        )

    @staticmethod
    def _default_reranker() -> CohereReranker | LLMReranker:
        """Create default reranker: Cohere if available, else LLM-based."""
        try:
            return CohereReranker()
        except (ImportError, Exception):
            return LLMReranker()

    @classmethod
    def from_model(
        cls,
        model: str,
        index: str,
        *,
        gen_model: str | None = None,
        configurable_fields: str | list[str] | None = None,
        **kwargs: Any,
    ) -> "AgenticRAG":
        """Create an AgenticRAG from a provider-prefixed model string.

        Wraps LangChain's ``init_chat_model`` so you never have to import
        provider-specific classes.  Pass ``"provider:model-name"`` and the
        right LangChain package is selected automatically.

        Supported providers (same as ``init_chat_model``):
            openai, anthropic, azure_openai, google_vertexai, google_genai,
            bedrock, mistralai, groq, ollama, fireworks, together, …

        Parameters
        ----------
        model:
            Model string, either ``"provider:model-name"`` (e.g.
            ``"openai:gpt-5.4"``, ``"anthropic:claude-sonnet-4-6"``,
            ``"ollama:llama3"``) or a bare model name when the provider can
            be inferred from installed packages.
        index:
            Collection / index name in the backend.
        gen_model:
            Optional separate generation model string in the same format.
            Defaults to ``model``.
        configurable_fields:
            Passed through to ``init_chat_model`` for runtime configurability.
        **kwargs:
            All other ``AgenticRAG.__init__`` keyword arguments (``backend``,
            ``reranker``, ``top_k``, ``instructions``, ``embed_fn``, …).

        Examples
        --------
        >>> from rag7 import Agent
        >>> rag = Agent.from_model("openai:gpt-5.4", index="docs")
        >>> rag = Agent.from_model(
        ...     "anthropic:claude-sonnet-4-6",
        ...     index="docs",
        ...     gen_model="anthropic:claude-opus-4-6",
        ...     instructions="Answer only in English.",
        ... )
        >>> rag = Agent.from_model("ollama:llama3", index="docs")
        """
        from langchain.chat_models import init_chat_model  # type: ignore[import-untyped]

        init_kwargs: dict[str, Any] = {"temperature": 0}
        if configurable_fields is not None:
            init_kwargs["configurable_fields"] = configurable_fields

        llm = init_chat_model(model, **init_kwargs)
        gen_llm = init_chat_model(gen_model, **init_kwargs) if gen_model else llm
        return cls(index=index, llm=llm, gen_llm=gen_llm, **kwargs)

    def __init__(
        self,
        index: str,
        *,
        backend: SearchBackend | None = None,
        collections: Mapping[str, SearchBackend] | None = None,
        collection_descriptions: Mapping[str, str] | None = None,
        llm: BaseChatModel | None = None,
        gen_llm: BaseChatModel | None = None,
        reranker: Reranker | CohereReranker | LLMReranker | None = None,
        top_k: int | None = None,
        rerank_top_n: int | None = None,
        retrieval_factor: int | None = None,
        max_iter: int | None = None,
        instructions: str = "",
        n_swarm_queries: int | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
        embedder_name: str | None = None,
        semantic_ratio: float | None = None,
        boost_fn: Callable[[dict], float] | None = None,
        sort_fields: list[str] | None = None,
        filter: str | None = None,  # noqa: A002
        base_filter: str | None = None,
        rerank_chars: int | None = None,
        hyde_min_words: int | None = None,
        hyde_style_hint: str = "",
        auto_strategy: bool = True,
        group_field: str = "",
        name_field: str = "",
        category_fields: list[str] | None = None,
        fusion: Literal["rrf", "dbsf"] | None = None,
        expert_reranker: Reranker | CohereReranker | LLMReranker | None = None,
        expert_top_n: int | None = None,
        expert_threshold: float | None = None,
        verbose: bool | None = None,
        checkpointer: Any = None,
        memory_store: Any = None,
    ):
        self.index = index
        if collections:
            self.collections: dict[str, SearchBackend] | None = dict(collections)
            self.collection_descriptions: dict[str, str] = dict(
                collection_descriptions or {name: name for name in collections}
            )
            self.backend = _MultiBackend(self.collections)
        else:
            self.collections = None
            self.collection_descriptions = {}
            self.backend = backend or InMemoryBackend()
        self.top_k = top_k or int(os.getenv("RAG_TOP_K", "10"))
        self.rerank_top_n = rerank_top_n or int(os.getenv("RAG_RERANK_TOP_N", "5"))
        self.retrieval_factor = retrieval_factor or int(
            os.getenv("RAG_RETRIEVAL_FACTOR", "4")
        )
        self.max_iter = max_iter or int(os.getenv("RAG_MAX_ITER", "3"))
        self.verbose = (
            verbose if verbose is not None else bool(int(os.getenv("RAG_VERBOSE", "0")))
        )
        self.n_swarm_queries = n_swarm_queries or int(
            os.getenv("RAG_N_SWARM_QUERIES", "4")
        )
        self.embedder_name = embedder_name or os.getenv(
            "RAG_EMBEDDER_NAME", "azure_openai"
        )
        self.semantic_ratio = (
            semantic_ratio
            if semantic_ratio is not None
            else float(os.getenv("RAG_SEMANTIC_RATIO", "0.5"))
        )
        self.rerank_chars = (
            rerank_chars
            if rerank_chars is not None
            else int(os.getenv("RAG_RERANK_CHARS", "2048"))
        )
        self.hyde_min_words = (
            hyde_min_words
            if hyde_min_words is not None
            else int(os.getenv("RAG_HYDE_MIN_WORDS", "8"))
        )
        self.fusion: Literal["rrf", "dbsf"] = fusion or os.getenv("RAG_FUSION", "rrf")  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        self.instructions = instructions
        self.embed_fn = embed_fn
        self.hyde_style_hint = hyde_style_hint
        self.group_field = group_field
        self.name_field = name_field
        self.category_fields = (
            list(category_fields)
            if category_fields is not None
            else _detect_category_fields(self.backend)
        )

        num_fields: list[str] = []
        if sort_fields is None and boost_fn is None:
            sort_fields, boost_fn, num_fields = _detect_index_signals(self.backend)
        self.boost_fn = boost_fn
        self.sort_fields = sort_fields
        self.num_fields = num_fields
        if base_filter is not None:
            warnings.warn(
                "`base_filter` is deprecated, use `filter` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if filter is None:
                filter = base_filter
        self.filter = filter

        self._llm = llm or self._default_llm()
        self._gen_llm = gen_llm or self._default_gen_llm()
        self._reranker = reranker or self._default_reranker()
        self._expert_reranker = expert_reranker
        self.expert_top_n = expert_top_n or int(os.getenv("RAG_EXPERT_TOP_N", "10"))
        self.expert_threshold = (
            expert_threshold
            if expert_threshold is not None
            else float(os.getenv("RAG_EXPERT_THRESHOLD", "0.15"))
        )

        self._domain_hint: str = ""
        self._field_values_cache: dict = {}

        if auto_strategy:
            self._auto_configure(override_hyde_min_words=hyde_min_words is None)

        self._search_query_chain = self._llm.with_structured_output(
            SearchQuery, method="json_schema"
        )
        self._quality_chain = self._llm.with_structured_output(
            QualityAssessment, method="json_schema"
        )
        self._multi_query_chain = self._llm.with_structured_output(
            MultiQuery, method="json_schema"
        )
        self._filter_intent_chain = self._llm.with_structured_output(
            FilterIntent, method="json_schema"
        )
        self._select_collection_chain = self._llm.with_structured_output(
            CollectionIntent, method="json_schema"
        )
        self._relevance_chain = self._llm.with_structured_output(
            RelevanceCheck, method="json_schema"
        )
        self._reasoning_chain = self._llm.with_structured_output(
            ReasoningVerdict, method="json_schema"
        )

        self._toolset = RAGToolset(self)
        self._tools = self._toolset.as_tools()
        self._checkpointer = checkpointer
        self._memory_store = memory_store
        self._graph = self._build_graph()
        self._agent = self._build_agent()

    @property
    def base_filter(self) -> str | None:
        """Deprecated alias for ``self.filter``."""
        warnings.warn(
            "`base_filter` attribute is deprecated, use `filter` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.filter

    @base_filter.setter
    def base_filter(self, value: str | None) -> None:
        warnings.warn(
            "`base_filter` attribute is deprecated, use `filter` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.filter = value

    def _auto_configure(self, override_hyde_min_words: bool = True) -> None:
        try:
            sample = self.backend.sample_documents(
                limit=5,
                filter_expr=self.filter,
            )
            if not sample:
                return
            previews: list[str] = []
            max_doc_chars = 0
            for h in sample[:5]:
                items = [
                    (k, str(v))
                    for k, v in h.items()
                    if k not in _SKIP_FIELDS and v and isinstance(v, (str, int, float))
                ]
                previews.append(str({k: v[:120] for k, v in items}))
                max_doc_chars = max(max_doc_chars, sum(len(v) for _, v in items))

            if max_doc_chars > self.rerank_chars:
                self.rerank_chars = min(8192, int(max_doc_chars * 1.2))
            result = self._llm.invoke(
                [
                    SystemMessage(
                        "Analyze these sample documents and return ONLY a JSON object (no markdown) with:\n"
                        "- hyde_style_hint: str — one short phrase for HyDE prompt, e.g. "
                        "'German legal ruling', 'product spec sheet', 'customer address record', "
                        "'Wikipedia article', 'medical Q&A'\n"
                        "- hyde_min_words: int 3–12 — min query words to trigger HyDE expansion. "
                        "Lower (3–5) for factual Q&A / legal where even short questions need expansion. "
                        "Higher (8–12) for keyword search / products where short queries are already precise.\n"
                        "- domain_hint: str — 1-2 sentences describing what kind of content is stored, "
                        "what identifiers or codes exist (e.g. article numbers, case IDs), what language "
                        "the content is in, and any query preprocessing tips specific to this domain "
                        "(e.g. verb-to-noun normalisation for German product queries, case number format, "
                        "medical abbreviations). Leave empty string if no domain-specific tips apply."
                    ),
                    HumanMessage("Sample documents:\n" + "\n---\n".join(previews)),
                ]
            )
            config = json.loads(str(result.content).strip())
            if "hyde_style_hint" in config:
                self.hyde_style_hint = str(config["hyde_style_hint"])
            if "hyde_min_words" in config and override_hyde_min_words:
                self.hyde_min_words = int(config["hyde_min_words"])
            if "domain_hint" in config:
                self._domain_hint = str(config["domain_hint"])
            print(f"  [{self.index}] strategy: {config}")
        except Exception as e:
            print(f"  [{self.index}] auto-strategy failed ({e}), using defaults")

    def _sys(self, prompt: str) -> SystemMessage:
        if self.instructions:
            return SystemMessage(f"{self.instructions}\n\n{prompt}")
        return SystemMessage(prompt)

    def _hit_to_text(self, h: dict) -> str:
        if "content" in h:
            return h["content"]
        return "\n".join(
            f"{k}: {v.strip() if isinstance(v, str) else v}"
            for k, v in h.items()
            if k not in _SKIP_FIELDS
            and (isinstance(v, str) and v.strip() or isinstance(v, (int, float)) and v)
        )

    def _make_search_request(
        self,
        query: str,
        limit: int,
        *,
        vector: list[float] | None = None,
        semantic_ratio: float | None = None,
        filter_expr: str | None = None,
        sort_fields: list[str] | None = None,
        show_ranking_score: bool = False,
    ) -> SearchRequest:
        filters = [f for f in [self.filter, filter_expr] if f]
        return SearchRequest(
            query=query,
            limit=limit,
            vector=vector,
            semantic_ratio=semantic_ratio
            if semantic_ratio is not None
            else self.semantic_ratio,
            filter_expr=" AND ".join(filters) if filters else None,
            sort_fields=sort_fields,
            show_ranking_score=show_ranking_score,
            embedder_name=self.embedder_name,
            index_uid=self.index,
        )

    @_llm_retry
    async def _ahypothetical_doc(self, query: str) -> str:
        cached = _cache.load("hyde-v1", self.hyde_style_hint or "", query)
        if cached:
            return str(cached)
        style = f" ({self.hyde_style_hint})" if self.hyde_style_hint else ""
        result = await self._llm.ainvoke(
            [
                self._sys(
                    f"Write 2-4 sentences from a relevant document{style}. "
                    "Use domain terminology, specific identifiers, citations, "
                    "or technical terms likely present in the source. "
                    "No questions — write as extracted text."
                ),
                HumanMessage(query),
            ]
        )
        text = str(result.content).strip() or query
        _cache.save("hyde-v1", self.hyde_style_hint or "", query, value=text)
        return text

    def _multi_search(
        self,
        queries: list[str],
        lang: str | None = None,
        vectors: list[list[float] | None] | None = None,
        filter_expr: str | None = None,
    ) -> list[Document]:
        limit = int(self.top_k * self.retrieval_factor)
        vecs: list[list[float] | None] = vectors if vectors else [None] * len(queries)
        # lang filter and explicit filter_expr can be combined
        parts = [p for p in [f"language = {lang}" if lang else None, filter_expr] if p]
        combined_filter = " AND ".join(f"({p})" for p in parts) if parts else None

        requests = [
            self._make_search_request(q, limit, vector=v, filter_expr=combined_filter)
            for q, v in zip(queries, vecs)
        ]
        results = self.backend.batch_search(requests)
        fused = _rrf_fuse(results)
        return [
            Document(page_content=self._hit_to_text(h), metadata=h)
            for h in fused[:limit]
        ]

    async def _apreprocess(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        # Short, keyword-like questions don't benefit from LLM rewrite: skip the call.
        if len(state.question.split()) <= 6:
            raw = _strip_stop_words(state.question) or state.question
            new = state.model_copy(
                update={
                    "query": raw,
                    "query_variants": [],
                    "adaptive_semantic_ratio": None,
                    "adaptive_fusion": None,
                }
            )
            self._trace(new, "preprocess", t0, path="short_skip", query=raw)
            return new
        cached = _cache.load("preprocess-v3", self._domain_hint or "", state.question)
        if cached:
            try:
                result = SearchQuery.model_validate(cached)
                raw = _strip_stop_words(state.question)
                variants = result.variants[:3]
                if raw and raw.lower() != result.query.lower():
                    variants = [raw] + variants
                updates: dict = {
                    "query": result.query,
                    "query_variants": variants[:4],
                    "adaptive_semantic_ratio": result.semantic_ratio,
                    "adaptive_fusion": result.fusion,
                }
                if result.alternative_to:
                    updates["alternative_to"] = result.alternative_to
                new = state.model_copy(update=updates)
                self._trace(new, "preprocess", t0, cached=True)
                return new
            except Exception:
                pass
        try:
            result = cast(
                SearchQuery,
                await self._search_query_chain.ainvoke(
                    [
                        self._sys(
                            "Extract 2-5 concise search keywords from the user's question. "
                            "Return ONLY essential nouns, entities, and technical terms. "
                            "Omit question words, common verbs, and stop words (e.g. find, search, what, how, give me). "
                            "Preserve identifiers, codes, and specialized terms exactly. "
                            + (
                                f"Domain context: {self._domain_hint} "
                                if self._domain_hint
                                else ""
                            )
                            + "Also set 'variants': 2-3 more specific alternative keyword phrasings — "
                            "use technical names, codes, or narrower terminology for the same concept. "
                            "Avoid broader synonyms. Keep keywords in the same language as the query. "
                            "Also set 'semantic_ratio' (0.0-1.0): how much to weight semantic/vector "
                            "search vs keyword/BM25. "
                            "- Exact identifiers, codes, names → 0.1-0.2 (BM25 excels) "
                            "- Specific nouns, mixed queries → 0.3-0.5 (balanced) "
                            "- Intent/need-based queries → 0.6-0.8 (semantic helps) "
                            "- Abstract/conceptual questions → 0.7-0.9 (semantic dominant) "
                            "Also set 'fusion' ('rrf' or 'dbsf'): how to fuse multi-arm search results. "
                            "- 'rrf': Reciprocal Rank Fusion — rank-based, best for keyword-heavy queries. "
                            "- 'dbsf': Distribution-Based Score Fusion — score-normalised, best for semantic queries. "
                            "Also set 'alternative_to' (string or null): if the user asks for alternatives, "
                            "replacements, substitutes, or competitors for a SPECIFIC named product/item, "
                            "set this to the product name/identifier. "
                            "Trigger words: 'alternative', 'ersatz', 'replacement', 'substitut', 'competitor', "
                            "'statt', 'anstelle', 'remplacer', 'remplaçant', 'à la place de'. "
                            "The query should then contain the CATEGORY/TYPE terms (what kind of product it is), "
                            "NOT the specific product name."
                        ),
                        HumanMessage(state.question),
                    ]
                ),
            )
            _cache.save(
                "preprocess-v3",
                self._domain_hint or "",
                state.question,
                value=result.model_dump(),
            )
            raw = _strip_stop_words(state.question)
            variants = result.variants[:3]
            if raw and raw.lower() != result.query.lower():
                variants = [raw] + variants
            updates: dict = {
                "query": result.query,
                "query_variants": variants[:4],
                "adaptive_semantic_ratio": result.semantic_ratio,
                "adaptive_fusion": result.fusion,
            }
            if result.alternative_to:
                updates["alternative_to"] = result.alternative_to
            new = state.model_copy(update=updates)
            self._trace(
                new,
                "preprocess",
                t0,
                query=result.query,
                semantic_ratio=result.semantic_ratio,
                fusion=result.fusion,
                variants=len(variants[:4]),
                alternative_to=result.alternative_to,
            )
            return new
        except Exception as e:
            new = state.model_copy(update={"query": state.question})
            self._trace(new, "preprocess", t0, error=type(e).__name__)
            return new

    async def _asearch(
        self,
        query: str,
        question: str | None = None,
        lang: str | None = None,
        extra_bm25: list[str] | None = None,
        factor: int | None = None,
        adaptive_semantic_ratio: float | None = None,
        adaptive_fusion: str | None = None,
        hyde_text: str | None = None,
    ) -> list[Document]:
        limit = int(self.top_k * (factor or self.retrieval_factor))
        hyde_source = question or query

        if hyde_text is None:
            if self.embed_fn and len(hyde_source.split()) >= self.hyde_min_words:
                try:
                    hyde_text = await self._ahypothetical_doc(hyde_source)
                except Exception:
                    hyde_text = hyde_source
            else:
                hyde_text = hyde_source
        loop = asyncio.get_running_loop()
        vector: list[float] | None = None
        if self.embed_fn:
            try:
                vector = await loop.run_in_executor(None, self.embed_fn, hyde_text)
            except Exception:
                vector = None

        lang_filter = f"language = {lang}" if lang else None
        hyde_bm25 = hyde_text if hyde_text != hyde_source else query
        hybrid_ratio = (
            adaptive_semantic_ratio
            if adaptive_semantic_ratio is not None
            else self.semantic_ratio
        )

        def _req(q: str, **extra: Any) -> SearchRequest:
            return self._make_search_request(
                q,
                limit,
                sort_fields=self.sort_fields,
                show_ranking_score=True,
                filter_expr=lang_filter,
                **extra,
            )

        requests = [
            _req(query),
            _req(hyde_bm25),
            _req(query, vector=vector, semantic_ratio=hybrid_ratio),
        ]
        if vector and hybrid_ratio < 1.0:
            requests.append(_req(query, vector=vector, semantic_ratio=1.0))
        seen = {query, hyde_bm25}
        for v in extra_bm25 or []:
            if v and v not in seen:
                requests.append(_req(v))
                seen.add(v)

        results = await loop.run_in_executor(None, self.backend.batch_search, requests)
        fuse_method = adaptive_fusion or self.fusion
        if results:
            if fuse_method == "dbsf":
                fused = _dbsf_fuse(results)
            else:
                fused = _rrf_fuse(results, ranking_score_weight=1.0)
        else:
            fused = await loop.run_in_executor(
                None,
                self.backend.search,
                self._make_search_request(query, limit, filter_expr=lang_filter),
            )

        return [
            Document(page_content=self._hit_to_text(h), metadata=h)
            for h in fused[:limit]
        ]

    async def _aroute_collections(self, state: RAGState) -> RAGState:
        if self.collections and not state.selected_collections:
            selected = await self._aselect_collections(state.question)
            _ACTIVE_COLLECTIONS.set(selected)
            return state.model_copy(update={"selected_collections": selected})
        if state.selected_collections:
            _ACTIVE_COLLECTIONS.set(state.selected_collections)
        return state

    def _trace(self, state: RAGState, node: str, t0: float, **info: Any) -> None:
        dur = round(time.perf_counter() - t0, 4)
        state.trace.append({"node": node, "dur_s": dur, **info})
        if self.verbose:
            extras = " ".join(f"{k}={v}" for k, v in info.items())
            msg = f"[rag {node}] {dur:.3f}s {extras}".rstrip()
            print(msg, file=sys.stderr, flush=True)

    async def _aparallel_start(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        state = await self._aroute_collections(state)
        state = await self._apreprocess(state)
        if state.alternative_to:
            _, docs = await self._aalternative_retrieve(
                state.query, state.alternative_to, self.top_k
            )
            new = state.model_copy(update={"documents": docs, "iterations": 1})
            self._trace(
                new, "parallel_start", t0, docs=len(docs), alt=state.alternative_to
            )
            return new
        new = await self._aretrieve_node(state)
        self._trace(new, "parallel_start", t0, docs=len(new.documents))
        return new

    async def _aswarm_retrieve(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        state = await self._aroute_collections(state)

        # Generate query variants AND detect filter intent in parallel
        async def _gen_variants() -> list[str]:
            cached = _cache.load("multi-query-v1", self.n_swarm_queries, state.question)
            if cached and isinstance(cached, list):
                return cached[: self.n_swarm_queries] or [state.query]
            try:
                result = cast(
                    MultiQuery,
                    await self._multi_query_chain.ainvoke(
                        [
                            self._sys(
                                f"Generate {self.n_swarm_queries} distinct search query variants for the question. "
                                "Each should use different keywords, synonyms, or angles to maximise recall. "
                                f"Return JSON with a 'queries' list of {self.n_swarm_queries} strings."
                            ),
                            HumanMessage(state.question),
                        ]
                    ),
                )
                queries = result.queries[: self.n_swarm_queries] or [state.query]
                if result.queries:
                    _cache.save(
                        "multi-query-v1",
                        self.n_swarm_queries,
                        state.question,
                        value=result.queries,
                    )
                return queries
            except Exception:
                return [state.query]

        queries, filter_intent = await asyncio.gather(
            _gen_variants(),
            self._adetect_filter_intent(state.question),
        )

        vectors: list[list[float] | None] | None = None
        if self.embed_fn:
            try:
                vectors = await _embed_all_async(self.embed_fn, queries)
            except Exception:
                vectors = None

        loop = asyncio.get_running_loop()

        def _search(filter_expr: str | None = None) -> list[Document]:
            return self._multi_search(queries, vectors=vectors, filter_expr=filter_expr)

        has_filter = (
            filter_intent.field and filter_intent.value and filter_intent.operator
        )
        if has_filter:
            f_expr = f'{filter_intent.field} {filter_intent.operator} "{filter_intent.value}"'
            broad_docs, filtered_docs = await asyncio.gather(
                loop.run_in_executor(None, _search),
                loop.run_in_executor(None, _search, f_expr),
            )
            docs = (
                self._merge_doc_lists(filtered_docs, broad_docs)
                if filtered_docs
                else broad_docs
            )
        else:
            docs = await loop.run_in_executor(None, _search)

        new = state.model_copy(
            update={"documents": docs, "iterations": state.iterations + 1}
        )
        self._trace(
            new,
            "swarm_retrieve",
            t0,
            variants=len(queries),
            docs=len(docs),
            has_filter=bool(has_filter),
        )
        return new

    async def _aretrieve_node(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        factor = (
            self.retrieval_factor * 2
            if state.iterations >= 1
            else self.retrieval_factor
        )
        intent: FilterIntent | None = state.filter_intent
        broad_coro = self._asearch(
            state.query,
            question=state.question,
            extra_bm25=state.query_variants,
            factor=factor,
            adaptive_semantic_ratio=state.adaptive_semantic_ratio,
            adaptive_fusion=state.adaptive_fusion,
        )
        if state.iterations == 0:
            broad_docs, (intent, filter_docs) = await asyncio.gather(
                broad_coro,
                self._afilter_search_with_intent(state.question, state.query),
            )
            if filter_docs and len(filter_docs) >= self.top_k:
                docs = filter_docs
            elif filter_docs:
                docs = self._merge_doc_lists(filter_docs, broad_docs)
            else:
                docs = broad_docs
        else:
            docs = await broad_coro
        new = state.model_copy(
            update={
                "documents": docs,
                "iterations": state.iterations + 1,
                "filter_intent": intent,
            }
        )
        self._trace(
            new,
            "retrieve",
            t0,
            iter=new.iterations,
            docs=len(docs),
            factor=factor,
            filter_field=(intent.field if intent else None),
        )
        return new

    async def _afilter_search_with_intent(
        self, question: str, query: str
    ) -> tuple[FilterIntent | None, list[Document]]:
        intent = await self._adetect_filter_intent(question)
        if not intent.field or not intent.value or not intent.operator:
            return None, []
        docs = await self._afilter_search_from_intent(query, intent)
        return intent, docs

    def _build_filter_expr(self, intent: FilterIntent) -> str:
        """Build a filter expression in the active backend's dialect.

        Delegates to backend.build_filter_expr so SQL backends (LanceDB,
        DuckDB, pgvector) get SQL syntax with ILIKE, Azure gets OData, and
        Meili keeps its CONTAINS-based syntax.
        """
        return self.backend.build_filter_expr(intent)

    async def _afilter_search_from_intent(
        self, query: str, intent: FilterIntent
    ) -> list[Document]:
        filter_expr = self._build_filter_expr(intent)
        limit = int(self.top_k * self.retrieval_factor)
        loop = asyncio.get_running_loop()
        vector: list[float] | None = None
        if self.embed_fn:
            try:
                vector = await loop.run_in_executor(None, self.embed_fn, query)
            except Exception:
                pass
        try:
            hits = await loop.run_in_executor(
                None,
                self.backend.search,
                self._make_search_request(
                    query, limit, vector=vector, filter_expr=filter_expr
                ),
            )
        except Exception:
            return []
        return [
            Document(page_content=self._hit_to_text(h), metadata=h)
            for h in hits[:limit]
        ]

    def _make_rerank_docs(self, docs: list[Document]) -> list[str]:
        return [d.page_content[: self.rerank_chars] for d in docs]

    def _apply_boost(
        self,
        docs: list[Document],
        indexed: list[tuple[int, float]],
        query: str = "",
    ) -> list[Document]:
        if not self.boost_fn:
            return [docs[i] for i, _ in indexed]
        confident = False
        if indexed and query and self.name_field:
            top_doc_idx = max(indexed, key=lambda p: p[1])[0]
            top_name = (docs[top_doc_idx].metadata.get(self.name_field) or "").lower()
            q_tokens = [t for t in query.lower().split() if len(t) > 3]
            confident = bool(q_tokens) and any(t in top_name for t in q_tokens)
        boost = (lambda _: 1.0) if confident else self.boost_fn
        scored = sorted(
            [(docs[i], s * boost(docs[i].metadata)) for i, s in indexed],
            key=lambda x: x[1],
            reverse=True,
        )
        if query and self.group_field and self.name_field and len(scored) >= 5:
            top_groups = {d.metadata.get(self.group_field, "") for d, _ in scored[:5]}
            if len(top_groups) == 1 and next(iter(top_groups)):
                tokens = {t for t in query.lower().split() if len(t) > 3}
                if tokens:
                    name_f = self.name_field

                    def _tok_boost(d: Document) -> float:
                        name = (d.metadata.get(name_f, "") or "").lower()
                        return 1.5 if all(t in name for t in tokens) else 1.0

                    scored = sorted(
                        ((d, s * _tok_boost(d)) for d, s in scored),
                        key=lambda x: x[1],
                        reverse=True,
                    )
        top = scored[: self.top_k]
        num_fields = getattr(self, "num_fields", None) or []
        if top and len(top) > 1 and num_fields:
            max_score = top[0][1]
            threshold = max_score * 0.9
            cluster = [(d, s) for d, s in top if s >= threshold]
            if len(cluster) > 1:

                def _num_signal(d: Document) -> float:
                    return sum(
                        float(d.metadata.get(f, 0) or 0)
                        for f in num_fields
                        if isinstance(d.metadata.get(f), (int, float))
                    )

                cluster.sort(key=lambda x: (_num_signal(x[0]), x[1]), reverse=True)
                top = cluster + [(d, s) for d, s in top if s < threshold]
        return [d for d, _ in top]

    @staticmethod
    def _doc_matches_intent(doc: Document, intent: FilterIntent) -> bool:
        field, value, op = intent.field, intent.value, intent.operator
        if not field:
            return True
        actual = doc.metadata.get(field)
        if actual is None:
            return op in ("NOT_CONTAINS", "!=")
        actual_str = str(actual).lower()
        value_str = str(value).lower()
        if op == "=":
            return actual_str == value_str
        if op == "CONTAINS":
            return value_str in actual_str
        if op == "NOT_CONTAINS":
            import re as _re

            all_values = [value_str] + [v.lower() for v in intent.extra_excludes]
            for val in all_values:
                if val in actual_str:
                    return False
                spaced = _re.sub(r"(?<=[a-z])(?=[A-Z])", " ", val).lower()
                if spaced != val and spaced in actual_str:
                    return False
                ws = val.split()
                if len(ws) >= 3 and " ".join(ws[:2]) in actual_str:
                    return False
                sw = spaced.split()
                if len(sw) >= 3:
                    sp2 = " ".join(sw[:2])
                    if sp2 in actual_str:
                        return False
            return True
        if op == "!=":
            return actual_str != value_str
        return True

    async def _aexpert_rerank(
        self,
        query: str,
        documents: list[Document],
        indexed: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """Rescore top-N of primary rerank output with expert reranker."""
        if not self._expert_reranker:
            return indexed
        head = indexed[: self.expert_top_n]
        tail = indexed[self.expert_top_n :]
        docs = [documents[i].page_content[: self.rerank_chars] for i, _ in head]
        idx_map = [i for i, _ in head]
        arerank_fn = getattr(self._expert_reranker, "arerank", None)
        try:
            if arerank_fn is not None:
                results = await asyncio.wait_for(
                    arerank_fn(query, docs, len(docs)), timeout=60.0
                )
            else:
                loop = asyncio.get_running_loop()
                results = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        self._expert_reranker.rerank,
                        query,
                        docs,
                        len(docs),
                    ),
                    timeout=60.0,
                )
        except (asyncio.TimeoutError, Exception):
            return indexed
        rescored = [(idx_map[r.index], r.relevance_score) for r in results]
        return rescored + tail

    async def _arerank(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        if not state.documents:
            return state
        # Selective skip: confident query + no filter intent + no expert → hybrid order good enough
        if (
            not (state.filter_intent and state.filter_intent.field)
            and not self._expert_reranker
            and self.name_field
            and state.query
        ):
            q_tokens = [t for t in state.query.lower().split() if len(t) > 3]
            if q_tokens:
                top_name = (
                    state.documents[0].metadata.get(self.name_field) or ""
                ).lower()
                token_hits = sum(1 for t in q_tokens if t in top_name)
                docs_scored = state.documents[:5]
                scores = [
                    float(d.metadata.get("_rankingScore", 0.0)) for d in docs_scored
                ]
                dominant = (
                    len(scores) >= 2
                    and scores[0] >= 0.85
                    and scores[0] - scores[-1] >= 0.1
                )
                if token_hits >= max(1, len(q_tokens) // 2) and dominant:
                    indexed = [(i, 1.0 / (i + 1)) for i in range(len(state.documents))]
                    ranked = self._apply_boost(state.documents, indexed, state.query)
                    new = state.model_copy(
                        update={"documents": ranked, "expert_fired": False}
                    )
                    self._trace(new, "rerank", t0, skipped="confident")
                    return new
        rerank_docs = self._make_rerank_docs(state.documents)
        effective_top_n = len(rerank_docs)
        rerank_query = state.question or state.query

        try:
            arerank_fn = getattr(self._reranker, "arerank", None)
            if arerank_fn is not None:
                coro = arerank_fn(rerank_query, rerank_docs, effective_top_n)
            else:
                coro = asyncio.get_running_loop().run_in_executor(
                    None,
                    self._reranker.rerank,
                    rerank_query,
                    rerank_docs,
                    effective_top_n,
                )
            results = await asyncio.wait_for(coro, timeout=30.0)
            indexed = [(r.index, r.relevance_score) for r in results]
            expert_fired = False
            if self._expert_reranker and len(results) >= 2:
                scores = sorted((r.relevance_score for r in results), reverse=True)
                top1 = scores[0]
                ref = scores[min(len(scores) - 1, 4)]
                gap = top1 - ref
                if gap < self.expert_threshold:
                    indexed = await self._aexpert_rerank(
                        rerank_query, state.documents, indexed
                    )
                    expert_fired = True
            ranked = self._apply_boost(state.documents, indexed, state.query)
            intent = state.filter_intent
            if intent and intent.field and ranked:
                is_negation = intent.operator in ("NOT_CONTAINS", "!=")
                if is_negation:
                    exclude_vals = [str(intent.value)] + list(intent.extra_excludes)
                    ranked = [
                        d
                        for d in ranked
                        if self._doc_matches_intent(d, intent)
                        and not any(
                            self._content_contains_exclusion(d.page_content, v)
                            for v in exclude_vals
                        )
                    ]
                else:
                    top_n = self.rerank_top_n
                    top_match_count = sum(
                        1 for d in ranked[:top_n] if self._doc_matches_intent(d, intent)
                    )
                    if top_match_count >= 1:
                        matching_all = [
                            state.documents[i]
                            for i, _ in indexed
                            if self._doc_matches_intent(state.documents[i], intent)
                        ]
                        seen_ids = {_doc_id(d.metadata) for d in matching_all}
                        rest = [
                            d for d in ranked if _doc_id(d.metadata) not in seen_ids
                        ]
                        ranked = matching_all + rest
            new = state.model_copy(
                update={"documents": ranked, "expert_fired": expert_fired}
            )
            self._trace(new, "rerank", t0, docs=len(ranked), expert_fired=expert_fired)
            return new
        except (asyncio.TimeoutError, Exception) as e:
            self._trace(state, "rerank", t0, error=type(e).__name__)
            return state

    async def _agenerate(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        numbered = "\n\n---\n\n".join(
            f"[{i + 1}] {d.page_content}"
            for i, d in enumerate(state.documents[: self.top_k])
        )
        messages: list = [
            self._sys(
                "Answer using only the provided context. "
                "Cite sources inline using [n] numbers that match the context blocks. "
                "Say so if the context is insufficient."
            ),
        ]
        for turn in state.history:
            messages.append(HumanMessage(turn.question))
            messages.append(AIMessage(turn.answer))
        messages.append(
            HumanMessage(f"Context:\n{numbered}\n\nQuestion: {state.question}")
        )
        response = await self._gen_llm.ainvoke(messages)
        update: dict[str, Any] = {"answer": str(response.content)}
        # Don't persist retrieved documents in the checkpoint — they're
        # re-fetched on every call. Keeps checkpointed state lean.
        if self._checkpointer is not None:
            update["documents"] = []
        new = state.model_copy(update=update)
        self._trace(
            new,
            "generate",
            t0,
            docs=len(state.documents[: self.top_k]),
            answer_chars=len(str(response.content)),
        )
        return new

    async def _arewrite(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        if not state.documents:
            if state.adaptive_semantic_ratio is None:
                pre = await self._apreprocess(state)
                state = state.model_copy(
                    update={
                        "query": pre.query or state.query,
                        "query_variants": pre.query_variants,
                        "adaptive_semantic_ratio": pre.adaptive_semantic_ratio,
                        "adaptive_fusion": pre.adaptive_fusion,
                    }
                )
            new = await self._aswarm_retrieve(state)
            self._trace(new, "rewrite", t0, path="swarm", docs=len(new.documents))
            return new

        if state.quality_ok is False and state.documents:
            top_snippet = state.documents[0].page_content[:150]
            prompt = (
                "The search returned documents but they are NOT relevant to the question. "
                f'Previous query: "{state.query}". '
                f'Top result snippet: "{top_snippet}...". '
                "Rewrite using different keywords, synonyms, or a narrower angle. "
                "Return only 1-4 keywords — no filler words."
            )
        else:
            prompt = (
                "The previous search returned NO results. Rewrite the query using different words, "
                "synonyms, or a broader term. Return only 1-3 short keywords — no filler."
            )

        try:
            rewrite_coro = self._search_query_chain.ainvoke(
                [self._sys(prompt), HumanMessage(state.question)]
            )
            if state.adaptive_semantic_ratio is None:
                pre, result = await asyncio.gather(
                    self._apreprocess(state), rewrite_coro
                )
                state = state.model_copy(
                    update={
                        "query_variants": pre.query_variants,
                        "adaptive_semantic_ratio": pre.adaptive_semantic_ratio,
                        "adaptive_fusion": pre.adaptive_fusion,
                    }
                )
            else:
                result = await rewrite_coro
            result = cast(SearchQuery, result)
            new = state.model_copy(update={"query": result.query, "quality_ok": None})
            self._trace(new, "rewrite", t0, path="llm", new_query=result.query)
            return new
        except Exception as e:
            self._trace(state, "rewrite", t0, path="llm", error=type(e).__name__)
            return state

    def _give_up(self, state: RAGState) -> RAGState:
        return state.model_copy(
            update={
                "answer": f"No relevant documents found after {state.iterations} attempts."
            }
        )

    @staticmethod
    def _content_contains_exclusion(text: str, value: str) -> bool:
        import re as _re

        low = text.lower()
        val = value.lower()
        if val in low:
            return True
        spaced = _re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value).lower()
        if spaced != val and spaced in low:
            return True
        words = val.split()
        if len(words) >= 3:
            prefix2 = " ".join(words[:2])
            if prefix2 in low:
                return True
            spaced_words = spaced.split()
            if len(spaced_words) >= 3:
                sp2 = " ".join(spaced_words[:2])
                if sp2 != prefix2 and sp2 in low:
                    return True
        return False

    def _needs_reasoning(self, state: RAGState) -> bool:
        if not state.documents:
            return False
        intent = state.filter_intent
        if intent and intent.operator in ("NOT_CONTAINS", "!="):
            exclude_vals = [str(intent.value)] + list(intent.extra_excludes)
            for d in state.documents[: self.rerank_top_n]:
                if any(
                    self._content_contains_exclusion(d.page_content, v)
                    for v in exclude_vals
                ):
                    return True
        if state.iterations > 1:
            return True
        return False

    async def _areason(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        if not self._needs_reasoning(state):
            self._trace(state, "reason", t0, path="skip")
            return state

        top_n = min(self.rerank_top_n, len(state.documents))
        snippets = "\n\n".join(
            f"[{i + 1}] {d.page_content[:400]}"
            for i, d in enumerate(state.documents[:top_n])
        )
        intent_hint = ""
        if state.filter_intent and state.filter_intent.field:
            fi = state.filter_intent
            intent_hint = (
                f'\nDetected filter: {fi.field} {fi.operator} "{fi.value}". '
                "Documents violating this filter should be removed.\n"
            )

        try:
            verdict = cast(
                ReasoningVerdict,
                await self._reasoning_chain.ainvoke(
                    [
                        self._sys(
                            "You are a retrieval result judge. Given a user question and "
                            "the top retrieved documents, decide:\n"
                            "1. Which documents (by 1-based index) are clearly IRRELEVANT "
                            "or VIOLATE the query intent (e.g., the user asked for "
                            "alternatives to product X but the results contain X itself).\n"
                            "   List their indices in `dominated_by`.\n"
                            "2. If ALL documents are poor, suggest a `rewritten_query` "
                            "that would find better results. Otherwise leave it null.\n"
                            "3. Provide brief `reasoning` for your verdict.\n\n"
                            "Be strict: only flag documents that clearly miss the intent. "
                            "If results look reasonable, return an empty `dominated_by`."
                            f"{intent_hint}"
                        ),
                        HumanMessage(
                            f"Question: {state.question}\n\n"
                            f"Retrieved documents:\n{snippets}"
                        ),
                    ]
                ),
            )
        except Exception as e:
            self._trace(state, "reason", t0, path="error", error=type(e).__name__)
            return state

        if verdict.dominated_by:
            drop_set = set(verdict.dominated_by)
            kept = [d for i, d in enumerate(state.documents) if (i + 1) not in drop_set]
            new = state.model_copy(update={"documents": kept})
        else:
            new = state

        if verdict.rewritten_query and not verdict.dominated_by:
            new = new.model_copy(
                update={"query": verdict.rewritten_query, "quality_ok": False}
            )

        self._trace(
            new,
            "reason",
            t0,
            path="fired",
            dropped=len(verdict.dominated_by),
            rewrite=bool(verdict.rewritten_query),
        )
        return new

    async def _aquality_gate(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        if not state.documents or state.iterations >= self.max_iter:
            new = state.model_copy(update={"quality_ok": bool(state.documents)})
            self._trace(
                new, "quality_gate", t0, path="shortcut_empty_or_max", ok=new.quality_ok
            )
            return new

        # Trust reranker: if iter 1 and we have a full top_k window, skip LLM gate.
        if state.iterations <= 1 and len(state.documents) >= self.top_k:
            new = state.model_copy(update={"quality_ok": True})
            self._trace(new, "quality_gate", t0, path="shortcut_topk", ok=True)
            return new

        # Iter 1: strict quality assessment — no shortcut.

        snippets = "\n\n".join(
            f"[{i + 1}] {d.page_content[:300]}"
            for i, d in enumerate(state.documents[:5])
        )
        try:
            assessment = cast(
                QualityAssessment,
                await self._quality_chain.ainvoke(
                    [
                        self._sys(
                            "You are a retrieval quality judge. "
                            "Given a question and the top-5 retrieved document snippets, "
                            "decide if the documents are sufficient to answer the question. "
                            "Return sufficient=true if at least one document directly addresses the question. "
                            "Return sufficient=false if the documents are off-topic, too vague, or clearly wrong. "
                            "For 'alternative to X' or 'not X' queries, documents that are DIFFERENT from X "
                            "are correct — they should be rated sufficient if they are in the same product category."
                        ),
                        HumanMessage(
                            f"Question: {state.question}\n\nRetrieved snippets:\n{snippets}"
                        ),
                    ]
                ),
            )
            ok = assessment.sufficient
            err = None
        except Exception as e:
            ok = True
            err = type(e).__name__

        new = state.model_copy(update={"quality_ok": ok})
        self._trace(
            new, "quality_gate", t0, path="llm", ok=ok, iter=state.iterations, error=err
        )
        return new

    def _route(self, state: RAGState) -> Literal["generate", "rewrite", "give_up"]:
        if not state.documents:
            return "give_up" if state.iterations >= self.max_iter else "rewrite"
        return "generate"

    def _quality_route(
        self, state: RAGState
    ) -> Literal["generate", "rewrite", "give_up"]:
        if state.quality_ok is not False:
            return "generate"
        return "give_up" if state.iterations >= self.max_iter else "rewrite"

    def _reason_route(self, state: RAGState) -> Literal["reason", "quality_gate"]:
        return "reason" if self._needs_reasoning(state) else "quality_gate"

    async def _aread_memory(self, state: RAGState, *, store: Any = None, config: Any = None) -> dict:
        """Inject long-term memories into instructions before retrieval."""
        if store is None:
            return {}
        try:
            user_id = (config or {}).get("configurable", {}).get("user_id", "default")
            memories = await store.asearch(("memories", user_id), query=state.question, limit=5)
            if not memories:
                return {}
            mem_text = "\n".join(f"- {m.value['text']}" for m in memories)
            # Memories are surfaced via trace so callers can inspect them
            return {"trace": state.trace + [{"node": "read_memory", "memories": mem_text}]}
        except Exception:
            return {}

    async def _awrite_memory(self, state: RAGState, *, store: Any = None, config: Any = None) -> dict:
        """Distil and save key facts from this exchange for future conversations."""
        if store is None or not state.answer:
            return {}
        try:
            user_id = (config or {}).get("configurable", {}).get("user_id", "default")
            import uuid, time as _time
            key = str(uuid.uuid4())
            await store.aput(
                ("memories", user_id),
                key,
                {"text": f"Q: {state.question}\nA: {state.answer[:300]}", "ts": _time.time()},
            )
        except Exception:
            pass
        return {}

    def _build_graph(self) -> Any:
        store = self._memory_store
        g = StateGraph(RAGState)
        for name, fn in [
            ("smart_entry", lambda state: state),
            ("parallel_start", self._aparallel_start),
            ("retrieve", self._aretrieve_node),
            ("rerank", self._arerank),
            ("reason", self._areason),
            ("quality_gate", self._aquality_gate),
            ("generate", self._agenerate),
            ("rewrite", self._arewrite),
            ("give_up", self._give_up),
        ]:
            g.add_node(name, fn)

        if store is not None:
            g.add_node("read_memory", self._aread_memory)
            g.add_node("write_memory", self._awrite_memory)

        if store is not None:
            g.add_edge(START, "read_memory")
            g.add_edge("read_memory", "smart_entry")
        else:
            g.add_edge(START, "smart_entry")

        g.add_conditional_edges(
            "smart_entry",
            lambda s: "rerank" if s.documents else "parallel_start",
            {"rerank": "rerank", "parallel_start": "parallel_start"},
        )
        g.add_conditional_edges(
            "parallel_start",
            self._route,
            {"generate": "rerank", "rewrite": "rewrite", "give_up": "give_up"},
        )
        g.add_conditional_edges(
            "retrieve",
            self._route,
            {"generate": "rerank", "rewrite": "rewrite", "give_up": "give_up"},
        )
        g.add_conditional_edges(
            "rerank",
            self._reason_route,
            {"reason": "reason", "quality_gate": "quality_gate"},
        )
        g.add_edge("reason", "quality_gate")
        g.add_conditional_edges(
            "quality_gate",
            self._quality_route,
            {"generate": "generate", "rewrite": "rewrite", "give_up": "give_up"},
        )
        g.add_edge("rewrite", "retrieve")
        if store is not None:
            g.add_edge("generate", "write_memory")
            g.add_edge("write_memory", END)
            g.add_edge("give_up", "write_memory")
        else:
            g.add_edge("generate", END)
            g.add_edge("give_up", END)
        compiled = g.compile(checkpointer=self._checkpointer, store=store if store is not None else None)
        compiled.step_timeout = 120.0 if self._expert_reranker else 30.0
        return compiled

    def _build_agent(self) -> Any:
        system_prompt = (
            "You are a retrieval agent with intelligent filtering. For each user question:\n"
            "\n"
            "1. Call get_index_settings to discover filterable, sortable, and boost fields.\n"
            "   Use ONLY the field names returned — never invent field names.\n"
            "\n"
            "2. DECIDE whether to apply a filter (fast precision boost):\n"
            "   - If the user names a specific entity that maps to a filterable field,\n"
            "     call get_filter_values(field) to see real stored values.\n"
            "     Then build a filter expression using the closest matching value.\n"
            "   - Use CONTAINS for partial name matches.\n"
            "   - Use exact = for booleans, IDs, and known precise values.\n"
            "   - Combine with AND/OR when multiple constraints apply.\n"
            "   - SKIP the filter step for broad or ambiguous queries where no entity is named.\n"
            "\n"
            "3. Call search_hybrid with the query and any filter_expr you decided to apply.\n"
            "   If the question implies popularity/preference, also pass sort_fields from\n"
            "   'boost_fields' (e.g. sort_fields=['field_name:desc']).\n"
            "\n"
            "4. If results are poor (< 3 hits), call search_bm25 as fallback (drop the filter\n"
            "   if the previous filtered search was empty).\n"
            "\n"
            "5. Call rerank_results to get the top semantically-ranked hits.\n"
            "\n"
            "6. Summarise results. When boost/filter fields influenced ranking, explain why.\n"
            f"{self.instructions}"
        )

        agent = create_agent(
            self._llm,
            self._tools,
            system_prompt=system_prompt,
            store=self._memory_store,
            middleware=[  # ty: ignore[invalid-argument-type]
                ToolCallLimitMiddleware(run_limit=10, exit_behavior="end"),  # type: ignore[arg-type]
                ToolRetryMiddleware(
                    max_retries=5,
                    tools=["search_hybrid", "rerank_results", "get_filter_values"],
                    retry_on=(Exception,),
                    backoff_factor=2.0,
                    initial_delay=1.0,
                    max_delay=30.0,
                    jitter=True,
                    on_failure="continue",
                ),
                ModelRetryMiddleware(
                    max_retries=5,
                    backoff_factor=2.0,
                    initial_delay=1.0,
                    max_delay=30.0,
                    on_failure="continue",
                ),
            ],
        )
        agent.step_timeout = 30.0
        return agent

    def _sample_field_values(
        self, fields: list[str], limit: int = 100
    ) -> dict[str, list]:
        """Sample real stored values for filterable fields (like agent's get_filter_values)."""
        cache_key = tuple(fields)
        cached = self._field_values_cache.get(cache_key)
        if cached is not None:
            return cached
        sample_limit = max(limit, 500)
        field_values: dict[str, list] = {}
        for field in fields:
            samples = self.backend.sample_documents(
                limit=sample_limit, attributes_to_retrieve=[field]
            )
            seen: set[str] = set()
            values: list = []
            for doc in samples:
                v = doc.get(field)
                if v is None:
                    continue
                for item in v if isinstance(v, list) else [v]:
                    k = str(item)
                    if k not in seen:
                        seen.add(k)
                        values.append(item)
            if values:
                field_values[field] = values[:limit]
        self._field_values_cache[cache_key] = field_values
        return field_values

    async def _adetect_filter_intent(self, question: str) -> FilterIntent:
        config = self.backend.get_index_config()
        filterable = config.filterable_attributes
        if not filterable:
            return FilterIntent(field=None, value="", operator="")

        # Skip LLM if no structured filter targets exist — id/content/text are
        # not real filter axes, filtering them is a substring match on the body.
        _GENERIC = {"id", "_id", "content", "text", "body", "document", "doc"}
        if all(f.lower() in _GENERIC for f in filterable):
            return FilterIntent(field=None, value="", operator="")

        # Cheap heuristic pre-filter: skip LLM if no entity-like signal.
        # Entity signals: capitalized word (not sentence-start), standalone numeric
        # token (year/version), or uppercase code. Embedded digits (bm25, oauth2)
        # are technical jargon, not entities — don't count them.
        words = question.strip().split()
        has_entity_signal = any(
            (i > 0 and w and w[0].isupper() and not w.isupper())
            or (w and w[0].isdigit())
            or (len(w) >= 3 and w.isupper())
            for i, w in enumerate(words)
        )
        if not has_entity_signal:
            return FilterIntent(field=None, value="", operator="")

        cached = _cache.load("filter-intent-v5", tuple(filterable), question)
        if cached:
            try:
                return FilterIntent.model_validate(cached)
            except Exception:
                pass

        # Sample real stored values so the LLM can match entities precisely
        loop = asyncio.get_running_loop()
        field_values = await loop.run_in_executor(
            None, self._sample_field_values, filterable
        )

        values_block = "\n".join(
            f"  {field}: {vals[:30]}"  # show up to 30 sample values
            for field, vals in field_values.items()
        )

        try:
            result = cast(
                FilterIntent,
                await self._filter_intent_chain.ainvoke(
                    [
                        self._sys(
                            f"You help build filter expressions for a search backend.\n"
                            f"Filterable fields in this index: {filterable}\n\n"
                            f"Sample values per field:\n{values_block}\n\n"
                            "TASK: Detect if the question mentions a SPECIFIC named entity "
                            "(brand, supplier, company, category, boolean flag, etc.) that should "
                            "be used as a filter.\n\n"
                            "RULES:\n"
                            "- If a proper noun / brand name appears that likely maps to a name field "
                            "(e.g. supplier_name, brand_name, company), use CONTAINS with that name — "
                            "even if the exact value is not in the samples above.\n"
                            "- For boolean fields (is_own_brand, active, etc.), use = with true/false.\n"
                            "- For exact IDs or codes, use =.\n"
                            "- For multiple values, use IN.\n"
                            "- EXCLUSION: If the question says 'nicht'/'not'/'aber nicht'/'sans'/'pas de'/"
                            "'exclude'/'except', use NOT_CONTAINS to EXCLUDE that entity.\n"
                            "  For multi-brand exclusion ('nicht X oder Y'), put the first brand in value "
                            "and additional brands in extra_excludes.\n"
                            "  Example: 'Mörtel nicht Weber' → field=supplier_name, value=Weber, operator=NOT_CONTAINS\n"
                            "  Example: 'nicht Sakret oder Fixit' → value=Sakret, extra_excludes=[Fixit]\n"
                            "- 'Alternative zu X' means 'find similar products but not X itself'. "
                            "Use NOT_CONTAINS on the name field to exclude the queried product.\n"
                            "  Example: 'Alternative zu PCI Polyfix' → field=article_name, value=PCI Polyfix, operator=NOT_CONTAINS\n"
                            "- If NO specific entity is named (broad/generic query), return field=null, value='', operator=''.\n"
                            "Operators: CONTAINS (partial match), = (exact), IN (multiple values), "
                            "NOT_CONTAINS (exclude partial match), != (exclude exact)"
                        ),
                        HumanMessage(question),
                    ]
                ),
            )
            result = self._patch_exclusion_intent(question, result, filterable)
            _cache.save(
                "filter-intent-v5",
                tuple(filterable),
                question,
                value=result.model_dump(),
            )
            return result
        except Exception:
            return FilterIntent(field=None, value="", operator="")

    @staticmethod
    def _patch_exclusion_intent(
        question: str, intent: FilterIntent, filterable: list[str]
    ) -> FilterIntent:
        import re

        alt_m = re.search(
            r"(?i)\balternative\s+(?:zu|à|to)\s+(.+?)$",
            question.strip(),
        )
        if alt_m and "article_name" in filterable:
            product = alt_m.group(1).strip()
            if intent.operator == "NOT_CONTAINS" and intent.field == "article_name":
                return intent
            return FilterIntent(
                field="article_name", value=product, operator="NOT_CONTAINS"
            )

        neg_m = re.search(
            r"(?i)\b(?:nicht|not|sans|pas de|except|exclude)\s+(?:von\s+)?(.+?)$",
            question.strip(),
        )
        if neg_m and intent.operator not in ("NOT_CONTAINS", "!="):
            raw_entity = neg_m.group(1).strip()
            # Split on "oder"/"or"/"ou" for multi-brand exclusion
            parts = re.split(r"\s+(?:oder|or|ou)\s+", raw_entity, flags=re.IGNORECASE)
            # Strip trailing clauses like "und keine Eigenmarke"
            cleaned = []
            for p in parts:
                p = re.sub(
                    r"\s+(?:und|and|et)\s+.*", "", p, flags=re.IGNORECASE
                ).strip()
                if p:
                    cleaned.append(p)
            entity = cleaned[0] if cleaned else raw_entity
            extras = cleaned[1:] if len(cleaned) > 1 else []
            if intent.field and intent.field != "supplier_name":
                return intent
            if "supplier_name" in filterable:
                return FilterIntent(
                    field="supplier_name",
                    value=entity,
                    operator="NOT_CONTAINS",
                    extra_excludes=extras,
                )

        return intent

    async def _aselect_collections(self, question: str) -> list[str]:
        """Ask the LLM which collection(s) to route the query to.

        Returns a list of collection names. Falls back to all collections on
        error, empty selection, or when only a single collection is registered.
        """
        if not self.collections:
            return []
        all_names = list(self.collections.keys())
        if len(all_names) == 1:
            return all_names

        descriptions = "\n".join(
            f"- {name}: {self.collection_descriptions.get(name, name)}"
            for name in all_names
        )
        cached = _cache.load("select-collection-v1", tuple(all_names), question)
        if cached:
            try:
                filtered = [n for n in cached if n in self.collections]
                if filtered:
                    return filtered
            except Exception:
                pass
        try:
            result = cast(
                CollectionIntent,
                await self._select_collection_chain.ainvoke(
                    [
                        self._sys(
                            "Pick the collections most likely to contain the answer.\n"
                            f"Available collections:\n{descriptions}\n\n"
                            "Return JSON with a 'collections' list containing ONLY names "
                            "from the list above.\n"
                            "- Prefer the minimum set that covers the question.\n"
                            "- If the question spans multiple topics, include all relevant.\n"
                            "- If unsure, include all candidates (better recall than miss).\n"
                            "- Never invent names not in the list."
                        ),
                        HumanMessage(question),
                    ]
                ),
            )
            filtered = [n for n in result.collections if n in self.collections]
            if filtered:
                _cache.save(
                    "select-collection-v1", tuple(all_names), question, value=filtered
                )
                return filtered
            return all_names
        except Exception:
            return all_names

    def _merge_doc_lists(self, *doc_lists: list[Document]) -> list[Document]:
        raw = [[d.metadata for d in dl] for dl in doc_lists]
        fused_meta = _rrf_fuse(raw)
        id_to_doc: dict[str, Document] = {}
        for dl in doc_lists:
            for d in dl:
                id_to_doc.setdefault(_doc_id(d.metadata), d)
        result = []
        for meta in fused_meta:
            key = _doc_id(meta)
            result.append(
                id_to_doc.get(key)
                or Document(page_content=self._hit_to_text(meta), metadata=meta)
            )
        return result

    async def _acompute_hyde(self, question: str) -> str:
        """Pre-compute HyDE text, to be run concurrently with preprocessing."""
        if self.embed_fn and len(question.split()) >= self.hyde_min_words:
            try:
                return await self._ahypothetical_doc(question)
            except Exception:
                return question
        return question

    async def _arelevance_check(
        self, query: str, docs: list[Document]
    ) -> RelevanceCheck:
        top_ids = tuple(
            d.metadata.get("id") or d.metadata.get("_id") or d.page_content[:40]
            for d in docs[:3]
        )
        cached = _cache.load("relevance-v2", query, top_ids)
        if cached:
            try:
                return RelevanceCheck.model_validate(cached)
            except Exception:
                pass
        snippets = "\n---\n".join(d.page_content[:300] for d in docs[:3])
        try:
            result = cast(
                RelevanceCheck,
                await self._relevance_chain.ainvoke(
                    [
                        self._sys(
                            "Strictly judge if snippets DIRECTLY answer the question. "
                            "Set makes_sense=true and confidence>=0.9 only if a snippet "
                            "clearly contains the answer. Otherwise lower confidence."
                        ),
                        HumanMessage(f"Question: {query}\n\nSnippets:\n{snippets}"),
                    ]
                ),
            )
            _cache.save("relevance-v2", query, top_ids, value=result.model_dump())
            return result
        except Exception:
            return RelevanceCheck(makes_sense=False, confidence=0.0)

    async def _afast_keyword_retrieve(self, query: str, limit: int) -> list[Document]:
        """Pure BM25, no LLM, single request. Returns docs with _rankingScore."""
        loop = asyncio.get_running_loop()
        req = self._make_search_request(
            query,
            limit,
            semantic_ratio=0.0,
            sort_fields=self.sort_fields,
            show_ranking_score=True,
        )
        results = await loop.run_in_executor(None, self.backend.batch_search, [req])
        hits = results[0] if results else []
        return [Document(page_content=self._hit_to_text(h), metadata=h) for h in hits]

    async def _aalternative_retrieve(
        self, query: str, alternative_to: str, top_k: int
    ) -> tuple[str, list[Document]]:
        """Find similar items, excluding the referenced product."""
        limit = int(top_k * self.retrieval_factor)
        ref_docs = await self._afast_keyword_retrieve(alternative_to, 3)
        if not ref_docs:
            ref_docs = await self._afast_keyword_retrieve(query, limit)
            return query, ref_docs[:top_k]

        ref = ref_docs[0]
        ref_text = ref.page_content
        ref_id = _doc_id(ref.metadata)
        # Category-aware rerank signal: prefer metadata category field if present,
        # fall back to page_content. Narrow signal keeps rerank on category, not brand.
        ref_category = ""
        for f in self.category_fields:
            v = ref.metadata.get(f)
            if v:
                ref_category = str(v)
                break
        rerank_signal = (
            f"{ref_category}: {ref.metadata.get(self.name_field) or ''}".strip(": ")
            if ref_category
            else ref_text
        )

        broad_docs = await self._asearch(
            query,
            question=ref_text,
            factor=self.retrieval_factor,
            adaptive_semantic_ratio=0.7,
            adaptive_fusion="dbsf",
        )
        filtered = [
            d
            for d in broad_docs
            if _doc_id(d.metadata) != ref_id
            and not self._content_contains_exclusion(d.page_content, alternative_to)
        ]
        state = RAGState(
            question=rerank_signal,
            query=query,
            documents=filtered,
            iterations=1,
        )
        state = await self._arerank(state)
        return state.query, state.documents[:top_k]

    async def _aretrieve_documents(
        self, query: str, top_k: int | None = None
    ) -> tuple[str, list[Document]]:
        k = top_k or self.top_k
        limit = int(k * self.retrieval_factor)

        # Fast path: cheap keyword search; if top-score confident → rerank + return.
        fast_docs = await self._afast_keyword_retrieve(query, limit)
        top_score = (
            float(fast_docs[0].metadata.get("_rankingScore", 0.0)) if fast_docs else 0.0
        )
        fast_accept = top_score >= 0.85 and len(fast_docs) >= k
        if not fast_accept and fast_docs and len(fast_docs) >= k:
            rc = await self._arelevance_check(query, fast_docs)
            fast_accept = rc.makes_sense and rc.confidence >= 0.9
        if fast_accept:
            state = RAGState(
                question=query, query=query, documents=fast_docs, iterations=1
            )
            state = await self._arerank(state)
            return state.query, state.documents[:k]

        # Slow path: full smart retrieve (preprocess + HyDE + filter-intent concurrent).
        init = RAGState(question=query, query=query)
        (state, hyde_text, (_, filter_docs)) = await asyncio.gather(
            self._apreprocess(init),
            self._acompute_hyde(query),
            self._afilter_search_with_intent(query, query),
        )

        # Alternative path: preprocess detected "find alternatives for X".
        if state.alternative_to:
            return await self._aalternative_retrieve(
                state.query, state.alternative_to, k
            )

        broad_docs = await self._asearch(
            state.query,
            question=state.question,
            extra_bm25=state.query_variants,
            factor=self.retrieval_factor,
            adaptive_semantic_ratio=state.adaptive_semantic_ratio,
            adaptive_fusion=state.adaptive_fusion,
            hyde_text=hyde_text,
        )
        docs = (
            self._merge_doc_lists(filter_docs, broad_docs)
            if filter_docs
            else broad_docs
        )
        # Merge fast-path docs for recall (cheap; dedup handled by _merge_doc_lists)
        if fast_docs:
            docs = self._merge_doc_lists(docs, fast_docs)
        state = state.model_copy(update={"documents": docs, "iterations": 1})
        if not state.documents:
            state = await self._aswarm_retrieve(state)
        state = await self._arerank(state)
        return state.query, state.documents[:k]

    def retrieve_documents(
        self, query: str, top_k: int | None = None
    ) -> tuple[str, list[Document]]:
        return _run_sync(self._aretrieve_documents(query, top_k=top_k))

    async def ainvoke(self, question: str) -> RAGState:
        init = RAGState(question=question, query=question)
        state = await self._aparallel_start(init)
        return cast(RAGState, await self._graph.ainvoke(state))

    async def astream(self, question: str):
        """Stream answer tokens for the final generate step.

        Runs the full retrieval/rerank pipeline, then yields chunks from
        `gen_llm.astream`. Earlier pipeline latency is unchanged; only the
        final LLM call streams for improved time-to-first-token.
        """
        _, docs = await self._aretrieve_documents(question)
        numbered = "\n\n---\n\n".join(
            f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs[: self.top_k])
        )
        messages: list = [
            self._sys(
                "Answer using only the provided context. "
                "Cite sources inline using [n] numbers that match the context blocks. "
                "Say so if the context is insufficient."
            ),
            HumanMessage(f"Context:\n{numbered}\n\nQuestion: {question}"),
        ]
        async for chunk in self._gen_llm.astream(messages):
            content = getattr(chunk, "content", None)
            if content:
                yield str(content)

    def invoke(self, question: str) -> RAGState:
        return _run_sync(self.ainvoke(question))

    async def abatch(self, questions: list[str]) -> list[RAGState]:
        return await asyncio.gather(*(self.ainvoke(q) for q in questions))

    def batch(self, questions: list[str]) -> list[RAGState]:
        return _run_sync(self.abatch(questions))

    async def achat(
        self,
        question: str,
        history: list[ConversationTurn] | None = None,
    ) -> RAGState:
        return cast(
            RAGState,
            await self._graph.ainvoke(
                RAGState(question=question, history=history or [])
            ),
        )

    def chat(
        self,
        question: str,
        history: list[ConversationTurn] | None = None,
    ) -> RAGState:
        return _run_sync(self.achat(question, history))

    async def ainvoke_agent(self, question: str) -> str:
        result = await self._agent.ainvoke({"messages": [HumanMessage(question)]})
        msgs = result.get("messages", [])
        return str(msgs[-1].content) if msgs else ""

    def invoke_agent(self, question: str) -> str:
        return _run_sync(self.ainvoke_agent(question))

    def query(self, question: str) -> str:
        return self.invoke(question).answer or ""
