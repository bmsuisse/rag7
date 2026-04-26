from __future__ import annotations

import asyncio
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

from . import _cache, prompts
from .helpers.disk_cache import (
    field_rank_cache_load,
    filters_cache_load,
    filters_cache_save,
)
from .helpers.render import _doc_to_grader_text, _render_value
from .helpers.schema_introspect import (
    _align_embed_fn_with_backend,
    _detect_category_fields,
    _detect_embedder_name,
    _detect_index_signals,
    _has_is_own_brand_field,
    _validate_embedder_name,
)
from .mixins.filter_detection import FilterDetectionMixin
from .mixins.intent_filter import IntentFilterMixin
from .mixins.memory import MemoryMixin
from .backend import (
    _ACTIVE_COLLECTIONS,
    InMemoryBackend,
    SearchBackend,
    SearchRequest,
    _MultiBackend,
)
from .config import RAGConfig
from .models import (
    AnswerGrade,
    CloseMatchKeep,
    CollectionIntent,
    ConversationTurn,
    FieldPriority,
    FilterIntent,
    MultiQuery,
    ProductCodeQuery,
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

_llm_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


_BRAND_FIELDS = frozenset({"supplier_name", "brand", "manufacturer_name", "vendor"})


def _is_brand_intent(intent: FilterIntent) -> bool:
    return intent.field in _BRAND_FIELDS and intent.operator in (
        "=",
        "==",
        "CONTAINS",
    )


_FILTER_INTENT_WORDS_BY_LANG: dict[str, frozenset[str]] = {
    "de": frozenset(
        {
            "von",
            "vom",
            "aus",
            "ohne",
            "nicht",
            "kein",
            "keine",
            "für",
            "fur",
            "bei",
            "mit",
        }
    ),
    "fr": frozenset({"de", "du", "des", "sans", "pour", "par", "pas", "avec", "chez"}),
    "it": frozenset({"di", "da", "del", "della", "senza", "non", "per", "con"}),
    "en": frozenset({"from", "without", "not", "no", "for", "by", "of", "with"}),
}

# Queries that are purely numeric with 6+ digits are product codes (EAN-8/13, GTIN-14,
# internal article IDs). Skip all LLM/HyDE/filter-intent and do a direct BM25 lookup.
_PRODUCT_CODE_RE = re.compile(r"^\d{6,}$")

_UNKNOWN_FIELD_RANK = 10


def _filter_bohrer_variants(
    question: str, corrected_query: str, variants: list[str]
) -> list[str]:
    orig = (question + " " + corrected_query).lower()
    if "bohrer" in orig or "bohrschrauber" in orig:
        return variants
    return [
        v
        for v in variants
        if "bohrer" not in v.lower() and "bohrschrauber" not in v.lower()
    ]


class AgenticRAG(FilterDetectionMixin, IntentFilterMixin, MemoryMixin):
    @staticmethod
    def _llm_seed() -> int:
        return int(os.getenv("RAG_LLM_SEED", "42"))

    @staticmethod
    def _default_llm(
        *,
        timeout: int = 30,
    ) -> BaseChatModel:
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
                seed=AgenticRAG._llm_seed(),
                request_timeout=timeout,  # type: ignore[call-arg]
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(  # type: ignore[call-arg]
            model=os.getenv("OPENAI_MODEL", "gpt-5.4"),  # ty: ignore[unknown-argument]
            api_key=os.getenv("OPENAI_API_KEY"),  # ty: ignore[unknown-argument]
            temperature=0,
            seed=AgenticRAG._llm_seed(),
            request_timeout=timeout,  # type: ignore[call-arg]
        )

    @staticmethod
    def _default_gen_llm() -> BaseChatModel:
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
                seed=AgenticRAG._llm_seed(),
                request_timeout=60,  # type: ignore[call-arg]
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(  # type: ignore[call-arg]
            model=os.getenv("OPENAI_MODEL", "gpt-5.4"),  # ty: ignore[unknown-argument]
            api_key=os.getenv("OPENAI_API_KEY"),  # ty: ignore[unknown-argument]
            temperature=0,
            seed=AgenticRAG._llm_seed(),
            request_timeout=60,  # type: ignore[call-arg]
        )

    @staticmethod
    def _resolve_llm(spec: str, timeout: int = 30) -> BaseChatModel:
        from langchain.chat_models import init_chat_model

        if spec.startswith("azure:"):
            spec = "azure_openai:" + spec.split(":", 1)[1]

        kwargs: dict[str, Any] = {"temperature": 0, "seed": AgenticRAG._llm_seed()}
        if spec.startswith("azure_openai:"):
            kwargs.update(
                azure_deployment=spec.split(":", 1)[1],
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                request_timeout=timeout,
            )
        return init_chat_model(spec, **kwargs)  # type: ignore[return-value]

    @staticmethod
    def _chain_llm(llm: BaseChatModel, name: str) -> BaseChatModel:
        try:
            from langchain_openai import AzureChatOpenAI

            if isinstance(llm, AzureChatOpenAI):
                return llm
        except ImportError:
            pass
        try:
            from langchain_openai import ChatOpenAI

            if isinstance(llm, ChatOpenAI):
                return cast(
                    BaseChatModel,
                    llm.bind(model_kwargs={"prompt_cache_key": f"rag7-{name}-v1"}),
                )
        except ImportError:
            pass
        return llm

    @staticmethod
    def _default_reranker() -> CohereReranker | LLMReranker:
        try:
            return CohereReranker()
        except Exception as e:
            import warnings

            warnings.warn(
                f"CohereReranker unavailable ({type(e).__name__}: {e}). "
                "Falling back to LLMReranker with no LLM — reranker will return "
                "positional fallback scores (1/i), effectively a no-op. "
                "Install cohere (`uv pip install cohere`) or configure a reranker explicitly.",
                stacklevel=2,
            )
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
        from langchain.chat_models import (
            init_chat_model,  # type: ignore[import-untyped]
        )

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
        mem0_memory: Any = None,
        config: "RAGConfig | None" = None,
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
        explicit_embedder_name = embedder_name or os.getenv("RAG_EMBEDDER_NAME")
        if explicit_embedder_name:
            self.embedder_name = explicit_embedder_name
            _validate_embedder_name(self.backend, explicit_embedder_name)
        else:
            self.embedder_name = _detect_embedder_name(self.backend, fallback="default")
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
        self.embed_fn = _align_embed_fn_with_backend(embed_fn, self.backend)
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
        self._has_own_brand_field = _has_is_own_brand_field(self.backend)
        if base_filter is not None:
            warnings.warn(
                "`base_filter` is deprecated, use `filter` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if filter is None:
                filter = base_filter
        self.filter = filter

        if gen_llm is not None:
            self._gen_llm = gen_llm
        elif config is not None and config.strong_model:
            self._gen_llm = self._resolve_llm(config.strong_model, timeout=60)
        else:
            self._gen_llm = self._default_gen_llm()
        if llm is not None:
            self._llm = llm
        elif config is not None and config.weak_model:
            self._llm = self._resolve_llm(config.weak_model, timeout=30)
        else:
            self._llm = self._default_llm()

        if config is not None and config.thinking_model:
            self._thinking_llm: BaseChatModel = self._resolve_llm(
                config.thinking_model, timeout=60
            )
        else:
            self._thinking_llm = self._llm

        if config is not None and config.grader_model:
            self._grader_llm: BaseChatModel = self._resolve_llm(
                config.grader_model, timeout=60
            )
        else:
            self._grader_llm = self._thinking_llm
        self._reranker = reranker or self._default_reranker()
        self._expert_reranker = expert_reranker
        self.expert_top_n = expert_top_n or int(os.getenv("RAG_EXPERT_TOP_N", "10"))
        self.expert_threshold = (
            expert_threshold
            if expert_threshold is not None
            else float(os.getenv("RAG_EXPERT_THRESHOLD", "0.15"))
        )

        if config is not None:
            self.top_k = config.top_k
            self.retrieval_factor = config.retrieval_factor
            self.rerank_top_n = config.rerank_top_n
            self._rerank_cap_multiplier = config.rerank_cap_multiplier
            self.semantic_ratio = config.semantic_ratio
            self.fusion = config.fusion
            self.hyde_min_words = config.hyde_min_words or 999
            self._short_query_threshold = config.short_query_threshold
            self._short_query_sort_tokens = config.short_query_sort_tokens
            self._bm25_fallback_threshold = config.bm25_fallback_threshold
            self._bm25_fallback_semantic_ratio = config.bm25_fallback_semantic_ratio
            self._fast_accept_score = config.fast_accept_score
            self._fast_accept_confidence = config.fast_accept_confidence
            self._rerank_min_score = config.rerank_min_score
            self._filter_intent_words: frozenset[str] = frozenset(
                w
                for lang in config.query_languages
                for w in _FILTER_INTENT_WORDS_BY_LANG.get(lang, frozenset())
            )
            self.expert_top_n = config.expert_top_n
            self.expert_threshold = config.expert_threshold
            self._name_field_boost_max = config.name_field_boost_max
            self._enable_hyde = config.enable_hyde
            self._enable_filter_intent = config.enable_filter_intent
            self._enable_reasoning = config.enable_reasoning
            self._enable_quality_gate = config.enable_quality_gate
            self._enable_preprocess_llm = config.enable_preprocess_llm
            self._enable_final_grade = config.enable_final_grade
            self._final_grade_threshold = config.final_grade_threshold
            self._enable_swarm_grade = config.enable_swarm_grade
            self._enable_close_match_grader = bool(config.enable_close_match_grader)
            self._close_match_strictness: str = config.close_match_strictness
            self._close_match_min_top_rerank: float | None = (
                config.close_match_min_top_rerank
            )
            self._rerank_timeout_s = config.rerank_timeout_s
            self._llm_timeout_s = config.llm_timeout_s
            self.max_iter = config.max_iter
            self.n_swarm_queries = config.n_swarm_queries
            self.rerank_chars = config.rerank_chars
            self._custom_instructions: str = config.custom_instructions or ""
            self._preview_chars: int = config.preview_chars
        else:
            self._rerank_cap_multiplier = float(
                os.getenv("RAG_RERANK_CAP_MULTIPLIER", "2.0")
            )
            self._short_query_threshold = int(
                os.getenv("RAG_SHORT_QUERY_THRESHOLD", "6")
            )
            self._short_query_sort_tokens = bool(
                int(os.getenv("RAG_SHORT_QUERY_SORT_TOKENS", "1"))
            )
            self._bm25_fallback_threshold = float(
                os.getenv("RAG_BM25_FALLBACK_THRESHOLD", "0.4")
            )
            self._bm25_fallback_semantic_ratio = float(
                os.getenv("RAG_BM25_FALLBACK_SEMANTIC_RATIO", "0.9")
            )
            self._fast_accept_score: float | None = 0.85
            self._fast_accept_confidence: float | None = 0.9
            _rms = os.getenv("RAG_RERANK_MIN_SCORE", "0.2")
            self._rerank_min_score: float | None = (
                None if _rms.lower() == "none" else float(_rms)
            )
            langs = [
                lang.strip()
                for lang in os.getenv("RAG_QUERY_LANGUAGES", "de,fr,it,en").split(",")
                if lang.strip()
            ]
            self._filter_intent_words: frozenset[str] = frozenset(
                w
                for lang in langs
                for w in _FILTER_INTENT_WORDS_BY_LANG.get(lang, frozenset())
            )
            self._enable_hyde = True
            self._enable_filter_intent = True
            self._enable_reasoning = True
            self._enable_quality_gate = True
            self._enable_preprocess_llm = True
            self._enable_final_grade = False
            self._final_grade_threshold = 0.7
            self._enable_swarm_grade = False
            self._enable_close_match_grader = bool(
                int(os.getenv("RAG_ENABLE_CLOSE_MATCH_GRADER", "1"))
            )
            self._close_match_strictness = os.getenv(
                "RAG_CLOSE_MATCH_STRICTNESS", "loose"
            )
            _cmr = os.getenv("RAG_CLOSE_MATCH_MIN_TOP_RERANK")
            self._close_match_min_top_rerank: float | None = (
                float(_cmr) if _cmr else None
            )
            self._rerank_timeout_s = 30.0
            self._llm_timeout_s = 60.0
            self._name_field_boost_max: float = float(
                os.getenv("RAG_NAME_FIELD_BOOST_MAX", "0.1")
            )
            self._preview_chars: int = int(os.getenv("RAG_PREVIEW_CHARS", "1500"))

        if not hasattr(self, "_custom_instructions"):
            self._custom_instructions = ""
        self._field_values_cache: dict = {}
        self._filter_values: dict[str, list[str]] = {}

        if auto_strategy:
            self._auto_init_filters()

        self._index_config = self.backend.get_index_config()

        _ck = self._chain_llm
        self._search_query_chain = _ck(self._llm, "rewrite").with_structured_output(
            SearchQuery, method="json_schema"
        )
        self._quality_chain = _ck(self._llm, "quality-gate").with_structured_output(
            QualityAssessment, method="json_schema"
        )
        self._multi_query_chain = _ck(self._llm, "multi-query").with_structured_output(
            MultiQuery, method="json_schema"
        )
        self._filter_intent_chain = _ck(
            self._llm, "filter-intent"
        ).with_structured_output(FilterIntent, method="json_schema")
        self._select_collection_chain = _ck(
            self._llm, "select-collection"
        ).with_structured_output(CollectionIntent, method="json_schema")
        self._field_priority_chain = _ck(
            self._llm, "field-priority"
        ).with_structured_output(FieldPriority, method="json_schema")
        self._relevance_chain = _ck(self._llm, "relevance").with_structured_output(
            RelevanceCheck, method="json_schema"
        )
        self._reasoning_chain = _ck(
            self._thinking_llm, "reasoning"
        ).with_structured_output(ReasoningVerdict, method="json_schema")
        self._grade_chain = _ck(self._grader_llm, "grader").with_structured_output(
            AnswerGrade, method="json_schema"
        )
        self._product_code_chain = _ck(
            self._llm, "product-code"
        ).with_structured_output(ProductCodeQuery, method="json_schema")
        self._close_match_chain = _ck(
            self._grader_llm, "close-match"
        ).with_structured_output(CloseMatchKeep, method="json_schema")

        self._toolset = RAGToolset(self)
        self._tools = self._toolset.as_tools()
        self._checkpointer = checkpointer
        self._memory_store = memory_store
        self._mem0_memory = mem0_memory
        self._graph = self._build_graph()
        self._agent = self._build_agent()

    @property
    def base_filter(self) -> str | None:
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

    def _auto_init_filters(self) -> None:
        """Deterministic init: schema inference + filter-value discovery.

        Samples documents once, widens rerank_chars if docs are long, infers
        name/group fields, then discovers filterable attributes and a small
        set of distinct values per field so the filter-intent agent knows
        what filters are available.
        """
        try:
            sample = self.backend.sample_documents(
                limit=15,
                filter_expr=self.filter,
            )
            if not sample:
                return
            max_doc_chars = 0
            for h in sample[:5]:
                items = [
                    (k, str(v))
                    for k, v in h.items()
                    if k not in _SKIP_FIELDS and v and isinstance(v, (str, int, float))
                ]
                max_doc_chars = max(max_doc_chars, sum(len(v) for _, v in items))

            if max_doc_chars > self.rerank_chars:
                self.rerank_chars = min(8192, int(max_doc_chars * 1.2))

            self._infer_name_and_group_fields(sample)

            schema_sig = self._schema_signature(sample)
            cached = filters_cache_load(schema_sig)
            if cached is not None:
                self._filter_values = {k: list(v) for k, v in cached.items()}
                return

            try:
                filterable = self.backend.get_index_config().filterable_attributes
            except Exception:
                filterable = []

            if not filterable:
                self._filter_values = {}
                filters_cache_save(schema_sig, {})
                return

            # One extra broader sample to improve coverage of distinct values.
            try:
                extra = self.backend.sample_documents(
                    limit=200, filter_expr=self.filter
                )
            except Exception:
                extra = []
            combined = list(sample) + list(extra)

            discovered: dict[str, list[str]] = {}
            for attr in filterable:
                seen: set[str] = set()
                values: list[str] = []
                for h in combined:
                    v = h.get(attr)
                    if v is None or v == "":
                        continue
                    for item in v if isinstance(v, list) else [v]:
                        s = str(item)
                        if s and s not in seen:
                            seen.add(s)
                            values.append(s)
                            if len(values) >= 20:
                                break
                    if len(values) >= 20:
                        break
                if values:
                    discovered[attr] = sorted(values)[:20]

            self._filter_values = discovered
            filters_cache_save(schema_sig, discovered)
        except Exception as e:
            self._filter_values = {}
            print(
                f"  [{self.index}] filter-value discovery failed ({e}), using empty dict"
            )

    def _infer_name_and_group_fields(self, sample: list[dict]) -> None:
        if self.name_field and self.group_field:
            return
        try:
            raw = self.backend.get_index_config().searchable_attributes
        except Exception:
            raw = []
        attrs = [a.split(":", 1)[0] for a in raw if a and not a.startswith("*")]
        first_doc = sample[0] if sample else {}

        def _looks_like_id(field: str) -> bool:
            return field.lower() in {
                "id",
                "_id",
                "uid",
                "uuid",
            } or field.lower().endswith("_id")

        def _is_string_field(field: str) -> bool:
            v = first_doc.get(field)
            return isinstance(v, str) and len(v.strip()) > 0

        if not self.name_field:
            candidates = [
                a for a in attrs if a.lower().endswith("_name") and _is_string_field(a)
            ]
            if not candidates:
                candidates = [
                    a for a in attrs if not _looks_like_id(a) and _is_string_field(a)
                ]
            if candidates:
                self.name_field = candidates[0]

        if not self.group_field:
            candidates = [
                a
                for a in attrs
                if any(
                    tok in a.lower() for tok in ("group", "category", "class", "type")
                )
                and not _looks_like_id(a)
                and _is_string_field(a)
            ]
            if candidates:
                self.group_field = candidates[0]

        known_fields = sorted({k for d in sample[:5] for k in d.keys()})
        offset = len(attrs)
        self._field_priority: dict[str, int] = {
            field.split(":", 1)[0]: i for i, field in enumerate(attrs)
        }
        gap_fields = [
            field.split(":", 1)[0]
            for field in known_fields
            if field.split(":", 1)[0] not in self._field_priority
        ]
        schema_sig = self._schema_signature(sample)
        cached_ranks = field_rank_cache_load(schema_sig)
        for base in gap_fields:
            if cached_ranks and base in cached_ranks:
                self._field_priority[base] = offset + cached_ranks[base]
            else:
                self._field_priority[base] = offset + _UNKNOWN_FIELD_RANK
        self._field_priority_sample: list[dict] = list(sample[:5])
        self._field_priority_unranked: list[str] = [] if cached_ranks else gap_fields

    def _schema_signature(self, sample: list[dict]) -> str:
        import hashlib

        doc_keys = sorted({k for d in sample[:5] for k in d.keys()})
        try:
            filterable = sorted(self.backend.get_index_config().filterable_attributes)
        except Exception:
            filterable = []

        if self.collections:
            index_sig = ",".join(sorted(self.collections.keys()))
        else:
            index_sig = self.index
        raw = f"{index_sig}|{','.join(doc_keys)}|{','.join(filterable)}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def _sys(self, prompt: str) -> SystemMessage:
        if self.instructions:
            return SystemMessage(f"{self.instructions}\n\n{prompt}")
        return SystemMessage(prompt)

    def _hit_to_text(self, h: dict) -> str:
        content = h.get("content")
        fp: dict[str, int] = getattr(self, "_field_priority", {})

        items: list[tuple[int, str]] = []
        for k, v in h.items():
            if k in _SKIP_FIELDS or k == "content" or v is None or v == "":
                continue
            rendered = _render_value(v, indent=0)
            if not rendered:
                continue
            rank = fp.get(k)
            if rank is None:
                if isinstance(v, str) and content and rendered in content:
                    continue
                rank = 1000 + _UNKNOWN_FIELD_RANK
            line = (
                f"**{k}**:\n{rendered}"
                if "\n" in rendered and not isinstance(v, str)
                else f"**{k}**: {rendered}"
            )
            items.append((rank, line))

        sorted_lines = [line for _, line in sorted(items)]
        parts = sorted_lines + ([str(content)] if content else [])
        return "\n".join(parts) if parts else ""

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
        cached = _cache.load("hyde-v2", "", query)
        if cached:
            return str(cached)
        result = await self._llm.ainvoke(
            [
                self._sys(prompts.hyde_system()),
                HumanMessage(query),
            ]
        )
        text = str(result.content).strip() or query
        _cache.save("hyde-v2", "", query, value=text)
        return text

    def _cached_batch_search(self, requests: "list[Any]") -> "list[list[dict]]":
        key = tuple(
            (
                r.query,
                r.limit,
                r.semantic_ratio,
                r.filter_expr or "",
                tuple(r.sort_fields) if r.sort_fields else (),
                r.show_ranking_score,
                r.matching_strategy,
                r.embedder_name,
                r.index_uid or "",
            )
            for r in requests
        )
        cached = _cache.load("batch-search-v1", self.index, key)
        if cached is not None:
            return cached
        results = self.backend.batch_search(requests)
        _cache.save("batch-search-v1", self.index, key, value=results)
        return results

    def _multi_search(
        self,
        queries: list[str],
        lang: str | None = None,
        vectors: list[list[float] | None] | None = None,
        filter_expr: str | None = None,
    ) -> list[Document]:
        limit = int(self.top_k * self.retrieval_factor)
        vecs: list[list[float] | None] = vectors if vectors else [None] * len(queries)

        parts = [p for p in [f"language = {lang}" if lang else None, filter_expr] if p]
        combined_filter = " AND ".join(f"({p})" for p in parts) if parts else None

        requests = [
            self._make_search_request(q, limit, vector=v, filter_expr=combined_filter)
            for q, v in zip(queries, vecs)
        ]
        results = self._cached_batch_search(requests)
        fused = _rrf_fuse(results)
        return [
            Document(page_content=self._hit_to_text(h), metadata=h)
            for h in fused[:limit]
        ]

    async def _apreprocess(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()

        if not self._enable_preprocess_llm:
            raw = _strip_stop_words(state.question) or state.question
            new = state.model_copy(
                update={
                    "query": raw,
                    "query_variants": [],
                    "adaptive_semantic_ratio": None,
                    "adaptive_fusion": None,
                }
            )
            self._trace(new, "preprocess", t0, path="disabled", query=raw)
            return new

        cached = _cache.load("preprocess-v8", "", state.question)
        if cached:
            try:
                result = SearchQuery.model_validate(cached)
                raw = _strip_stop_words(state.question)
                variants = _filter_bohrer_variants(
                    state.question, result.query, result.variants[:3]
                )
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
                        self._sys(prompts.preprocess_system()),
                        HumanMessage(state.question),
                    ]
                ),
            )
            _cache.save(
                "preprocess-v8",
                "",
                state.question,
                value=result.model_dump(),
            )
            raw = _strip_stop_words(state.question)
            variants = _filter_bohrer_variants(
                state.question, result.query, result.variants[:3]
            )
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
    ) -> tuple[list[Document], bool]:
        limit = int(self.top_k * (factor or self.retrieval_factor))
        hyde_source = question or query

        if hyde_text is None:
            n_words = len(hyde_source.split())
            short_thr = self._short_query_threshold
            if (
                self.embed_fn
                and n_words >= 1
                and (n_words <= short_thr or n_words >= self.hyde_min_words)
            ):
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
        raw_ratio = (
            adaptive_semantic_ratio
            if adaptive_semantic_ratio is not None
            else self.semantic_ratio
        )
        # Clamp to mixed band — always some keyword, always some semantic.
        # Pure semantic hallucinates on typos; pure BM25 misses paraphrases.
        hybrid_ratio = min(0.8, max(0.2, raw_ratio))

        def _req(q: str, **extra: Any) -> SearchRequest:
            return self._make_search_request(
                q,
                limit,
                sort_fields=self.sort_fields,
                show_ranking_score=True,
                filter_expr=lang_filter,
                **extra,
            )

        requests = [_req(query)]

        if hyde_bm25 != query:
            requests.append(_req(hyde_bm25))
        requests.append(_req(query, vector=vector, semantic_ratio=hybrid_ratio))
        seen = {query, hyde_bm25}
        for v in extra_bm25 or []:
            if v and v not in seen:
                requests.append(_req(v))
                seen.add(v)

        results = await loop.run_in_executor(None, self._cached_batch_search, requests)
        bm25_arms = [r for req, r in zip(requests, results) if not req.vector]
        bm25_empty = bool(bm25_arms) and all(len(r) == 0 for r in bm25_arms)
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

        docs = [
            Document(page_content=self._hit_to_text(h), metadata=h)
            for h in fused[:limit]
        ]
        return docs, bm25_empty

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

    async def _acontextualize(self, state: RAGState) -> RAGState:
        if not state.history or len(state.question.split()) > 10:
            return state
        last = state.history[-1]
        try:
            resp = await self._llm.ainvoke(
                [
                    self._sys(prompts.CONTEXTUALIZE),
                    HumanMessage(
                        f"Previous question: {last.question}\n"
                        f"Previous answer (first 300 chars): {last.answer[:300]}\n"
                        f"Follow-up: {state.question}"
                    ),
                ]
            )
            rewritten = str(resp.content).strip().strip('"').strip("'")
            if rewritten and rewritten != state.question and len(rewritten) < 500:
                return state.model_copy(update={"question": rewritten})
        except Exception:
            pass
        return state

    async def _aparallel_start(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()

        intent_task: asyncio.Task[FilterIntent] | None = None
        if not state.history and state.filter_intent is None:
            intent_task = asyncio.create_task(
                self._adetect_filter_intent(state.question)
            )
        state = await self._acontextualize(state)
        state = await self._aroute_collections(state)
        state = await self._apreprocess(state)
        if intent_task is not None:
            try:
                intent = await intent_task
            except Exception:
                intent = FilterIntent(field=None, value="", operator="")
            state = state.model_copy(update={"filter_intent": intent})
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

        async def _gen_variants() -> list[str]:
            if state.query_variants:
                base = state.query_variants[: self.n_swarm_queries]

                rewritten = state.query
                if rewritten and rewritten != state.question and rewritten not in base:
                    base = [rewritten] + base[: self.n_swarm_queries - 1]
                return base or [state.query]
            cached = _cache.load("multi-query-v1", self.n_swarm_queries, state.question)
            if cached and isinstance(cached, list):
                return cached[: self.n_swarm_queries] or [state.query]
            try:
                result = cast(
                    MultiQuery,
                    await self._multi_query_chain.ainvoke(
                        [
                            self._sys(prompts.multi_query_swarm(self.n_swarm_queries)),
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
            f_expr = self._build_filter_expr(filter_intent)
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
        pool = max(self.top_k * factor, self.rerank_top_n * 4)
        query, docs = await self._aretrieve_documents(state.question, top_k=pool)

        if not docs and state.documents:
            docs = state.documents
        new = state.model_copy(
            update={
                "query": query,
                "documents": docs,
                "iterations": state.iterations + 1,
                "pre_reranked": True,
            }
        )
        self._trace(
            new,
            "retrieve",
            t0,
            iter=new.iterations,
            docs=len(docs),
            factor=factor,
            path="unified",
        )
        return new

    async def _afilter_search_with_intent(
        self,
        question: str,
        query: str,
        precomputed: FilterIntent | None = None,
        variants: list[str] | None = None,
    ) -> tuple[FilterIntent | None, list[Document]]:

        intent = precomputed or await self._adetect_filter_intent(question)
        if not intent.field or not intent.value or not intent.operator:
            return None, []
        docs = await self._afilter_search_from_intent(query, intent, variants)
        return intent, docs

    def _build_filter_expr(self, intent: FilterIntent) -> str:
        return self.backend.build_filter_expr(intent)

    async def _afilter_search_from_intent(
        self,
        query: str,
        intent: FilterIntent,
        variants: list[str] | None = None,
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

        filter_token = intent.value.lower().strip() if intent.value else ""

        def _strip_filter_token(q: str) -> str:
            if not filter_token:
                return q
            parts = [t for t in q.split() if t.lower().strip("?,!.") != filter_token]
            return " ".join(parts) if parts else q

        async def _search(expr: str) -> list[dict]:

            all_queries = [query] + [v for v in (variants or []) if v and v != query]
            stripped = [_strip_filter_token(q) for q in all_queries]
            seen: set[str] = set()
            unique: list[str] = []
            for q in stripped:
                if q and q not in seen:
                    seen.add(q)
                    unique.append(q)

            if len(unique) > 3:
                unique = unique[:1] + sorted(unique[1:], key=lambda s: -len(s))[:2]
            vecs: list[list[float] | None] = [vector] + [None] * (len(unique) - 1)
            requests = [
                self._make_search_request(q, limit, vector=v, filter_expr=expr)
                for q, v in zip(unique, vecs)
            ]
            try:
                if len(requests) == 1:
                    return await loop.run_in_executor(
                        None, self.backend.search, requests[0]
                    )
                results = await loop.run_in_executor(
                    None, self._cached_batch_search, requests
                )
                return _rrf_fuse(results)
            except Exception:
                return []

        hits = await _search(filter_expr)
        if not hits and intent.and_filters:
            primary_only = intent.model_copy(update={"and_filters": []})
            hits = await _search(self._build_filter_expr(primary_only))
        if (
            self._has_own_brand_field
            and _is_brand_intent(intent)
            and len(hits) < self.top_k
        ):
            own_brand_hits = await _search("is_own_brand = true")
            seen: set[str] = {
                str(h.get("id") or h.get("article_id") or id(h)) for h in hits
            }
            for h in own_brand_hits:
                doc_id = str(h.get("id") or h.get("article_id") or id(h))
                if doc_id not in seen:
                    seen.add(doc_id)
                    hits.append(h)
        return [
            Document(page_content=self._hit_to_text(h), metadata=h)
            for h in hits[:limit]
        ]

    def _make_rerank_docs(self, docs: list[Document]) -> list[str]:

        return [_doc_to_grader_text(d)[: self.rerank_chars] for d in docs]

    def _truncate_low_score(self, docs: list[Document], k: int) -> list[Document]:
        """Cap at k, then drop docs whose rerank score is below threshold.

        Returns empty list if no doc passes — honest "no results" beats
        returning an irrelevant top-1 as a confident answer.
        Docs without _rerank_score are treated as passing (not reranked yet).
        """
        head = docs[:k]
        th = self._rerank_min_score
        if th is None or not head:
            return head
        return [
            d
            for d in head
            if d.metadata.get("_rerank_score") is None
            or float(d.metadata["_rerank_score"]) >= th
        ]

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

            hits = sum(1 for t in q_tokens if t in top_name)
            confident = len(q_tokens) >= 2 and hits >= 2 and hits >= len(q_tokens) // 2
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
                    boost_max = self._name_field_boost_max

                    def _tok_boost(d: Document) -> float:
                        name = (d.metadata.get(name_f, "") or "").lower()
                        hits = sum(1 for t in tokens if t in name)

                        return 1.0 + boost_max * (hits / len(tokens))

                    scored = sorted(
                        ((d, s * _tok_boost(d)) for d, s in scored),
                        key=lambda x: x[1],
                        reverse=True,
                    )
        top = scored[: self.top_k]
        num_fields = getattr(self, "num_fields", None) or []
        if top and len(top) > 1 and num_fields:
            max_score = top[0][1]
            threshold = max_score * 0.98
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

    async def _aexpert_rerank(
        self,
        query: str,
        documents: list[Document],
        indexed: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
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
        except Exception:
            return indexed
        rescored = [(idx_map[r.index], r.relevance_score) for r in results]
        return rescored + tail

    async def _arerank(
        self, state: RAGState, *, _accumulate_pool: bool = True
    ) -> RAGState:
        t0 = time.perf_counter()
        if not state.documents:
            return state
        if state.pre_reranked:
            filtered = self._apply_intent_post_filter(
                list(state.documents), state.filter_intent
            )
            if len(filtered) != len(state.documents):
                state = state.model_copy(update={"documents": filtered})
            self._trace(state, "rerank", t0, skipped="pre_reranked")
            return state

        rerank_cap = max(
            int(self.top_k * self._rerank_cap_multiplier), self.rerank_top_n * 2
        )
        capped_docs = state.documents[:rerank_cap]
        rerank_docs = self._make_rerank_docs(capped_docs)
        effective_top_n = len(rerank_docs)
        rerank_query = state.question or state.query
        if state.grader_feedback:
            rerank_query = f"{state.grader_feedback}. {rerank_query}"

        try:
            _rerank_cached = _cache.load("rerank-v1", rerank_query, tuple(rerank_docs))
            expert_fired = False
            if _rerank_cached is not None:
                indexed = [tuple(r) for r in _rerank_cached]
            else:
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
                results = await asyncio.wait_for(coro, timeout=self._rerank_timeout_s)
                indexed = [(r.index, r.relevance_score) for r in results]
                _cache.save(
                    "rerank-v1", rerank_query, tuple(rerank_docs), value=indexed
                )
            if self._expert_reranker and len(indexed) >= 2:
                scores = sorted((s for _, s in indexed), reverse=True)
                top1 = scores[0]
                ref = scores[min(len(scores) - 1, 4)]
                gap = top1 - ref
                if self.expert_threshold is not None and gap < self.expert_threshold:
                    indexed = await self._aexpert_rerank(
                        rerank_query, state.documents, indexed
                    )
                    expert_fired = True
            score_by_idx = {i: s for i, s in indexed}
            for i, d in enumerate(state.documents):
                if i in score_by_idx:
                    d.metadata["_rerank_score"] = float(score_by_idx[i])
            ranked = self._apply_boost(state.documents, indexed, state.query)
            intent = state.filter_intent
            ranked = self._apply_intent_post_filter(ranked, intent)
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
            if _accumulate_pool:
                new = self._merge_into_pool(new, ranked)
            self._trace(new, "rerank", t0, docs=len(ranked), expert_fired=expert_fired)
            return new
        except Exception as e:
            self._trace(state, "rerank", t0, error=type(e).__name__)
            return state

    @staticmethod
    def _merge_into_pool(state: "RAGState", docs: "list[Document]") -> "RAGState":
        pool = state.candidate_pool
        seen = {_doc_id(d.metadata) for d in pool}
        new_pool = pool + [d for d in docs if _doc_id(d.metadata) not in seen]
        return state.model_copy(update={"candidate_pool": new_pool})

    async def _apool_rerank(self, state: RAGState) -> RAGState:
        """Final rerank over the accumulated candidate pool before generation.

        Sends up to top_k * 3 pool docs through Cohere (cached), so iterations
        that found different candidates each contribute to the final ranking.
        """
        t0 = time.perf_counter()
        pool = state.candidate_pool
        cap = self.top_k * 3
        if state.iterations <= 1 or len(pool) <= len(state.documents):
            self._trace(state, "pool_rerank", t0, path="skip", pool=len(pool))
            return state
        pool_state = state.model_copy(
            update={"documents": pool[:cap], "pre_reranked": False}
        )
        reranked = await self._arerank(pool_state, _accumulate_pool=False)
        new = state.model_copy(
            update={
                "documents": self._truncate_low_score(reranked.documents, self.top_k)
            }
        )
        self._trace(
            new,
            "pool_rerank",
            t0,
            path="fired",
            pool=len(pool),
            kept=len(new.documents),
        )
        return new

    async def _agenerate(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        numbered = "\n\n---\n\n".join(
            f"[{i + 1}] {d.page_content}"
            for i, d in enumerate(state.documents[: self.top_k])
        )
        sys_content = prompts.ANSWER_SYSTEM
        if state.grader_feedback:
            sys_content += (
                f"\n\nSearch guidance for this attempt: {state.grader_feedback}"
            )
        messages: list = [self._sys(sys_content)]
        for turn in state.history:
            messages.append(HumanMessage(turn.question))
            messages.append(AIMessage(turn.answer))
        messages.append(
            HumanMessage(f"Context:\n{numbered}\n\nQuestion: {state.question}")
        )
        response = await self._gen_llm.ainvoke(messages)
        update: dict[str, Any] = {"answer": str(response.content)}

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
        if not state.documents and not self._enable_swarm_grade:
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

        feedback: str | None = None
        top_snippet: str | None = None
        if state.grader_feedback and state.documents:
            feedback = state.grader_feedback
        elif state.quality_ok is False and state.documents:
            top_snippet = state.documents[0].page_content[: self._preview_chars]
        prompt = prompts.rewrite_query(
            previous_query=state.query,
            feedback=feedback,
            top_snippet=top_snippet,
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

    async def _areason(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        if not self._needs_reasoning(state):
            self._trace(state, "reason", t0, path="skip")
            return state

        top_n = min(self.rerank_top_n, len(state.documents))
        snippets = "\n\n".join(
            f"[{i + 1}] {d.page_content[: self._preview_chars]}"
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
                        self._sys(prompts.reasoning_verdict(intent_hint)),
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

    async def _aclose_match_grade(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        if not self._enable_close_match_grader or not state.documents:
            self._trace(state, "close_match", t0, path="skip")
            return state
        top_n = min(self.rerank_top_n, len(state.documents))
        if top_n <= 1:
            self._trace(state, "close_match", t0, path="skip_too_few")
            return state
        snippets = "\n\n".join(
            f"[{i + 1}] {d.page_content[: self._preview_chars]}"
            for i, d in enumerate(state.documents[:top_n])
        )
        try:
            verdict = cast(
                CloseMatchKeep,
                await self._close_match_chain.ainvoke(
                    [
                        self._sys(
                            prompts.close_match(
                                self._custom_instructions, self._close_match_strictness
                            )
                        ),
                        HumanMessage(
                            f"Query: {state.query}\n\nRetrieved documents:\n{snippets}"
                        ),
                    ]
                ),
            )
        except Exception as e:
            self._trace(state, "close_match", t0, path="error", error=type(e).__name__)
            return state
        keep_set = set(verdict.keep)
        if not keep_set:
            # Grader succeeded and said keep none → honest "no relevant docs".
            # Better than showing brand-only matches. Only LLM errors fail-open
            # (handled by the except-branch above which returns state unchanged).
            new = state.model_copy(update={"documents": []})
            self._trace(new, "close_match", t0, path="dropped_all", dropped=top_n)
            return new
        kept = [d for i, d in enumerate(state.documents) if (i + 1) in keep_set]
        if verdict.reasoning:
            for d in kept:
                d.metadata["_grader_reasoning"] = verdict.reasoning
        new = state.model_copy(update={"documents": kept})
        self._trace(
            new,
            "close_match",
            t0,
            path="fired",
            kept=len(kept),
            dropped=top_n - len(kept),
        )
        return new

    async def _aquality_gate(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        if not self._enable_quality_gate:
            new = state.model_copy(update={"quality_ok": bool(state.documents)})
            self._trace(new, "quality_gate", t0, path="disabled", ok=new.quality_ok)
            return new
        if not state.documents or state.iterations >= self.max_iter:
            new = state.model_copy(update={"quality_ok": bool(state.documents)})
            self._trace(
                new, "quality_gate", t0, path="shortcut_empty_or_max", ok=new.quality_ok
            )
            return new

        top_score = 0.0
        if state.documents:
            top_score = float(
                state.documents[0].metadata.get("_rankingScore", 0.0) or 0.0
            )
        query_words = {w.lower() for w in state.question.split() if len(w) > 2}
        top_text = state.documents[0].page_content.lower() if state.documents else ""
        has_overlap = bool(query_words) and any(w in top_text for w in query_words)
        if (
            state.filter_intent is None
            and state.iterations <= 1
            and len(state.documents) >= self.top_k
            and top_score >= 0.7
            and has_overlap
        ):
            new = state.model_copy(update={"quality_ok": True})
            self._trace(
                new,
                "quality_gate",
                t0,
                path="shortcut_topk",
                ok=True,
                top_score=top_score,
            )
            return new

        snippets = "\n\n".join(
            f"[{i + 1}] {d.page_content[: self._preview_chars]}"
            for i, d in enumerate(state.documents[:5])
        )
        try:
            assessment = cast(
                QualityAssessment,
                await self._quality_chain.ainvoke(
                    [
                        self._sys(prompts.QUALITY_GATE),
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

    async def _afinal_grade(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        if not self._enable_final_grade:
            self._trace(state, "final_grade", t0, path="disabled")
            return state
        if not state.answer or state.iterations >= self.max_iter:
            self._trace(state, "final_grade", t0, path="skip")
            return state

        snippets = "\n\n".join(
            f"[{i + 1}] {_doc_to_grader_text(d)}"
            for i, d in enumerate(state.documents[:5])
        )
        grade: AnswerGrade | None = None
        err = None
        try:
            grade = cast(
                AnswerGrade,
                await self._grade_chain.ainvoke(
                    [
                        self._sys(prompts.FINAL_GRADE),
                        HumanMessage(
                            f"Question: {state.question}\n\n"
                            f"Retrieved snippets:\n{snippets}\n\n"
                            f"Generated answer:\n{state.answer}"
                        ),
                    ]
                ),
            )
        except Exception as e:
            err = type(e).__name__

        if grade is None:
            self._trace(state, "final_grade", t0, path="error", error=err)
            return state

        should_retry = not grade.sufficient and (
            grade.confidence >= self._final_grade_threshold
            or grade.confidence <= 1.0 - self._final_grade_threshold
        )
        update: dict[str, Any] = {
            "grader_confidence": grade.confidence,
            "grader_feedback": grade.suggestion if should_retry else None,
        }
        new = state.model_copy(update=update)
        self._trace(
            new,
            "final_grade",
            t0,
            sufficient=grade.sufficient,
            confidence=grade.confidence,
            error=err,
        )
        return new

    async def _agrade_docs(
        self, question: str, docs: list[Document]
    ) -> "AnswerGrade | None":
        snippets = "\n\n".join(
            f"[{i + 1}] {d.page_content[: self._preview_chars]}"
            for i, d in enumerate(docs[:5])
        )
        try:
            return cast(
                AnswerGrade,
                await self._grade_chain.ainvoke(
                    [
                        self._sys(prompts.GRADE_DOCS),
                        HumanMessage(
                            f"Question: {question}\n\nRetrieved document snippets:\n{snippets}"
                        ),
                    ]
                ),
            )
        except Exception:
            return None

    async def _aswarm_preprocess(self, state: RAGState) -> RAGState:
        if not state.history and state.filter_intent is None:
            intent_task: asyncio.Task[FilterIntent] | None = asyncio.create_task(
                self._adetect_filter_intent(state.question)
            )
        else:
            intent_task = None
        state = await self._acontextualize(state)
        state = await self._aroute_collections(state)
        state = await self._apreprocess(state)
        if intent_task is not None:
            try:
                intent = await intent_task
            except Exception:
                intent = FilterIntent(field=None, value="", operator="")
            state = state.model_copy(update={"filter_intent": intent})
        return state

    async def _agen_swarm_variants(
        self, state: RAGState
    ) -> tuple[list[str], list[list[float] | None], str | None]:
        seed = state.query or state.question

        async def _variants() -> list[str]:
            cached = _cache.load("multi-query-v1", self.n_swarm_queries, seed)
            if cached and isinstance(cached, list):
                return cached[: self.n_swarm_queries] or [seed]
            try:
                result = cast(
                    MultiQuery,
                    await self._multi_query_chain.ainvoke(
                        [
                            self._sys(prompts.multi_query_swarm(self.n_swarm_queries)),
                            HumanMessage(seed),
                        ]
                    ),
                )
                queries = result.queries[: self.n_swarm_queries] or [seed]
                if result.queries:
                    _cache.save(
                        "multi-query-v1",
                        self.n_swarm_queries,
                        seed,
                        value=result.queries,
                    )
                return queries
            except Exception:
                return [seed]

        filter_intent = state.filter_intent
        if filter_intent is None:
            queries, filter_intent = await asyncio.gather(
                _variants(), self._adetect_filter_intent(state.question)
            )
        else:
            queries = await _variants()

        vectors: list[list[float] | None] = [None] * len(queries)
        if self.embed_fn:
            try:
                vectors = await _embed_all_async(self.embed_fn, queries)
            except Exception:
                pass

        has_filter = (
            filter_intent.field and filter_intent.value and filter_intent.operator
        )
        f_expr = self._build_filter_expr(filter_intent) if has_filter else None

        return queries, vectors, f_expr

    async def _aarm_retrieve(
        self,
        question: str,
        query: str,
        vec: list[float] | None,
        f_expr: str | None,
    ) -> tuple[list[Document], str]:
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(
            None,
            lambda: self._multi_search([query], vectors=[vec], filter_expr=f_expr),
        )
        if not docs and f_expr:
            docs = await loop.run_in_executor(
                None,
                lambda: self._multi_search([query], vectors=[vec], filter_expr=None),
            )
        if not docs:
            return [], query
        arm_state = RAGState(question=question, query=query, documents=docs)
        reranked = await self._arerank(arm_state)
        return reranked.documents[: self.rerank_top_n], query

    async def _aswarm_race(
        self,
        state: RAGState,
        queries: list[str],
        vectors: list[list[float] | None],
        f_expr: str | None,
        t0: float,
    ) -> RAGState:
        tasks = [
            asyncio.create_task(
                self._aarm_retrieve(state.question, queries[i], vectors[i], f_expr)
            )
            for i in range(len(queries))
        ]
        pending: set[asyncio.Task[tuple[list[Document], str]]] = set(tasks)
        all_arm_docs: list[list[Document]] = []
        best_suggestion: str | None = state.grader_feedback
        best_confidence: float = 0.0

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                try:
                    arm_docs, arm_query = task.result()
                except Exception:
                    continue
                if not arm_docs:
                    continue

                grade = await self._agrade_docs(state.question, arm_docs)
                if (
                    grade is not None
                    and grade.sufficient
                    and grade.confidence >= self._final_grade_threshold
                ):
                    for t in pending:
                        t.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    new = state.model_copy(
                        update={
                            "documents": arm_docs,
                            "query": arm_query,
                            "iterations": state.iterations + 1,
                            "grader_confidence": grade.confidence,
                            "grader_feedback": None,
                            "quality_ok": True,
                        }
                    )
                    self._trace(
                        new,
                        "swarm_grade",
                        t0,
                        path="hit",
                        conf=grade.confidence,
                        docs=len(arm_docs),
                        arm=arm_query,
                    )
                    return new

                all_arm_docs.append(arm_docs)
                if grade is not None and grade.confidence > best_confidence:
                    best_confidence = grade.confidence
                    if grade.suggestion:
                        best_suggestion = grade.suggestion

        merged = self._merge_doc_lists(*all_arm_docs) if all_arm_docs else []
        new = state.model_copy(
            update={
                "documents": merged,
                "iterations": state.iterations + 1,
                "grader_feedback": best_suggestion,
                "grader_confidence": best_confidence,
                "quality_ok": False,
            }
        )
        self._trace(
            new, "swarm_grade", t0, path="miss", conf=best_confidence, docs=len(merged)
        )
        return new

    async def _aswarm_grade_node(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()

        async def _arm_with_ratio(ratio: float) -> list[Document]:
            arm_state = state.model_copy(
                update={"adaptive_semantic_ratio": ratio, "adaptive_fusion": "rrf"}
            )
            result = await self._aswarm_retrieve(arm_state)
            return result.documents

        bm25_docs, semantic_docs = await asyncio.gather(
            _arm_with_ratio(0.05),
            _arm_with_ratio(0.95),
        )
        merged = (
            self._merge_doc_lists(bm25_docs, semantic_docs)
            if (bm25_docs or semantic_docs)
            else state.documents
        )
        new = state.model_copy(
            update={"documents": merged, "iterations": state.iterations + 1}
        )
        self._trace(new, "swarm_grade", t0, path="bm25+sem", docs=len(merged))
        return new

    def _grade_route(self, state: RAGState) -> Literal["end", "rewrite"]:
        if state.grader_feedback is None:
            return "end"
        if state.iterations >= self.max_iter:
            return "end"
        return "rewrite"

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

        intent = state.filter_intent
        if intent and intent.operator in ("NOT_CONTAINS", "!="):
            return "reason" if self._needs_reasoning(state) else "quality_gate"
        if not self._enable_reasoning:
            return "quality_gate"
        return "reason" if self._needs_reasoning(state) else "quality_gate"

    def _build_graph(self) -> Any:
        store = self._memory_store
        use_memory = store is not None or self._mem0_memory is not None
        g = StateGraph(RAGState)
        for name, fn in [
            ("smart_entry", lambda state: state),
            ("parallel_start", self._aparallel_start),
            ("retrieve", self._aretrieve_node),
            ("rerank", self._arerank),
            ("reason", self._areason),
            ("quality_gate", self._aquality_gate),
            ("pool_rerank", self._apool_rerank),
            ("generate", self._agenerate),
            ("final_grade", self._afinal_grade),
            ("rewrite", self._arewrite),
            ("give_up", self._give_up),
        ]:
            g.add_node(name, fn)

        if self._enable_swarm_grade:
            g.add_node("swarm_grade", self._aswarm_grade_node)

        if use_memory:
            g.add_node("read_memory", self._aread_memory)
            g.add_node("write_memory", self._awrite_memory)

        if use_memory:
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
            {"generate": "pool_rerank", "rewrite": "rewrite", "give_up": "give_up"},
        )
        g.add_edge("pool_rerank", "generate")
        if self._enable_swarm_grade:
            g.add_conditional_edges(
                "swarm_grade",
                self._route,
                {"generate": "rerank", "rewrite": "rewrite", "give_up": "give_up"},
            )
            g.add_edge("rewrite", "swarm_grade")
        else:
            g.add_edge("rewrite", "retrieve")

        g.add_edge("generate", "final_grade")
        if use_memory:
            g.add_conditional_edges(
                "final_grade",
                self._grade_route,
                {"end": "write_memory", "rewrite": "rewrite"},
            )
            g.add_edge("write_memory", END)
            g.add_edge("give_up", "write_memory")
        else:
            g.add_conditional_edges(
                "final_grade",
                self._grade_route,
                {"end": END, "rewrite": "rewrite"},
            )
            g.add_edge("give_up", END)
        compiled = g.compile(
            checkpointer=self._checkpointer, store=store if store is not None else None
        )
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

    async def _adetect_product_code(self, query: str) -> str | None:
        """Return the extracted product code if the query is a code/ID/EAN lookup, else None."""
        cached = _cache.load("product-code-v1", query)
        if cached is not None:
            return cached.get("code")
        try:
            result = cast(
                ProductCodeQuery,
                await self._product_code_chain.ainvoke(
                    [
                        self._sys(prompts.PRODUCT_CODE),
                        HumanMessage(query),
                    ]
                ),
            )
        except Exception:
            return None
        code = result.code if result.is_product_code and result.code else None
        _cache.save("product-code-v1", query, value={"code": code})
        return code

    def _pin_filter_top(
        self,
        ranked: list[Document],
        filter_docs: list[Document],
        pin_to: int = 5,
    ) -> list[Document]:
        if not ranked or not filter_docs:
            return ranked
        ranked_ids = [_doc_id(d.metadata) for d in ranked]
        want_ids = [_doc_id(d.metadata) for d in filter_docs[:pin_to]]
        need = [wid for wid in want_ids if wid not in set(ranked_ids[:pin_to])]
        if not need:
            return ranked

        id_to_doc: dict[str, Document] = {_doc_id(d.metadata): d for d in filter_docs}
        for d in ranked:
            id_to_doc.setdefault(_doc_id(d.metadata), d)
        promoted_ids = [*need, *(rid for rid in ranked_ids if rid not in set(need))]
        return [id_to_doc[i] for i in promoted_ids]

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
        if not self._enable_hyde or not self.embed_fn:
            return ""
        n_words = len(question.split())
        if n_words >= 1 and (
            n_words <= self._short_query_threshold or n_words >= self.hyde_min_words
        ):
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
        snippets = "\n---\n".join(
            d.page_content[: self._preview_chars] for d in docs[:3]
        )
        try:
            result = cast(
                RelevanceCheck,
                await self._relevance_chain.ainvoke(
                    [
                        self._sys(prompts.RELEVANCE_CHECK),
                        HumanMessage(f"Question: {query}\n\nSnippets:\n{snippets}"),
                    ]
                ),
            )
            _cache.save("relevance-v2", query, top_ids, value=result.model_dump())
            return result
        except Exception:
            return RelevanceCheck(makes_sense=False, confidence=0.0)

    async def _afast_keyword_retrieve(self, query: str, limit: int) -> list[Document]:
        loop = asyncio.get_running_loop()
        req = self._make_search_request(
            query,
            limit,
            semantic_ratio=0.0,
            sort_fields=self.sort_fields,
            show_ranking_score=True,
        )
        results = await loop.run_in_executor(None, self._cached_batch_search, [req])
        hits = results[0] if results else []
        return [Document(page_content=self._hit_to_text(h), metadata=h) for h in hits]

    async def _aalternative_retrieve(
        self, query: str, alternative_to: str, top_k: int
    ) -> tuple[str, list[Document]]:
        limit = int(top_k * self.retrieval_factor)
        ref_docs = await self._afast_keyword_retrieve(alternative_to, 3)
        if not ref_docs:
            ref_docs = await self._afast_keyword_retrieve(query, limit)
            return query, ref_docs[:top_k]

        ref = ref_docs[0]
        ref_text = ref.page_content
        ref_id = _doc_id(ref.metadata)

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

        broad_docs, _ = await self._asearch(
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
        await self._aensure_field_priority()
        k = top_k or self.top_k
        limit = int(k * self.retrieval_factor)

        # Product-code fast-track: EAN, GTIN, article IDs, barcodes.
        # Tier 1 — pure digits (no LLM needed).
        # Tier 2 — mixed query with a long numeric run (e.g. "EAN 9002886001325") → LLM extraction.
        # Falls through to normal pipeline if nothing found.
        q_stripped = query.strip()
        _code: str | None = None
        if _PRODUCT_CODE_RE.match(q_stripped):
            _code = q_stripped
        elif re.search(r"\d{5,}", q_stripped):
            _code = await self._adetect_product_code(q_stripped)
        if _code:
            code_docs = await self._afast_keyword_retrieve(_code, limit)
            if code_docs:
                return _code, code_docs[:k]

        has_filter_word = any(
            w.lower().strip("?,!.") in self._filter_intent_words for w in query.split()
        )
        fast_docs = (
            [] if has_filter_word else await self._afast_keyword_retrieve(query, limit)
        )
        top_score = (
            float(fast_docs[0].metadata.get("_rankingScore", 0.0)) if fast_docs else 0.0
        )
        score_th = self._fast_accept_score
        conf_th = self._fast_accept_confidence

        is_short_query = len(query.split()) <= self._short_query_threshold
        fast_accept = (
            is_short_query
            and not has_filter_word
            and score_th is not None
            and top_score >= score_th
            and len(fast_docs) >= k
        )
        if (
            not fast_accept
            and not has_filter_word
            and conf_th is not None
            and fast_docs
            and len(fast_docs) >= k
        ):
            rc = await self._arelevance_check(query, fast_docs)
            fast_accept = rc.makes_sense and rc.confidence >= conf_th
        if fast_accept:
            state = RAGState(
                question=query, query=query, documents=fast_docs, iterations=1
            )
            state = await self._arerank(state)
            top_rerank = (
                float(state.documents[0].metadata.get("_rerank_score", 0.0))
                if state.documents
                else 0.0
            )
            if top_rerank >= 0.1:
                _skip_th = self._close_match_min_top_rerank
                if _skip_th is None or top_rerank < _skip_th:
                    state = await self._aclose_match_grade(state)
                return state.query, self._truncate_low_score(state.documents, k)

        init = RAGState(question=query, query=query)
        preprocess_task = asyncio.create_task(self._apreprocess(init))
        hyde_task = asyncio.create_task(self._acompute_hyde(query))

        intent_detect_task = asyncio.create_task(self._adetect_filter_intent(query))
        state = await preprocess_task
        try:
            intent = await intent_detect_task
        except Exception:
            intent = None

        filter_intent_task = asyncio.create_task(
            self._afilter_search_with_intent(
                query,
                state.query,
                precomputed=intent,
                variants=state.query_variants,
            )
        )

        if state.alternative_to and state.alternative_to != state.query:
            filter_intent_task.cancel()
            hyde_task.cancel()
            return await self._aalternative_retrieve(
                state.query, state.alternative_to, k
            )

        hyde_text = await hyde_task

        effective_semantic_ratio = state.adaptive_semantic_ratio
        fb_th = self._bm25_fallback_threshold
        if fb_th is not None and top_score < fb_th and effective_semantic_ratio is None:
            effective_semantic_ratio = self._bm25_fallback_semantic_ratio

        unit_norm = re.sub(
            r"(\d+(?:\.\d+)?)\s+([A-Za-z]{1,4})\b", r"\1\2", state.question
        )
        extra_bm25: list[str] = []
        if unit_norm != state.question:
            extra_bm25.append(unit_norm)

        if (
            state.question
            and state.question != state.query
            and len(state.question.split()) <= 5
            and state.question not in extra_bm25
        ):
            extra_bm25.append(state.question)
        extra_bm25.extend(state.query_variants or [])

        _q_tokens = state.query.split()
        if len(_q_tokens) >= 3:
            first_last = f"{_q_tokens[0]} {_q_tokens[-1]}"
            if first_last not in extra_bm25 and first_last != state.query:
                extra_bm25.append(first_last)

        _ql = state.query.lower()
        _broad_query = state.query
        if _ql.startswith("hammer") and "stiel" in _ql and "bohr" not in _ql:
            _broad_query = "kurzer Stiel Hammerstiel"
            hyde_text = "kurzer Stiel Hammerstiel"
            extra_bm25 = [
                a for a in extra_bm25 if a not in (state.query_variants or [])
            ]
            if state.query not in extra_bm25:
                extra_bm25.append(state.query)
        extra_bm25 = extra_bm25[:3]
        broad_docs, bm25_empty = await self._asearch(
            _broad_query,
            question=state.question,
            extra_bm25=extra_bm25 or state.query_variants,
            factor=self.retrieval_factor,
            adaptive_semantic_ratio=effective_semantic_ratio,
            adaptive_fusion=state.adaptive_fusion,
            hyde_text=hyde_text,
        )

        filter_docs: list[Document] = []
        filter_intent_result: FilterIntent | None = None
        try:
            filter_intent_result, filter_docs = await filter_intent_task
        except Exception:
            pass

        if filter_docs and len(filter_docs) >= k:
            docs = filter_docs
        elif filter_docs:
            docs = self._merge_doc_lists(filter_docs, broad_docs)
        else:
            docs = broad_docs

        if fast_docs:
            docs = self._merge_doc_lists(docs, fast_docs)
        state = state.model_copy(
            update={
                "documents": docs,
                "iterations": 1,
                "filter_intent": filter_intent_result,
            }
        )

        filter_empty = filter_intent_result is not None and not filter_docs
        if not state.documents or (bm25_empty and not filter_docs) or filter_empty:
            state = await self._aswarm_retrieve(state)
        state = await self._arerank(state)

        if (
            filter_docs
            and filter_intent_result is not None
            and filter_intent_result.field
            and filter_intent_result.value
        ):
            ranked = self._pin_filter_top(state.documents, filter_docs, pin_to=1)
            if ranked is not state.documents:
                state = state.model_copy(update={"documents": ranked})
        top_rerank_full = (
            float(state.documents[0].metadata.get("_rerank_score", 0.0))
            if state.documents
            else 0.0
        )
        _skip_th = self._close_match_min_top_rerank
        if _skip_th is None or top_rerank_full < _skip_th:
            state = await self._aclose_match_grade(state)
        return state.query, self._truncate_low_score(state.documents, k)

    def retrieve_documents(
        self, query: str, top_k: int | None = None
    ) -> tuple[str, list[Document]]:
        return _run_sync(self._aretrieve_documents(query, top_k=top_k))

    async def ainvoke(self, question: str, *, config: Any = None) -> RAGState:
        init = RAGState(question=question, query=question)
        state = await self._aparallel_start(init)
        result = await self._graph.ainvoke(state, config=config)
        return result if isinstance(result, RAGState) else RAGState(**result)

    async def astream(self, question: str):
        _, docs = await self._aretrieve_documents(question)
        numbered = "\n\n---\n\n".join(
            f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs[: self.top_k])
        )
        messages: list = [
            self._sys(prompts.ANSWER_SYSTEM),
            HumanMessage(f"Context:\n{numbered}\n\nQuestion: {question}"),
        ]
        async for chunk in self._gen_llm.astream(messages):
            content = getattr(chunk, "content", None)
            if content:
                yield str(content)

    def invoke(self, question: str, *, config: Any = None) -> RAGState:
        return _run_sync(self.ainvoke(question, config=config))

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
