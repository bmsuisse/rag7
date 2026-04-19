"""Agentic RAG — LangGraph + multi-backend retrieval-augmented generation."""

import warnings as _warnings

_warnings.filterwarnings(
    "ignore",
    message=r".*Pydantic V1 functionality.*",
    category=UserWarning,
)

from .factory import init_agent  # noqa: E402
from .backend import (  # noqa: E402
    AzureAISearchBackend,
    ChromaDBBackend,
    DuckDBBackend,
    InMemoryBackend,
    IndexConfig,
    LanceDBBackend,
    MeilisearchBackend,
    PgvectorBackend,
    QdrantBackend,
    SearchBackend,
    SearchRequest,
)
from .config import RAGConfig  # noqa: E402
from .core import AgenticRAG  # noqa: E402
from .models import ConversationTurn, RAGState, Reranker, RerankResult  # noqa: E402
from .rerankers import (  # noqa: E402
    CohereReranker,
    HuggingFaceReranker,
    JinaReranker,
    LLMReranker,
    RerankersReranker,
)

try:  # noqa: E402
    from .embedder import EmbedAnythingEmbedder
    from .rerankers import EmbedAnythingReranker
except ImportError:
    pass
from .utils import _dbsf_fuse, _make_azure_embed_fn, _rrf_fuse  # noqa: E402

Agent = AgenticRAG

__all__ = [
    "Agent",
    "AgenticRAG",
    "RAGConfig",
    "init_agent",
    "AzureAISearchBackend",
    "ChromaDBBackend",
    "CohereReranker",
    "HuggingFaceReranker",
    "JinaReranker",
    "RerankersReranker",
    "ConversationTurn",
    "DuckDBBackend",
    "EmbedAnythingEmbedder",
    "EmbedAnythingReranker",
    "InMemoryBackend",
    "IndexConfig",
    "LLMReranker",
    "LanceDBBackend",
    "MeilisearchBackend",
    "PgvectorBackend",
    "QdrantBackend",
    "RAGState",
    "Reranker",
    "RerankResult",
    "SearchBackend",
    "SearchRequest",
    "_dbsf_fuse",
    "_make_azure_embed_fn",
    "_rrf_fuse",
]
