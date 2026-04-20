from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field


@dataclass
class RerankResult:
    """Single reranking result."""

    index: int
    relevance_score: float


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranking backends.

    Any object exposing this signature can be used as a reranker —
    no inheritance required.
    """

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]: ...


class ConversationTurn(BaseModel):
    question: str
    answer: str


class RAGState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str = ""
    query: str = ""
    query_variants: list[str] = Field(default_factory=list)
    adaptive_semantic_ratio: float | None = None
    adaptive_fusion: str | None = None
    documents: list[Document] = Field(default_factory=list)
    answer: str | None = None
    iterations: int = 0
    history: list[ConversationTurn] = Field(default_factory=list)
    quality_ok: bool | None = None
    filter_intent: "FilterIntent | None" = None
    alternative_to: str | None = None
    selected_collections: list[str] = Field(default_factory=list)
    expert_fired: bool = False
    pre_reranked: bool = False
    trace: list[dict[str, Any]] = Field(default_factory=list)


class SearchQuery(BaseModel):
    reasoning: str = ""
    query: str
    variants: list[str] = []
    semantic_ratio: float = 0.5
    fusion: str = "rrf"
    alternative_to: str | None = None


class QualityAssessment(BaseModel):
    reason: str
    sufficient: bool


class RelevanceCheck(BaseModel):
    confidence: float
    makes_sense: bool


class MultiQuery(BaseModel):
    queries: list[str]


class ReasoningVerdict(BaseModel):
    """Per-document relevance verdict from the reasoning node."""

    reasoning: str = Field(
        default="",
        description="Brief chain-of-thought explaining the verdict.",
    )
    dominated_by: list[int] = Field(
        default_factory=list,
        description="1-based indices of docs that are clearly irrelevant or violate the query intent. Empty if all are fine.",
    )
    rewritten_query: str | None = Field(
        default=None,
        description="If the current docs are all poor, suggest a better search query. Null if docs are acceptable.",
    )


class FilterIntent(BaseModel):
    reasoning: str = ""
    field: str | None
    value: str
    operator: str
    extra_excludes: list[str] = Field(
        default_factory=list,
        description="Additional values to exclude when operator is NOT_CONTAINS (e.g., 'nicht X oder Y' → value=X, extra_excludes=[Y]).",
    )


class CollectionIntent(BaseModel):
    """LLM-selected subset of collection names to query."""

    collections: list[str] = Field(default_factory=list)


RAGState.model_rebuild()
