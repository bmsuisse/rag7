from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field


@dataclass
class RerankResult:
    index: int
    relevance_score: float


@runtime_checkable
class Reranker(Protocol):
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
    candidate_pool: list[Document] = Field(default_factory=list)
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
    grader_feedback: str | None = None
    grader_confidence: float | None = None


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


class AnswerGrade(BaseModel):
    sufficient: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    suggestion: str


class RelevanceCheck(BaseModel):
    confidence: float
    makes_sense: bool


class MultiQuery(BaseModel):
    queries: list[str]


class ReasoningVerdict(BaseModel):
    reasoning: str = Field(
        default="",
        description="Brief chain-of-thought explaining the verdict.",
    )
    sufficient: bool = Field(
        default=True,
        description="True if the retrieved documents contain a direct, relevant answer to the user's question. False if off-topic, too vague, or wrong.",
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
    and_filters: list["FilterIntent"] = Field(
        default_factory=list,
        description="Additional AND conditions for compound queries (e.g. supplier CONTAINS brand AND article_name NOT_CONTAINS type).",
    )


class FieldRank(BaseModel):
    name: str = Field(description="Field name as listed in the input.")
    rank: int = Field(
        description="Importance rank: 0 = most informative for representing the document, 9 = least."
    )


class FieldPriority(BaseModel):
    ranks: list[FieldRank] = Field(
        default_factory=list,
        description="One entry per input field, every input field must appear exactly once.",
    )


class CollectionIntent(BaseModel):
    collections: list[str] = Field(default_factory=list)


class ProductCodeQuery(BaseModel):
    is_product_code: bool = Field(
        description="True if the query is asking to look up a product by a specific code, ID, EAN, GTIN, barcode, or article number."
    )
    code: str | None = Field(
        default=None,
        description="The extracted numeric code (digits only, no prefix words). Null if not a product-code query.",
    )


class StandaloneQuery(BaseModel):
    query: str = Field(
        description="The rewritten standalone search query. If already standalone, return it unchanged."
    )


class HypotheticalDoc(BaseModel):
    text: str = Field(
        description="2-4 sentences written as if extracted from a relevant document. Domain terminology, specific identifiers, technical terms. No questions."
    )


class RelevanceScore(BaseModel):
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance score between 0.0 and 1.0.",
    )


class CloseMatchKeep(BaseModel):
    reasoning: str = Field(
        default="",
        description="Brief explanation of which docs are genuinely relevant vs. which match only on brand/lexical overlap.",
    )
    keep: list[int] = Field(
        default_factory=list,
        description="1-based indices of docs that are genuinely relevant to the query's semantic intent. Drop docs that only match on brand name or lexical overlap without matching the user's concept.",
    )


RAGState.model_rebuild()
