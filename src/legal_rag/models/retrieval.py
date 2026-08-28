"""
Models for retrieval results, evidence, citations, and final answers.
These are separate objects — never collapse them.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    INSUFFICIENT = "insufficient"
    OUT_OF_SCOPE = "out_of_scope"


class QueryIntent(str, Enum):
    SPECIFIC_CLAUSE_LOOKUP = "specific_clause_lookup"
    DEFINITION_INQUIRY = "definition_inquiry"
    OBLIGATION_QUERY = "obligation_query"
    THRESHOLD_PARAMETER = "threshold_parameter"
    CROSS_CONTRACT_COMPARISON = "cross_contract_comparison"
    COMPLIANCE_AUDIT = "compliance_audit"
    GENERAL_LEGAL = "general_legal"
    UNKNOWN = "unknown"


# ------------------------------------------------------------------ #
# Retrieval-layer objects
# ------------------------------------------------------------------ #

class RetrievalResult(BaseModel):
    """A single candidate from vector or BM25 search."""
    chunk_id: str
    parent_id: str | None = None
    document_id: str = ""
    document_title: str | None = None
    category: str = ""
    section_number: str | None = None
    section_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    text: str = ""
    dense_score: float | None = None   # vector similarity score
    sparse_score: float | None = None  # BM25 score
    rrf_score: float | None = None     # after RRF fusion
    reranker_score: float | None = None
    source: str = "unknown"            # "dense", "sparse", "fused"


class ExpandedEvidence(BaseModel):
    """
    A child chunk + its parent context, ready for context assembly.
    Both are preserved; neither replaces the other.
    """
    child: RetrievalResult
    parent_text: str | None = None
    related_provisions: list[RetrievalResult] = Field(default_factory=list)


class Citation(BaseModel):
    """
    A grounded citation traceable to actual indexed source content.
    Must never be invented by the LLM — derived from chunk metadata.
    """
    citation_id: str = ""
    document_id: str = ""
    document_title: str | None = None
    chunk_id: str = ""
    section: str | None = None
    section_title: str | None = None
    page: int | None = None
    category: str = ""
    excerpt: str = ""           # the supporting text snippet (from chunk, not LLM)


class QueryAnalysis(BaseModel):
    """Structured output from the query understanding stage."""
    original_query: str
    normalized_query: str | None = None
    intent: QueryIntent = QueryIntent.UNKNOWN
    section_refs: list[str] = Field(default_factory=list)   # ["Section 138", "Article III"]
    act_names: list[str] = Field(default_factory=list)      # ["Negotiable Instruments Act"]
    exact_terms: list[str] = Field(default_factory=list)    # quoted/key terms
    jurisdictions: list[str] = Field(default_factory=list)
    category_hints: list[str] = Field(default_factory=list) # ["Finance", "employment"]
    dates: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)        # percentages, days, amounts


class QueryResponse(BaseModel):
    """
    The structured response returned by the RAG engine.
    Machine-readable and suitable for the future contract-analysis layer.
    """
    query: str
    answer: str
    evidence_status: EvidenceStatus = EvidenceStatus.INSUFFICIENT
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    citations: list[Citation] = Field(default_factory=list)
    supporting_chunks: list[RetrievalResult] = Field(default_factory=list)
    query_analysis: QueryAnalysis | None = None
    retrieval_attempts: int = 1
    rewritten_query: str | None = None
    processing_metadata: dict[str, Any] = Field(default_factory=dict)
