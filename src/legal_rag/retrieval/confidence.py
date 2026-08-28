"""
Retrieval confidence scoring.
Multi-signal approach: reranker scores, score separation, coverage, metadata match.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from legal_rag.models.retrieval import ConfidenceLevel, RetrievalResult, ExpandedEvidence

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceConfig:
    high_threshold: float = 0.75
    low_threshold: float = 0.40


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)



def score_confidence(
    results: list[Any],
    query_analysis=None,     # QueryAnalysis | None
    config: ConfidenceConfig | None = None,
) -> tuple[float, ConfidenceLevel]:
    """
    Calculate retrieval confidence from multiple signals.
    Returns (raw_score 0-1, ConfidenceLevel).
    """
    cfg = config or ConfidenceConfig()

    if not results:
        return 0.0, ConfidenceLevel.LOW

    # Signal 1: Top reranker score
    top_reranker = _get_val(results[0], "reranker_score", _get_val(results[0], "protected_score", None))
    if top_reranker is None:
        top_rrf = _get_val(results[0], "rrf_score", 0.0) or 0.0
        top_reranker_norm = min(top_rrf * 10, 1.0)
    else:
        top_reranker_norm = max(0.0, min((top_reranker + 5) / 10.0, 1.0))

    signal_1 = top_reranker_norm * 0.40

    # Signal 2: Score separation (top-1 vs top-3)
    rrf_0 = _get_val(results[0], "rrf_score", 0.0)
    rrf_2 = _get_val(results[2], "rrf_score", 0.0) if len(results) >= 3 else 0.0
    rrf_1 = _get_val(results[1], "rrf_score", 0.0) if len(results) >= 2 else 0.0

    if len(results) >= 3 and rrf_0 and rrf_2:
        sep = rrf_0 - rrf_2
        sep_norm = min(sep / 0.05, 1.0)
    elif len(results) >= 2 and rrf_0 and rrf_1:
        sep = rrf_0 - rrf_1
        sep_norm = min(sep / 0.05, 1.0)
    else:
        sep_norm = 0.3
    signal_2 = sep_norm * 0.20


    # Signal 3: Number of agreeing chunks (multiple top results = stronger evidence)
    count_factor = min(len(results) / 3.0, 1.0)
    signal_3 = count_factor * 0.20

    # Signal 4: Metadata match from query analysis
    signal_4 = 0.0
    if query_analysis is not None:
        category_hints = set(h.lower() for h in (query_analysis.category_hints or []))
        section_refs = set(query_analysis.section_refs or [])

        matched_categories = sum(
            1 for r in results[:3]
            if any(hint in str(_get_val(r, "category", _get_val(r, "category_hint", ""))).lower() for hint in category_hints)
        )
        matched_sections = sum(
            1 for r in results[:3]
            if _get_val(r, "section_number", None) and any(ref in str(_get_val(r, "section_number", "")) for ref in section_refs)
        )
        meta_factor = min((matched_categories + matched_sections) / 3.0, 1.0)
        signal_4 = meta_factor * 0.20
    else:
        signal_4 = 0.10  # neutral when no query analysis


    raw_score = signal_1 + signal_2 + signal_3 + signal_4
    raw_score = min(max(raw_score, 0.0), 1.0)

    if raw_score >= cfg.high_threshold:
        level = ConfidenceLevel.HIGH
    elif raw_score >= cfg.low_threshold:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW

    logger.debug(
        "Confidence: raw=%.3f (s1=%.3f s2=%.3f s3=%.3f s4=%.3f) → %s",
        raw_score, signal_1, signal_2, signal_3, signal_4, level.value,
    )
    return raw_score, level


def expand_to_parents(
    results: list[Any],
    qdrant_store,  # QdrantVectorStore
) -> list[ExpandedEvidence]:
    """
    For each child result, fetch the parent chunk for context.
    Returns ExpandedEvidence with both child + parent text.
    """
    expanded: list[ExpandedEvidence] = []
    for result in results:
        if isinstance(result, dict):
            ret_res = RetrievalResult(
                chunk_id=result.get("chunk_id", ""),
                parent_id=result.get("parent_id"),
                document_id=result.get("document_id", ""),
                document_title=result.get("document_title"),
                category=result.get("category", ""),
                section_number=result.get("section_number"),
                section_title=result.get("section_title"),
                page_start=result.get("page_start"),
                page_end=result.get("page_end"),
                text=result.get("text", ""),
                dense_score=result.get("dense_score"),
                sparse_score=result.get("sparse_score"),
                rrf_score=result.get("rrf_score"),
                reranker_score=result.get("reranker_score"),
                source=result.get("source", "unknown"),
            )
        else:
            ret_res = result

        parent_text: str | None = None
        if ret_res.parent_id:
            parent_payload = qdrant_store.get_parent_by_chunk_id(ret_res.parent_id)
            if parent_payload:
                parent_text = parent_payload.get("text")

        expanded.append(ExpandedEvidence(child=ret_res, parent_text=parent_text))

    return expanded

