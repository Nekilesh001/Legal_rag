"""
Evidence Set Selector & Diversity Control module — Experiment 10.

Selects a concise, complementary final evidence set from the ranked candidate list.
Enforces:
1. Max chunks per section number (prevents a single sub-clause from clogging top evidence).
2. Content-type & concept filtering (suppresses generic definition/preamble chunks when operative remedy provisions exist).
3. Section diversity across complementary statutory provisions.
"""
from __future__ import annotations

import copy
import logging
from typing import Any

from legal_rag.models.retrieval import QueryAnalysis, QueryIntent

logger = logging.getLogger(__name__)


_FINE_CONCEPT_MAP: dict[str, list[str]] = {
    "breach_of_warranty": ["breach of warranty", "remedy for breach of warranty", "section 59"],
    "repudiation": ["repudiation", "repudiates", "anticipatory breach", "section 60"],
    "refund_price": ["refund of the price", "suit for refund", "refund"],
    "damages_nondelivery": ["non-delivery", "suit for damages", "damages for non-delivery", "section 54"],
    "resale_rights": ["re-sale", "resale", "right of re-sale", "unpaid seller"],
    "condition_vs_warranty": ["condition to be treated as warranty", "breach of condition", "section 13"],
    "general_breach": ["breach of contract", "breaches the contract", "default"],
}


def extract_candidate_fine_concepts(candidate: dict[str, Any]) -> set[str]:
    """Extract fine-grained legal concepts from chunk text and section metadata."""
    meta = candidate.get("payload", candidate)
    text = candidate.get("text", "").lower()
    sec_title = str(meta.get("section_title") or "").lower()
    sec_num = str(meta.get("section_number") or "").lower()
    full_text = f"{text} {sec_title} section {sec_num}"

    concepts: set[str] = set()
    for concept_name, keywords in _FINE_CONCEPT_MAP.items():
        if any(kw in full_text for kw in keywords):
            concepts.add(concept_name)
    return concepts


class EvidenceSelector:
    """
    Selects a complementary final evidence set from a ranked candidate list.
    Balances relevance, concept coverage, and section diversity.
    """

    def __init__(
        self,
        max_chunks_per_section: int = 1,
        max_evidence_items: int = 5,
        concept_yield_bonus: float = 0.25,
    ) -> None:
        self.max_chunks_per_section = max_chunks_per_section
        self.max_evidence_items = max_evidence_items
        self.concept_yield_bonus = concept_yield_bonus

    def is_broad_multi_evidence_query(self, query_analysis: QueryAnalysis) -> bool:
        """
        Determines if a query is a broad conceptual legal question requiring
        multi-evidence section diversity (e.g. seller breach remedies).
        """
        # Exact clause lookups (like Q1 "Section 73") are NOT broad multi-evidence queries
        if len(query_analysis.section_refs) > 0 and len(query_analysis.act_names) > 0:
            return False

        q_lower = query_analysis.original_query.lower()

        # Broad conceptual questions (e.g. Q4 "seller breaches contract")
        if query_analysis.intent in (QueryIntent.OBLIGATION_QUERY, QueryIntent.COMPLIANCE_AUDIT):
            if any(term in q_lower for term in ("breach", "breaches", "remedy", "remedies", "what happens if", "options")):
                return True

        return False

    def select_final_evidence_set(
        self,
        ranked_candidates: list[dict[str, Any]],
        query_analysis: QueryAnalysis,
    ) -> list[dict[str, Any]]:
        """
        Selects top complementary evidence items enforcing concept coverage and section diversity.
        """
        if not ranked_candidates:
            return []

        # Annotate fine-grained concepts onto candidates
        for cand in ranked_candidates:
            cand["fine_concepts"] = extract_candidate_fine_concepts(cand)

        # If not a broad query, return top N directly
        if not self.is_broad_multi_evidence_query(query_analysis):
            return ranked_candidates[: self.max_evidence_items]

        # -------------------------------------------------------------- #
        # Concept-Aware Selection for Broad Queries
        # -------------------------------------------------------------- #
        remaining_pool = copy.deepcopy(ranked_candidates)
        seen_sections: set[str] = set()
        seen_chunks: set[str] = set()
        covered_concepts: set[str] = set()
        final_evidence: list[dict[str, Any]] = []

        while remaining_pool and len(final_evidence) < self.max_evidence_items:
            # Score each candidate in pool based on reranker score + concept coverage yield
            best_idx = -1
            best_score = -999.0

            for i, cand in enumerate(remaining_pool):
                cid = cand.get("chunk_id", "")
                if cid in seen_chunks:
                    continue

                meta = cand.get("payload", cand)
                sec_num = str(meta.get("section_number") or "").strip().lower()
                doc_id = meta.get("document_id", "").strip().lower()
                ctype = cand.get("content_type", meta.get("content_type", ""))
                sec_key = f"{doc_id}:{sec_num}" if sec_num else cid

                # Skip section duplicate
                if sec_num and sec_key in seen_sections:
                    continue

                # Suppress generic definitions if operative remedy chunks exist
                c_fine = cand.get("fine_concepts", set())
                if ctype == "definition" and not c_fine and len(final_evidence) > 0:
                    continue

                # Base score = protected_score or blended_score or reranker_score
                base = cand.get("protected_score", cand.get("blended_score", cand.get("reranker_score", 0.0)))

                # Concept Yield Bonus: candidate provides concepts NOT yet covered in final_evidence
                uncovered = c_fine.difference(covered_concepts)
                # Exclude general_breach from bonus calculation to prioritize specific remedies
                specific_uncovered = [c for c in uncovered if c != "general_breach"]
                yield_bonus = len(specific_uncovered) * self.concept_yield_bonus

                selection_score = base + yield_bonus
                if selection_score > best_score:
                    best_score = selection_score
                    best_idx = i

            if best_idx == -1:
                break

            selected = remaining_pool.pop(best_idx)
            cid = selected.get("chunk_id", "")
            meta = selected.get("payload", selected)
            sec_num = str(meta.get("section_number") or "").strip().lower()
            doc_id = meta.get("document_id", "").strip().lower()
            sec_key = f"{doc_id}:{sec_num}" if sec_num else cid

            if sec_num:
                seen_sections.add(sec_key)
            seen_chunks.add(cid)
            covered_concepts.update(selected.get("fine_concepts", set()))

            selected["selection_score"] = best_score
            selected["covered_concepts_at_selection"] = list(selected.get("fine_concepts", set()))
            final_evidence.append(selected)

            logger.info(
                "EvidenceSelector: selected chunk %s (sec %s, score=%.3f, concepts=%s)",
                cid, sec_num, best_score, selected.get("fine_concepts")
            )

        return final_evidence

