"""
Metadata-Aware Retrieval Policy Layer — Experiment 3.
Implements deterministic structural text formatting, metadata constraints,
and exact legal section / document / category / jurisdiction boosting.
Uses LegalDocumentRegistry for canonical act name resolution.
"""
from __future__ import annotations

import logging
from typing import Any
from legal_rag.config import RagConfig
from legal_rag.models.retrieval import QueryAnalysis

logger = logging.getLogger(__name__)


def build_structural_retrieval_text(meta: dict[str, Any], raw_text: str) -> str:
    """
    Constructs deterministic structural context from metadata without LLM.
    Preserves original raw text separately.
    """
    doc_title = meta.get("document_title") or meta.get("document_id") or "Unknown Document"
    sec_num = meta.get("section_number") or ""
    sec_title = meta.get("section_title") or ""
    category = meta.get("category") or "general"

    # Resolve jurisdiction
    jurisdiction = "India"
    doc_lower = doc_title.lower()
    cat_lower = str(category).lower()
    if "tamil nadu" in doc_lower or "tamil nadu" in cat_lower:
        jurisdiction = "Tamil Nadu"

    sec_str = f"{sec_num} ({sec_title})" if sec_num and sec_title else (sec_num or sec_title or "N/A")

    return (
        f"Document: {doc_title}\n"
        f"Section: {sec_str}\n"
        f"Category: {category}\n"
        f"Jurisdiction: {jurisdiction}\n\n"
        f"Original chunk text:\n{raw_text}"
    )


class MetadataRetrievalPolicy:
    """
    Modular policy layer for metadata-aware retrieval.
    Applies exact statutory section boosts, act/document matching,
    category boosting, and jurisdiction signals.
    Uses LegalDocumentRegistry for canonical act name → document_id resolution.
    """

    def __init__(self, config: RagConfig) -> None:
        self.config = config
        self._registry_loaded = False

    def _ensure_registry(self, bm25_metadata: list[dict[str, Any]]) -> None:
        if not self._registry_loaded:
            from legal_rag.retrieval.legal_identity import registry as _reg
            if not _reg._loaded:
                _reg.bootstrap(bm25_metadata)
            self._reg = _reg
            self._registry_loaded = True

    def apply_policy(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        candidates: list[dict[str, Any]],
        bm25_metadata: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Applies metadata-aware boosting and ranking adjustments to candidate chunks.
        """
        if not self.config.rag_metadata_aware_retrieval or not candidates:
            return candidates

        if bm25_metadata:
            self._ensure_registry(bm25_metadata)
        else:
            # Fall back without registry if no metadata provided
            self._reg = None

        sec_refs = [s.lower().replace("section", "").strip() for s in query_analysis.section_refs]
        act_names = [a.lower().strip() for a in query_analysis.act_names]
        category_hints = [c.lower().strip() for c in query_analysis.category_hints]
        jurisdictions = [j.lower().strip() for j in query_analysis.jurisdictions]

        # Pre-compute matched document IDs via registry
        matched_doc_ids: set[str] = set()
        if self._reg is not None:
            for act in act_names:
                for did in self._reg.resolve_act_name(act):
                    matched_doc_ids.add(did.lower())

        boosted_candidates = []

        for cand in candidates:
            item = cand.copy()
            meta = item.get("payload", item)

            doc_title = (meta.get("document_title") or "").lower()
            doc_id = (meta.get("document_id") or "").lower()
            sec_num = str(meta.get("section_number") or "").lower()
            sec_title = (meta.get("section_title") or "").lower()
            category = (meta.get("category") or "").lower()

            policy_boost = 0.0
            is_authoritative_source = False

            # 1. Exact Legal Section Matching & Authoritative Source Boost
            # Document match: via canonical registry OR direct title substring match
            doc_matches_act = (doc_id in matched_doc_ids) if matched_doc_ids else False
            if not doc_matches_act:
                for act in act_names:
                    if act in doc_title or act in doc_id:
                        doc_matches_act = True
                        break

            if doc_matches_act:
                policy_boost += self.config.rag_document_act_boost

                # Check if this is the authoritative Act (not rules / secondary commentary)
                if "rules" not in doc_title and "commentary" not in doc_title:
                    is_authoritative_source = True
                    policy_boost += 2.0

                for sref in sec_refs:
                    if sref and (sref == sec_num or sref in sec_num or sref in sec_title):
                        policy_boost += self.config.rag_exact_section_boost
                        if is_authoritative_source:
                            policy_boost += 5.0  # Authoritative source + exact section

            # 2. Category Boosting
            for cat_hint in category_hints:
                if (
                    cat_hint in category
                    or cat_hint in doc_title
                    or (cat_hint == "nda" and ("nda" in doc_title or "mandatory" in doc_title))
                ):
                    policy_boost += self.config.rag_category_boost

            # 3. Jurisdiction Boosting
            for jur in jurisdictions:
                if jur in doc_title or jur in doc_id:
                    policy_boost += self.config.rag_jurisdiction_boost

            item["policy_boost"] = policy_boost
            item["is_authoritative"] = is_authoritative_source

            base_score = item.get("rrf_score", item.get("score", 0.0))
            item["adjusted_score"] = base_score + (policy_boost * 0.01)

            boosted_candidates.append(item)

        # Re-sort candidates by adjusted score descending
        boosted_candidates.sort(key=lambda x: x.get("adjusted_score", 0.0), reverse=True)
        return boosted_candidates
