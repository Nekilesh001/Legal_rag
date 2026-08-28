"""
Structured Query Retrieval module — Experiment 3.
Executes exact legal section/document lookups and metadata candidate generation BEFORE top-k cutoffs.
Uses LegalDocumentRegistry for canonical act name resolution (no hardcoded filename conditions).
"""
from __future__ import annotations

import logging
from typing import Any

from legal_rag.indexing.bm25_store import BM25Store
from legal_rag.indexing.qdrant_store import QdrantVectorStore
from legal_rag.models.retrieval import QueryAnalysis
from legal_rag.retrieval.legal_identity import LegalDocumentRegistry

import re

logger = logging.getLogger(__name__)


def _section_matches(sref: str, sec_num: str, sec_title: str) -> bool:
    """
    Returns True if the queried section reference matches a chunk's section metadata.
    Uses word-boundary-aware matching to avoid '73' matching '173' or '730'.
    """
    if not sref:
        return False
    # Exact numeric match
    if sref == sec_num:
        return True
    # Word-boundary match in section number: e.g. "73" in "73(a)" but not in "173"
    if sec_num and re.search(r"(?<!\d)" + re.escape(sref) + r"(?!\d)", sec_num):
        return True
    # Substring match in section title (titles are descriptive, substring is safe)
    if sec_title and sref in sec_title:
        return True
    return False



class StructuredQueryRetriever:
    """
    Retrieves candidates using structured metadata rules before top-k truncation.
    Uses LegalDocumentRegistry to resolve canonical act names to document IDs.
    """

    def __init__(
        self,
        qdrant_store: QdrantVectorStore,
        bm25_store: BM25Store,
        registry: LegalDocumentRegistry | None = None,
    ) -> None:
        self.qdrant = qdrant_store
        self.bm25 = bm25_store
        self._registry = registry

    def _get_registry(self) -> LegalDocumentRegistry:
        """Lazily bootstrap the registry from BM25 metadata."""
        if self._registry is None:
            from legal_rag.retrieval.legal_identity import registry as _global_registry
            self._registry = _global_registry
        if not self._registry._loaded:
            self._registry.bootstrap(self.bm25._chunk_metadata)
        return self._registry

    def retrieve_structured_candidates(
        self, query_analysis: QueryAnalysis
    ) -> list[dict[str, Any]]:
        """
        Executes exact section/act lookups and metadata category lookups directly against the index metadata.
        Resolves act names through the canonical LegalDocumentRegistry so opaque filenames are matched correctly.
        """
        reg = self._get_registry()

        sec_refs = [s.lower().replace("section", "").strip() for s in query_analysis.section_refs]
        act_names = [a.lower().strip() for a in query_analysis.act_names]
        category_hints = [c.lower().strip() for c in query_analysis.category_hints]

        # Resolve canonical document IDs for all queried act names
        matched_doc_ids: set[str] = set()
        for act in act_names:
            for did in reg.resolve_act_name(act):
                matched_doc_ids.add(did.lower())

        structured_results: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()

        # Iterate over all indexed BM25 metadata (1-to-1 with Qdrant canonical chunks)
        for i, meta in enumerate(self.bm25._chunk_metadata):
            cid = meta.get("chunk_id")
            if not cid or cid in seen_chunk_ids:
                continue

            doc_id = (meta.get("document_id") or "").lower()
            doc_title = (meta.get("document_title") or "").lower()
            sec_num = str(meta.get("section_number") or "").lower()
            sec_title = (meta.get("section_title") or "").lower()
            category = (meta.get("category") or "").lower()
            raw_text = self.bm25._chunk_texts[i]

            match_reason = None

            # 1. Exact Act + Section Lookup (via canonical registry)
            if sec_refs and matched_doc_ids:
                if doc_id in matched_doc_ids:
                    for sref in sec_refs:
                        if sref and _section_matches(sref, sec_num, sec_title):
                            match_reason = f"exact_section_lookup (doc={doc_id} sec={sref})"
                            break

            # Fallback: direct title-based act name match if registry produced no IDs
            if not match_reason and sec_refs and act_names and not matched_doc_ids:
                for act in act_names:
                    if act in doc_title:
                        for sref in sec_refs:
                            if sref and (sref == sec_num or sref in sec_num or sref in sec_title):
                                match_reason = f"exact_section_lookup_title ({act} sec={sref})"
                                break

            # 2. Metadata Category / Act Lookup
            if not match_reason and category_hints:
                for cat in category_hints:
                    if cat == "nda" and (
                        "nda" in doc_title
                        or "mandatory" in doc_title
                        or "playbook" in doc_title
                    ):
                        match_reason = f"category_lookup ({cat})"
                        break
                    elif cat in category or cat in doc_title:
                        match_reason = f"category_lookup ({cat})"
                        break

            if match_reason:
                seen_chunk_ids.add(cid)
                item = meta.copy()
                item["text"] = raw_text
                item["chunk_id"] = cid
                item["retrieval_source"] = match_reason
                item["structured_matched"] = True
                structured_results.append(item)

        logger.info(
            "Structured query retrieval: %d candidates "
            "(registry resolved doc_ids=%s, sec_refs=%s, cats=%s)",
            len(structured_results),
            list(matched_doc_ids)[:5],
            sec_refs[:3],
            category_hints[:3],
        )
        return structured_results
