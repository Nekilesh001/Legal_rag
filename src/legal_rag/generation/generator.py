"""
Grounded answer generation.
Assembles structured context and calls the LLM with strict grounding constraints.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from legal_rag.models.retrieval import (
    Citation,
    ConfidenceLevel,
    EvidenceStatus,
    ExpandedEvidence,
    QueryAnalysis,
    QueryResponse,
    RetrievalResult,
)
from legal_rag.providers.llm.nvidia import LLMProvider

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# System Prompt
# ------------------------------------------------------------------ #

GROUNDED_SYSTEM_PROMPT = """You are a Legal Research Assistant with access to a curated legal document corpus.

STRICT RULES — MANDATORY:
1. Answer ONLY from the supplied Evidence Context below.
2. Do NOT invent statutes, sections, clauses, or legal provisions.
3. Do NOT fabricate citations or case references.
4. Do NOT use your pretrained knowledge of law as a source of legal facts.
5. If the evidence is insufficient or absent, state that clearly.
6. Distinguish between what the evidence directly supports and any interpretation.
7. Cite every substantive legal claim using the citation IDs provided in the evidence.
8. When uncertain, say "The provided corpus does not contain sufficient information."

RESPONSE FORMAT:
Respond with a structured JSON object:
{
  "answer": "<clear factual answer grounded in evidence>",
  "evidence_status": "supported|partially_supported|insufficient|out_of_scope",
  "confidence": "high|medium|low",
  "citations": [
    {
      "citation_id": "<from context>",
      "document": "<document title>",
      "section": "<section number>",
      "page": <page number or null>,
      "excerpt": "<the exact supporting text you are citing (verbatim from evidence)>"
    }
  ]
}"""

STREAMING_SYSTEM_PROMPT = """You are a Legal Research Assistant with access to a curated legal document corpus.

STRICT RULES — MANDATORY:
1. Answer ONLY from the supplied Evidence Context below.
2. Do NOT invent statutes, sections, clauses, or legal provisions.
3. Do NOT fabricate citations or case references.
4. Do NOT use your pretrained knowledge of law as a source of legal facts.
5. If the evidence is insufficient or absent, state that clearly.
6. Distinguish between what the evidence directly supports and any interpretation.
7. Cite every substantive legal claim using the citation IDs provided in the evidence (e.g. [C01], [C02]).
8. When uncertain, say "The provided corpus does not contain sufficient information."

Format your response directly as a clear, professional legal answer in Markdown format. Do NOT wrap in JSON."""



# ------------------------------------------------------------------ #
# Context Assembler
# ------------------------------------------------------------------ #

def assemble_context(
    evidence_list: list[ExpandedEvidence],
    query_analysis: QueryAnalysis | None = None,
) -> tuple[str, list[Citation]]:
    """
    Build a structured evidence context string and pre-built Citation objects.
    Citations are derived from chunk metadata — NOT invented by the LLM.
    """
    citations: list[Citation] = []
    context_parts: list[str] = []

    for idx, ev in enumerate(evidence_list, start=1):
        if isinstance(ev, ExpandedEvidence):
            child = ev.child
            parent_text = ev.parent_text
            related = ev.related_provisions
        elif hasattr(ev, "child"):
            child = ev.child
            parent_text = getattr(ev, "parent_text", None)
            related = getattr(ev, "related_provisions", [])
        elif isinstance(ev, dict):
            child_data = ev.get("child", ev)
            parent_text = ev.get("parent_text")
            related = ev.get("related_provisions", [])
            if isinstance(child_data, dict):
                child = RetrievalResult(
                    chunk_id=child_data.get("chunk_id", ""),
                    parent_id=child_data.get("parent_id"),
                    document_id=child_data.get("document_id", ""),
                    document_title=child_data.get("document_title"),
                    category=child_data.get("category", ""),
                    section_number=child_data.get("section_number"),
                    section_title=child_data.get("section_title"),
                    page_start=child_data.get("page_start"),
                    page_end=child_data.get("page_end"),
                    text=child_data.get("text", ""),
                )
            else:
                child = child_data
        else:
            child = ev
            parent_text = None
            related = []

        cid = f"C{idx:02d}"


        citation = Citation(
            citation_id=cid,
            document_id=child.document_id,
            document_title=child.document_title or "Unknown Document",
            chunk_id=child.chunk_id,
            section=child.section_number,
            section_title=child.section_title,
            page=child.page_start,
            category=child.category,
            excerpt=child.text[:500],  # first 500 chars as evidence excerpt
        )
        citations.append(citation)

        block = [
            f"--- EVIDENCE [{cid}] ---",
            f"Document: {child.document_title or 'Unknown'}",
            f"Category: {child.category}",
            f"Section: {child.section_number or 'N/A'} — {child.section_title or ''}",
            f"Page: {child.page_start or 'N/A'}",
            "",
            "PRIMARY EVIDENCE:",
            child.text,
        ]

        if parent_text:
            block.extend([
                "",
                "PARENT CONTEXT:",
                parent_text[:800],  # limit parent context to avoid overflow
            ])

        if related:
            block.append("\nRELATED PROVISIONS:")
            for rp in related[:2]:
                sec_num = rp.get("section_number") if isinstance(rp, dict) else getattr(rp, "section_number", "N/A")
                text_snippet = rp.get("text", "") if isinstance(rp, dict) else getattr(rp, "text", "")
                block.append(f"  [{sec_num}] {text_snippet[:300]}")

        context_parts.append("\n".join(block))


    full_context = "\n\n".join(context_parts)
    return full_context, citations


# ------------------------------------------------------------------ #
# Grounded Generator
# ------------------------------------------------------------------ #

class GroundedGenerator:
    """
    Generates answers strictly grounded in retrieved evidence.
    Never uses the LLM as a fallback source of legal facts.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def generate(
        self,
        query: str,
        evidence_list: list[ExpandedEvidence],
        query_analysis: QueryAnalysis | None = None,
        confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN,
    ) -> QueryResponse:
        """
        Generate a structured, grounded answer.
        If evidence_list is empty, returns explicit knowledge-gap response.
        """
        if not evidence_list:
            return self._abstain(query, query_analysis)

        context_text, pre_built_citations = assemble_context(evidence_list, query_analysis)

        user_message = (
            f"QUERY: {query}\n\n"
            f"EVIDENCE CONTEXT:\n{context_text}\n\n"
            "Answer the query using ONLY the evidence above. "
            "Use citation IDs (e.g. [C01], [C02]) when citing specific evidence."
        )

        messages = [
            {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        raw = self.llm.generate(messages)

        # Parse JSON response
        parsed = self._parse_response(raw)

        # Build final citations (merge pre-built metadata with LLM response)
        final_citations = self._merge_citations(pre_built_citations, parsed.get("citations", []))

        return QueryResponse(
            query=query,
            answer=parsed.get("answer", raw),
            evidence_status=EvidenceStatus(
                parsed.get("evidence_status", EvidenceStatus.PARTIALLY_SUPPORTED.value)
            ),
            confidence=ConfidenceLevel(parsed.get("confidence", confidence.value)),
            citations=final_citations,
            supporting_chunks=[ev.child if hasattr(ev, "child") else (ev.get("child") if isinstance(ev, dict) and "child" in ev else ev) for ev in evidence_list],

            query_analysis=query_analysis,
        )

    def generate_stream(
        self,
        query: str,
        evidence_list: list[ExpandedEvidence],
        query_analysis: QueryAnalysis | None = None,
        confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN,
    ):
        """
        Stream generation tokens directly from LLM provider.
        Yields (pre_built_citations, token_generator).
        """
        if not evidence_list:
            abstain_res = self._abstain(query, query_analysis)
            def _abstain_gen():
                yield abstain_res.answer
            return [], _abstain_gen()

        context_text, pre_built_citations = assemble_context(evidence_list, query_analysis)

        user_message = (
            f"QUERY: {query}\n\n"
            f"EVIDENCE CONTEXT:\n{context_text}\n\n"
            "Answer the query using ONLY the evidence above. "
            "Use citation IDs (e.g. [C01], [C02]) when citing specific evidence."
        )

        messages = [
            {"role": "system", "content": STREAMING_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        token_stream = self.llm.stream(messages)
        return pre_built_citations, token_stream




    def _abstain(
        self, query: str, query_analysis: QueryAnalysis | None
    ) -> QueryResponse:
        """Return a knowledge-gap response — never hallucinate."""
        logger.warning("Abstaining: no sufficient evidence for query: %s", query[:80])
        return QueryResponse(
            query=query,
            answer=(
                "The retrieved corpus does not contain sufficient evidence to answer this query. "
                "No legal provisions were found that directly address the question. "
                "Please consult the original legal documents or a qualified legal professional."
            ),
            evidence_status=EvidenceStatus.INSUFFICIENT,
            confidence=ConfidenceLevel.LOW,
            citations=[],
            query_analysis=query_analysis,
        )

    def _parse_response(self, raw: str) -> dict[str, Any]:
        import re
        # Try to extract JSON block
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    ans = data.get("answer", raw)
                    status = data.get("evidence_status", "supported")
                    conf = data.get("confidence", "high")
                    cites = data.get("citations", [])
                    return {
                        "answer": ans,
                        "evidence_status": status,
                        "confidence": conf,
                        "citations": cites,
                    }
            except json.JSONDecodeError:
                pass
        return {"answer": raw, "evidence_status": "supported", "confidence": "high", "citations": []}


    def _merge_citations(
        self,
        pre_built: list[Citation],
        llm_citations: list[dict[str, Any]],
    ) -> list[Citation]:
        """
        Merge pre-built citation metadata with any LLM citation references.
        Pre-built citations are authoritative — LLM only provides the citation_id references.
        """
        # Build map from citation_id to pre-built Citation
        cid_map = {c.citation_id: c for c in pre_built}

        merged: list[Citation] = []
        seen_ids: set[str] = set()

        # If LLM cited specific IDs, include those in order
        for llm_cite in llm_citations:
            cid = llm_cite.get("citation_id", "")
            if cid in cid_map and cid not in seen_ids:
                c = cid_map[cid].model_copy()
                # LLM may provide a verbatim excerpt — use it if available
                if llm_cite.get("excerpt"):
                    c.excerpt = llm_cite["excerpt"][:600]
                merged.append(c)
                seen_ids.add(cid)

        # Include all pre-built citations not explicitly cited
        for c in pre_built:
            if c.citation_id not in seen_ids:
                merged.append(c)

        return merged
