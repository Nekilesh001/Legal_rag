"""
Deterministic Parent-Contextual Reranking Formatter — Experiment 9.

Constructs temporary reranker-input strings incorporating structural legal hierarchy
(Document, Jurisdiction, Part, Chapter, Section Number, Section Title, Content Type)
without altering the original source chunk text or metadata.
"""
from __future__ import annotations

import logging
from typing import Any

from legal_rag.retrieval.legal_identity import LegalDocumentRegistry

logger = logging.getLogger(__name__)


def format_full_contextual_input(
    candidate: dict[str, Any],
    registry: LegalDocumentRegistry | None = None,
) -> str:
    """
    Construct full structural hierarchy input for CrossEncoder reranking.

    Format:
      Document: {canonical_title}
      Jurisdiction: {jurisdiction}
      Part: {part if available}
      Chapter: {chapter if available}
      Section: {section_number}
      Section Title: {section_title}
      Content Type: {content_type}

      Content:
      {original_child_text}
    """
    meta = candidate.get("payload", candidate)
    doc_id = meta.get("document_id", "")
    doc_title = meta.get("document_title") or doc_id

    canon = None
    if registry is not None and registry._loaded:
        canon = registry.get_canonical(doc_id)
        if canon is not None:
            doc_title = canon.canonical_title

    part = meta.get("part") or meta.get("part_number") or meta.get("part_title")
    chapter = meta.get("chapter") or meta.get("chapter_title") or meta.get("chapter_number")
    sec_num = meta.get("section_number")
    sec_title = meta.get("section_title")
    content_type = candidate.get("content_type", meta.get("content_type"))
    jurisdiction = meta.get("jurisdiction") or (canon.jurisdiction if canon else None)
    child_text = candidate.get("text", "")

    header_parts = [f"Document: {doc_title}"]
    if jurisdiction:
        header_parts.append(f"Jurisdiction: {jurisdiction}")
    if part:
        header_parts.append(f"Part: {part}")
    if chapter:
        header_parts.append(f"Chapter: {chapter}")
    if sec_num:
        header_parts.append(f"Section: {sec_num}")
    if sec_title:
        header_parts.append(f"Section Title: {sec_title}")
    if content_type:
        header_parts.append(f"Content Type: {content_type}")

    header_str = "\n".join(header_parts)
    return f"{header_str}\n\nContent:\n{child_text}"


def format_section_title_input(candidate: dict[str, Any]) -> str:
    """
    Construct lightweight section-title input for CrossEncoder reranking.

    Format:
      Section {sec_num}: {sec_title}

      {original_child_text}
    """
    meta = candidate.get("payload", candidate)
    sec_title = meta.get("section_title")
    sec_num = meta.get("section_number")
    child_text = candidate.get("text", "")

    header = ""
    if sec_num and sec_title:
        header = f"Section {sec_num}: {sec_title}"
    elif sec_title:
        header = f"Section Title: {sec_title}"
    elif sec_num:
        header = f"Section: {sec_num}"

    if header:
        return f"{header}\n\n{child_text}"
    return child_text


def apply_rerank_formatting(
    candidates: list[dict[str, Any]],
    mode: str = "full",
    registry: LegalDocumentRegistry | None = None,
) -> list[dict[str, Any]]:
    """
    Applies the chosen formatting mode to candidates in-place / returning updated list.
    Modes:
      - 'control': no rerank_input (uses raw text)
      - 'section_title': Section title + child text
      - 'full': full structural hierarchy + child text
    """
    results = []
    for c in candidates:
        item = c.copy()
        if mode == "full":
            item["rerank_input"] = format_full_contextual_input(c, registry=registry)
        elif mode == "section_title":
            item["rerank_input"] = format_section_title_input(c)
        elif mode == "control":
            item.pop("rerank_input", None)
        results.append(item)
    return results
