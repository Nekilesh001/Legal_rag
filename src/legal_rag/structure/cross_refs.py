"""
Cross-reference extraction and normalization.
Deterministic pattern matching — no LLM used here.
"""
from __future__ import annotations

import logging
import re

from legal_rag.models.reference import CrossReference, ReferenceIndex, ReferenceType

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Normalization mapping
# ------------------------------------------------------------------ #

# Aliases that should be normalized to canonical forms
_ALIASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bSec\.\s*", re.IGNORECASE),     "Section "),
    (re.compile(r"\bS\.\s+(?=\d)",  re.IGNORECASE), "Section "),
    (re.compile(r"\bArt\.\s*",      re.IGNORECASE), "Article "),
    (re.compile(r"\bCh\.\s*",       re.IGNORECASE), "Chapter "),
    (re.compile(r"\bSch\.\s*",      re.IGNORECASE), "Schedule "),
]

# Extraction patterns → (ReferenceType, regex)
_EXTRACT_PATTERNS: list[tuple[ReferenceType, re.Pattern[str]]] = [
    (ReferenceType.SECTION,  re.compile(
        r"\b(?:Section|Sec\.|S\.)\s*(\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))*)", re.IGNORECASE)),
    (ReferenceType.ARTICLE,  re.compile(
        r"\bArticle\s+([IVXLCDM]+|\d+)(?:\.\d+)*\b", re.IGNORECASE)),
    (ReferenceType.CHAPTER,  re.compile(
        r"\bChapter\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)),
    (ReferenceType.SCHEDULE, re.compile(
        r"\bSchedule\s+([IVXLCDM]+|\d+[A-Z]?)\b", re.IGNORECASE)),
    (ReferenceType.CLAUSE,   re.compile(
        r"\bClause\s+(\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))*)\b", re.IGNORECASE)),
    (ReferenceType.ORDER,    re.compile(
        r"\bOrder\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)),
    (ReferenceType.EXHIBIT,  re.compile(
        r"\bExhibit\s+([A-Z\d]+)\b", re.IGNORECASE)),
]


def normalize_reference_text(raw: str) -> str:
    """Normalize aliases to canonical forms."""
    result = raw.strip()
    for pattern, replacement in _ALIASES:
        result = pattern.sub(replacement, result)
    return result.strip()


def make_ref_id(ref_type: ReferenceType, identifier: str) -> str:
    """Create a stable normalized reference ID."""
    clean = identifier.strip().replace(" ", "_")
    return f"{ref_type.value}_{clean}"


def extract_references_from_text(
    text: str,
    source_document_id: str,
    source_chunk_id: str,
) -> list[CrossReference]:
    """
    Extract all explicit cross-references from a block of text.
    Returns a list of CrossReference objects (unresolved).
    """
    refs: list[CrossReference] = []
    seen_raw: set[str] = set()

    for ref_type, pattern in _EXTRACT_PATTERNS:
        for m in pattern.finditer(text):
            raw_text = m.group(0)
            if raw_text in seen_raw:
                continue
            seen_raw.add(raw_text)

            identifier = m.group(1)
            norm_text = normalize_reference_text(raw_text)
            ref_id = make_ref_id(ref_type, identifier)

            refs.append(CrossReference(
                ref_id=ref_id,
                raw_text=raw_text,
                ref_type=ref_type,
                source_document_id=source_document_id,
                source_chunk_id=source_chunk_id,
                target_section_number=identifier,
                is_resolved=False,
                resolution_confidence=0.0,
            ))

    return refs


def resolve_references(
    ref_index: ReferenceIndex,
    section_map: dict[str, str],  # ref_id -> chunk_id
) -> None:
    """
    Attempt to resolve cross-references against a section map.
    Mutates CrossReference objects in-place.
    section_map keys should be built from make_ref_id().
    """
    for refs in ref_index.index.values():
        for ref in refs:
            target_chunk_id = section_map.get(ref.ref_id)
            if target_chunk_id:
                ref.target_chunk_id = target_chunk_id
                ref.is_resolved = True
                ref.resolution_confidence = 1.0
            else:
                ref.is_resolved = False
                ref.resolution_confidence = 0.0

    resolved = sum(
        1 for refs in ref_index.index.values() for r in refs if r.is_resolved
    )
    total = sum(len(refs) for refs in ref_index.index.values())
    logger.info("Cross-reference resolution: %d/%d resolved", resolved, total)
