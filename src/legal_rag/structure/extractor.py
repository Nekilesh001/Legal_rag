"""
Legal structure extractor.
Detects Section/Article/Chapter/Schedule/Appendix numbering in parsed text
and converts flat paragraphs into a proper legal hierarchy.
"""
from __future__ import annotations

import logging
import re
from typing import Pattern

from legal_rag.models.document import (
    Document,
    LegalHierarchyLevel,
    LegalSection,
    Paragraph,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Patterns — ordered by specificity
# ------------------------------------------------------------------ #

_PATTERNS: list[tuple[LegalHierarchyLevel, Pattern[str]]] = [
    (LegalHierarchyLevel.PART,       re.compile(r"^\s*PART\s+([IVXLCDM]+|\d+)[\.\s\-—]", re.IGNORECASE)),
    (LegalHierarchyLevel.CHAPTER,    re.compile(r"^\s*CHAPTER\s+([IVXLCDM]+|\d+)[\.\s\-—]", re.IGNORECASE)),
    (LegalHierarchyLevel.ARTICLE,    re.compile(r"^\s*ARTICLE\s+([IVXLCDM]+|\d+)[\.\s\-—]?", re.IGNORECASE)),
    (LegalHierarchyLevel.SCHEDULE,   re.compile(r"^\s*SCHEDULE\s+([IVXLCDM]+|\d+[A-Z]?)\b", re.IGNORECASE)),
    (LegalHierarchyLevel.APPENDIX,   re.compile(r"^\s*APPENDIX\s+([IVXLCDM]+|\d+|[A-Z])\b", re.IGNORECASE)),
    (LegalHierarchyLevel.EXHIBIT,    re.compile(r"^\s*EXHIBIT\s+([A-Z\d]+)\b", re.IGNORECASE)),
    (LegalHierarchyLevel.SECTION,    re.compile(r"^\s*(?:SECTION|SEC\.?)\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE)),
    (LegalHierarchyLevel.SECTION,    re.compile(r"^\s*(\d+)\.\s+[A-Z]")),   # "73. Compensation..."
    (LegalHierarchyLevel.SUBSECTION, re.compile(r"^\s*(\d+\.\d+(?:\.\d+)*)\s+")),  # "73.2 ..."
    (LegalHierarchyLevel.CLAUSE,     re.compile(r"^\s*\(([a-z]{1,3}|[ivx]+|\d+)\)\s")),  # "(a) ...", "(i) ..."
]

_LEGAL_ID_RE = re.compile(
    r"(?:Section|Sec\.|S\.|Article|Art\.|Chapter|Ch\.|Schedule|Sch\.|Appendix|Clause|Order)\s+"
    r"(\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))*|[IVXLCDM]+(?:\.\d+)*)",
    re.IGNORECASE,
)


def detect_legal_id(line: str) -> tuple[LegalHierarchyLevel, str | None]:
    """
    Check whether a line starts a new legal section.
    Returns (hierarchy_level, legal_id) or (UNKNOWN, None).
    """
    for level, pattern in _PATTERNS:
        m = pattern.match(line)
        if m:
            return level, m.group(1) if m.lastindex else None
    return LegalHierarchyLevel.UNKNOWN, None


def extract_cross_references(text: str) -> list[str]:
    """
    Extract all explicit legal cross-references from a block of text.
    Returns a list of raw reference strings (e.g. "Section 73(2)").
    """
    return _LEGAL_ID_RE.findall(text)


def refine_document_structure(doc: Document) -> Document:
    """
    Walk a Document's flat paragraphs and split them into
    proper LegalSection hierarchy based on detected legal identifiers.
    Operates in-place (modifies doc.sections).
    """
    if not doc.sections:
        return doc

    new_sections: list[LegalSection] = []

    for flat_sec in doc.sections:
        # Combine all paragraph texts for analysis
        all_paragraphs = flat_sec.paragraphs.copy()

        if not all_paragraphs:
            new_sections.append(flat_sec)
            continue

        current_section: LegalSection | None = None
        # Preserve original section as root if it has a real heading
        root_sec = LegalSection(
            hierarchy_level=flat_sec.hierarchy_level,
            legal_id=flat_sec.legal_id,
            heading=flat_sec.heading,
            page_start=flat_sec.page_start,
            page_end=flat_sec.page_end,
            tables=flat_sec.tables,
        )

        for para in all_paragraphs:
            first_line = para.text.split("\n")[0].strip()
            level, legal_id = detect_legal_id(first_line)

            if level != LegalHierarchyLevel.UNKNOWN and legal_id is not None:
                # Start a new sub-section
                new_sub = LegalSection(
                    hierarchy_level=level,
                    legal_id=legal_id,
                    heading=first_line[:120],
                    page_start=para.page_number,
                    page_end=para.page_number,
                )
                new_sub.paragraphs.append(para)

                # Extract cross-refs from paragraph
                xrefs = extract_cross_references(para.text)
                new_sub.cross_references.extend(xrefs)

                root_sec.subsections.append(new_sub)
                current_section = new_sub
            else:
                # Belongs to current section or root
                xrefs = extract_cross_references(para.text)
                if current_section is not None:
                    current_section.paragraphs.append(para)
                    current_section.cross_references.extend(xrefs)
                    if para.page_number:
                        current_section.page_end = para.page_number
                else:
                    root_sec.paragraphs.append(para)

        new_sections.append(root_sec)

    doc.sections = new_sections
    logger.debug(
        "Structure refinement: document %s → %d top sections, %d subsections",
        doc.metadata.document_id,
        len(doc.sections),
        sum(len(s.subsections) for s in doc.sections),
    )
    return doc
