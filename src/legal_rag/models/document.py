"""
Pydantic models for all document-related structures.
These are the canonical internal representations — all parsers produce these.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
# Enums
# ------------------------------------------------------------------ #

class ExtractionMethod(str, Enum):
    NATIVE_PDF = "native_pdf"
    OCR = "ocr"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class DocumentStatus(str, Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    DUPLICATE = "duplicate"
    OCR_REQUIRED = "ocr_required"
    OCR_FAILED = "ocr_failed"
    PARSE_FAILED = "parse_failed"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"
    CORRUPT = "corrupt"


class LegalHierarchyLevel(str, Enum):
    PART = "part"
    CHAPTER = "chapter"
    ARTICLE = "article"
    SECTION = "section"
    SUBSECTION = "subsection"
    CLAUSE = "clause"
    SCHEDULE = "schedule"
    APPENDIX = "appendix"
    EXHIBIT = "exhibit"
    PREAMBLE = "preamble"
    PARAGRAPH = "paragraph"
    UNKNOWN = "unknown"


# ------------------------------------------------------------------ #
# Sub-models
# ------------------------------------------------------------------ #

class TableCell(BaseModel):
    row: int
    col: int
    text: str


class Table(BaseModel):
    """Extracted table from a document."""
    table_id: str = Field(default_factory=lambda: f"tbl_{uuid.uuid4().hex[:8]}")
    page_number: int
    rows: int
    cols: int
    cells: list[TableCell] = Field(default_factory=list)
    raw_text: str = ""  # flattened fallback

    def to_text(self) -> str:
        if self.raw_text:
            return self.raw_text
        rows: dict[int, list[str]] = {}
        for c in self.cells:
            rows.setdefault(c.row, []).append(c.text)
        return "\n".join(" | ".join(r) for r in rows.values())


class Paragraph(BaseModel):
    """A paragraph of text within a section."""
    paragraph_id: str = Field(default_factory=lambda: f"par_{uuid.uuid4().hex[:8]}")
    text: str
    page_number: int | None = None
    is_list_item: bool = False
    list_marker: str | None = None  # "•", "1.", "(a)", etc.


class LegalSection(BaseModel):
    """
    A hierarchical legal unit — Section, Article, Chapter, Schedule, etc.
    Sections can nest (subsections are child LegalSections).
    """
    section_id: str = Field(default_factory=lambda: f"sec_{uuid.uuid4().hex[:8]}")
    hierarchy_level: LegalHierarchyLevel = LegalHierarchyLevel.SECTION
    legal_id: str | None = None        # e.g. "73", "73(2)", "III", "Schedule II"
    heading: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    paragraphs: list[Paragraph] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    subsections: list["LegalSection"] = Field(default_factory=list)
    cross_references: list[str] = Field(default_factory=list)  # raw reference strings

    def full_text(self) -> str:
        """Return concatenated text of all paragraphs and tables."""
        parts: list[str] = []
        if self.heading:
            parts.append(self.heading)
        for p in self.paragraphs:
            parts.append(p.text)
        for t in self.tables:
            parts.append(t.to_text())
        for sub in self.subsections:
            parts.append(sub.full_text())
        return "\n".join(parts)


class PageContent(BaseModel):
    """Raw content extracted from a single PDF page."""
    page_number: int
    text: str
    word_count: int = 0
    image_count: int = 0
    extraction_method: ExtractionMethod = ExtractionMethod.UNKNOWN
    ocr_confidence: float | None = None  # 0-100, None if not OCR


# ------------------------------------------------------------------ #
# Document-level metadata
# ------------------------------------------------------------------ #

class DocumentMetadata(BaseModel):
    """
    Metadata attached to every document.
    Separated into: directly extracted, inferred (high confidence), NOT guessed.
    """
    # --- Directly extracted ---
    document_id: str = Field(default_factory=lambda: f"doc_{uuid.uuid4().hex[:12]}")
    title: str | None = None
    source_type: str = "unknown"         # "statute", "case_law", "rulebook", etc.
    file_type: str = "unknown"           # "pdf", "markdown"
    content_hash: str = ""               # SHA-256 of file bytes
    source_paths: list[str] = Field(default_factory=list)
    source_categories: list[str] = Field(default_factory=list)
    source_file_names: list[str] = Field(default_factory=list)
    page_count: int = 0
    extraction_method: ExtractionMethod = ExtractionMethod.UNKNOWN
    ingestion_timestamp: datetime = Field(default_factory=datetime.utcnow)

    # --- Inferred with high confidence from content ---
    jurisdiction: str | None = None       # e.g. "India", "Tamil Nadu"
    enactment_year: int | None = None

    # --- Version tracking ---
    document_family_id: str | None = None
    document_version_id: str = Field(default_factory=lambda: f"ver_{uuid.uuid4().hex[:8]}")
    effective_from: str | None = None     # kept as string; not guessed
    effective_to: str | None = None
    status: str = "active"               # "active", "superseded", "unknown"

    # NOTE: We do NOT infer: parties, governing_law, payment_terms, or clause-level
    # legal facts from filenames alone.


# ------------------------------------------------------------------ #
# Top-level Document
# ------------------------------------------------------------------ #

class Document(BaseModel):
    """
    Canonical internal representation of any legal document.
    All parsers produce this before chunking.
    """
    metadata: DocumentMetadata
    pages: list[PageContent] = Field(default_factory=list)
    sections: list[LegalSection] = Field(default_factory=list)

    def full_text(self) -> str:
        return "\n\n".join(s.full_text() for s in self.sections)

    def word_count(self) -> int:
        return sum(p.word_count for p in self.pages)


# ------------------------------------------------------------------ #
# Ingestion Report
# ------------------------------------------------------------------ #

class FileIngestionResult(BaseModel):
    """Result for a single file during the ingestion run."""
    file_path: str
    status: DocumentStatus
    document_id: str | None = None
    content_hash: str | None = None
    canonical_doc_id: str | None = None   # if duplicate, points to canonical
    notes: str = ""
    error: str | None = None
    processing_time_seconds: float | None = None


class IngestionReport(BaseModel):
    """Summary report for an entire ingestion run."""
    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:8]}")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    corpus_path: str = ""
    total_files_discovered: int = 0
    results: list[FileIngestionResult] = Field(default_factory=list)

    @property
    def summary(self) -> dict[str, int]:
        from collections import Counter
        counts: Counter[str] = Counter(r.status.value for r in self.results)
        return dict(counts)
