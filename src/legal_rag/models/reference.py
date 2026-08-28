"""
Models for cross-references within and across legal documents.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ReferenceType(str, Enum):
    SECTION = "section"
    ARTICLE = "article"
    CHAPTER = "chapter"
    SCHEDULE = "schedule"
    CLAUSE = "clause"
    ORDER = "order"
    EXHIBIT = "exhibit"
    UNKNOWN = "unknown"


class CrossReference(BaseModel):
    """A single extracted cross-reference from a legal document."""
    ref_id: str = ""                    # normalized form e.g. "Section_73"
    raw_text: str = ""                  # original string e.g. "Sec. 73(2)"
    ref_type: ReferenceType = ReferenceType.UNKNOWN
    source_document_id: str = ""
    source_chunk_id: str = ""
    target_section_number: str | None = None
    target_document_id: str | None = None   # None if within same doc
    target_chunk_id: str | None = None      # filled after resolution
    resolution_confidence: float = 0.0      # 0.0 = unresolved, 1.0 = exact match
    is_resolved: bool = False


class ReferenceIndex(BaseModel):
    """
    Lightweight cross-reference index for the corpus.
    Maps source_chunk_id -> list of CrossReferences.
    """
    index: dict[str, list[CrossReference]] = Field(default_factory=dict)

    def add(self, ref: CrossReference) -> None:
        self.index.setdefault(ref.source_chunk_id, []).append(ref)

    def get_refs_for_chunk(self, chunk_id: str) -> list[CrossReference]:
        return self.index.get(chunk_id, [])

    def get_resolved(self) -> list[CrossReference]:
        out = []
        for refs in self.index.values():
            out.extend(r for r in refs if r.is_resolved)
        return out

    def get_unresolved(self) -> list[CrossReference]:
        out = []
        for refs in self.index.values():
            out.extend(r for r in refs if not r.is_resolved)
        return out
