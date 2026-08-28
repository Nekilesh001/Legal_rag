"""
Chunk models — ParentChunk and ChildChunk.
These are the units that get embedded and indexed.
"""
from __future__ import annotations

import uuid
from enum import Enum

from pydantic import BaseModel, Field

from legal_rag.models.document import ExtractionMethod


class ChunkType(str, Enum):
    PARENT = "parent"
    CHILD = "child"


class BaseChunk(BaseModel):
    """Common fields for both parent and child chunks."""
    chunk_id: str = Field(default_factory=lambda: f"chk_{uuid.uuid4().hex[:12]}")
    document_id: str
    document_version_id: str
    document_title: str | None = None
    source_type: str = "unknown"        # "statute", "rulebook", "case_law", etc.
    category: str = "unknown"           # folder category
    legal_domain: str | None = None
    section_number: str | None = None   # e.g. "73(2)"
    section_title: str | None = None
    subsection: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    extraction_method: ExtractionMethod = ExtractionMethod.UNKNOWN
    content_hash: str = ""              # SHA-256 of chunk text
    chunk_type: ChunkType
    embedding_model: str = ""
    text: str = ""

    # Contextual enrichment (optional, stored separately)
    context_prefix: str | None = None
    embedding_text: str | None = None   # = context_prefix + text if enrichment enabled


class ParentChunk(BaseChunk):
    """
    A larger coherent legal unit (whole section / article / clause group).
    Used for context expansion after child retrieval.
    """
    chunk_type: ChunkType = ChunkType.PARENT
    child_ids: list[str] = Field(default_factory=list)
    token_count: int = 0


class ChildChunk(BaseChunk):
    """
    A smaller retrieval unit (subsection / clause).
    These are the units indexed in the vector store and BM25.
    """
    chunk_type: ChunkType = ChunkType.CHILD
    parent_id: str = ""
    token_count: int = 0
    ocr_confidence: float | None = None
    cross_references: list[str] = Field(default_factory=list)
