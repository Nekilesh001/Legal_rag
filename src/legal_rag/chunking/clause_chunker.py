"""
Structure-aware, clause-aware parent-child chunker for legal documents.

Strategy:
1. Walk the document's LegalSection hierarchy.
2. Each top-level section becomes a candidate ParentChunk.
3. Subsections and clauses within it become ChildChunks.
4. Token budget enforced WITHOUT splitting legal sentences/clauses.
5. If a parent section is too large, it is subdivided at subsection boundaries.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Generator

from legal_rag.chunking.token_utils import count_tokens
from legal_rag.models.chunk import ChildChunk, ParentChunk, ChunkType
from legal_rag.models.document import (
    Document,
    ExtractionMethod,
    LegalHierarchyLevel,
    LegalSection,
)

logger = logging.getLogger(__name__)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _section_text(section: LegalSection, include_heading: bool = True) -> str:
    parts: list[str] = []
    if include_heading and section.heading:
        parts.append(section.heading)
    for p in section.paragraphs:
        if p.text.strip():
            parts.append(p.text.strip())
    for t in section.tables:
        row_text = t.to_text()
        if row_text.strip():
            parts.append(row_text.strip())
    return "\n\n".join(parts)


def _collect_cross_refs(section: LegalSection) -> list[str]:
    refs = list(section.cross_references)
    for sub in section.subsections:
        refs.extend(_collect_cross_refs(sub))
    return refs


class ClauseChunker:
    """
    Structure-aware parent-child chunker for legal documents.
    Produces ParentChunks and ChildChunks without merging them.
    """

    def __init__(
        self,
        parent_max_tokens: int = 1800,
        child_max_tokens: int = 350,
        overlap_tokens: int = 50,
        embedding_model: str = "",
    ) -> None:
        self.parent_max_tokens = parent_max_tokens
        self.child_max_tokens = child_max_tokens
        self.overlap_tokens = overlap_tokens
        self.embedding_model = embedding_model

    def chunk_document(
        self, doc: Document
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """
        Chunk a full document into parent and child chunks.
        Returns (parents, children).
        """
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []

        for section in doc.sections:
            sec_parents, sec_children = self._chunk_section(doc, section)
            parents.extend(sec_parents)
            children.extend(sec_children)

        logger.info(
            "Chunking %s: %d parents, %d children",
            doc.metadata.title or doc.metadata.document_id,
            len(parents),
            len(children),
        )
        return parents, children

    def _chunk_section(
        self, doc: Document, section: LegalSection
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """
        Process one top-level section into parent + children.
        If too large, subdivides at subsection boundaries or paragraph blocks.
        """
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        meta = doc.metadata

        # Build candidate parent text (full section)
        full_text = _section_text(section)
        full_token_count = count_tokens(full_text)

        if not full_text.strip():
            return parents, children

        if full_token_count <= self.parent_max_tokens:
            # The whole section fits in one parent
            parent = self._make_parent(doc, section, full_text)
            child_list = self._make_children_from_section(doc, section, parent.chunk_id)
            parent.child_ids = [c.chunk_id for c in child_list]
            parents.append(parent)
            children.extend(child_list)
        else:
            # Too large — split at subsection boundaries or paragraph blocks
            sub_groups = self._split_large_section(doc, section)
            for group_text, group_subs, page_start, page_end in sub_groups:
                partial_section = LegalSection(
                    section_id=section.section_id,
                    hierarchy_level=section.hierarchy_level,
                    legal_id=section.legal_id,
                    heading=section.heading,
                    page_start=page_start,
                    page_end=page_end,
                    paragraphs=section.paragraphs,
                    subsections=group_subs,
                )
                parent = self._make_parent(doc, partial_section, group_text)

                if group_subs:
                    child_list = self._make_children_from_subsections(
                        doc, group_subs, parent.chunk_id, meta
                    )
                else:
                    child_list = self._split_paragraphs_into_children(
                        doc, partial_section, parent.chunk_id
                    )

                # INVARIANT: Non-empty legal content MUST produce at least one child chunk
                if not child_list and group_text.strip():
                    child_list = [self._make_child(doc, partial_section, parent.chunk_id, group_text)]

                parent.child_ids = [c.chunk_id for c in child_list]
                parents.append(parent)
                children.extend(child_list)

        return parents, children

    def _split_large_section(
        self, doc: Document, section: LegalSection
    ) -> list[tuple[str, list[LegalSection], int | None, int | None]]:
        """
        Split a large section into groups of subsections that fit within parent_max_tokens.
        Returns list of (group_text, subsections, page_start, page_end).
        """
        groups: list[tuple[str, list[LegalSection], int | None, int | None]] = []
        current_subs: list[LegalSection] = []
        current_tokens = 0
        page_start = section.page_start

        for sub in section.subsections:
            sub_text = _section_text(sub)
            sub_tokens = count_tokens(sub_text)

            if current_tokens + sub_tokens > self.parent_max_tokens and current_subs:
                # Flush current group
                group_text = "\n\n".join(_section_text(s) for s in current_subs)
                page_end = current_subs[-1].page_end
                groups.append((group_text, current_subs, page_start, page_end))
                page_start = sub.page_start
                current_subs = []
                current_tokens = 0

            current_subs.append(sub)
            current_tokens += sub_tokens

        if current_subs:
            group_text = "\n\n".join(_section_text(s) for s in current_subs)
            page_end = current_subs[-1].page_end
            groups.append((group_text, current_subs, page_start, page_end))

        # Edge: no subsections at all (just one very long paragraph)
        if not groups:
            groups.append((_section_text(section), [], section.page_start, section.page_end))

        return groups

    def _make_parent(
        self, doc: Document, section: LegalSection, text: str
    ) -> ParentChunk:
        meta = doc.metadata
        return ParentChunk(
            document_id=meta.document_id,
            document_version_id=meta.document_version_id,
            document_title=meta.title,
            source_type=meta.source_type,
            category=meta.source_categories[0] if meta.source_categories else "unknown",
            section_number=section.legal_id,
            section_title=section.heading,
            page_start=section.page_start,
            page_end=section.page_end,
            extraction_method=meta.extraction_method,
            content_hash=_sha256_text(text),
            embedding_model=self.embedding_model,
            text=text,
            token_count=count_tokens(text),
        )

    def _make_children_from_section(
        self, doc: Document, section: LegalSection, parent_id: str
    ) -> list[ChildChunk]:
        """
        Create children from a section's subsections.
        If no subsections, split paragraphs into children.
        """
        if section.subsections:
            child_list = self._make_children_from_subsections(
                doc, section.subsections, parent_id, doc.metadata
            )
        else:
            child_list = self._split_paragraphs_into_children(doc, section, parent_id)

        sec_text = _section_text(section)
        if not child_list and sec_text.strip():
            child_list = [self._make_child(doc, section, parent_id, sec_text)]

        return child_list

    def _make_children_from_subsections(
        self,
        doc: Document,
        subsections: list[LegalSection],
        parent_id: str,
        meta,  # DocumentMetadata
    ) -> list[ChildChunk]:
        children: list[ChildChunk] = []
        for sub in subsections:
            text = _section_text(sub)
            token_count = count_tokens(text)

            if token_count <= self.child_max_tokens:
                children.append(self._make_child(doc, sub, parent_id, text))
            else:
                # Sub too large: split at its own paragraphs
                children.extend(self._split_paragraphs_into_children(doc, sub, parent_id))

        return children

    def _split_paragraphs_into_children(
        self, doc: Document, section: LegalSection, parent_id: str
    ) -> list[ChildChunk]:
        """
        Split section paragraphs into children at token boundaries.
        Never splits a paragraph mid-sentence.
        """
        children: list[ChildChunk] = []
        meta = doc.metadata
        buffer: list[str] = []
        buffer_tokens = 0
        page_start = section.page_start

        def _get_ocr_conf(doc: Document) -> float | None:
            if doc.pages:
                confs = [p.ocr_confidence for p in doc.pages if p.ocr_confidence is not None]
                if confs:
                    return round(sum(confs) / len(confs), 2)
            return None

        def flush_buffer(page_end: int | None) -> None:
            if not buffer:
                return
            text = "\n\n".join(buffer)
            child = ChildChunk(
                parent_id=parent_id,
                document_id=meta.document_id,
                document_version_id=meta.document_version_id,
                document_title=meta.title,
                source_type=meta.source_type,
                category=meta.source_categories[0] if meta.source_categories else "unknown",
                section_number=section.legal_id,
                section_title=section.heading,
                page_start=page_start,
                page_end=page_end,
                extraction_method=meta.extraction_method,
                ocr_confidence=_get_ocr_conf(doc),
                content_hash=_sha256_text(text),
                embedding_model=self.embedding_model,
                text=text,
                token_count=count_tokens(text),
                cross_references=_collect_cross_refs(section),
            )
            children.append(child)
            buffer.clear()

        for para in section.paragraphs:
            para_text = para.text.strip()
            if not para_text:
                continue
            para_tokens = count_tokens(para_text)

            if buffer_tokens + para_tokens > self.child_max_tokens and buffer:
                flush_buffer(para.page_number)
                buffer_tokens = 0
                page_start = para.page_number

            buffer.append(para_text)
            buffer_tokens += para_tokens

        flush_buffer(section.page_end)
        return children

    def _make_child(
        self, doc: Document, section: LegalSection, parent_id: str, text: str
    ) -> ChildChunk:
        meta = doc.metadata
        xrefs = _collect_cross_refs(section)
        confs = [p.ocr_confidence for p in doc.pages if p.ocr_confidence is not None] if doc.pages else []
        avg_ocr_conf = round(sum(confs) / len(confs), 2) if confs else None

        return ChildChunk(
            parent_id=parent_id,
            document_id=meta.document_id,
            document_version_id=meta.document_version_id,
            document_title=meta.title,
            source_type=meta.source_type,
            category=meta.source_categories[0] if meta.source_categories else "unknown",
            section_number=section.legal_id,
            section_title=section.heading,
            page_start=section.page_start,
            page_end=section.page_end,
            extraction_method=meta.extraction_method,
            ocr_confidence=avg_ocr_conf,
            content_hash=_sha256_text(text),
            embedding_model=self.embedding_model,
            text=text,
            token_count=count_tokens(text),
            cross_references=xrefs,
        )
