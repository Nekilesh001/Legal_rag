"""
Parser base class and PDF parser using PyMuPDF (fitz).
Preserves page boundaries, text, and detects scanned PDFs.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import fitz  # PyMuPDF

from legal_rag.models.document import (
    Document,
    DocumentMetadata,
    ExtractionMethod,
    LegalHierarchyLevel,
    LegalSection,
    PageContent,
    Paragraph,
    Table,
    TableCell,
)

logger = logging.getLogger(__name__)

SCANNED_WORDS_PER_PAGE_THRESHOLD = 20  # overridden by config


class BaseParser(ABC):
    """Abstract base for all parsers. Every parser produces a Document."""

    @abstractmethod
    def can_parse(self, path: Path) -> bool: ...

    @abstractmethod
    def parse(self, path: Path, metadata: DocumentMetadata) -> Document: ...


def _extract_paragraphs_from_page(page_text: str, page_number: int) -> list[Paragraph]:
    """
    Split page text into logical paragraph blocks.
    Ensures mid-page legal headers (Section, Chapter, Part, Schedule)
    are separated so structure extraction detects them cleanly.
    """
    if not page_text.strip():
        return []

    from legal_rag.structure.extractor import detect_legal_id
    from legal_rag.models.document import LegalHierarchyLevel

    raw_lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    paragraphs: list[Paragraph] = []
    current_lines: list[str] = []

    for line in raw_lines:
        level, legal_id = detect_legal_id(line)
        if level != LegalHierarchyLevel.UNKNOWN and legal_id is not None and current_lines:
            # Flush existing paragraph buffer before heading line
            text_block = "\n".join(current_lines).strip()
            if text_block:
                paragraphs.append(Paragraph(text=text_block, page_number=page_number))
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        text_block = "\n".join(current_lines).strip()
        if text_block:
            paragraphs.append(Paragraph(text=text_block, page_number=page_number))

    return paragraphs


class PDFParser(BaseParser):
    """
    Extracts text from native (text-based) PDFs using PyMuPDF.
    Preserves page boundaries and extracts paragraph/line blocks.
    Does NOT perform OCR — scanned pages are flagged and left empty.
    """

    def __init__(self, words_per_page_threshold: int = SCANNED_WORDS_PER_PAGE_THRESHOLD) -> None:
        self.words_per_page_threshold = words_per_page_threshold

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def parse(self, path: Path, metadata: DocumentMetadata) -> Document:
        doc = Document(metadata=metadata)

        try:
            pdf = fitz.open(str(path))
        except Exception as e:
            logger.error("Cannot open PDF %s: %s", path.name, e)
            return doc

        metadata.page_count = len(pdf)
        pages_need_ocr: list[int] = []
        all_page_contents: list[PageContent] = []
        all_paragraphs: list[Paragraph] = []

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text("text")  # plain text, preserves layout order
            words = text.split()
            word_count = len(words)
            images = page.get_images(full=False)

            is_scanned = word_count < self.words_per_page_threshold and len(images) > 0
            if is_scanned:
                pages_need_ocr.append(page_num + 1)
                logger.debug(
                    "Page %d of %s appears scanned (words=%d, images=%d)",
                    page_num + 1, path.name, word_count, len(images),
                )

            # Extract tables on this page
            tables: list[Table] = []
            try:
                tab_finder = page.find_tables()
                for t in tab_finder.tables:
                    cells: list[TableCell] = []
                    raw_rows: list[list[str]] = t.extract()
                    for r_idx, row in enumerate(raw_rows):
                        for c_idx, cell_text in enumerate(row):
                            cells.append(TableCell(row=r_idx, col=c_idx, text=cell_text or ""))
                    tbl = Table(
                        page_number=page_num + 1,
                        rows=len(raw_rows),
                        cols=len(raw_rows[0]) if raw_rows else 0,
                        cells=cells,
                    )
                    tables.append(tbl)
            except Exception:
                pass  # table extraction is best-effort

            pc = PageContent(
                page_number=page_num + 1,
                text=text,
                word_count=word_count,
                image_count=len(images),
                extraction_method=(
                    ExtractionMethod.UNKNOWN if is_scanned else ExtractionMethod.NATIVE_PDF
                ),
            )
            all_page_contents.append(pc)

            if not is_scanned and text.strip():
                page_paras = _extract_paragraphs_from_page(text, page_num + 1)
                all_paragraphs.extend(page_paras)

        pdf.close()
        doc.pages = all_page_contents

        # Determine overall extraction method
        all_scanned = all(
            p.extraction_method == ExtractionMethod.UNKNOWN for p in all_page_contents
        ) and len(all_page_contents) > 0

        if all_scanned:
            metadata.extraction_method = ExtractionMethod.UNKNOWN  # will be OCR'd
            logger.info(
                "%s is fully scanned (%d pages need OCR)", path.name, len(all_page_contents)
            )
        elif pages_need_ocr:
            metadata.extraction_method = ExtractionMethod.NATIVE_PDF  # partially
            logger.info(
                "%s has %d scanned pages out of %d",
                path.name, len(pages_need_ocr), len(all_page_contents),
            )
        else:
            metadata.extraction_method = ExtractionMethod.NATIVE_PDF

        # Build section structure from extracted paragraphs
        if all_paragraphs:
            root_section = LegalSection(
                hierarchy_level=LegalHierarchyLevel.UNKNOWN,
                heading=metadata.title or path.stem,
                page_start=1,
                page_end=metadata.page_count,
                paragraphs=all_paragraphs,
            )
            doc.sections = [root_section]

        return doc


def is_scanned_pdf(doc: Document) -> bool:
    """Return True if a document has no extractable text (needs OCR)."""
    total_words = sum(p.word_count for p in doc.pages)
    return total_words == 0 and len(doc.pages) > 0
