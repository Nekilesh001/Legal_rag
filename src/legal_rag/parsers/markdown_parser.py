"""
Markdown parser using mistune AST.
Preserves heading hierarchy, paragraphs, bullet lists, numbered lists, and tables.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mistune

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
from legal_rag.parsers.pdf_parser import BaseParser

logger = logging.getLogger(__name__)


def _heading_to_level(level: int) -> LegalHierarchyLevel:
    """Map Markdown heading level to LegalHierarchyLevel."""
    mapping = {
        1: LegalHierarchyLevel.PART,      # H1 = top-level category
        2: LegalHierarchyLevel.CHAPTER,   # H2 = section group
        3: LegalHierarchyLevel.SECTION,   # H3 = individual clause
        4: LegalHierarchyLevel.SUBSECTION,
        5: LegalHierarchyLevel.CLAUSE,
        6: LegalHierarchyLevel.CLAUSE,
    }
    return mapping.get(level, LegalHierarchyLevel.UNKNOWN)


def _extract_text(token: dict[str, Any]) -> str:
    """Recursively extract text from a mistune token."""
    if token.get("type") == "text":
        return token.get("raw", "")
    children = token.get("children") or []
    return "".join(_extract_text(c) for c in children)


def _parse_table_token(token: dict[str, Any], page_num: int = 1) -> Table:
    """Convert a mistune table token into a Table model."""
    cells: list[TableCell] = []
    head = token.get("children", [{}])[0].get("children", [])
    body = token.get("children", [{}])[1].get("children", []) if len(token.get("children", [])) > 1 else []

    row_idx = 0
    for cell in head:
        text = _extract_text(cell)
        cells.append(TableCell(row=row_idx, col=len(cells), text=text))
    row_idx += 1
    for row in body:
        col_idx = 0
        for cell in row.get("children", []):
            text = _extract_text(cell)
            cells.append(TableCell(row=row_idx, col=col_idx, text=text))
            col_idx += 1
        row_idx += 1

    return Table(
        page_number=page_num,
        rows=row_idx,
        cols=max((c.col for c in cells), default=0) + 1,
        cells=cells,
    )


class MarkdownParser(BaseParser):
    """
    Parses Markdown files using mistune AST.
    Converts headings → LegalSections, paragraphs → Paragraphs.
    """

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in {".md", ".markdown"}

    def parse(self, path: Path, metadata: DocumentMetadata) -> Document:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.error("Cannot read Markdown %s: %s", path.name, e)
            return Document(metadata=metadata)

        metadata.extraction_method = ExtractionMethod.MARKDOWN
        metadata.page_count = 1

        # Parse into AST
        md = mistune.create_markdown(renderer=None)  # AST mode
        tokens: list[dict[str, Any]] = md(raw)  # type: ignore[assignment]

        doc = Document(metadata=metadata)

        # Single page representation for Markdown
        doc.pages = [
            PageContent(
                page_number=1,
                text=raw,
                word_count=len(raw.split()),
                extraction_method=ExtractionMethod.MARKDOWN,
            )
        ]

        # Build section hierarchy
        sections: list[LegalSection] = []
        current_stack: list[tuple[int, LegalSection]] = []  # (heading_level, section)

        def push_section(level: int, heading: str) -> LegalSection:
            sec = LegalSection(
                hierarchy_level=_heading_to_level(level),
                heading=heading,
                page_start=1,
                page_end=1,
            )
            # Pop stack back to parent level
            while current_stack and current_stack[-1][0] >= level:
                current_stack.pop()

            if current_stack:
                current_stack[-1][1].subsections.append(sec)
            else:
                sections.append(sec)

            current_stack.append((level, sec))
            return sec

        def current_section() -> LegalSection | None:
            return current_stack[-1][1] if current_stack else None

        # If there's no H1, create a root section from title
        has_h1 = any(t.get("type") == "heading" and t.get("attrs", {}).get("level") == 1 for t in tokens)
        if not has_h1:
            root = push_section(1, metadata.title or path.stem)

        for token in tokens:
            t_type = token.get("type", "")

            if t_type == "heading":
                level = token.get("attrs", {}).get("level", 1)
                heading_text = _extract_text(token)
                push_section(level, heading_text)

            elif t_type == "paragraph":
                text = _extract_text(token)
                if text.strip():
                    para = Paragraph(text=text.strip(), page_number=1)
                    sec = current_section()
                    if sec is not None:
                        sec.paragraphs.append(para)
                    else:
                        # Create implicit root section
                        root = push_section(1, metadata.title or path.stem)
                        root.paragraphs.append(para)

            elif t_type == "list":
                items = token.get("children", [])
                for item in items:
                    item_text = _extract_text(item)
                    if item_text.strip():
                        para = Paragraph(text=item_text.strip(), page_number=1, is_list_item=True)
                        sec = current_section()
                        if sec is not None:
                            sec.paragraphs.append(para)

            elif t_type == "table":
                tbl = _parse_table_token(token, page_num=1)
                sec = current_section()
                if sec is not None:
                    sec.tables.append(tbl)

            elif t_type == "block_code":
                code_text = token.get("raw", "")
                if code_text.strip():
                    para = Paragraph(text=f"```\n{code_text.strip()}\n```", page_number=1)
                    sec = current_section()
                    if sec is not None:
                        sec.paragraphs.append(para)

        doc.sections = sections
        logger.info(
            "Markdown parsed: %s — %d top-level sections, %d words",
            path.name, len(sections), len(raw.split()),
        )
        return doc
