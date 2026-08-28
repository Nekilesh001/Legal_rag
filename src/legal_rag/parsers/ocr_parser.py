"""
OCR pipeline for scanned PDFs using pytesseract.
Isolated module — only called for documents flagged as scanned.
"""
from __future__ import annotations

import logging
from pathlib import Path

from legal_rag.models.document import (
    Document,
    DocumentMetadata,
    ExtractionMethod,
    LegalHierarchyLevel,
    LegalSection,
    PageContent,
    Paragraph,
)

logger = logging.getLogger(__name__)


def _check_tesseract() -> bool:
    """
    Configures pytesseract to locate the tesseract executable.
    Resolution order:
    1. config.rag_tesseract_cmd (from .env / config)
    2. Standard Windows install locations
    3. PATH lookup (shutil.which("tesseract"))
    """
    import shutil
    import pytesseract
    from legal_rag.config import get_config

    candidates: list[str] = []
    try:
        cfg = get_config()
        if cfg.rag_tesseract_cmd:
            candidates.append(cfg.rag_tesseract_cmd)
    except Exception:
        pass

    candidates.extend([
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ])

    for cmd in candidates:
        if cmd and Path(cmd).is_file():
            pytesseract.pytesseract.tesseract_cmd = cmd
            try:
                pytesseract.get_tesseract_version()
                logger.info("Tesseract configured at: %s", cmd)
                return True
            except Exception as e:
                logger.debug("Failed to invoke tesseract at %s: %s", cmd, e)

    # Fall back to PATH lookup
    which_cmd = shutil.which("tesseract")
    if which_cmd:
        pytesseract.pytesseract.tesseract_cmd = which_cmd
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract found in PATH at: %s", which_cmd)
            return True
        except Exception:
            pass

    logger.error(
        "Tesseract OCR executable not found. "
        "Configured path: %s. Please verify installation or set RAG_TESSERACT_CMD.",
        candidates[0] if candidates else "N/A"
    )
    return False


def ocr_pdf(
    path: Path,
    metadata: DocumentMetadata,
    confidence_threshold: float = 60.0,
    dpi: int = 300,
) -> Document:
    """
    Perform OCR on every page of a scanned PDF.
    Uses pytesseract with image rendering via fitz.

    Returns a Document with page-level text and confidence scores.
    Marks extraction_method = "ocr" on every page.
    """
    import fitz  # PyMuPDF — for page rendering
    import pytesseract
    from PIL import Image

    doc = Document(metadata=metadata)
    metadata.extraction_method = ExtractionMethod.OCR

    if not _check_tesseract():
        logger.error(
            "Tesseract not found in PATH. Cannot OCR: %s", path.name
        )
        return doc

    try:
        pdf = fitz.open(str(path))
    except Exception as e:
        logger.error("Cannot open PDF for OCR %s: %s", path.name, e)
        return doc

    metadata.page_count = len(pdf)
    all_pages: list[PageContent] = []
    all_paragraphs: list[Paragraph] = []

    for page_num in range(len(pdf)):
        page = pdf[page_num]

        # Render page to image at given DPI
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)

        try:
            # Run OCR with confidence data
            ocr_data = pytesseract.image_to_data(
                img,
                output_type=pytesseract.Output.DICT,
                config="--oem 3 --psm 6",
            )
            words = [
                w for w, conf in zip(ocr_data["text"], ocr_data["conf"])
                if str(w).strip() and int(conf) > 0
            ]
            confidences = [
                int(c) for c, w in zip(ocr_data["conf"], ocr_data["text"])
                if str(w).strip() and int(c) > 0
            ]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            page_text = pytesseract.image_to_string(img, config="--oem 3 --psm 6")

        except Exception as e:
            logger.warning("OCR failed on page %d of %s: %s", page_num + 1, path.name, e)
            page_text = ""
            avg_conf = 0.0

        if avg_conf < confidence_threshold and page_text.strip():
            logger.warning(
                "Low OCR confidence %.1f%% on page %d of %s",
                avg_conf, page_num + 1, path.name,
            )

        pc = PageContent(
            page_number=page_num + 1,
            text=page_text,
            word_count=len(page_text.split()),
            extraction_method=ExtractionMethod.OCR,
            ocr_confidence=round(avg_conf, 2),
        )
        all_pages.append(pc)

        if page_text.strip():
            from legal_rag.parsers.pdf_parser import _extract_paragraphs_from_page
            page_paras = _extract_paragraphs_from_page(page_text, page_num + 1)
            all_paragraphs.extend(page_paras)

    pdf.close()
    doc.pages = all_pages

    if all_paragraphs:
        doc.sections = [
            LegalSection(
                hierarchy_level=LegalHierarchyLevel.UNKNOWN,
                heading=metadata.title or path.stem,
                page_start=1,
                page_end=metadata.page_count,
                paragraphs=all_paragraphs,
            )
        ]

    total_words = sum(p.word_count for p in all_pages)
    logger.info(
        "OCR complete: %s — %d pages, %d words extracted",
        path.name, len(all_pages), total_words,
    )
    return doc
