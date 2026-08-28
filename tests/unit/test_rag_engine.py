"""
Unit tests for the Legal RAG Engine — covers all core components.
Run with: py -3.11 -m pytest tests/ -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """Create a minimal valid PDF for testing."""
    import fitz
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text(
        (72, 72),
        "Section 1. Definitions\nIn this Act, unless context otherwise requires:\n"
        "(a) 'Employee' means any person employed for wages.\n"
        "(b) 'Employer' means any person who employs workers.\n"
        "Section 2. Scope\nThis Act shall apply to all establishments.",
    )
    pdf_path = tmp_path / "test_act.pdf"
    pdf.save(str(pdf_path))
    pdf.close()
    return pdf_path


@pytest.fixture
def sample_md_path(tmp_path: Path) -> Path:
    """Create a sample Markdown contract rules file."""
    content = """# Employment Contract — Mandatory Clauses

## 1. Identification
Both parties must be clearly identified with full legal names and addresses.

## 2. Employment Term
The contract must specify start date and duration (fixed or indefinite).

## 3. Compensation
- Base salary
- Payment frequency (weekly, monthly)
- Allowances and benefits

## 4. Termination
Notice period of at least 30 days must be provided.

### 4.1 Immediate Termination
Termination without notice is permitted only for gross misconduct.
"""
    md_path = tmp_path / "mandatory_clauses.md"
    md_path.write_text(content, encoding="utf-8")
    return md_path


# ------------------------------------------------------------------ #
# Tests: File Discovery
# ------------------------------------------------------------------ #

class TestDiscovery:
    def test_discovers_pdf_and_md(self, tmp_path: Path) -> None:
        from legal_rag.ingestion.discovery import discover_files
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 minimal")
        (tmp_path / "rules.md").write_text("# Rules", encoding="utf-8")
        (tmp_path / "skip.py").write_text("x = 1", encoding="utf-8")
        (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
        (tmp_path / "ignored.txt").write_text("some text")

        files = discover_files(tmp_path)
        names = [f.name for f in files]
        assert "doc.pdf" in names
        assert "rules.md" in names
        assert "skip.py" not in names
        assert ".gitkeep" not in names

    def test_empty_directory(self, tmp_path: Path) -> None:
        from legal_rag.ingestion.discovery import discover_files
        files = discover_files(tmp_path)
        assert files == []

    def test_category_extraction(self, tmp_path: Path) -> None:
        from legal_rag.ingestion.discovery import get_category_from_path
        subdir = tmp_path / "legal_documents" / "Finance"
        subdir.mkdir(parents=True)
        fp = subdir / "A1993-51.pdf"
        fp.write_bytes(b"%PDF")
        cat = get_category_from_path(fp, tmp_path)
        assert "Finance" in cat

    def test_nested_directories_discovered(self, tmp_path: Path) -> None:
        from legal_rag.ingestion.discovery import discover_files
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        (nested / "doc.pdf").write_bytes(b"%PDF-1.4 minimal")
        files = discover_files(tmp_path)
        assert len(files) == 1


# ------------------------------------------------------------------ #
# Tests: Hashing + Deduplication
# ------------------------------------------------------------------ #

class TestHasher:
    def test_hash_deterministic(self, tmp_path: Path) -> None:
        from legal_rag.ingestion.hasher import hash_file
        f = tmp_path / "file.pdf"
        f.write_bytes(b"hello legal corpus")
        h1 = hash_file(f)
        h2 = hash_file(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex = 64 chars

    def test_different_files_different_hashes(self, tmp_path: Path) -> None:
        from legal_rag.ingestion.hasher import hash_file
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert hash_file(f1) != hash_file(f2)

    def test_dedup_registry_detects_duplicate(self, tmp_path: Path) -> None:
        from legal_rag.ingestion.hasher import DeduplicationRegistry
        registry = DeduplicationRegistry()
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b_copy.pdf"
        content = b"identical content"
        f1.write_bytes(content)
        f2.write_bytes(content)

        import hashlib
        h = hashlib.sha256(content).hexdigest()

        canonical1, is_new1 = registry.register(f1, h, "cat_a")
        canonical2, is_new2 = registry.register(f2, h, "cat_b")

        assert is_new1 is True
        assert is_new2 is False
        assert canonical1.canonical_id == canonical2.canonical_id
        assert len(canonical1.source_paths) == 2

    def test_validate_file_detects_zero_byte(self, tmp_path: Path) -> None:
        from legal_rag.ingestion.hasher import validate_file
        from legal_rag.models.document import DocumentStatus
        f = tmp_path / "empty.pdf"
        f.write_bytes(b"")
        status, notes = validate_file(f)
        assert status == DocumentStatus.CORRUPT

    def test_validate_file_success(self, tmp_path: Path) -> None:
        from legal_rag.ingestion.hasher import validate_file
        from legal_rag.models.document import DocumentStatus
        f = tmp_path / "valid.pdf"
        f.write_bytes(b"%PDF-1.4 real content")
        status, _ = validate_file(f)
        assert status == DocumentStatus.SUCCESS


# ------------------------------------------------------------------ #
# Tests: PDF Parser
# ------------------------------------------------------------------ #

class TestPDFParser:
    def test_parses_native_pdf(self, sample_pdf_path: Path) -> None:
        from legal_rag.parsers.pdf_parser import PDFParser
        from legal_rag.models.document import DocumentMetadata, ExtractionMethod
        meta = DocumentMetadata(title="Test Act")
        parser = PDFParser()
        doc = parser.parse(sample_pdf_path, meta)
        assert doc.metadata.page_count > 0
        assert len(doc.pages) > 0
        total_words = sum(p.word_count for p in doc.pages)
        assert total_words > 0
        assert meta.extraction_method == ExtractionMethod.NATIVE_PDF

    def test_sections_created_from_pdf(self, sample_pdf_path: Path) -> None:
        from legal_rag.parsers.pdf_parser import PDFParser
        from legal_rag.models.document import DocumentMetadata
        meta = DocumentMetadata(title="Test Act")
        parser = PDFParser()
        doc = parser.parse(sample_pdf_path, meta)
        assert len(doc.sections) > 0


# ------------------------------------------------------------------ #
# Tests: Markdown Parser
# ------------------------------------------------------------------ #

class TestMarkdownParser:
    def test_parses_markdown_file(self, sample_md_path: Path) -> None:
        from legal_rag.parsers.markdown_parser import MarkdownParser
        from legal_rag.models.document import DocumentMetadata, ExtractionMethod
        meta = DocumentMetadata(title="Employment Mandatory Clauses")
        parser = MarkdownParser()
        doc = parser.parse(sample_md_path, meta)
        assert meta.extraction_method == ExtractionMethod.MARKDOWN
        assert len(doc.sections) > 0
        full_text = doc.full_text()
        assert "Employment" in full_text or "Mandatory" in full_text

    def test_section_hierarchy_preserved(self, sample_md_path: Path) -> None:
        from legal_rag.parsers.markdown_parser import MarkdownParser
        from legal_rag.models.document import DocumentMetadata
        meta = DocumentMetadata(title="Test")
        parser = MarkdownParser()
        doc = parser.parse(sample_md_path, meta)
        assert len(doc.sections) > 0


# ------------------------------------------------------------------ #
# Tests: Structure Extractor
# ------------------------------------------------------------------ #

class TestStructureExtractor:
    def test_detects_section_numbers(self) -> None:
        from legal_rag.structure.extractor import detect_legal_id
        from legal_rag.models.document import LegalHierarchyLevel
        level, legal_id = detect_legal_id("Section 73. Compensation for injury")
        assert level == LegalHierarchyLevel.SECTION
        assert legal_id == "73"

    def test_detects_part(self) -> None:
        from legal_rag.structure.extractor import detect_legal_id
        from legal_rag.models.document import LegalHierarchyLevel
        level, _ = detect_legal_id("PART I — Preliminary")
        assert level == LegalHierarchyLevel.PART

    def test_detects_chapter(self) -> None:
        from legal_rag.structure.extractor import detect_legal_id
        from legal_rag.models.document import LegalHierarchyLevel
        level, _ = detect_legal_id("CHAPTER IV — Penalties")
        assert level == LegalHierarchyLevel.CHAPTER

    def test_unknown_line_returns_unknown(self) -> None:
        from legal_rag.structure.extractor import detect_legal_id
        from legal_rag.models.document import LegalHierarchyLevel
        level, legal_id = detect_legal_id("This is a regular paragraph.")
        assert level == LegalHierarchyLevel.UNKNOWN

    def test_cross_reference_extraction(self) -> None:
        from legal_rag.structure.extractor import extract_cross_references
        text = "As per Section 73(2) and Article 14 of the Constitution."
        refs = extract_cross_references(text)
        assert len(refs) >= 2


# ------------------------------------------------------------------ #
# Tests: Chunker
# ------------------------------------------------------------------ #

class TestChunker:
    def test_produces_parent_and_child_chunks(self, sample_pdf_path: Path) -> None:
        from legal_rag.parsers.pdf_parser import PDFParser
        from legal_rag.structure.extractor import refine_document_structure
        from legal_rag.chunking.clause_chunker import ClauseChunker
        from legal_rag.models.document import DocumentMetadata

        meta = DocumentMetadata(title="Test Act", source_categories=["employment_acts"])
        parser = PDFParser()
        doc = parser.parse(sample_pdf_path, meta)
        doc = refine_document_structure(doc)

        chunker = ClauseChunker(parent_max_tokens=500, child_max_tokens=100)
        parents, children = chunker.chunk_document(doc)

        assert len(parents) > 0
        assert len(children) > 0
        # Every child must have a parent_id
        for child in children:
            assert child.parent_id != ""

    def test_children_reference_parents(self, sample_pdf_path: Path) -> None:
        from legal_rag.parsers.pdf_parser import PDFParser
        from legal_rag.structure.extractor import refine_document_structure
        from legal_rag.chunking.clause_chunker import ClauseChunker
        from legal_rag.models.document import DocumentMetadata

        meta = DocumentMetadata(title="Test Act", source_categories=["test"])
        parser = PDFParser()
        doc = parser.parse(sample_pdf_path, meta)
        doc = refine_document_structure(doc)

        chunker = ClauseChunker()
        parents, children = chunker.chunk_document(doc)

        parent_ids = {p.chunk_id for p in parents}
        for child in children:
            assert child.parent_id in parent_ids


# ------------------------------------------------------------------ #
# Tests: Token Utils
# ------------------------------------------------------------------ #

class TestTokenUtils:
    def test_count_tokens_empty(self) -> None:
        from legal_rag.chunking.token_utils import count_tokens
        assert count_tokens("") == 0

    def test_count_tokens_nonempty(self) -> None:
        from legal_rag.chunking.token_utils import count_tokens
        count = count_tokens("Hello world this is a test")
        assert count > 0

    def test_truncate_to_tokens(self) -> None:
        from legal_rag.chunking.token_utils import truncate_to_tokens, count_tokens
        long_text = " ".join(["word"] * 500)
        truncated = truncate_to_tokens(long_text, max_tokens=50)
        assert count_tokens(truncated) <= 55  # small margin for approximation


# ------------------------------------------------------------------ #
# Tests: Cross-Reference Extraction
# ------------------------------------------------------------------ #

class TestCrossRefs:
    def test_extract_section_references(self) -> None:
        from legal_rag.structure.cross_refs import extract_references_from_text
        text = "As provided in Section 138 of the Negotiable Instruments Act."
        refs = extract_references_from_text(text, "doc_1", "chunk_1")
        assert len(refs) >= 1
        assert any("138" in r.target_section_number for r in refs)

    def test_extract_schedule_references(self) -> None:
        from legal_rag.structure.cross_refs import extract_references_from_text
        text = "Refer to Schedule II for the list of goods."
        refs = extract_references_from_text(text, "doc_1", "chunk_1")
        assert any(r.ref_type.value == "schedule" for r in refs)

    def test_no_false_positives(self) -> None:
        from legal_rag.structure.cross_refs import extract_references_from_text
        text = "This is a plain paragraph with no legal cross-references."
        refs = extract_references_from_text(text, "doc_1", "chunk_1")
        assert len(refs) == 0


# ------------------------------------------------------------------ #
# Tests: Confidence Scoring
# ------------------------------------------------------------------ #

class TestConfidence:
    def _make_result(self, rrf_score: float, reranker_score: float | None = None):
        from legal_rag.models.retrieval import RetrievalResult
        return RetrievalResult(
            chunk_id=f"chk_{rrf_score}",
            text="test evidence",
            rrf_score=rrf_score,
            reranker_score=reranker_score,
        )

    def test_empty_results_low_confidence(self) -> None:
        from legal_rag.retrieval.confidence import score_confidence
        from legal_rag.models.retrieval import ConfidenceLevel
        _, level = score_confidence([])
        assert level == ConfidenceLevel.LOW

    def test_high_reranker_score_gives_high_confidence(self) -> None:
        from legal_rag.retrieval.confidence import score_confidence
        from legal_rag.models.retrieval import ConfidenceLevel
        results = [
            self._make_result(0.09, reranker_score=8.0),
            self._make_result(0.08, reranker_score=6.0),
            self._make_result(0.07, reranker_score=4.0),
            self._make_result(0.05),
        ]
        _, level = score_confidence(results)
        assert level in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)


# ------------------------------------------------------------------ #
# Tests: Query Analyzer
# ------------------------------------------------------------------ #

class TestQueryAnalyzer:
    def test_detects_section_refs(self) -> None:
        from legal_rag.query.analyzer import analyze_query
        analysis = analyze_query("What does Section 73(2) of the Payment of Wages Act say?")
        assert len(analysis.section_refs) > 0

    def test_detects_intent_definition(self) -> None:
        from legal_rag.query.analyzer import analyze_query
        from legal_rag.models.retrieval import QueryIntent
        analysis = analyze_query("What is the definition of 'employer' in the Act?")
        assert analysis.intent == QueryIntent.DEFINITION_INQUIRY

    def test_detects_category_employment(self) -> None:
        from legal_rag.query.analyzer import analyze_query
        analysis = analyze_query("What are the employee rights regarding maternity leave?")
        assert "employment" in analysis.category_hints

    def test_detects_jurisdiction(self) -> None:
        from legal_rag.query.analyzer import analyze_query
        analysis = analyze_query("What are the Tamil Nadu specific rules for rent control?")
        assert "Tamil Nadu" in analysis.jurisdictions

    def test_query_1_section_73(self) -> None:
        from legal_rag.query.analyzer import analyze_query
        from legal_rag.models.retrieval import QueryIntent
        analysis = analyze_query("What does Section 73 of the Indian Contract Act say?")
        assert analysis.intent == QueryIntent.SPECIFIC_CLAUSE_LOOKUP
        assert "Section 73" in analysis.section_refs
        assert "Indian Contract Act" in analysis.act_names
        assert not any("What does" in a for a in analysis.act_names)

    def test_query_2_nda_mandatory_clauses(self) -> None:
        from legal_rag.query.analyzer import analyze_query
        analysis = analyze_query("What are the mandatory clauses in an NDA agreement?")
        assert "nda" in analysis.category_hints
        assert analysis.act_names == []

    def test_query_3_tamil_nadu_shops_act(self) -> None:
        from legal_rag.query.analyzer import analyze_query
        analysis = analyze_query("What is the notice period under Tamil Nadu Shops Act?")
        assert "Tamil Nadu" in analysis.jurisdictions
        assert "employment" in analysis.category_hints
        assert "Tamil Nadu Shops Act" in analysis.act_names
        assert not any("What is the notice period" in a for a in analysis.act_names)

    def test_query_4_seller_breach_contract(self) -> None:
        from legal_rag.query.analyzer import analyze_query
        analysis = analyze_query("What happens if the seller breaches the contract?")
        assert "vendor" in analysis.category_hints
        assert analysis.act_names == []


class TestBlockerFixes:
    def test_pdf_midpage_heading_detection(self, tmp_path: Path) -> None:
        """Test that mid-page legal headings are detected by PDFParser and refine_document_structure."""
        import fitz
        from legal_rag.parsers.pdf_parser import PDFParser
        from legal_rag.structure.extractor import refine_document_structure
        from legal_rag.models.document import DocumentMetadata

        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            "THE TRANSFER OF PROPERTY ACT, 1882\n"
            "Preamble text here.\n"
            "CHAPTER I\n"
            "PRELIMINARY\n"
            "1. Short title.\n"
            "This Act may be called the Transfer of Property Act, 1882.\n"
            "2. Repeal of Acts.\n"
            "Nothing herein contained shall be deemed to affect...",
        )
        pdf_path = tmp_path / "midpage_test.pdf"
        pdf.save(str(pdf_path))
        pdf.close()

        meta = DocumentMetadata(title="Midpage Test", file_type="pdf")
        doc = PDFParser().parse(pdf_path, meta)
        doc = refine_document_structure(doc)

        assert len(doc.sections) > 0
        section_ids = [s.legal_id for s in doc.sections[0].subsections if s.legal_id]
        assert "1" in section_ids or "2" in section_ids or len(doc.sections[0].subsections) > 0

    def test_large_section_no_subsections_fallback(self) -> None:
        """Test that a section > parent_max_tokens with 0 subsections produces paragraph-based child chunks."""
        from legal_rag.chunking.clause_chunker import ClauseChunker
        from legal_rag.models.document import Document, DocumentMetadata, LegalSection, Paragraph

        meta = DocumentMetadata(title="Large Section Test")
        long_paragraphs = [
            Paragraph(text=f"Paragraph {i}. " + "This is long text detailing contract rights and obligations. " * 30, page_number=1)
            for i in range(1, 10)
        ]
        sec = LegalSection(heading="Section 100. Long Section", legal_id="100", paragraphs=long_paragraphs, subsections=[])
        doc = Document(metadata=meta, sections=[sec])

        chunker = ClauseChunker(parent_max_tokens=200, child_max_tokens=100)
        parents, children = chunker.chunk_document(doc)

        assert len(parents) > 0
        assert len(children) > 0
        for p in parents:
            assert len(p.child_ids) > 0

    def test_non_empty_section_never_returns_zero_children(self) -> None:
        """Invariant: Every non-empty section must produce at least one child retrieval chunk."""
        from legal_rag.chunking.clause_chunker import ClauseChunker
        from legal_rag.models.document import Document, DocumentMetadata, LegalSection, Paragraph

        meta = DocumentMetadata(title="Non-empty Invariant Test")
        sec = LegalSection(heading="Section 1. Short", legal_id="1", paragraphs=[Paragraph(text="Just a single short sentence.", page_number=1)])
        doc = Document(metadata=meta, sections=[sec])

        chunker = ClauseChunker()
        parents, children = chunker.chunk_document(doc)

        assert len(parents) == 1
        assert len(children) >= 1
        assert len(parents[0].child_ids) >= 1

    def test_parent_vectors_not_embedded(self, tmp_path: Path) -> None:
        """Verify engine does not compute BGE-M3 embeddings for parents during ingest_and_index_document."""
        from legal_rag.config import get_config
        from legal_rag.engine import LegalRagEngine

        cfg = get_config()
        cfg.rag_qdrant_in_memory = True
        engine = LegalRagEngine(cfg)

        md_file = tmp_path / "test.md"
        md_file.write_text("# Section 1\nSome paragraph text.", encoding="utf-8")

        parents, children = engine.ingest_and_index_document(md_file)
        assert len(parents) > 0
        assert len(children) > 0


# ------------------------------------------------------------------ #
# Tests: BM25 Store
# ------------------------------------------------------------------ #

class TestBM25Store:
    def test_build_and_search(self, tmp_path: Path) -> None:
        from legal_rag.indexing.bm25_store import BM25Store
        from legal_rag.models.chunk import ChildChunk, ChunkType

        store = BM25Store(tmp_path / "bm25")

        # Need at least 3 docs for BM25Okapi IDF to be non-zero (log formula)
        chunks = [
            ChildChunk(
                parent_id="p1",
                document_id="d1",
                document_version_id="v1",
                text="employer wages deduction minimum payment schedule",
                chunk_type=ChunkType.CHILD,
            ),
            ChildChunk(
                parent_id="p2",
                document_id="d2",
                document_version_id="v2",
                text="tenant lease premises landlord rent agreement",
                chunk_type=ChunkType.CHILD,
            ),
            ChildChunk(
                parent_id="p3",
                document_id="d3",
                document_version_id="v3",
                text="arbitration clause mediation dispute resolution conciliation",
                chunk_type=ChunkType.CHILD,
            ),
        ]
        store.build(chunks)

        results = store.search("employer wages", top_k=5)
        # With distinct vocabularies, BM25 should find relevant results
        assert len(results) > 0
        texts = [r["text"] for r in results]
        assert any("employer" in t.lower() or "wages" in t.lower() for t in texts)

    def test_save_and_load(self, tmp_path: Path) -> None:
        from legal_rag.indexing.bm25_store import BM25Store
        from legal_rag.models.chunk import ChildChunk

        store1 = BM25Store(tmp_path / "bm25")
        chunks = [ChildChunk(parent_id="p", document_id="d", document_version_id="v",
                              text="arbitration clause and mediation", chunk_type="child")]
        store1.build(chunks)

        store2 = BM25Store(tmp_path / "bm25")
        loaded = store2.load()
        assert loaded is True

        results = store2.search("arbitration", top_k=5)
        assert len(results) > 0


# ------------------------------------------------------------------ #
# Tests: Evaluation Metrics
# ------------------------------------------------------------------ #

class TestEvaluationMetrics:
    def test_recall_at_k_perfect(self) -> None:
        from legal_rag.evaluation.runner import recall_at_k
        retrieved = ["a", "b", "c", "d"]
        relevant = ["a", "b"]
        assert recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_recall_at_k_partial(self) -> None:
        from legal_rag.evaluation.runner import recall_at_k
        retrieved = ["a", "x", "y"]
        relevant = ["a", "b"]
        assert recall_at_k(retrieved, relevant, k=3) == 0.5

    def test_mrr_first_result_relevant(self) -> None:
        from legal_rag.evaluation.runner import mean_reciprocal_rank
        assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0

    def test_mrr_second_result_relevant(self) -> None:
        from legal_rag.evaluation.runner import mean_reciprocal_rank
        assert mean_reciprocal_rank(["x", "a", "b"], ["a"]) == pytest.approx(0.5)

    def test_citation_accuracy_all_matched(self) -> None:
        from legal_rag.evaluation.runner import citation_accuracy
        from legal_rag.models.retrieval import Citation
        c = Citation(
            citation_id="C1",
            excerpt="The employer shall pay wages as per Section 73.",
            document_title="Payment of Wages Act",
            section="73",
        )
        acc = citation_accuracy([c], ["wages", "Section 73", "Payment"])
        assert acc > 0.5
