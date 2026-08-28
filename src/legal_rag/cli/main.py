"""
Typer CLI for the Legal RAG Engine.
Commands: ingest, query, inspect, status, serve
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

app = typer.Typer(
    name="rag",
    help="Legal RAG Engine — ingestion, querying, and inspection CLI",
    add_completion=False,
)
console = Console()

# ------------------------------------------------------------------ #
# Logging setup
# ------------------------------------------------------------------ #

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ------------------------------------------------------------------ #
# Commands
# ------------------------------------------------------------------ #

@app.command()
def ingest(
    corpus: Optional[str] = typer.Option(None, "--corpus", "-c", help="Corpus path override"),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Ingest a single file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    Run the full ingestion pipeline or ingest a single file.
    Discovery → Validation → Dedup → Parse → OCR → Chunk → Embed → Index
    """
    _setup_logging(verbose)

    from legal_rag.config import get_config
    from legal_rag.engine import LegalRagEngine

    config = get_config()
    engine = LegalRagEngine(config)

    if file:
        fp = Path(file)
        if not fp.exists():
            console.print(f"[red]File not found: {fp}[/red]")
            raise typer.Exit(1)

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      console=console) as progress:
            task = progress.add_task(f"Ingesting {fp.name}...", total=None)
            parents, children = engine.ingest_and_index_document(fp)
            progress.stop()

        console.print(Panel(
            f"[green][OK][/green] {fp.name}\n"
            f"  Parents: {len(parents)}\n"
            f"  Children: {len(children)}",
            title="Ingestion Complete",
        ))
    else:
        corpus_path = Path(corpus) if corpus else None

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      console=console) as progress:
            task = progress.add_task("Running full corpus ingestion...", total=None)
            report, _, _ = engine.run_ingestion(corpus_path)
            progress.stop()

        # Display summary table
        table = Table(title="Ingestion Report", show_header=True, header_style="bold magenta")
        table.add_column("Status", style="cyan")
        table.add_column("Count", justify="right")

        for status, count in report.summary.items():
            color = "green" if status == "success" else "yellow" if status in ("skipped", "duplicate") else "red"
            table.add_row(f"[{color}]{status}[/{color}]", str(count))

        console.print(table)

        # Save report
        report_path = config.reports_dir / f"{report.run_id}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2, default=str)
        console.print(f"\n[dim]Report saved: {report_path}[/dim]")


@app.command()
def query(
    q: str = typer.Argument(..., help="Legal research query"),
    category: Optional[str] = typer.Option(None, "--category", "-c",
                                             help="Filter by document category"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output raw JSON"),
) -> None:
    """Query the legal corpus and get a grounded answer with citations."""
    _setup_logging(verbose)

    from legal_rag.config import get_config
    from legal_rag.engine import LegalRagEngine

    config = get_config()
    engine = LegalRagEngine(config)

    filters = {"category": category} if category else None

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console) as progress:
        task = progress.add_task("Querying corpus...", total=None)
        response = engine.query(q, filters=filters)
        progress.stop()

    if json_output:
        console.print_json(json.dumps(response.model_dump(mode="json"), default=str))
        return

    # Pretty output
    conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(
        response.confidence.value, "white"
    )
    status_color = {"supported": "green", "partially_supported": "yellow"}.get(
        response.evidence_status.value, "red"
    )

    console.print(Panel(response.answer, title=f"[bold]Answer[/bold]", border_style="blue"))
    console.print(
        f"  Confidence: [{conf_color}]{response.confidence.value}[/{conf_color}]  |  "
        f"Evidence: [{status_color}]{response.evidence_status.value}[/{status_color}]  |  "
        f"Attempts: {response.retrieval_attempts}"
    )

    if response.citations:
        console.print("\n[bold]Citations:[/bold]")
        for c in response.citations:
            console.print(
                f"  [{c.citation_id}] {c.document_title} — {c.section or 'N/A'} "
                f"(p. {c.page or 'N/A'})"
            )
            if c.excerpt:
                console.print(f"       [dim]{c.excerpt[:150]}...[/dim]")


@app.command()
def inspect(
    corpus: Optional[str] = typer.Option(None, "--corpus", "-c", help="Corpus path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Inspect the corpus: discover and classify all files without modifying anything."""
    _setup_logging(verbose)
    from legal_rag.config import get_config
    from legal_rag.ingestion.discovery import discover_files, get_category_from_path
    from legal_rag.ingestion.hasher import hash_file, validate_file

    config = get_config()
    corpus_path = Path(corpus) if corpus else config.rag_corpus_path

    files = discover_files(corpus_path)

    table = Table(title=f"Corpus Inspection: {corpus_path}", show_header=True,
                  header_style="bold cyan")
    table.add_column("File", style="white")
    table.add_column("Category", style="dim")
    table.add_column("Size", justify="right")
    table.add_column("Valid?", justify="center")

    for f in files:
        status, _ = validate_file(f)
        size_kb = f.stat().st_size // 1024
        valid = "[green][OK][/green]" if status.value == "success" else f"[red]{status.value}[/red]"
        cat = get_category_from_path(f, corpus_path)
        table.add_row(f.name, cat, f"{size_kb} KB", valid)

    console.print(table)
    console.print(f"\n[bold]Total:[/bold] {len(files)} files discovered in {corpus_path}")


@app.command(name="inspect-chunks")
def inspect_chunks(
    corpus: Optional[str] = typer.Option(None, "--corpus", "-c", help="Corpus path override"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    Dry-run chunk inspection: discover → validate → dedup → parse → OCR → structure → xref → chunk.
    Runs WITHOUT embeddings, Qdrant indexing, BM25, or LLM generation.
    """
    _setup_logging(verbose)

    from legal_rag.config import get_config
    from legal_rag.ingestion.discovery import discover_files, get_category_from_path, get_source_domain
    from legal_rag.ingestion.hasher import hash_file, validate_file, DeduplicationRegistry
    from legal_rag.models.document import DocumentMetadata, DocumentStatus
    from legal_rag.parsers.markdown_parser import MarkdownParser
    from legal_rag.parsers.pdf_parser import PDFParser, is_scanned_pdf
    from legal_rag.parsers.ocr_parser import ocr_pdf, _check_tesseract
    from legal_rag.structure.extractor import refine_document_structure
    from legal_rag.structure.cross_refs import extract_references_from_text
    from legal_rag.chunking.clause_chunker import ClauseChunker

    config = get_config()
    corpus_path = Path(corpus) if corpus else config.rag_corpus_path

    console.print(Panel(f"[bold cyan]RAG DRY-RUN CHUNK INSPECTION[/bold cyan]\nCorpus: {corpus_path}", border_style="cyan"))

    files = discover_files(corpus_path)
    dedup_registry = DeduplicationRegistry()

    valid_count = 0
    dup_count = 0
    corrupt_count = 0
    unsupported_count = 0
    ocr_count = 0

    total_pages = 0
    total_sections = 0
    total_parents = 0
    total_children = 0

    doc_zero_parents = 0
    doc_zero_children = 0
    sections_zero_children = 0
    parents_zero_children = 0

    children_missing_parent_id = 0
    children_missing_doc_id = 0
    children_missing_page_meta = 0

    all_extracted_xrefs = []
    sample_docs_chunks = {}  # key -> dict of sample data

    pdf_parser = PDFParser(words_per_page_threshold=config.rag_ocr_words_per_page_threshold)
    md_parser = MarkdownParser()
    chunker = ClauseChunker(
        parent_max_tokens=config.rag_parent_max_tokens,
        child_max_tokens=config.rag_child_max_tokens,
        overlap_tokens=config.rag_chunk_overlap_tokens,
    )

    tesseract_available = _check_tesseract()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Processing corpus files...", total=len(files))

        for f in files:
            progress.update(task, description=f"Processing {f.name[:30]}...")
            v_status, v_notes = validate_file(f)

            if v_status == DocumentStatus.CORRUPT:
                corrupt_count += 1
                progress.advance(task)
                continue
            elif v_status == DocumentStatus.UNSUPPORTED:
                unsupported_count += 1
                progress.advance(task)
                continue
            elif v_status != DocumentStatus.SUCCESS:
                progress.advance(task)
                continue

            content_hash = hash_file(f)
            category = get_category_from_path(f, corpus_path)
            canonical, is_new = dedup_registry.register(f, content_hash, category)

            if not is_new:
                dup_count += 1
                progress.advance(task)
                continue

            valid_count += 1
            meta = DocumentMetadata(
                document_id=canonical.canonical_id,
                title=f.stem.replace("_", " ").title(),
                file_type=f.suffix.lower().lstrip("."),
                content_hash=content_hash,
                source_paths=canonical.source_paths,
                source_categories=canonical.source_categories,
                source_file_names=canonical.source_file_names,
            )

            # Parse
            ext = f.suffix.lower()
            if ext in {".md", ".markdown"}:
                doc = md_parser.parse(f, meta)
            else:
                doc = pdf_parser.parse(f, meta)
                if is_scanned_pdf(doc):
                    ocr_count += 1
                    if tesseract_available:
                        doc = ocr_pdf(f, meta, confidence_threshold=config.rag_ocr_confidence_threshold)

            total_pages += doc.metadata.page_count

            # Structure extraction
            doc = refine_document_structure(doc)
            total_sections += len(doc.sections)

            # Cross-reference extraction
            xrefs = extract_references_from_text(doc.full_text(), doc.metadata.document_id, "doc_root")
            all_extracted_xrefs.extend(xrefs)

            # Chunking
            parents, children = chunker.chunk_document(doc)
            total_parents += len(parents)
            total_children += len(children)

            if not parents:
                doc_zero_parents += 1
            if not children and doc.full_text().strip():
                doc_zero_children += 1

            for p in parents:
                if not p.child_ids:
                    parents_zero_children += 1

            for c in children:
                if not c.parent_id:
                    children_missing_parent_id += 1
                if not c.document_id:
                    children_missing_doc_id += 1
                if c.page_start is None:
                    children_missing_page_meta += 1

            # Keep sample chunks for key doc types
            if "mandatory_clauses.md" in f.name and "markdown_rulebook" not in sample_docs_chunks:
                sample_docs_chunks["markdown_rulebook"] = (doc, parents, children)
            elif f.name == "A1882-04.pdf" and "normal_statute" not in sample_docs_chunks:
                sample_docs_chunks["normal_statute"] = (doc, parents, children)
            elif f.name == "the_code_of_civil_procedure,_1908.pdf" and "large_statute" not in sample_docs_chunks:
                sample_docs_chunks["large_statute"] = (doc, parents, children)
            elif f.name == "satyam-final-judgment.pdf" and "case_law" not in sample_docs_chunks:
                sample_docs_chunks["case_law"] = (doc, parents, children)
            elif f.name == "G.O.72_Milk_and_Milk_Industry.pdf" and "ocr_doc" not in sample_docs_chunks:
                sample_docs_chunks["ocr_doc"] = (doc, parents, children)

            progress.advance(task)

    # ---------------------------------------------------------------- #
    # Print Inspection Report
    # ---------------------------------------------------------------- #

    corpus_table = Table(title="Corpus Ingestion Summary", show_header=True, header_style="bold green")
    corpus_table.add_column("Metric", style="cyan")
    corpus_table.add_column("Value", justify="right")
    corpus_table.add_row("Total Files Discovered", str(len(files)))
    corpus_table.add_row("Valid Canonical Files", str(valid_count))
    corpus_table.add_row("Duplicates Collapsed", str(dup_count))
    corpus_table.add_row("Corrupt / Blank (0-byte)", str(corrupt_count))
    corpus_table.add_row("OCR Scanned PDFs", str(ocr_count))
    corpus_table.add_row("OCR Engine Available?", "[green]YES (Tesseract)[/green]" if tesseract_available else "[yellow]NO[/yellow]")
    console.print(corpus_table)

    struct_table = Table(title="Structure & Chunk Invariants", show_header=True, header_style="bold magenta")
    struct_table.add_column("Metric", style="cyan")
    struct_table.add_column("Value", justify="right")
    struct_table.add_row("Total Document Pages", str(total_pages))
    struct_table.add_row("Total Extracted Legal Sections", str(total_sections))
    struct_table.add_row("Total Parent Chunks Created", str(total_parents))
    struct_table.add_row("Total Child Chunks Created", str(total_children))
    struct_table.add_row("Extracted Cross-References", str(len(all_extracted_xrefs)))

    # Invariants (Desired: ALL ZERO)
    c_zero_children = "[green]0[/green]" if doc_zero_children == 0 else f"[red]{doc_zero_children}[/red]"
    p_zero_children = "[green]0[/green]" if parents_zero_children == 0 else f"[red]{parents_zero_children}[/red]"
    missing_p_id = "[green]0[/green]" if children_missing_parent_id == 0 else f"[red]{children_missing_parent_id}[/red]"

    struct_table.add_row("Non-Empty Docs with 0 Children (INVARIANT)", c_zero_children)
    struct_table.add_row("Parents with 0 Children (INVARIANT)", p_zero_children)
    struct_table.add_row("Children Missing Parent ID (INVARIANT)", missing_p_id)
    console.print(struct_table)

    # Print 5 Representative Chunks
    console.print("\n[bold yellow]=== REPRESENTATIVE REAL CORPUS CHUNKS ===[/bold yellow]\n")

    samples_meta = [
        ("1. Markdown Rulebook", "markdown_rulebook"),
        ("2. Normal Statute PDF", "normal_statute"),
        ("3. Large Statute PDF", "large_statute"),
        ("4. Case Law PDF", "case_law"),
        ("5. OCR Document PDF", "ocr_doc"),
    ]

    for label, key in samples_meta:
        console.print(f"[bold underline cyan]{label}[/bold underline cyan]")
        if key in sample_docs_chunks:
            doc, parents, children = sample_docs_chunks[key]
            parent = parents[0] if parents else None
            child = children[0] if children else None

            p_id = parent.chunk_id if parent else "N/A"
            p_title = parent.section_title if parent else "N/A"
            p_toks = parent.token_count if parent else 0

            c_id = child.chunk_id if child else "N/A"
            c_toks = child.token_count if child else 0
            c_sec = (child.section_number or child.section_title) if child else "N/A"
            c_page = child.page_start if child else "N/A"
            c_method = child.extraction_method.value if child else "N/A"
            c_text = child.text[:300] if child else "N/A (Scanned PDF - OCR engine unavailable)"

            panel_content = (
                f"[bold]Document:[/bold] {doc.metadata.title} ({doc.metadata.file_type})\n"
                f"[bold]Parent ID:[/bold] {p_id} | [bold]Parent Title:[/bold] {p_title} | [bold]Tokens:[/bold] {p_toks}\n"
                f"[bold]Child ID:[/bold] {c_id} | [bold]Tokens:[/bold] {c_toks} | [bold]Section:[/bold] {c_sec} | [bold]Page:[/bold] {c_page} | [bold]Method:[/bold] {c_method}\n\n"
                f"[bold]Original Child Text Snippet:[/bold]\n[dim]{c_text}...[/dim]"
            )
            console.print(Panel(panel_content, border_style="blue"))
        else:
            console.print(f"[dim]Sample for {key} not found in this run.[/dim]\n")

    console.print("\n[bold green][OK] Dry-run chunk inspection complete. No embeddings or index calls were executed.[/bold green]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", "-r"),
) -> None:
    """Start the FastAPI server."""
    import uvicorn
    console.print(f"[bold green]Starting Legal RAG API on {host}:{port}[/bold green]")
    uvicorn.run(
        "legal_rag.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def status() -> None:
    """Show current system configuration and index status."""
    from legal_rag.config import get_config
    config = get_config()

    table = Table(title="Legal RAG Engine Status", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    rows = [
        ("Corpus Path", str(config.rag_corpus_path)),
        ("Data Dir", str(config.rag_data_dir)),
        ("Qdrant URL", config.rag_qdrant_url),
        ("Collection", config.rag_qdrant_collection),
        ("Embedding Model", config.rag_embedding_model),
        ("LLM Model", config.rag_llm_model),
        ("Dense Top-K", str(config.rag_dense_top_k)),
        ("Child Max Tokens", str(config.rag_child_max_tokens)),
        ("Parent Max Tokens", str(config.rag_parent_max_tokens)),
        ("XRef Expansion", str(config.rag_xref_expansion)),
        ("OCR Provider", config.rag_ocr_provider),
    ]

    for k, v in rows:
        table.add_row(k, v)

    console.print(table)


if __name__ == "__main__":
    app()
