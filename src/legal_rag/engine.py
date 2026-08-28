"""
Main RAG Engine — orchestrates the complete query and ingestion pipelines.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from legal_rag.config import RagConfig, get_config
from legal_rag.chunking.clause_chunker import ClauseChunker
from legal_rag.embedding.provider import EmbeddingProvider, get_embedding_provider
from legal_rag.generation.generator import GroundedGenerator
from legal_rag.indexing.bm25_store import BM25Store
from legal_rag.indexing.qdrant_store import QdrantVectorStore
from legal_rag.ingestion.discovery import discover_files, get_category_from_path, get_source_domain
from legal_rag.ingestion.hasher import DeduplicationRegistry, hash_file, validate_file
from legal_rag.models.chunk import ChildChunk, ParentChunk
from legal_rag.models.document import Document, DocumentMetadata, DocumentStatus, FileIngestionResult, IngestionReport
from legal_rag.models.retrieval import (
    ConfidenceLevel,
    ExpandedEvidence,
    EvidenceStatus,
    QueryResponse,
)
from legal_rag.parsers.markdown_parser import MarkdownParser
from legal_rag.parsers.pdf_parser import PDFParser, is_scanned_pdf
from legal_rag.providers.llm.nvidia import NvidiaLLMProvider, get_llm_provider
from legal_rag.query.analyzer import analyze_query
from legal_rag.retrieval.confidence import ConfidenceConfig, expand_to_parents, score_confidence
from legal_rag.retrieval.hybrid import CrossEncoderReranker, HybridRetriever
from legal_rag.structure.extractor import refine_document_structure

logger = logging.getLogger(__name__)


class LegalRagEngine:
    """
    Main orchestration engine for the Legal RAG System.
    Separates the offline indexing pipeline from the online query pipeline.
    """

    def __init__(self, config: RagConfig | None = None) -> None:
        self.config = config or get_config()
        self.config.ensure_data_dirs()

        # Shared components (lazy-loaded)
        self._embedder: EmbeddingProvider | None = None
        self._qdrant: QdrantVectorStore | None = None
        self._bm25: BM25Store | None = None
        self._retriever: HybridRetriever | None = None
        self._llm: NvidiaLLMProvider | None = None
        self._generator: GroundedGenerator | None = None

    # ------------------------------------------------------------------ #
    # Lazy-loaded components
    # ------------------------------------------------------------------ #

    @property
    def embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            self._embedder = get_embedding_provider(
                provider=self.config.rag_embedding_provider,
                model_name=self.config.rag_embedding_model,
                batch_size=self.config.rag_embedding_batch_size,
            )
        return self._embedder

    @property
    def qdrant(self) -> QdrantVectorStore:
        if self._qdrant is None:
            self._qdrant = QdrantVectorStore(
                collection_name=self.config.rag_qdrant_collection,
                embedding_dim=self.embedder.dimension,
                url=self.config.rag_qdrant_url,
                in_memory=self.config.rag_qdrant_in_memory,
                embedding_model=self.config.rag_embedding_model,
            )
            self._qdrant.ensure_collections()
        return self._qdrant

    @property
    def bm25(self) -> BM25Store:
        if self._bm25 is None:
            self._bm25 = BM25Store(self.config.bm25_dir)
            self._bm25.load()
        return self._bm25

    @property
    def llm(self) -> NvidiaLLMProvider:
        if self._llm is None:
            self._llm = get_llm_provider(self.config)
        return self._llm

    @property
    def generator(self) -> GroundedGenerator:
        if self._generator is None:
            self._generator = GroundedGenerator(self.llm)
        return self._generator

    @property
    def retriever(self) -> HybridRetriever:
        return self._get_retriever()

    def _get_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            try:
                reranker = CrossEncoderReranker()
            except Exception as e:
                logger.warning("Reranker not available (%s) — proceeding without reranking", e)
                reranker = None

            self._retriever = HybridRetriever(
                qdrant_store=self.qdrant,
                bm25_store=self.bm25,
                embedding_provider=self.embedder,
                reranker=reranker,
                dense_top_k=self.config.rag_dense_top_k,
                sparse_top_k=self.config.rag_sparse_top_k,
                rerank_top_k=self.config.rag_rerank_top_k,
                rrf_k=self.config.rag_rrf_k,
            )
        return self._retriever

    # ------------------------------------------------------------------ #
    # A. OFFLINE INGESTION PIPELINE
    # ------------------------------------------------------------------ #

    def run_ingestion(self, corpus_path: Path | None = None) -> tuple[IngestionReport, list[ParentChunk], list[ChildChunk]]:
        """
        Full offline ingestion pipeline:
        Discovery → Validation → Hashing → Dedup → Parsing → OCR → Normalization
        → Structure Extraction → Cross-refs → Chunking → BGE-M3 Embeddings (CHILDREN ONLY)
        → Qdrant Indexing → BM25 Indexing
        """
        corpus_path = corpus_path or self.config.rag_corpus_path
        report = IngestionReport(corpus_path=str(corpus_path))

        logger.info("Starting baseline ingestion from: %s", corpus_path)
        files = discover_files(corpus_path)
        report.total_files_discovered = len(files)

        dedup_registry = DeduplicationRegistry()
        all_children: list[ChildChunk] = []
        all_parents: list[ParentChunk] = []

        self.qdrant.ensure_collections()

        for file_path in files:
            res = self._process_file(file_path, corpus_path, dedup_registry)
            report.results.append(res)

            if res.status != DocumentStatus.SUCCESS:
                continue

            try:
                parents, children = self.ingest_and_index_document(file_path, corpus_path)
                all_parents.extend(parents)
                all_children.extend(children)
            except Exception as e:
                logger.error("Failed to ingest document %s: %s", file_path.name, e)
                res.status = DocumentStatus.PARSE_FAILED
                res.error = str(e)

        # Build BM25 index on all children
        if all_children:
            logger.info("BM25: building index with %d child chunks", len(all_children))
            self.bm25.build(all_children)

        import datetime
        report.finished_at = datetime.datetime.utcnow()
        logger.info("Ingestion complete. Summary: %s", report.summary)
        return report, all_parents, all_children

    def ingest_and_index_document(
        self,
        file_path: Path,
        corpus_path: Path | None = None,
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """
        Parse, chunk, embed, and index a single document.
        Suitable for incremental indexing.
        """
        corpus_path = corpus_path or self.config.rag_corpus_path
        category = get_category_from_path(file_path, corpus_path)
        source_domain = get_source_domain(file_path, corpus_path)

        content_hash = hash_file(file_path)
        metadata = DocumentMetadata(
            title=file_path.stem.replace("_", " ").title(),
            file_type=file_path.suffix.lower().lstrip("."),
            content_hash=content_hash,
            source_paths=[str(file_path)],
            source_categories=[category],
            source_file_names=[file_path.name],
            source_type=self._classify_source_type(source_domain, category),
        )

        # Parse
        doc = self._parse_document(file_path, metadata)
        if not doc.sections:
            logger.warning("No content extracted from: %s", file_path.name)
            return [], []

        # Structure extraction
        doc = refine_document_structure(doc)

        # Chunk
        chunker = ClauseChunker(
            parent_max_tokens=self.config.rag_parent_max_tokens,
            child_max_tokens=self.config.rag_child_max_tokens,
            overlap_tokens=self.config.rag_chunk_overlap_tokens,
            embedding_model=self.config.rag_embedding_model,
        )
        parents, children = chunker.chunk_document(doc)

        if not children:
            logger.warning("No chunks produced from: %s", file_path.name)
            return parents, children

        # Embed children (retrieval units)
        child_texts = [c.embedding_text or c.text for c in children]
        child_embeddings = self.embedder.embed_documents(child_texts)

        # Parents are context-expansion payload records — DO NOT compute BGE-M3 vector embeddings
        dummy_parent_embeddings = [[0.0] * self.embedder.dimension for _ in parents]

        # Index in Qdrant
        self.qdrant.upsert_children(children, child_embeddings)
        self.qdrant.upsert_parents(parents, dummy_parent_embeddings)

        logger.info(
            "Indexed %s: %d parents, %d children",
            file_path.name, len(parents), len(children),
        )
        return parents, children

    def _process_file(
        self,
        file_path: Path,
        corpus_path: Path,
        dedup_registry: DeduplicationRegistry,
    ) -> FileIngestionResult:
        start_time = time.time()
        status, notes = validate_file(file_path)
        if status != DocumentStatus.SUCCESS:
            return FileIngestionResult(
                file_path=str(file_path),
                status=status,
                notes=notes,
                processing_time_seconds=time.time() - start_time,
            )

        content_hash = hash_file(file_path)
        category = get_category_from_path(file_path, corpus_path)
        canonical, is_new = dedup_registry.register(file_path, content_hash, category)

        if not is_new:
            return FileIngestionResult(
                file_path=str(file_path),
                status=DocumentStatus.DUPLICATE,
                content_hash=content_hash,
                canonical_doc_id=canonical.canonical_id,
                notes=f"Duplicate of {canonical.source_paths[0]}",
                processing_time_seconds=time.time() - start_time,
            )

        return FileIngestionResult(
            file_path=str(file_path),
            status=DocumentStatus.SUCCESS,
            content_hash=content_hash,
            processing_time_seconds=time.time() - start_time,
        )

    def _parse_document(self, file_path: Path, metadata: DocumentMetadata) -> Document:
        """Route to the correct parser based on file type."""
        ext = file_path.suffix.lower()

        if ext in {".md", ".markdown"}:
            parser = MarkdownParser()
            return parser.parse(file_path, metadata)

        elif ext == ".pdf":
            pdf_parser = PDFParser(
                words_per_page_threshold=self.config.rag_ocr_words_per_page_threshold
            )
            doc = pdf_parser.parse(file_path, metadata)

            # Route to OCR if scanned
            if is_scanned_pdf(doc):
                logger.info("Routing %s to OCR pipeline", file_path.name)
                from legal_rag.parsers.ocr_parser import ocr_pdf
                doc = ocr_pdf(
                    file_path,
                    metadata,
                    confidence_threshold=self.config.rag_ocr_confidence_threshold,
                )
            return doc

        else:
            logger.warning("Unsupported format: %s", file_path)
            return Document(metadata=metadata)

    def _classify_source_type(self, source_domain: str, category: str) -> str:
        if source_domain == "contract_rules":
            return "rulebook"
        cat_lower = category.lower()
        if "case_law" in cat_lower:
            return "case_law"
        if "finance" in cat_lower:
            return "statute_finance"
        if "employment" in cat_lower:
            return "statute_employment"
        if "ip" in cat_lower:
            return "statute_ip"
        if "lease" in cat_lower:
            return "statute_lease"
        if "vendor" in cat_lower:
            return "statute_vendor"
        if "dispute" in cat_lower:
            return "statute_dispute"
        return "statute"

    # ------------------------------------------------------------------ #
    # B. ONLINE QUERY PIPELINE
    # ------------------------------------------------------------------ #

    def query(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        conversation_context: list[dict[str, Any]] | None = None,
        model_mode: str = "quality",
    ) -> QueryResponse:
        """
        Full online query pipeline:
        Conversation Router → Analysis → Retrieval → Confidence → Parent Expansion → Generation
        """
        logger.info("Query received: %s (mode=%s)", user_query[:80], model_mode)

        from legal_rag.query.conversation_router import classify_conversation_turn, TurnType
        from legal_rag.providers.llm.nvidia import get_llm_provider_by_model

        # Determine target model
        target_model = self.config.rag_fast_llm_model if model_mode == "fast" else self.config.rag_llm_model
        llm = get_llm_provider_by_model(self.config, target_model)
        generator = GroundedGenerator(llm)

        # Route conversation turn
        routing = classify_conversation_turn(user_query, conversation_context)

        # TYPE A: Contextual follow-up without new retrieval
        if routing.turn_type == TurnType.CONTEXTUAL_FOLLOWUP and routing.previous_evidence:
            logger.info("TYPE A TURN: Reusing previous evidence context (0 ms retrieval)")
            followup_prompt = (
                f"PREVIOUS QUESTION: {routing.resolved_query}\n"
                f"PREVIOUS ANSWER: {routing.previous_answer}\n\n"
                f"FOLLOW-UP REQUEST: {user_query}"
            )
            resp = generator.generate(followup_prompt, routing.previous_evidence)
            resp.query = user_query
            return resp

        # TYPE B or TYPE C: Run retrieval pipeline
        query_text = routing.resolved_query if routing.turn_type == TurnType.RETRIEVAL_FOLLOWUP else user_query

        # 1. Query understanding
        query_analysis = analyze_query(query_text)

        # 2. Initial retrieval
        retriever = self._get_retriever()
        results = retriever.retrieve(query_text, filters=filters)

        # 3. Confidence scoring
        conf_cfg = ConfidenceConfig(
            high_threshold=self.config.rag_confidence_high_threshold,
            low_threshold=self.config.rag_confidence_low_threshold,
        )
        raw_score, confidence = score_confidence(results, query_analysis, conf_cfg)
        retry_count = 0
        rewritten_query: str | None = None

        # 4. Retry on low confidence
        while confidence == ConfidenceLevel.LOW and retry_count < self.config.rag_max_retry_attempts:
            retry_count += 1
            logger.info("Low confidence (%.3f) — retry %d", raw_score, retry_count)
            rewritten_query = self._rewrite_query(query_text, query_analysis, retry_count)
            results = retriever.retrieve(rewritten_query, filters=filters)
            raw_score, confidence = score_confidence(results, query_analysis, conf_cfg)

        # 5. Abstain if still insufficient
        if not results or confidence == ConfidenceLevel.LOW:
            response = generator._abstain(user_query, query_analysis)
            response.retrieval_attempts = retry_count + 1
            return response

        # 6. Parent expansion
        expanded = expand_to_parents(results, self.qdrant)

        # 7. Cross-reference expansion (if enabled)
        if self.config.rag_xref_expansion:
            expanded = self._expand_cross_references(expanded, max_extra=self.config.rag_xref_max_extra)

        # 8. Generate grounded answer
        response = generator.generate(
            query=user_query,
            evidence_list=expanded,
            query_analysis=query_analysis,
            confidence=confidence,
        )
        response.retrieval_attempts = retry_count + 1
        response.rewritten_query = rewritten_query
        response.query_analysis = query_analysis

        logger.info(
            "Query complete: confidence=%s evidence=%s citations=%d",
            confidence.value, response.evidence_status.value, len(response.citations),
        )
        return response

    def query_stream(
        self,
        user_query: str,
        filters: dict[str, Any] | None = None,
        conversation_context: list[dict[str, Any]] | None = None,
        model_mode: str = "quality",
    ):
        """
        Stream query execution. Yields dictionary events for SSE streaming.
        """
        import time
        t0_start = time.perf_counter()

        yield {"type": "status", "status": "analyzing", "message": "Analyzing query & routing..."}

        from legal_rag.query.conversation_router import classify_conversation_turn, TurnType
        from legal_rag.providers.llm.nvidia import get_llm_provider_by_model

        target_model = self.config.rag_fast_llm_model if model_mode == "fast" else self.config.rag_llm_model
        llm = get_llm_provider_by_model(self.config, target_model)
        generator = GroundedGenerator(llm)

        routing = classify_conversation_turn(user_query, conversation_context)

        # TYPE A: Contextual follow-up without new retrieval
        if routing.turn_type == TurnType.CONTEXTUAL_FOLLOWUP and routing.previous_evidence:
            yield {"type": "status", "status": "generating", "message": "Continuing from previous evidence..."}
            followup_prompt = (
                f"PREVIOUS QUESTION: {routing.resolved_query}\n"
                f"PREVIOUS ANSWER: {routing.previous_answer}\n\n"
                f"FOLLOW-UP REQUEST: {user_query}"
            )
            pre_built_citations, token_stream = generator.generate_stream(followup_prompt, routing.previous_evidence)

            # Metadata event
            yield {
                "type": "metadata",
                "citations": [c.model_dump() for c in pre_built_citations],
                "confidence": "high",
                "evidence_status": "supported",
            }

            accumulated = ""
            for token in token_stream:
                accumulated += token
                yield {"type": "token", "token": token}

            yield {
                "type": "complete",
                "answer": accumulated,
                "citations": [c.model_dump() for c in pre_built_citations],
                "confidence": "high",
                "evidence_status": "supported",
            }
            return

        # TYPE B or C: Run retrieval pipeline
        yield {"type": "status", "status": "retrieving", "message": "Retrieving legal evidence..."}

        query_text = routing.resolved_query if routing.turn_type == TurnType.RETRIEVAL_FOLLOWUP else user_query
        query_analysis = analyze_query(query_text)

        retriever = self._get_retriever()
        results = retriever.retrieve(query_text, filters=filters)

        conf_cfg = ConfidenceConfig(
            high_threshold=self.config.rag_confidence_high_threshold,
            low_threshold=self.config.rag_confidence_low_threshold,
        )
        raw_score, confidence = score_confidence(results, query_analysis, conf_cfg)

        if not results or confidence == ConfidenceLevel.LOW:
            abstain_resp = generator._abstain(user_query, query_analysis)
            yield {
                "type": "complete",
                "answer": abstain_resp.answer,
                "citations": [],
                "confidence": "low",
                "evidence_status": "insufficient",
            }
            return

        expanded = expand_to_parents(results, self.qdrant)
        if self.config.rag_xref_expansion:
            expanded = self._expand_cross_references(expanded, max_extra=self.config.rag_xref_max_extra)

        pre_built_citations, token_stream = generator.generate_stream(user_query, expanded, query_analysis, confidence)

        yield {
            "type": "metadata",
            "citations": [c.model_dump() for c in pre_built_citations],
            "confidence": confidence.value,
            "evidence_status": "supported",
        }

        yield {"type": "status", "status": "generating", "message": "Generating answer..."}

        accumulated = ""
        for token in token_stream:
            accumulated += token
            yield {"type": "token", "token": token}

        yield {
            "type": "complete",
            "answer": accumulated,
            "citations": [c.model_dump() for c in pre_built_citations],
            "confidence": confidence.value,
            "evidence_status": "supported",
        }


    def _rewrite_query(
        self, original_query: str, analysis, retry: int
    ) -> str:
        """
        Simple query rewriting: expand with extracted act names and section refs.
        For retry > 1, use LLM-based rewriting.
        """
        if retry == 1 and (analysis.section_refs or analysis.act_names):
            # Add structured terms to query
            additions = " ".join(analysis.section_refs[:2] + analysis.act_names[:1])
            return f"{original_query} {additions}".strip()
        return original_query

    def _expand_cross_references(
        self,
        evidence: list[ExpandedEvidence],
        max_extra: int = 5,
    ) -> list[ExpandedEvidence]:
        """
        If a retrieved chunk has cross-references, fetch target chunks.
        Bounded: max_extra additional chunks, no recursive expansion.
        """
        extra: list[ExpandedEvidence] = []
        seen_ids: set[str] = {ev.child.chunk_id for ev in evidence}

        for ev in evidence[:3]:  # only expand from top 3 results
            xrefs = (getattr(ev.child, "cross_references", None) or [])[:self.config.rag_xref_max_extra]
            for xref in xrefs:

                if len(extra) >= max_extra:
                    break
                # Attempt to find chunk by section number in BM25
                xref_results = self.bm25.search(xref, top_k=1)
                if xref_results:
                    chunk_id = xref_results[0].get("chunk_id", "")
                    if chunk_id and chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        from legal_rag.models.retrieval import RetrievalResult
                        xref_result = RetrievalResult(
                            chunk_id=chunk_id,
                            parent_id=xref_results[0].get("parent_id"),
                            document_id=xref_results[0].get("document_id", ""),
                            document_title=xref_results[0].get("document_title"),
                            category=xref_results[0].get("category", ""),
                            section_number=xref_results[0].get("section_number"),
                            section_title=xref_results[0].get("section_title"),
                            page_start=xref_results[0].get("page_start"),
                            text=xref_results[0].get("text", ""),
                            source="cross_ref_expansion",
                        )
                        parent_text = None
                        if xref_result.parent_id:
                            parent_payload = self.qdrant.get_parent_by_chunk_id(xref_result.parent_id)
                            if parent_payload:
                                parent_text = parent_payload.get("text")
                        extra.append(ExpandedEvidence(child=xref_result, parent_text=parent_text))

        return evidence + extra
