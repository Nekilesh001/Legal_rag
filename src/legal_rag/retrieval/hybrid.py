"""
Hybrid retrieval: dense + BM25 + RRF fusion + reranking + parent expansion.
"""
from __future__ import annotations

import logging
from typing import Any

from legal_rag.embedding.provider import EmbeddingProvider
from legal_rag.indexing.bm25_store import BM25Store
from legal_rag.indexing.qdrant_store import QdrantVectorStore
from legal_rag.models.retrieval import ExpandedEvidence, RetrievalResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# RRF Fusion
# ------------------------------------------------------------------ #

def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Combine dense and sparse result lists using Reciprocal Rank Fusion.
    Returns unified list sorted by RRF score descending.
    k = 60 is the standard default (Cormack et al., 2009).
    """
    rrf_scores: dict[str, float] = {}
    all_payloads: dict[str, dict[str, Any]] = {}

    for rank, result in enumerate(dense_results, start=1):
        cid = result.get("payload", result).get("chunk_id", "")
        if cid:
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            all_payloads[cid] = result.get("payload", result)
            all_payloads[cid]["dense_score"] = result.get("score")

    for rank, result in enumerate(sparse_results, start=1):
        cid = result.get("chunk_id", "")
        if cid:
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in all_payloads:
                all_payloads[cid] = result.copy()
            all_payloads[cid]["sparse_score"] = result.get("score")

    fused = []
    for cid, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        payload = all_payloads[cid].copy()
        payload["rrf_score"] = rrf_score
        payload["chunk_id"] = cid
        fused.append(payload)

    return fused


# ------------------------------------------------------------------ #
# Reranker
# ------------------------------------------------------------------ #

class CrossEncoderReranker:
    """
    Local cross-encoder reranker using sentence_transformers.
    Replaceable: swap with any other scoring model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading reranker model: %s", model_name)
        self._model = CrossEncoder(model_name)
        self._model_name = model_name
        logger.info("Reranker loaded: %s", model_name)

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        pairs = [
            (query, c.get("rerank_input", c.get("text", "")))
            for c in candidates
        ]
        scores = self._model.predict(pairs)
        for c, score in zip(candidates, scores):
            c["reranker_score"] = float(score)
        ranked = sorted(candidates, key=lambda x: x.get("reranker_score", 0.0), reverse=True)
        return ranked[:top_k]



from legal_rag.query.analyzer import analyze_query
from legal_rag.retrieval.policy import MetadataRetrievalPolicy

# ------------------------------------------------------------------ #
# Hybrid Retriever
# ------------------------------------------------------------------ #

from legal_rag.retrieval.structured import StructuredQueryRetriever

class HybridRetriever:
    """
    Combines structured metadata lookup + dense vector search + BM25 + RRF + metadata policy + optional reranking.
    """

    def __init__(
        self,
        qdrant_store: QdrantVectorStore,
        bm25_store: BM25Store,
        embedding_provider: EmbeddingProvider,
        reranker: CrossEncoderReranker | None = None,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        rerank_top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        self.qdrant = qdrant_store
        self.bm25 = bm25_store
        self.embedder = embedding_provider
        self.reranker = reranker
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rerank_top_k = rerank_top_k
        self.rrf_k = rrf_k
        
        from legal_rag.config import get_config
        self.policy = MetadataRetrievalPolicy(get_config())
        self.structured_retriever = StructuredQueryRetriever(qdrant_store, bm25_store)

    def retrieve(
        self, query: str, filters: dict[str, Any] | None = None
    ) -> list[RetrievalResult]:
        """
        Full hybrid retrieval pipeline (Experiment 2):
        1. Query analysis
        2. Structured Query Retrieval (exact section & metadata category lookup BEFORE top-k cutoff)
        3. Dense search + BM25 search
        4. Candidate Union & Deduplication
        5. RRF fusion + Metadata-Aware Policy adjustment
        6. Reranking
        """
        import time
        t0_total = time.perf_counter()

        # 1. Analyze query for metadata signals
        t0 = time.perf_counter()
        query_analysis = analyze_query(query)
        t_qa = (time.perf_counter() - t0) * 1000

        # 2. Structured candidate lookup (pre-top-k cutoff)
        t0 = time.perf_counter()
        structured_raw = []
        if self.policy.config.rag_metadata_aware_retrieval:
            structured_raw = self.structured_retriever.retrieve_structured_candidates(query_analysis)
        t_struct = (time.perf_counter() - t0) * 1000

        # 3. Dense search
        t0 = time.perf_counter()
        query_vector = self.embedder.embed_query(query)
        qdrant_filter = self._build_qdrant_filter(filters) if filters else None
        dense_raw = self.qdrant.search_children(query_vector, self.dense_top_k, qdrant_filter)
        t_dense = (time.perf_counter() - t0) * 1000

        # 4. Sparse search
        t0 = time.perf_counter()
        sparse_raw = self.bm25.search(query, self.sparse_top_k)
        t_sparse = (time.perf_counter() - t0) * 1000

        # 5. RRF fusion (incorporating structured_raw candidates)
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(dense_raw, sparse_raw, k=self.rrf_k)
        fused_ids = {item.get("chunk_id") for item in fused}
        for s_item in structured_raw:
            cid = s_item.get("chunk_id")
            if cid not in fused_ids:
                s_copy = s_item.copy()
                s_copy["rrf_score"] = 1.0 / (self.rrf_k + 1)
                fused.append(s_copy)
                fused_ids.add(cid)
        t_rrf = (time.perf_counter() - t0) * 1000

        # 6. Metadata-Aware Policy adjustment (exact statutory source priority)
        t0 = time.perf_counter()
        fused = self.policy.apply_policy(
            query, query_analysis, fused,
            bm25_metadata=self.bm25._chunk_metadata,
        )
        t_policy = (time.perf_counter() - t0) * 1000

        # 7. Frozen Legal RAG Pipeline (LegalAwareRanker -> Context Format -> BGE Rerank -> Blend -> Protect -> Evidence Select)
        t0 = time.perf_counter()
        from legal_rag.retrieval.legal_ranker import LegalAwareRanker, LegalRankerWeights
        from legal_rag.retrieval.context_formatter import apply_rerank_formatting
        from legal_rag.retrieval.legal_identity import registry as global_registry
        from legal_rag.retrieval.blender import ScoreBlender, ProtectedEvidenceHandler
        from legal_rag.retrieval.evidence_selector import EvidenceSelector

        global_registry.bootstrap(self.bm25._chunk_metadata)

        legal_ranker = LegalAwareRanker(
            weights=LegalRankerWeights(intent_content_pref=8.0, concept_match=4.0),
            registry=global_registry,
        )
        legal_sorted = legal_ranker.rank(fused, query_analysis)[:50]
        t_legal_rank = (time.perf_counter() - t0) * 1000

        # Apply Parent-Contextual Reranking Formatting (Experiment 9)
        t0 = time.perf_counter()
        formatted = apply_rerank_formatting(legal_sorted, mode="full", registry=global_registry)
        t_format = (time.perf_counter() - t0) * 1000

        # Dynamic Rerank Window Budget Optimization (Part 6):
        # Exact query (section + act) -> top 10; Normal query (act or section) -> top 15-20; Broad -> top 25-30
        if query_analysis.section_refs and query_analysis.act_names:
            rerank_budget = min(10, self.rerank_top_k)
        elif query_analysis.act_names or query_analysis.section_refs:
            rerank_budget = min(20, self.rerank_top_k)
        else:
            rerank_budget = min(30, self.rerank_top_k)

        t0 = time.perf_counter()
        if self.reranker and formatted:
            to_rerank = formatted[:rerank_budget]
            bge_ranked = self.reranker.rerank(query, to_rerank, top_k=rerank_budget)
            # Append remaining candidates without extra reranking overhead
            bge_ranked.extend(formatted[rerank_budget:])
        else:
            bge_ranked = formatted[:self.rerank_top_k]
        t_rerank = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        blender = ScoreBlender(lambda_weight=0.50)
        protection_handler = ProtectedEvidenceHandler(tier_1_boost=0.35, tier_2_boost=0.20)
        evidence_selector = EvidenceSelector(max_chunks_per_section=1, max_evidence_items=5, concept_yield_bonus=0.25)

        blended = blender.blend_batch(bge_ranked)
        protected = protection_handler.apply_protection(blended, query_analysis, score_key="blended_score")
        final_evidence = evidence_selector.select_final_evidence_set(protected, query_analysis)
        t_post = (time.perf_counter() - t0) * 1000

        t_total = (time.perf_counter() - t0_total) * 1000
        logger.info(
            "HYBRID TIMING: QA=%.1fms Struct=%.1fms Dense=%.1fms Sparse=%.1fms RRF=%.1fms Policy=%.1fms LegalRank=%.1fms Format=%.1fms Rerank(%d)=%.1fms Post=%.1fms -> Total=%.1fms",
            t_qa, t_struct, t_dense, t_sparse, t_rrf, t_policy, t_legal_rank, t_format, rerank_budget, t_rerank, t_post, t_total
        )

        # Convert to RetrievalResult models
        results: list[RetrievalResult] = []
        for item in final_evidence:
            results.append(RetrievalResult(
                chunk_id=item.get("chunk_id", ""),
                parent_id=item.get("parent_id"),
                document_id=item.get("document_id", ""),
                document_title=item.get("document_title"),
                category=item.get("category", ""),
                section_number=item.get("section_number"),
                section_title=item.get("section_title"),
                page_start=item.get("page_start"),
                page_end=item.get("page_end"),
                text=item.get("text", ""),
                dense_score=item.get("dense_score"),
                sparse_score=item.get("sparse_score"),
                rrf_score=item.get("rrf_score"),
                reranker_score=item.get("protected_score", item.get("blended_score", item.get("reranker_score"))),
                source=item.get("source", "unknown"),
            ))

        return results

    def _build_qdrant_filter(self, filters: dict[str, Any]):
        from qdrant_client.http import models as qm
        conditions = [
            qm.FieldCondition(key=k, match=qm.MatchValue(value=v))
            for k, v in filters.items()
        ]
        return qm.Filter(must=conditions) if conditions else None
