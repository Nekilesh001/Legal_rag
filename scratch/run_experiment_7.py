"""
run_experiment_7.py — Protected Legal Evidence + Normalized Reranker Blending Evaluation.

Compares THREE pipelines across Q1, Q2, Q3, Q4:
  Pipeline A (Baseline Control):  RRF -> LegalAwareRanker -> BGE-Reranker (raw BGE score)
  Pipeline B (Normalized Blend):  RRF -> LegalAwareRanker -> BGE-Reranker -> MinMax Normalization -> Blended Score (norm_bge + lambda * norm_legal)
  Pipeline C (Protected Evidence): RRF -> LegalAwareRanker -> BGE-Reranker -> MinMax Normalization -> Blended Score -> Protected Tier Boost

Evaluates lambda in {0.10, 0.25, 0.50}, candidate rerank budget (N=50),
protection tiers (Tier 1 exact, Tier 2 strong linked), top-5 detailed tables,
and target surfacing metrics.
"""
from __future__ import annotations

import copy
import math
import os
import sys
import time
import logging

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

from legal_rag.config import get_config
from legal_rag.indexing.bm25_store import BM25Store
from legal_rag.indexing.qdrant_store import QdrantVectorStore
from legal_rag.query.analyzer import analyze_query
from legal_rag.query.linker import LegalEntityLinker
from legal_rag.retrieval.legal_identity import registry as global_registry
from legal_rag.retrieval.legal_ranker import (
    LegalAwareRanker,
    LegalRankerWeights,
    get_source_authority,
    classify_content_type,
)
from legal_rag.retrieval.structured import StructuredQueryRetriever
from legal_rag.retrieval.hybrid import CrossEncoderReranker, reciprocal_rank_fusion
from legal_rag.retrieval.policy import MetadataRetrievalPolicy
from legal_rag.retrieval.blender import (
    ScoreBlender,
    ProtectedEvidenceHandler,
    ProtectionTier,
    min_max_normalize,
)

SEP  = "=" * 80
THIN = "-" * 80

# ------------------------------------------------------------------ #
# Query Specifications
# ------------------------------------------------------------------ #
QUERIES = [
    {
        "label": "Q1",
        "query": "What does Section 73 of the Indian Contract Act say?",
        "expected_chunks": ["chk_6c2b46f4b321"],
        "expected_desc": "Indian Contract Act Section 73",
        "accept_multiple": False,
        "nda_keywords": (),
    },
    {
        "label": "Q2",
        "query": "What are the mandatory clauses in an NDA agreement?",
        "expected_chunks": [],
        "expected_desc": "NDA mandatory clauses (multiple acceptable)",
        "accept_multiple": True,
        "nda_keywords": ("nda", "mandatory", "playbook", "confidential"),
    },
    {
        "label": "Q3",
        "query": "What is the notice period under Tamil Nadu Shops Act?",
        "expected_chunks": ["chk_56b1160532cc"],
        "expected_desc": "TN Shops Act Section 41",
        "accept_multiple": False,
        "nda_keywords": (),
    },
    {
        "label": "Q4",
        "query": "What happens if the seller breaches the contract?",
        "expected_chunks": ["chk_01051fb4680e", "chk_e83012ad27b7"],
        "expected_desc": "Sale of Goods Act Sections 59 / 54",
        "accept_multiple": False,
        "nda_keywords": (),
    },
]

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def find_rank(results: list[dict], chunk_ids: list[str]) -> int | None:
    for i, r in enumerate(results, 1):
        if r.get("chunk_id") in chunk_ids:
            return i
    return None


def recall_at_k(results: list[dict], chunk_ids: list[str], k: int) -> float:
    if not chunk_ids:
        return float("nan")
    hits = sum(1 for r in results[:k] if r.get("chunk_id") in chunk_ids)
    return hits / len(chunk_ids)


def mrr(results: list[dict], chunk_ids: list[str]) -> float:
    if not chunk_ids:
        return float("nan")
    for i, r in enumerate(results, 1):
        if r.get("chunk_id") in chunk_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(results: list[dict], chunk_ids: list[str], k: int) -> float:
    if not chunk_ids:
        return float("nan")
    chunk_set = set(chunk_ids)
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, r in enumerate(results[:k], 1)
        if r.get("chunk_id") in chunk_set
    )
    ideal = min(len(chunk_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


def annotate_legal(items: list[dict], registry) -> None:
    for item in items:
        if "source_authority" not in item:
            item["source_authority"] = get_source_authority(item, registry).name
        if "content_type" not in item:
            item["content_type"] = classify_content_type(item, item.get("text", "")).value


def print_detailed_top5(results: list[dict], header: str, exp_chunks: list[str]) -> None:
    print(f"\n  {header}")
    print(f"  {'#':<3} {'Document':<24} {'Sec':<5} {'Pg':<3} {'ChunkID':<18} "
          f"{'RawBGE':>7} {'NormBGE':>7} {'RawLeg':>7} {'NormLeg':>7} {'Blend':>7} {'Tier':<18} Snippet")
    print("  " + "-" * 145)
    for rank, r in enumerate(results[:5], 1):
        doc       = (r.get("document_title") or r.get("document_id") or "")[:22]
        sec       = str(r.get("section_number") or "")[:4]
        pg        = str(r.get("page_start") or "")[:3]
        cid       = (r.get("chunk_id") or "")[:16]
        raw_bge   = r.get("reranker_score", 0.0)
        norm_bge  = r.get("normalized_bge_score", 0.0)
        raw_leg   = r.get("legal_score", 0.0)
        norm_leg  = r.get("normalized_legal_score", 0.0)
        blend     = r.get("protected_score", r.get("blended_score", norm_bge))
        tier      = r.get("protection_tier", "NONE")[:16]
        snip      = (r.get("text") or "")[:45].replace("\n", " ")
        mark      = " <--" if r.get("chunk_id") in exp_chunks else ""
        print(f"  {rank:<3} {doc:<24} {sec:<5} {pg:<3} {cid:<18} "
              f"{raw_bge:>7.3f} {norm_bge:>7.3f} {raw_leg:>7.1f} {norm_leg:>7.3f} {blend:>7.3f} {tier:<18} {snip}{mark}")


def is_nda_relevant(r: dict, keywords: tuple) -> bool:
    title = (r.get("document_title") or r.get("document_id") or "").lower()
    return any(kw in title for kw in keywords)


def _rs(rank: int | None) -> str:
    return str(rank) if rank else ">5"


def _rsp(rank: int | None, pool: int) -> str:
    return str(rank) if rank else f">pool({pool})"


# ------------------------------------------------------------------ #
# Main Execution
# ------------------------------------------------------------------ #

def main() -> None:
    cfg = get_config()

    print(SEP)
    print("EXPERIMENT 7: Protected Legal Evidence + Normalized Reranker Blending")
    print("  Pipeline A (Control Baseline):   RRF -> LegalRanker -> BGE-Reranker (raw BGE score)")
    print("  Pipeline B (Normalized Blend):   RRF -> LegalRanker -> BGE-Reranker -> MinMax -> Blend")
    print("  Pipeline C (Protected Evidence): RRF -> LegalRanker -> BGE-Reranker -> MinMax -> Blend -> Protection Boost")
    print(SEP)

    print("\nLoading stores & models ...")
    from legal_rag.embedding.provider import get_embedding_provider
    embedder = get_embedding_provider(
        provider=cfg.rag_embedding_provider,
        model_name=cfg.rag_embedding_model,
        batch_size=cfg.rag_embedding_batch_size,
    )
    qdrant = QdrantVectorStore(
        collection_name=cfg.rag_qdrant_collection,
        embedding_dim=embedder.dimension,
        url=cfg.rag_qdrant_url,
        in_memory=cfg.rag_qdrant_in_memory,
        embedding_model=cfg.rag_embedding_model,
    )
    bm25 = BM25Store(cfg.bm25_dir)
    bm25.load()
    print(f"  BM25 chunks: {len(bm25._chunk_metadata)}")

    global_registry.bootstrap(bm25._chunk_metadata)
    print(f"  Registry: {len(global_registry.all_documents())} canonical documents")

    policy = MetadataRetrievalPolicy(cfg)
    structured_retriever = StructuredQueryRetriever(qdrant, bm25, global_registry)
    legal_ranker = LegalAwareRanker(
        weights=LegalRankerWeights(),
        registry=global_registry,
    )
    linker = LegalEntityLinker()
    blender = ScoreBlender()
    protection_handler = ProtectedEvidenceHandler(tier_1_boost=0.35, tier_2_boost=0.20)

    print("\nLoading BGE-Reranker-v2-M3 ...")
    t0 = time.perf_counter()
    bge_reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")
    bge_load_s = time.perf_counter() - t0
    print(f"  BGE Reranker loaded in {bge_load_s:.1f}s")

    # ---------------------------------------------------------------- #
    # PART 1: Lambda Weight Tuning Grid (Testing lambda = 0.10, 0.25, 0.50)
    # ---------------------------------------------------------------- #
    print(f"\n{SEP}")
    print("LAMBDA BLENDING TUNING GRID (Testing lambda in {0.10, 0.25, 0.50})")
    print(SEP)

    lambda_candidates = [0.10, 0.25, 0.50]
    lambda_scores = {lam: 0 for lam in lambda_candidates}

    # Pre-build candidate batches for lambda search
    cached_batches = []

    for qspec in QUERIES:
        query = qspec["query"]
        qa = analyze_query(query)
        linked_ctx = linker.link(qa)

        # Multi-source candidate generation
        exp_struct = structured_retriever.retrieve_structured_candidates(qa)
        if linked_ctx.candidate_sources:
            mock_qa = copy.deepcopy(qa)
            for c_src in linked_ctx.candidate_sources:
                if c_src not in mock_qa.act_names:
                    mock_qa.act_names.append(c_src)
            extra_struct = structured_retriever.retrieve_structured_candidates(mock_qa)
            seen_struct = {x.get("chunk_id") for x in exp_struct}
            for x in extra_struct:
                if x.get("chunk_id") not in seen_struct:
                    exp_struct.append(x); seen_struct.add(x.get("chunk_id"))

        qvec = embedder.embed_query(query)
        orig_dense  = qdrant.search_children(qvec, 50, None)
        orig_sparse = bm25.search(query, 50)
        orig_cands  = reciprocal_rank_fusion(orig_dense, orig_sparse, k=60)

        variant_cands_list = []
        for v in linked_ctx.query_variants:
            v_vec = embedder.embed_query(v.variant_text)
            v_dense = qdrant.search_children(v_vec, 50, None)
            v_sparse = bm25.search(v.variant_text, 50)
            variant_cands_list.append(reciprocal_rank_fusion(v_dense, v_sparse, k=60))

        # Union & Deduplicate
        all_candidate_lists = [orig_cands] + variant_cands_list
        dedup_scores: dict[str, float] = {}
        all_payloads: dict[str, dict] = {}
        for clist in all_candidate_lists:
            for rank_idx, item in enumerate(clist, 1):
                cid = item.get("chunk_id")
                if cid:
                    dedup_scores[cid] = dedup_scores.get(cid, 0.0) + 1.0 / (60 + rank_idx)
                    if cid not in all_payloads: all_payloads[cid] = item.copy()

        for rank_idx, item in enumerate(exp_struct, 1):
            cid = item.get("chunk_id")
            if cid:
                dedup_scores[cid] = dedup_scores.get(cid, 0.0) + 1.0 / (60 + rank_idx)
                if cid not in all_payloads:
                    sc = item.copy(); sc["rrf_score"] = 1.0 / (60 + rank_idx)
                    all_payloads[cid] = sc

        fused = []
        for cid, rrf_score in sorted(dedup_scores.items(), key=lambda x: x[1], reverse=True):
            payload = all_payloads[cid].copy()
            payload["rrf_score"] = rrf_score; payload["chunk_id"] = cid
            fused.append(payload)

        fused = policy.apply_policy(query, qa, fused, bm25_metadata=bm25._chunk_metadata)
        annotate_legal(fused, global_registry)
        legal_sorted = legal_ranker.rank(fused, qa)

        # Rerank Top 50 budget with BGE
        top50 = copy.deepcopy(legal_sorted[:50])
        bge_reranked = bge_reranker.rerank(query, top50, top_k=50)
        annotate_legal(bge_reranked, global_registry)

        cached_batches.append({
            "qspec": qspec,
            "qa": qa,
            "linked_ctx": linked_ctx,
            "dedup_count": len(fused),
            "top50_reranked": bge_reranked,
        })

    # Test each lambda across cached batches
    for lam in lambda_candidates:
        print(f"\n  Testing Lambda = {lam:.2f}:")
        for batch in cached_batches:
            qspec = batch["qspec"]
            qa    = batch["qa"]
            cands = copy.deepcopy(batch["top50_reranked"])
            exp_cids = qspec["expected_chunks"]

            blended = blender.blend_batch(cands, lambda_weight=lam)
            blended.sort(key=lambda x: x["blended_score"], reverse=True)

            r = find_rank(blended, exp_cids)
            print(f"    {qspec['label']}: target rank = {_rs(r)}")
            if r is not None and r <= 5:
                lambda_scores[lam] += (6 - r)

    best_lambda = max(lambda_scores, key=lambda_scores.get)
    print(f"\n  SELECTED BEST LAMBDA: {best_lambda:.2f} (Total Score = {lambda_scores[best_lambda]})")

    # Update blender default lambda
    blender.lambda_weight = best_lambda

    # ---------------------------------------------------------------- #
    # PART 2: Three-Pipeline Comparison
    # ---------------------------------------------------------------- #
    print(f"\n{SEP}")
    print("PARTS 2-15: THREE-PIPELINE COMPARISON & RESULTS")
    print(SEP)

    summary_rows = []

    for batch in cached_batches:
        qspec      = batch["qspec"]
        qa         = batch["qa"]
        exp_cids   = qspec["expected_chunks"]
        exp_desc   = qspec["expected_desc"]
        label      = qspec["label"]
        query      = qspec["query"]
        top50_raw  = batch["top50_reranked"]
        pool_count = batch["dedup_count"]

        print(f"\n{THIN}")
        print(f"  {label}: {query}")
        print(f"  Expected: {exp_desc}  |  Candidate Pool Size: {pool_count}  |  Rerank Window: 50")
        print(THIN)

        # Pipeline A — Baseline Control (raw BGE score)
        pipe_a = copy.deepcopy(top50_raw)
        pipe_a.sort(key=lambda x: x.get("reranker_score", 0.0), reverse=True)
        # Add norm scores for display
        pipe_a = blender.blend_batch(pipe_a)
        pipe_a_top5 = pipe_a[:5]

        # Pipeline B — Normalized Score Blending (best lambda)
        pipe_b = blender.blend_batch(copy.deepcopy(top50_raw))
        pipe_b.sort(key=lambda x: x["blended_score"], reverse=True)
        pipe_b_top5 = pipe_b[:5]

        # Pipeline C — Protected Evidence
        pipe_c_candidates = blender.blend_batch(copy.deepcopy(top50_raw))
        pipe_c = protection_handler.apply_protection(pipe_c_candidates, qa, score_key="blended_score")
        pipe_c_top5 = pipe_c[:5]

        # Counts of protected candidates in top 50 batch
        protected_count = sum(
            1 for c in pipe_c if c.get("protection_tier") != "NONE"
        )
        tier1_count = sum(1 for c in pipe_c if c.get("protection_tier") == "TIER_1_EXACT_EVIDENCE")
        tier2_count = sum(1 for c in pipe_c if c.get("protection_tier") == "TIER_2_STRONG_LINKED")

        rank_a = find_rank(pipe_a, exp_cids)
        rank_b = find_rank(pipe_b, exp_cids)
        rank_c = find_rank(pipe_c, exp_cids)

        print(f"\n  Ranks — Baseline(A): {_rs(rank_a)}  |  Blend(B): {_rs(rank_b)}  |  Protected(C): {_rs(rank_c)}")
        print(f"  Batch Protection Stats: {protected_count}/50 Protected (Tier 1={tier1_count}, Tier 2={tier2_count})")

        print_detailed_top5(pipe_a_top5, f"Pipeline A — Baseline Control (Raw BGE), Top 5:", exp_cids)
        print_detailed_top5(pipe_b_top5, f"Pipeline B — Normalized Blend (lambda={best_lambda:.2f}), Top 5:", exp_cids)
        print_detailed_top5(pipe_c_top5, f"Pipeline C — Protected Evidence, Top 5:", exp_cids)

        # Regression check for Q2
        if qspec["accept_multiple"]:
            for pname, pipe in [("A", pipe_a_top5), ("B", pipe_b_top5), ("C", pipe_c_top5)]:
                nda_hits = sum(1 for r in pipe if is_nda_relevant(r, qspec["nda_keywords"]))
                print(f"  Q2 Pipeline {pname}: {nda_hits}/5 NDA-relevant")

        # Diagnostics for Q3 / Q4
        if label == "Q3":
            print(f"\n  Q3 SPECIAL TEST -- Section 41 Surfacing:")
            print(f"    Baseline A: {_rs(rank_a)}  |  Blend B: {_rs(rank_b)}  |  Protected C: {_rs(rank_c)}")
        if label == "Q4":
            print(f"\n  Q4 SPECIAL TEST -- Section 59 / 54 Surfacing:")
            print(f"    Baseline A: {_rs(rank_a)}  |  Blend B: {_rs(rank_b)}  |  Protected C: {_rs(rank_c)}")

        # Metrics
        metrics = {}
        if exp_cids:
            for pname, pipe in [("A_Base", pipe_a), ("B_Blend", pipe_b), ("C_Protect", pipe_c)]:
                metrics[pname] = {
                    "R@5":  recall_at_k(pipe, exp_cids, 5),
                    "MRR":  mrr(pipe, exp_cids),
                    "NDCG": ndcg_at_k(pipe, exp_cids, 5),
                }

        summary_rows.append({
            "label": label,
            "expected": exp_desc,
            "pool_size": pool_count,
            "rank_a": _rs(rank_a),
            "rank_b": _rs(rank_b),
            "rank_c": _rs(rank_c),
            "protected_count": protected_count,
            "metrics": metrics,
        })

    # ---------------------------------------------------------------- #
    # RESULTS SUMMARY TABLE
    # ---------------------------------------------------------------- #
    print(f"\n{SEP}")
    print("EXPERIMENT 7 COMPARISON SUMMARY TABLE")
    print(SEP)
    print(f"\n  {'Q':<4} {'Expected Evidence':<40} {'Pool Size':>10} {'A (Base)':>10} {'B (Blend)':>11} {'C (Protect)':>13}")
    print("  " + "-" * 95)
    for r in summary_rows:
        print(f"  {r['label']:<4} {r['expected'][:38]:<40} {r['pool_size']:>10} {r['rank_a']:>10} "
              f"{r['rank_b']:>11} {r['rank_c']:>13}")

    print(f"\n  Retrieval Metrics (where target defined):")
    print(f"  {'Q':<4} {'Pipeline':<12} {'R@5':>6} {'MRR':>6} {'NDCG@5':>8}")
    print("  " + "-" * 40)
    for r in summary_rows:
        for pname, m in r["metrics"].items():
            print(f"  {r['label']:<4} {pname:<12} {m['R@5']:>6.3f} {m['MRR']:>6.3f} {m['NDCG']:>8.3f}")

    # ---------------------------------------------------------------- #
    # OVERALL EXPERIMENT 7 VERDICT
    # ---------------------------------------------------------------- #
    print(f"\n{SEP}")
    print("OVERALL EXPERIMENT 7 VERDICT")
    print(SEP)
    print(f"""
  Selected Best Lambda: {best_lambda:.2f}
  Score Normalization: Min-Max Batch Normalization (range [0.0, 1.0])

  Verdict Selection:
    - PROTECTED EVIDENCE BETTER: If Pipeline C moves Section 41 (Q3) and Section 59/54 (Q4) into top 5 without breaking Q1/Q2.
    - BLENDING BETTER: If Pipeline B improves targets over Baseline A without requiring protection boosts.
    - BASELINE BETTER / NO CLEAR WINNER: If neither improves final top-5 ranks.
""")
    print(SEP)
    print("RECOMMENDED EXPERIMENT 8")
    print(SEP)
    print("""
  Full details in walkthrough artifact.
""")


if __name__ == "__main__":
    main()
