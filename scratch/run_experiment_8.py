"""
run_experiment_8.py — Intent-Aware Content-Type and Legal-Concept Ranking Evaluation.

Compares:
  Pipeline A (Exp 7 Baseline): LegalAwareRanker (Exp 7) + Protected Evidence + BGE-Reranker
  Pipeline B (Exp 8):          LegalAwareRanker (with Intent-Aware Content-Type Preference & Concept Alignment)
                               + Protected Evidence + BGE-Reranker

Evaluates intra-document ranking improvements for Q3 (Section 41 vs Schedule III/Sec 1/Sec 2)
and Q4 (Section 59/54 vs Sec 4/Sec 2), while verifying Q1 and Q2 regressions.
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
    extract_concepts_from_text,
)
from legal_rag.retrieval.structured import StructuredQueryRetriever
from legal_rag.retrieval.hybrid import CrossEncoderReranker, reciprocal_rank_fusion
from legal_rag.retrieval.policy import MetadataRetrievalPolicy
from legal_rag.retrieval.blender import (
    ScoreBlender,
    ProtectedEvidenceHandler,
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
          f"{'ContentType':<20} {'Concepts':<18} {'LegalScore':>10} {'RawBGE':>7} {'Tier':<18} {'FinalScore':>10} Snippet")
    print("  " + "-" * 155)
    for rank, r in enumerate(results[:5], 1):
        doc       = (r.get("document_title") or r.get("document_id") or "")[:22]
        sec       = str(r.get("section_number") or "")[:4]
        pg        = str(r.get("page_start") or "")[:3]
        cid       = (r.get("chunk_id") or "")[:16]
        ctype     = (r.get("content_type") or "")[:18]
        concepts  = ",".join(r.get("legal_breakdown", {}).get("matched_concepts", []))[:16] or "-"
        leg_score = r.get("legal_score", 0.0)
        raw_bge   = r.get("reranker_score", 0.0)
        tier      = r.get("protection_tier", "NONE")[:16]
        final_sc  = r.get("protected_score", r.get("blended_score", raw_bge))
        snip      = (r.get("text") or "")[:45].replace("\n", " ")
        mark      = " <--" if r.get("chunk_id") in exp_chunks else ""
        print(f"  {rank:<3} {doc:<24} {sec:<5} {pg:<3} {cid:<18} "
              f"{ctype:<20} {concepts:<18} {leg_score:>10.1f} {raw_bge:>7.3f} {tier:<18} {final_sc:>10.3f} {snip}{mark}")


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
    print("EXPERIMENT 8: Intent-Aware Content-Type and Legal-Concept Ranking")
    print("  Pipeline A (Exp 7 Baseline): LegalAwareRanker (Exp 7) + Protected Evidence + BGE-Reranker")
    print("  Pipeline B (Exp 8):          LegalAwareRanker (with Intent Preference & Concept Alignment)")
    print("                               + Protected Evidence + BGE-Reranker")
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

    # Ranker for Exp 7 (Baseline Control): disable intent_content_pref and concept_match
    exp7_weights = LegalRankerWeights(intent_content_pref=0.0, concept_match=0.0)
    exp7_ranker  = LegalAwareRanker(weights=exp7_weights, registry=global_registry)

    # Ranker for Exp 8: active intent_content_pref=8.0 and concept_match=4.0
    exp8_weights = LegalRankerWeights(intent_content_pref=8.0, concept_match=4.0)
    exp8_ranker  = LegalAwareRanker(weights=exp8_weights, registry=global_registry)

    linker = LegalEntityLinker()
    blender = ScoreBlender(lambda_weight=0.50)
    protection_handler = ProtectedEvidenceHandler(tier_1_boost=0.35, tier_2_boost=0.20)

    print("\nLoading BGE-Reranker-v2-M3 ...")
    t0 = time.perf_counter()
    bge_reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")
    bge_load_s = time.perf_counter() - t0
    print(f"  BGE Reranker loaded in {bge_load_s:.1f}s")

    summary_rows = []

    for qspec in QUERIES:
        label    = qspec["label"]
        query    = qspec["query"]
        exp_cids = qspec["expected_chunks"]
        exp_desc = qspec["expected_desc"]

        print(f"\n{THIN}")
        print(f"  {label}: {query}")
        print(f"  Expected: {exp_desc}  |  Rerank Window: 50")
        print(THIN)

        qa = analyze_query(query)
        linked_ctx = linker.link(qa)

        # Multi-source candidate generation (shared candidates for exact comparison)
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
        pool_size = len(fused)

        # ============================================================ #
        # PIPELINE A: Exp 7 Baseline (without intent & concept ranking)
        # ============================================================ #
        t0_a = time.perf_counter()
        legal_sorted_a = exp7_ranker.rank(copy.deepcopy(fused), qa)
        top50_a = copy.deepcopy(legal_sorted_a[:50])
        bge_a = bge_reranker.rerank(query, top50_a, top_k=50)
        annotate_legal(bge_a, global_registry)
        blend_a = blender.blend_batch(bge_a)
        pipe_a  = protection_handler.apply_protection(blend_a, qa, score_key="blended_score")
        lat_a   = time.perf_counter() - t0_a

        # ============================================================ #
        # PIPELINE B: Exp 8 (with Intent Preference & Concept Alignment)
        # ============================================================ #
        t0_b = time.perf_counter()
        legal_sorted_b = exp8_ranker.rank(copy.deepcopy(fused), qa)
        top50_b = copy.deepcopy(legal_sorted_b[:50])
        bge_b = bge_reranker.rerank(query, top50_b, top_k=50)
        annotate_legal(bge_b, global_registry)
        blend_b = blender.blend_batch(bge_b)
        pipe_b  = protection_handler.apply_protection(blend_b, qa, score_key="blended_score")
        lat_b   = time.perf_counter() - t0_b

        rank_a = find_rank(pipe_a, exp_cids)
        rank_b = find_rank(pipe_b, exp_cids)

        print(f"\n  Candidate Pool Size: {pool_size}")
        print(f"  Exp 7 Rank: {_rs(rank_a)}  |  Exp 8 Rank: {_rs(rank_b)}"
              f"  |  Latency Exp7={lat_a:.2f}s  Exp8={lat_b:.2f}s")

        print_detailed_top5(pipe_a, "Pipeline A — Exp 7 Baseline, Top 5:", exp_cids)
        print_detailed_top5(pipe_b, "Pipeline B — Exp 8 Intent & Concept Alignment, Top 5:", exp_cids)

        # Regression check for Q2
        if qspec["accept_multiple"]:
            for name, pipe in [("Exp 7", pipe_a[:5]), ("Exp 8", pipe_b[:5])]:
                nda_hits = sum(1 for r in pipe if is_nda_relevant(r, qspec["nda_keywords"]))
                titles = [(r.get("document_title") or r.get("document_id") or "")[:30] for r in pipe]
                unique_docs = len(set(titles))
                print(f"  Q2 {name}: {nda_hits}/5 NDA-relevant, {unique_docs} unique docs")

        # Q3 Section Ranks Validation
        if label == "Q3":
            print(f"\n  Q3 INTRA-DOCUMENT SECTION RANKINGS (Tamil Nadu Shops Act):")
            for pname, pipe in [("Exp 7 Baseline", pipe_a), ("Exp 8", pipe_b)]:
                r_sec41 = find_rank(pipe, ["chk_56b1160532cc"])
                r_sec1  = next((i for i, r in enumerate(pipe, 1) if str(r.get("section_number")) == "1"), None)
                r_sec2  = next((i for i, r in enumerate(pipe, 1) if str(r.get("section_number")) == "2"), None)
                r_sch3  = next((i for i, r in enumerate(pipe, 1) if "III" in str(r.get("section_number"))), None)
                print(f"    {pname:<15}: Sec 41={_rs(r_sec41)}  Sec 1={_rs(r_sec1)}  Sec 2={_rs(r_sec2)}  Sch III={_rs(r_sch3)}")

        # Q4 Section Ranks Validation
        if label == "Q4":
            print(f"\n  Q4 INTRA-DOCUMENT SECTION RANKINGS (Sale of Goods Act):")
            for pname, pipe in [("Exp 7 Baseline", pipe_a), ("Exp 8", pipe_b)]:
                r_sec59 = find_rank(pipe, ["chk_01051fb4680e"])
                r_sec54 = find_rank(pipe, ["chk_e83012ad27b7"])
                r_sec4  = next((i for i, r in enumerate(pipe, 1) if str(r.get("section_number")) == "4"), None)
                r_sec2  = next((i for i, r in enumerate(pipe, 1) if str(r.get("section_number")) == "2"), None)
                print(f"    {pname:<15}: Sec 59={_rs(r_sec59)}  Sec 54={_rs(r_sec54)}  Sec 4={_rs(r_sec4)}  Sec 2={_rs(r_sec2)}")

        metrics: dict[str, dict] = {}
        if exp_cids:
            for pname, pipe in [("Exp7_Base", pipe_a), ("Exp8", pipe_b)]:
                metrics[pname] = {
                    "R@5":  recall_at_k(pipe, exp_cids, 5),
                    "MRR":  mrr(pipe, exp_cids),
                    "NDCG": ndcg_at_k(pipe, exp_cids, 5),
                }

        summary_rows.append({
            "label": label,
            "expected": exp_desc,
            "pool_size": pool_size,
            "rank_a": _rs(rank_a),
            "rank_b": _rs(rank_b),
            "lat_a": lat_a,
            "lat_b": lat_b,
            "metrics": metrics,
        })

    # ============================================================ #
    # RESULTS REPORTING
    # ============================================================ #

    print(f"\n{SEP}")
    print("EXPERIMENT 8 COMPARISON SUMMARY TABLE")
    print(SEP)
    print(f"\n  {'Q':<4} {'Expected Evidence':<40} {'Pool Size':>10} {'Exp 7 Rank':>11} {'Exp 8 Rank':>11}")
    print("  " + "-" * 80)
    for r in summary_rows:
        print(f"  {r['label']:<4} {r['expected'][:38]:<40} {r['pool_size']:>10} {r['rank_a']:>11} {r['rank_b']:>11}")

    print(f"\n  Retrieval Metrics (where target defined):")
    print(f"  {'Q':<4} {'Pipeline':<12} {'R@5':>6} {'MRR':>6} {'NDCG@5':>8}")
    print("  " + "-" * 40)
    for r in summary_rows:
        for pname, m in r["metrics"].items():
            print(f"  {r['label']:<4} {pname:<12} {m['R@5']:>6.3f} {m['MRR']:>6.3f} {m['NDCG']:>8.3f}")

    print(f"\n{SEP}")
    print("PERFORMANCE & OVERHEAD REPORT")
    print(SEP)
    print(f"\n  {'Q':<4} {'Pool Size':>10} {'Rerank Window':>14} {'Exp 7 Latency':>14} {'Exp 8 Latency':>14} {'Overhead':>10}")
    print("  " + "-" * 72)
    for r in summary_rows:
        oh = r['lat_b'] - r['lat_a']
        print(f"  {r['label']:<4} {r['pool_size']:>10} {50:>14} {r['lat_a']:>13.2f}s {r['lat_b']:>13.2f}s {oh:>9.2f}s")

    print(f"\n{SEP}")
    print("EXPERIMENT 8 OVERALL VERDICT & EXPERIMENT 9 RECOMMENDATION")
    print(SEP)
    print("""
  Full details in walkthrough artifact.
""")


if __name__ == "__main__":
    main()
