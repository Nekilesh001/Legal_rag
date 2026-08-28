"""
run_experiment_10.py — Multi-Evidence Retrieval for Broad Breach/Remedy Queries.

Compares:
  Experiment 9 (Control):  Parent-Contextual Reranking -> Top 5 ranked candidates directly
  Experiment 10:           Multi-Evidence Retrieval -> Parent-Contextual Reranking
                           -> EvidenceSelector Diversity Filtering -> Final Evidence Set

Evaluates broad-query detection, section diversity, suppression of definition noise,
and final evidence set composition for Q4, while ensuring zero regression for Q1, Q2, Q3.
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
)
from legal_rag.retrieval.context_formatter import apply_rerank_formatting
from legal_rag.retrieval.evidence_selector import EvidenceSelector

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
        "expected_desc": "Sale of Goods Act Sections 59 / 54 / 60",
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


def print_evidence_table(results: list[dict], header: str, exp_chunks: list[str]) -> None:
    print(f"\n  {header}")
    print(f"  {'#':<3} {'Document':<24} {'Sec':<5} {'Pg':<3} {'ChunkID':<18} "
          f"{'ContentType':<18} {'Matched Concepts':<22} {'Score':>7} Snippet")
    print("  " + "-" * 135)
    for rank, r in enumerate(results[:5], 1):
        doc       = (r.get("document_title") or r.get("document_id") or "")[:22]
        sec       = str(r.get("section_number") or "")[:4]
        pg        = str(r.get("page_start") or "")[:3]
        cid       = (r.get("chunk_id") or "")[:16]
        ctype     = (r.get("content_type") or "")[:16]
        concepts  = ",".join(r.get("legal_breakdown", {}).get("matched_concepts", []))[:20] or "-"
        final_sc  = r.get("protected_score", r.get("blended_score", r.get("reranker_score", 0.0)))
        snip      = (r.get("text") or "")[:45].replace("\n", " ")
        mark      = " <--" if r.get("chunk_id") in exp_chunks else ""
        print(f"  {rank:<3} {doc:<24} {sec:<5} {pg:<3} {cid:<18} "
              f"{ctype:<18} {concepts:<22} {final_sc:>7.3f} {snip}{mark}")


def is_nda_relevant(r: dict, keywords: tuple) -> bool:
    title = (r.get("document_title") or r.get("document_id") or "").lower()
    return any(kw in title for kw in keywords)


def _rs(rank: int | None) -> str:
    return str(rank) if rank else ">5"


# ------------------------------------------------------------------ #
# Main Execution
# ------------------------------------------------------------------ #

def main() -> None:
    cfg = get_config()

    print(SEP)
    print("EXPERIMENT 10: Multi-Evidence Retrieval for Broad Breach/Remedy Queries")
    print("  Experiment 9 (Control):  Parent-Contextual Reranking -> Top 5 ranked candidates directly")
    print("  Experiment 10:           Multi-Evidence Retrieval -> Parent-Contextual Reranking")
    print("                           -> EvidenceSelector Diversity Filtering -> Final Evidence Set")
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

    exp8_weights = LegalRankerWeights(intent_content_pref=8.0, concept_match=4.0)
    legal_ranker = LegalAwareRanker(weights=exp8_weights, registry=global_registry)

    linker = LegalEntityLinker()
    blender = ScoreBlender(lambda_weight=0.50)
    protection_handler = ProtectedEvidenceHandler(tier_1_boost=0.35, tier_2_boost=0.20)
    evidence_selector  = EvidenceSelector(max_chunks_per_section=1, max_evidence_items=5)

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
        is_broad = evidence_selector.is_broad_multi_evidence_query(qa)

        print(f"\n  Broad Query Detection -> is_broad={is_broad}  intent={qa.intent.value}  cats={qa.category_hints}")
        if is_broad:
            print("  Generated Controlled Retrieval Concepts:")
            for idx, v in enumerate(linked_ctx.query_variants, 1):
                print(f"    Concept {idx}: \"{v.variant_text}\" (Reason: {v.reason})")

        # Multi-source candidate generation
        t0_ret = time.perf_counter()
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

        # LegalAwareRanker -> Top 50
        legal_sorted = legal_ranker.rank(fused, qa)
        top50        = legal_sorted[:50]

        # Apply Parent-Contextual Formatting (Exp 9 mode)
        cands_formatted = apply_rerank_formatting(copy.deepcopy(top50), mode="full", registry=global_registry)

        # Rerank with BGE Cross-Encoder
        bge_reranked = bge_reranker.rerank(query, cands_formatted, top_k=50)
        annotate_legal(bge_reranked, global_registry)

        # Score blending & Protection handling
        blended  = blender.blend_batch(bge_reranked)
        protected = protection_handler.apply_protection(blended, qa, score_key="blended_score")
        tot_lat  = time.perf_counter() - t0_ret

        # ============================================================ #
        # EXPERIMENT 9 (Control): Top 5 directly from ranked list
        # ============================================================ #
        exp9_top5 = protected[:5]

        # ============================================================ #
        # EXPERIMENT 10: EvidenceSelector Diversity Filtering -> Final Evidence Set
        # ============================================================ #
        exp10_evidence = evidence_selector.select_final_evidence_set(protected, qa)

        rank_exp9  = find_rank(exp9_top5, exp_cids)
        rank_exp10 = find_rank(exp10_evidence, exp_cids)

        print(f"\n  Candidate Pool Size: {pool_size}  |  Total Latency: {tot_lat:.2f}s")
        print(f"  Exp 9 Top-5 Target Rank: {_rs(rank_exp9)}  |  Exp 10 Final Evidence Set Target Rank: {_rs(rank_exp10)}")

        print_evidence_table(exp9_top5, "Experiment 9 (Control): Top-5 Ranked List:", exp_cids)
        print_evidence_table(exp10_evidence, "Experiment 10: Final Evidence Set (Diversity Filtered):", exp_cids)

        # Q2 special NDA check
        if qspec["accept_multiple"]:
            for name, ev_set in [("Exp 9 Top5", exp9_top5), ("Exp 10 Final Evidence", exp10_evidence)]:
                nda_hits = sum(1 for r in ev_set if is_nda_relevant(r, qspec["nda_keywords"]))
                titles = [(r.get("document_title") or r.get("document_id") or "")[:30] for r in ev_set]
                print(f"  Q2 {name}: {nda_hits}/5 NDA-relevant, {len(set(titles))} unique docs")

        # Q4 Special Evidence Breakdown
        if label == "Q4":
            print(f"\n  Q4 FINAL EVIDENCE SET SECTION DIVERSITY BREAKDOWN:")
            secs_selected = [
                f"Sec {str(r.get('section_number'))} ({r.get('section_title') or r.get('document_title')})"
                for r in exp10_evidence
            ]
            for idx, sec_info in enumerate(secs_selected, 1):
                print(f"    Evidence #{idx}: {sec_info}")

        metrics = {}
        if exp_cids:
            for name, ev_set in [("Exp9_Control", exp9_top5), ("Exp10_Final", exp10_evidence)]:
                metrics[name] = {
                    "R@5":  recall_at_k(ev_set, exp_cids, 5),
                    "MRR":  mrr(ev_set, exp_cids),
                    "NDCG": ndcg_at_k(ev_set, exp_cids, 5),
                }

        summary_rows.append({
            "label": label,
            "expected": exp_desc,
            "pool_size": pool_size,
            "is_broad": is_broad,
            "rank_exp9": _rs(rank_exp9),
            "rank_exp10": _rs(rank_exp10),
            "tot_lat": tot_lat,
            "metrics": metrics,
        })

    # ============================================================ #
    # RESULTS REPORTING
    # ============================================================ #

    print(f"\n{SEP}")
    print("EXPERIMENT 10 COMPARISON SUMMARY TABLE")
    print(SEP)
    print(f"\n  {'Q':<4} {'Expected Evidence':<36} {'Broad?':>7} {'Pool Size':>10} {'Exp 9 Rank':>11} {'Exp 10 Rank':>12}")
    print("  " + "-" * 88)
    for r in summary_rows:
        print(f"  {r['label']:<4} {r['expected'][:34]:<36} {str(r['is_broad']):>7} {r['pool_size']:>10} "
              f"{r['rank_exp9']:>11} {r['rank_exp10']:>12}")

    print(f"\n  Retrieval Metrics (where target defined):")
    print(f"  {'Q':<4} {'Pipeline':<14} {'R@5':>6} {'MRR':>6} {'NDCG@5':>8}")
    print("  " + "-" * 42)
    for r in summary_rows:
        for mname, m in r["metrics"].items():
            print(f"  {r['label']:<4} {mname:<14} {m['R@5']:>6.3f} {m['MRR']:>6.3f} {m['NDCG']:>8.3f}")

    print(f"\n{SEP}")
    print("EXPERIMENT 10 OVERALL VERDICT & EXPERIMENT 11 RECOMMENDATION")
    print(SEP)
    print("""
  Full details in walkthrough artifact.
""")


if __name__ == "__main__":
    main()
