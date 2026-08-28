"""
run_experiment_6.py — Implicit Legal Query Candidate Recall Evaluation.

Compares:
  Baseline (Experiment 5): Original Query -> RRF -> Policy -> LegalAwareRanker -> BGE-Reranker
  Experiment 6:            LegalEntityLinker -> Multi-Source & Variant Retrieval
                           -> Candidate Union & Deduplication -> RRF -> Policy
                           -> LegalAwareRanker -> BGE-Reranker

Evaluates candidate pool diagnostics, query variants, target recall, final ranks,
and latency overhead across Q1, Q2, Q3, Q4.
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
from legal_rag.query.linker import LegalEntityLinker, LinkedQueryContext
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


def print_top5(results: list[dict], header: str, exp_chunks: list[str]) -> None:
    print(f"\n  {header}")
    print(f"  {'#':<4} {'Document':<30} {'Sec':<6} {'Pg':<4} {'ChunkID':<20}"
          f" {'Authority':<18} {'ContentType':<22} {'Score':>8}  Snippet")
    print("  " + "-" * 140)
    for rank, r in enumerate(results[:5], 1):
        doc   = (r.get("document_title") or r.get("document_id") or "")[:28]
        sec   = str(r.get("section_number") or "")[:5]
        pg    = str(r.get("page_start") or "")[:3]
        cid   = (r.get("chunk_id") or "")[:18]
        auth  = (r.get("source_authority") or "")[:16]
        ctype = (r.get("content_type") or "")[:20]
        score = (
            r.get("reranker_score")
            or r.get("legal_combined_score")
            or r.get("adjusted_score")
            or 0.0
        )
        snip  = (r.get("text") or "")[:55].replace("\n", " ")
        mark  = " <--" if r.get("chunk_id") in exp_chunks else ""
        print(f"  {rank:<4} {doc:<30} {sec:<6} {pg:<4} {cid:<20}"
              f" {auth:<18} {ctype:<22} {score:>8.4f}  {snip}{mark}")


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
    print("EXPERIMENT 6: Improve Candidate Recall for Implicit Legal Queries")
    print("  Baseline (Exp 5): Original Query -> RRF -> Policy -> LegalRanker -> BGE-Reranker")
    print("  Experiment 6:    Linker -> Multi-Source/Variant Retrieval -> Union & Dedup")
    print("                   -> RRF -> Policy -> LegalRanker -> BGE-Reranker")
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

    print("\nLoading BGE-Reranker-v2-M3 ...")
    t0 = time.perf_counter()
    bge_reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")
    bge_load_s = time.perf_counter() - t0
    print(f"  BGE Reranker loaded in {bge_load_s:.1f}s")

    summary_rows = []
    variant_reports = []

    for qspec in QUERIES:
        label    = qspec["label"]
        query    = qspec["query"]
        exp_cids = qspec["expected_chunks"]
        exp_desc = qspec["expected_desc"]
        nda_kw   = qspec["nda_keywords"]

        print(f"\n{THIN}")
        print(f"  {label}: {query}")
        print(f"  Expected: {exp_desc}")
        print(THIN)

        qa = analyze_query(query)
        linked_ctx = linker.link(qa)

        print(f"\n  QueryAnalyzer  -> intent={qa.intent.value}  acts={qa.act_names}  sec_refs={qa.section_refs}  cats={qa.category_hints}")
        print(f"  LegalEntityLinker -> implicit={linked_ctx.is_implicit}  action={linked_ctx.legal_action}  "
              f"sources={linked_ctx.candidate_sources}  variants={len(linked_ctx.query_variants)}")

        # ============================================================ #
        # BASELINE PIPELINE (Experiment 5)
        # ============================================================ #
        t0_base = time.perf_counter()
        base_struct = structured_retriever.retrieve_structured_candidates(qa)
        qvec_base   = embedder.embed_query(query)
        dense_base  = qdrant.search_children(qvec_base, 50, None)
        sparse_base = bm25.search(query, 50)
        fused_base  = reciprocal_rank_fusion(dense_base, sparse_base, k=60)

        fused_base_ids = {x.get("chunk_id") for x in fused_base}
        for s in base_struct:
            cid = s.get("chunk_id")
            if cid not in fused_base_ids:
                sc = s.copy(); sc["rrf_score"] = 1.0 / 61
                fused_base.append(sc); fused_base_ids.add(cid)

        fused_base = policy.apply_policy(query, qa, fused_base, bm25_metadata=bm25._chunk_metadata)
        annotate_legal(fused_base, global_registry)
        legal_base = legal_ranker.rank(fused_base, qa)
        top50_base = copy.deepcopy(legal_base[:50])
        final_base = bge_reranker.rerank(query, top50_base, top_k=5)
        annotate_legal(final_base, global_registry)
        lat_base = time.perf_counter() - t0_base

        base_pool_rank  = find_rank(fused_base, exp_cids)
        base_legal_rank = find_rank(legal_base, exp_cids)
        base_final_rank = find_rank(final_base, exp_cids)

        # ============================================================ #
        # EXPERIMENT 6 PIPELINE
        # ============================================================ #
        t0_exp = time.perf_counter()

        # 1. Candidate sources structured retrieval
        exp_struct = structured_retriever.retrieve_structured_candidates(qa)
        if linked_ctx.candidate_sources:
            # Create a mock QA with candidate sources as act_names to trigger structured lookup
            mock_qa = copy.deepcopy(qa)
            for c_src in linked_ctx.candidate_sources:
                if c_src not in mock_qa.act_names:
                    mock_qa.act_names.append(c_src)
            extra_struct = structured_retriever.retrieve_structured_candidates(mock_qa)
            seen_struct = {x.get("chunk_id") for x in exp_struct}
            for x in extra_struct:
                if x.get("chunk_id") not in seen_struct:
                    exp_struct.append(x)
                    seen_struct.add(x.get("chunk_id"))

        # 2. Original query retrieval
        orig_dense  = qdrant.search_children(qvec_base, 50, None)
        orig_sparse = bm25.search(query, 50)
        orig_cands  = reciprocal_rank_fusion(orig_dense, orig_sparse, k=60)

        # 3. Variant retrievals
        variant_cands_list = []
        variant_diagnostics = []

        for v in linked_ctx.query_variants:
            v_vec = embedder.embed_query(v.variant_text)
            v_dense = qdrant.search_children(v_vec, 50, None)
            v_sparse = bm25.search(v.variant_text, 50)
            v_fused = reciprocal_rank_fusion(v_dense, v_sparse, k=60)
            variant_cands_list.append(v_fused)

            v_has_exp = find_rank(v_fused, exp_cids) is not None if exp_cids else False
            variant_diagnostics.append({
                "variant_text": v.variant_text,
                "reason": v.reason,
                "target_source": v.target_source or "All",
                "retrieved_count": len(v_fused),
                "has_expected": v_has_exp,
            })

        # 4. Multi-Source Union & Deduplication
        all_candidate_lists = [orig_cands] + variant_cands_list

        # Track pool diagnostics
        orig_count = len(orig_cands)
        expanded_count = sum(len(vc) for vc in variant_cands_list)
        candidate_source_count = len(exp_struct)
        union_count = orig_count + expanded_count + candidate_source_count

        # Multi-list RRF fusion / deduplication
        dedup_scores: dict[str, float] = {}
        all_payloads: dict[str, dict] = {}

        for clist in all_candidate_lists:
            for rank_idx, item in enumerate(clist, 1):
                cid = item.get("chunk_id")
                if cid:
                    dedup_scores[cid] = dedup_scores.get(cid, 0.0) + 1.0 / (60 + rank_idx)
                    if cid not in all_payloads:
                        all_payloads[cid] = item.copy()

        for rank_idx, item in enumerate(exp_struct, 1):
            cid = item.get("chunk_id")
            if cid:
                dedup_scores[cid] = dedup_scores.get(cid, 0.0) + 1.0 / (60 + rank_idx)
                if cid not in all_payloads:
                    sc = item.copy()
                    sc["rrf_score"] = 1.0 / (60 + rank_idx)
                    all_payloads[cid] = sc

        fused_exp = []
        for cid, rrf_score in sorted(dedup_scores.items(), key=lambda x: x[1], reverse=True):
            payload = all_payloads[cid].copy()
            payload["rrf_score"] = rrf_score
            payload["chunk_id"] = cid
            fused_exp.append(payload)

        dedup_count = len(fused_exp)

        # 5. Policy adjustment
        fused_exp = policy.apply_policy(query, qa, fused_exp, bm25_metadata=bm25._chunk_metadata)
        annotate_legal(fused_exp, global_registry)

        # 6. LegalAwareRanker
        legal_exp = legal_ranker.rank(fused_exp, qa)

        # 7. BGE Reranker
        top50_exp = copy.deepcopy(legal_exp[:50])
        final_exp = bge_reranker.rerank(query, top50_exp, top_k=5)
        annotate_legal(final_exp, global_registry)
        lat_exp = time.perf_counter() - t0_exp

        exp_pool_rank  = find_rank(fused_exp, exp_cids)
        exp_legal_rank = find_rank(legal_exp, exp_cids)
        exp_final_rank = find_rank(final_exp, exp_cids)

        # Output diagnostics for this query
        print(f"\n  BASELINE (Exp 5)   -> Pool Size={len(fused_base)}  RRF Rank={_rsp(base_pool_rank, len(fused_base))}  Legal Rank={_rsp(base_legal_rank, len(fused_base))}  Final Rank={_rs(base_final_rank)}  Lat={lat_base:.2f}s")
        print(f"  EXPERIMENT 6       -> Pool Size={dedup_count}  RRF Rank={_rsp(exp_pool_rank, dedup_count)}  Legal Rank={_rsp(exp_legal_rank, dedup_count)}  Final Rank={_rs(exp_final_rank)}  Lat={lat_exp:.2f}s")

        print(f"\n  Candidate Pool Diagnostics:")
        print(f"    - Original query candidates:       {orig_count}")
        print(f"    - Expanded query candidates:       {expanded_count}")
        print(f"    - Candidate-source candidates:     {candidate_source_count}")
        print(f"    - Total union count (pre-dedup):   {union_count}")
        print(f"    - Deduplicated candidate count:    {dedup_count}")

        # Top-5 comparisons
        print_top5(final_base, "Baseline Pipeline (Exp 5) Top 5:", exp_cids)
        print_top5(final_exp,  "Experiment 6 Pipeline Top 5:", exp_cids)

        # Q2 special NDA check
        if qspec["accept_multiple"]:
            for name, pipe in [("Baseline", final_base), ("Exp 6", final_exp)]:
                nda_hits   = sum(1 for r in pipe if is_nda_relevant(r, nda_kw))
                titles     = [(r.get("document_title") or r.get("document_id") or "")[:30] for r in pipe]
                unique_docs = len(set(titles))
                print(f"  Q2 {name}: {nda_hits}/5 NDA-relevant, {unique_docs} unique docs")

        # Q3 special check
        if label == "Q3":
            print(f"\n  Q3 SPECIAL CHECK -- Section 41 Surfacing:")
            print(f"    Baseline Section 41 rank: {_rs(base_final_rank)}")
            print(f"    Exp 6 Section 41 rank:    {_rs(exp_final_rank)}")

        # Q4 special check
        if label == "Q4":
            sec59_base = find_rank(final_base, ["chk_01051fb4680e"])
            sec54_base = find_rank(final_base, ["chk_e83012ad27b7"])
            sec59_exp  = find_rank(final_exp,  ["chk_01051fb4680e"])
            sec54_exp  = find_rank(final_exp,  ["chk_e83012ad27b7"])
            print(f"\n  Q4 SPECIAL CHECK -- Section 59 / 54 Surfacing:")
            print(f"    Baseline: sec59={_rs(sec59_base)}  sec54={_rs(sec54_base)}")
            print(f"    Exp 6:    sec59={_rs(sec59_exp)}  sec54={_rs(sec54_exp)}")

        summary_rows.append({
            "label": label,
            "expected": exp_desc,
            "base_pool_rank": _rsp(base_pool_rank, len(fused_base)),
            "exp_pool_rank":  _rsp(exp_pool_rank, dedup_count),
            "base_legal_rank": _rsp(base_legal_rank, len(fused_base)),
            "exp_legal_rank":  _rsp(exp_legal_rank, dedup_count),
            "base_final_rank": _rs(base_final_rank),
            "exp_final_rank":  _rs(exp_final_rank),
            "orig_count": orig_count,
            "expanded_count": expanded_count,
            "candidate_source_count": candidate_source_count,
            "dedup_count": dedup_count,
            "lat_base": lat_base,
            "lat_exp": lat_exp,
        })

        if variant_diagnostics:
            variant_reports.append({
                "label": label,
                "variants": variant_diagnostics,
            })

    # ============================================================ #
    # RESULTS REPORTING
    # ============================================================ #

    print(f"\n{SEP}")
    print("EXPERIMENT 6 COMPARISON SUMMARY TABLE")
    print(SEP)
    print(f"\n  {'Q':<4} {'Expected Evidence':<40} {'Base RRF':>9} {'Exp6 RRF':>9} {'Base Legal':>11} {'Exp6 Legal':>11} {'Base Final':>11} {'Exp6 Final':>11}")
    print("  " + "-" * 110)
    for r in summary_rows:
        print(f"  {r['label']:<4} {r['expected'][:38]:<40} {r['base_pool_rank']:>9} {r['exp_pool_rank']:>9} "
              f"{r['base_legal_rank']:>11} {r['exp_legal_rank']:>11} {r['base_final_rank']:>11} {r['exp_final_rank']:>11}")

    print(f"\n{SEP}")
    print("QUERY VARIANT REPORT (Q3 & Q4)")
    print(SEP)
    for vrep in variant_reports:
        print(f"\n  Query: {vrep['label']}")
        for idx, v in enumerate(vrep['variants'], 1):
            print(f"    Variant {idx}: \"{v['variant_text']}\"")
            print(f"      Reason:           {v['reason']}")
            print(f"      Target Source:    {v['target_source']}")
            print(f"      Candidates:       {v['retrieved_count']}")
            print(f"      Contains Target:  {v['has_expected']}")

    print(f"\n{SEP}")
    print("PERFORMANCE & LATENCY OVERHEAD")
    print(SEP)
    print(f"\n  {'Q':<4} {'Base Pool':>10} {'Exp6 Pool':>10} {'Extra Calls':>12} {'Base Latency':>13} {'Exp6 Latency':>13}")
    print("  " + "-" * 66)
    for r in summary_rows:
        extra_calls = 2 * (1 if r['candidate_source_count'] > 0 else 0) + 2 * (1 if r['expanded_count'] > 0 else 0)
        print(f"  {r['label']:<4} {r['orig_count']:>10} {r['dedup_count']:>10} {extra_calls:>12} {r['lat_base']:>12.2f}s {r['lat_exp']:>12.2f}s")

    print(f"\n{SEP}")
    print("EXPERIMENT 6 OVERALL VERDICT & EXPERIMENT 7 RECOMMENDATION")
    print(SEP)
    print("""
  Full details in walkthrough artifact.
""")


if __name__ == "__main__":
    main()
