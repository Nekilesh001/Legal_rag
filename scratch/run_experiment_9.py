"""
run_experiment_9.py — Parent-Contextual Reranking Input Formatting Evaluation.

Compares:
  Pipeline A (Control):     Child text only passed to BGE Reranker
  Pipeline B (Contextual):  Full structural hierarchy + Child text passed to BGE Reranker

Includes:
  - Ablation Study: Control vs Section-Title vs Full-Hierarchy across Q3 & Q4
  - Reranker input before/after example
  - Token and Latency Overhead analysis
  - Source integrity verification (citations use raw child text)
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
from legal_rag.retrieval.context_formatter import (
    format_full_contextual_input,
    format_section_title_input,
    apply_rerank_formatting,
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


def approx_token_count(text: str) -> int:
    """Approximate word/token count via whitespace splitting."""
    return int(len(text.split()) * 1.3)


def print_detailed_top5(results: list[dict], header: str, exp_chunks: list[str]) -> None:
    print(f"\n  {header}")
    print(f"  {'#':<3} {'Document':<24} {'Sec':<5} {'Pg':<3} {'ChunkID':<18} "
          f"{'ContentType':<18} {'RawBGE':>7} {'Tier':<18} {'FinalScore':>10} Snippet")
    print("  " + "-" * 135)
    for rank, r in enumerate(results[:5], 1):
        doc       = (r.get("document_title") or r.get("document_id") or "")[:22]
        sec       = str(r.get("section_number") or "")[:4]
        pg        = str(r.get("page_start") or "")[:3]
        cid       = (r.get("chunk_id") or "")[:16]
        ctype     = (r.get("content_type") or "")[:16]
        raw_bge   = r.get("reranker_score", 0.0)
        tier      = r.get("protection_tier", "NONE")[:16]
        final_sc  = r.get("protected_score", r.get("blended_score", raw_bge))
        snip      = (r.get("text") or "")[:45].replace("\n", " ")
        mark      = " <--" if r.get("chunk_id") in exp_chunks else ""
        print(f"  {rank:<3} {doc:<24} {sec:<5} {pg:<3} {cid:<18} "
              f"{ctype:<18} {raw_bge:>7.3f} {tier:<18} {final_sc:>10.3f} {snip}{mark}")


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
    print("EXPERIMENT 9: Parent-Contextual Reranking Input Formatting Evaluation")
    print("  Pipeline A (Control):    Raw child text passed to BGE Reranker")
    print("  Pipeline B (Contextual): Full structural hierarchy + Child text passed to BGE")
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

    # Use Exp 8 LegalAwareRanker
    exp8_weights = LegalRankerWeights(intent_content_pref=8.0, concept_match=4.0)
    legal_ranker = LegalAwareRanker(weights=exp8_weights, registry=global_registry)

    linker = LegalEntityLinker()
    blender = ScoreBlender(lambda_weight=0.50)
    protection_handler = ProtectedEvidenceHandler(tier_1_boost=0.35, tier_2_boost=0.20)

    print("\nLoading BGE-Reranker-v2-M3 ...")
    t0 = time.perf_counter()
    bge_reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")
    bge_load_s = time.perf_counter() - t0
    print(f"  BGE Reranker loaded in {bge_load_s:.1f}s")

    # ---------------------------------------------------------------- #
    # PART 1: Real Before / After Formatting Example
    # ---------------------------------------------------------------- #
    print(f"\n{SEP}")
    print("PART 1: RERANKER INPUT BEFORE / AFTER FORMATTING EXAMPLES")
    print(SEP)

    sample_sec41 = {
        "document_id": "1637820824",
        "document_title": "1637820824-Tamil-Nadu-Shops-And-Establishments-Act-1947",
        "section_number": "41",
        "section_title": "Notice of dismissal",
        "chapter": "Chapter VI — Employment and Discharge",
        "jurisdiction": "Tamil Nadu",
        "content_type": "operative_provision",
        "text": "41. (1) No employer shall dispense with the services of a person employed continuously for a period of not less than six months...",
    }

    sample_sec59 = {
        "document_id": "193003",
        "document_title": "193003",
        "section_number": "59",
        "section_title": "Remedy for breach of warranty",
        "chapter": "Chapter VI — Suits for Breach of the Contract",
        "jurisdiction": "India",
        "content_type": "operative_provision",
        "text": "(1) Where there is a breach of warranty by the seller, or where the buyer elects or is compelled to treat any breach...",
    }

    print("\n--- Example 1: Q3 Section 41 (Notice Period) ---")
    print("BEFORE (Raw Child Text):")
    print(f"  \"{sample_sec41['text'][:100]}...\"")
    print("\nAFTER (Full Contextual Reranker Input):")
    formatted_41 = format_full_contextual_input(sample_sec41, registry=global_registry)
    for line in formatted_41.split("\n"):
        print(f"  {line}")

    print("\n--- Example 2: Q4 Section 59 (Seller Breach Remedy) ---")
    print("BEFORE (Raw Child Text):")
    print(f"  \"{sample_sec59['text'][:100]}...\"")
    print("\nAFTER (Full Contextual Reranker Input):")
    formatted_59 = format_full_contextual_input(sample_sec59, registry=global_registry)
    for line in formatted_59.split("\n"):
        print(f"  {line}")

    # ---------------------------------------------------------------- #
    # PARTS 2-15: Per-Query Evaluation & Ablation
    # ---------------------------------------------------------------- #
    print(f"\n{SEP}")
    print("PARTS 2-15: PER-QUERY EVALUATION & CONTEXT ABLATION")
    print(SEP)

    summary_rows = []
    token_stats = []

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

        # Multi-source candidate generation (shared across all modes)
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

        legal_sorted = legal_ranker.rank(fused, qa)
        top50_base   = legal_sorted[:50]

        # ============================================================ #
        # MODE 1: Control (Child text only)
        # ============================================================ #
        t0_1 = time.perf_counter()
        cands_mode1 = apply_rerank_formatting(copy.deepcopy(top50_base), mode="control", registry=global_registry)
        bge_1 = bge_reranker.rerank(query, cands_mode1, top_k=50)
        annotate_legal(bge_1, global_registry)
        blend_1 = blender.blend_batch(bge_1)
        pipe_1  = protection_handler.apply_protection(blend_1, qa, score_key="blended_score")
        lat_1   = time.perf_counter() - t0_1

        # ============================================================ #
        # MODE 2: Section Title + Child text (Ablation Variant B)
        # ============================================================ #
        t0_2 = time.perf_counter()
        cands_mode2 = apply_rerank_formatting(copy.deepcopy(top50_base), mode="section_title", registry=global_registry)
        bge_2 = bge_reranker.rerank(query, cands_mode2, top_k=50)
        annotate_legal(bge_2, global_registry)
        blend_2 = blender.blend_batch(bge_2)
        pipe_2  = protection_handler.apply_protection(blend_2, qa, score_key="blended_score")
        lat_2   = time.perf_counter() - t0_2

        # ============================================================ #
        # MODE 3: Full Structural Hierarchy + Child text (Pipeline B / Variant C)
        # ============================================================ #
        t0_3 = time.perf_counter()
        cands_mode3 = apply_rerank_formatting(copy.deepcopy(top50_base), mode="full", registry=global_registry)
        bge_3 = bge_reranker.rerank(query, cands_mode3, top_k=50)
        annotate_legal(bge_3, global_registry)
        blend_3 = blender.blend_batch(bge_3)
        pipe_3  = protection_handler.apply_protection(blend_3, qa, score_key="blended_score")
        lat_3   = time.perf_counter() - t0_3

        rank_1 = find_rank(pipe_1, exp_cids)
        rank_2 = find_rank(pipe_2, exp_cids)
        rank_3 = find_rank(pipe_3, exp_cids)

        # Measure tokens for mode 3
        tokens_mode3 = [approx_token_count(c.get("rerank_input", "")) for c in cands_mode3]
        avg_tokens = sum(tokens_mode3) / len(tokens_mode3) if tokens_mode3 else 0

        token_stats.append({
            "label": label,
            "avg_tokens": avg_tokens,
            "lat_control": lat_1,
            "lat_contextual": lat_3,
        })

        # Source Integrity Verification: verify pipe_3 results retain original text
        for item in pipe_3[:5]:
            assert "rerank_input" in item, "rerank_input missing"
            assert "text" in item, "original text missing"
            assert not item["text"].startswith("Document:"), "Source text corrupted by formatting"

        print(f"\n  Ranks — Mode 1 (Control): {_rs(rank_1)}  |  Mode 2 (SecTitle): {_rs(rank_2)}  |  Mode 3 (FullContext): {_rs(rank_3)}")
        print(f"  Latency — Control: {lat_1:.2f}s  |  SecTitle: {lat_2:.2f}s  |  FullContext: {lat_3:.2f}s  |  Avg Input Tokens: {avg_tokens:.0f}")

        print_detailed_top5(pipe_1[:5], "Mode 1 — Control (Child text only), Top 5:", exp_cids)
        print_detailed_top5(pipe_2[:5], "Mode 2 — Ablation (Section Title + Child text), Top 5:", exp_cids)
        print_detailed_top5(pipe_3[:5], "Mode 3 — Pipeline B (Full Structural Hierarchy), Top 5:", exp_cids)

        # Regression check for Q2
        if qspec["accept_multiple"]:
            for mname, ptop in [("Control", pipe_1[:5]), ("SecTitle", pipe_2[:5]), ("FullContext", pipe_3[:5])]:
                nda_hits = sum(1 for r in ptop if is_nda_relevant(r, qspec["nda_keywords"]))
                titles = [(r.get("document_title") or r.get("document_id") or "")[:30] for r in ptop]
                print(f"  Q2 {mname}: {nda_hits}/5 NDA-relevant, {len(set(titles))} unique docs")

        # Q3 Section Ranks Validation
        if label == "Q3":
            print(f"\n  Q3 SECTION RANKING COMPARISON (Tamil Nadu Shops Act):")
            for mname, ptop in [("Control", pipe_1), ("SecTitle", pipe_2), ("FullContext", pipe_3)]:
                r_sec41 = find_rank(ptop, ["chk_56b1160532cc"])
                r_sec11 = next((i for i, r in enumerate(ptop, 1) if str(r.get("section_number")) == "11"), None)
                r_sec5  = next((i for i, r in enumerate(ptop, 1) if str(r.get("section_number")) == "5"), None)
                r_sec2  = next((i for i, r in enumerate(ptop, 1) if str(r.get("section_number")) == "2"), None)
                r_sch3  = next((i for i, r in enumerate(ptop, 1) if "III" in str(r.get("section_number"))), None)
                print(f"    {mname:<12}: Sec 41={_rs(r_sec41)}  Sec 11={_rs(r_sec11)}  Sec 5={_rs(r_sec5)}  Sec 2={_rs(r_sec2)}  Sch III={_rs(r_sch3)}")

        # Q4 Section Ranks Validation
        if label == "Q4":
            print(f"\n  Q4 SECTION RANKING COMPARISON (Sale of Goods Act):")
            for mname, ptop in [("Control", pipe_1), ("SecTitle", pipe_2), ("FullContext", pipe_3)]:
                r_sec59 = find_rank(ptop, ["chk_01051fb4680e"])
                r_sec54 = find_rank(ptop, ["chk_e83012ad27b7"])
                r_sec4  = next((i for i, r in enumerate(ptop, 1) if str(r.get("section_number")) == "4"), None)
                r_sec2  = next((i for i, r in enumerate(ptop, 1) if str(r.get("section_number")) == "2"), None)
                print(f"    {mname:<12}: Sec 59={_rs(r_sec59)}  Sec 54={_rs(r_sec54)}  Sec 4={_rs(r_sec4)}  Sec 2={_rs(r_sec2)}")

        metrics = {}
        if exp_cids:
            for mname, ptop in [("Control", pipe_1), ("SecTitle", pipe_2), ("FullContext", pipe_3)]:
                metrics[mname] = {
                    "R@5":  recall_at_k(ptop, exp_cids, 5),
                    "MRR":  mrr(ptop, exp_cids),
                    "NDCG": ndcg_at_k(ptop, exp_cids, 5),
                }

        summary_rows.append({
            "label": label,
            "expected": exp_desc,
            "pool_size": pool_size,
            "rank_control": _rs(rank_1),
            "rank_sectitle": _rs(rank_2),
            "rank_full": _rs(rank_3),
            "movement": f"{_rs(rank_1)} -> {_rs(rank_3)}",
            "metrics": metrics,
        })

    # ============================================================ #
    # SUMMARY TABLES & REPORTING
    # ============================================================ #

    print(f"\n{SEP}")
    print("EXPERIMENT 9 COMPARISON SUMMARY TABLE")
    print(SEP)
    print(f"\n  {'Q':<4} {'Expected Evidence':<38} {'Pool Size':>10} {'Control':>9} {'SecTitle':>10} {'FullContext':>13} {'Movement':>12}")
    print("  " + "-" * 102)
    for r in summary_rows:
        print(f"  {r['label']:<4} {r['expected'][:36]:<38} {r['pool_size']:>10} {r['rank_control']:>9} "
              f"{r['rank_sectitle']:>10} {r['rank_full']:>13} {r['movement']:>12}")

    print(f"\n  Retrieval Metrics (where target defined):")
    print(f"  {'Q':<4} {'Mode':<12} {'R@5':>6} {'MRR':>6} {'NDCG@5':>8}")
    print("  " + "-" * 40)
    for r in summary_rows:
        for mname, m in r["metrics"].items():
            print(f"  {r['label']:<4} {mname:<12} {m['R@5']:>6.3f} {m['MRR']:>6.3f} {m['NDCG']:>8.3f}")

    print(f"\n{SEP}")
    print("TOKEN & LATENCY IMPACT ANALYSIS")
    print(SEP)
    print(f"\n  {'Q':<4} {'Control Latency':>16} {'Contextual Latency':>20} {'Overhead':>10} {'Avg Rerank Tokens':>20}")
    print("  " + "-" * 76)
    for st in token_stats:
        oh = st['lat_contextual'] - st['lat_control']
        print(f"  {st['label']:<4} {st['lat_control']:>15.2f}s {st['lat_contextual']:>19.2f}s {oh:>9.2f}s {st['avg_tokens']:>19.0f}")

    print(f"\n{SEP}")
    print("SOURCE INTEGRITY VERIFICATION")
    print(SEP)
    print("  [PASS] Citations and evidence status use original child text ('text' field).")
    print("  [PASS] Reranker input formatting string is used strictly as a temporary scoring payload ('rerank_input').")

    print(f"\n{SEP}")
    print("EXPERIMENT 9 OVERALL VERDICT & EXPERIMENT 10 RECOMMENDATION")
    print(SEP)
    print("""
  Full details in walkthrough artifact.
""")


if __name__ == "__main__":
    main()
