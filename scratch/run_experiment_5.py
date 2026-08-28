"""
run_experiment_5.py — Reranker Replacement Benchmark.

Compares:
  Pipeline A (Control):   RRF + LegalAwareRanker + MS-MARCO MiniLM
  Pipeline B (Experiment): RRF + LegalAwareRanker + BAAI/bge-reranker-v2-m3

The candidate pool is captured once per query and shared between both rerankers,
so any ranking difference is attributable solely to the reranker.

Also verifies:
  - 193003 → Sale of Goods Act, 1930 → PRIMARY_ACT  (authority fix)
  - Q1 Section 73 preserved through BGE reranker
  - Q3 Section 41 vs Form A under BGE
  - Q4 Section 59/54 vs Section 4/2 under BGE
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
# Query specifications
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
    """Annotate source_authority and content_type onto each result (in-place)."""
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
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    cfg = get_config()

    print(SEP)
    print("EXPERIMENT 5: Reranker Replacement Benchmark")
    print("  Pipeline A (Control):    RRF + LegalAwareRanker + MS-MARCO MiniLM")
    print("  Pipeline B (Experiment): RRF + LegalAwareRanker + BAAI/bge-reranker-v2-m3")
    print(SEP)

    # ---------------------------------------------------------------- #
    # Load stores and models
    # ---------------------------------------------------------------- #
    print("\nLoading stores ...")
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

    # Load MS-MARCO (control)
    print("\n  Loading MS-MARCO (control) ...")
    t0 = time.perf_counter()
    msmarco = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    msmarco_load_s = time.perf_counter() - t0
    print(f"  MS-MARCO loaded in {msmarco_load_s:.1f}s")

    # Load BGE-Reranker-v2-M3 (experiment)
    print("\n  Loading BAAI/bge-reranker-v2-m3 (experiment) ...")
    t0 = time.perf_counter()
    bge = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")
    bge_load_s = time.perf_counter() - t0
    print(f"  BGE reranker loaded in {bge_load_s:.1f}s")

    # ---------------------------------------------------------------- #
    # PART 1 — Authority fix verification
    # ---------------------------------------------------------------- #
    print(f"\n{SEP}")
    print("PART 1: AUTHORITY FIX VERIFICATION — 193003 -> Sale of Goods Act")
    print(SEP)

    test_meta_193003 = {"document_id": "doc_PLACEHOLDER_193003", "document_title": "193003"}
    # Find the actual doc_id for 193003 from the registry
    soga_doc = None
    for doc in global_registry.all_documents():
        if "sale of goods" in doc.canonical_title.lower():
            soga_doc = doc
            break

    if soga_doc:
        soga_meta = {"document_id": soga_doc.document_id, "document_title": "193003"}
        auth_with_reg = get_source_authority(soga_meta, registry=global_registry)
        auth_no_reg   = get_source_authority(soga_meta, registry=None)
        print(f"\n  Sale of Goods Act document_id: {soga_doc.document_id}")
        print(f"  Canonical title:               {soga_doc.canonical_title}")
        print(f"  Authority WITHOUT registry:    {auth_no_reg.name}  (raw filename '193003' has no keywords)")
        print(f"  Authority WITH registry:       {auth_with_reg.name}  (canonical title contains 'Act')")
        ok = auth_with_reg.name == "PRIMARY_ACT"
        print(f"\n  [{'PASS' if ok else 'FAIL'}] 193003 -> {soga_doc.canonical_title} -> {auth_with_reg.name}")
        if not ok:
            print("  ERROR: Fix did not work — check get_source_authority() registry branch.")
    else:
        print("  WARNING: Sale of Goods Act not found in registry — seed may not have matched.")

    # ---------------------------------------------------------------- #
    # Per-query evaluation
    # ---------------------------------------------------------------- #
    print(f"\n{SEP}")
    print("PARTS 2-11: PER-QUERY PIPELINE COMPARISON")
    print(SEP)

    summary_rows  = []
    latency_a: list[float] = []
    latency_b: list[float] = []

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
        print(f"\n  QueryAnalyzer -> intent={qa.intent.value}  "
              f"act_names={qa.act_names}  section_refs={qa.section_refs}  "
              f"cats={qa.category_hints}")

        # ---- Build candidate pool (shared for both rerankers) ----
        struct_cands = structured_retriever.retrieve_structured_candidates(qa)
        qvec = embedder.embed_query(query)
        dense_raw  = qdrant.search_children(qvec, 50, None)
        sparse_raw = bm25.search(query, 50)
        fused      = reciprocal_rank_fusion(dense_raw, sparse_raw, k=60)

        fused_ids = {x.get("chunk_id") for x in fused}
        for s in struct_cands:
            cid = s.get("chunk_id")
            if cid not in fused_ids:
                sc = s.copy(); sc["rrf_score"] = 1.0 / 61
                fused.append(sc); fused_ids.add(cid)

        fused = policy.apply_policy(
            query, qa, fused, bm25_metadata=bm25._chunk_metadata
        )
        pool_size = len(fused)

        # Annotate with authority + content_type before passing to ranker
        annotate_legal(fused, global_registry)

        # ---- Legal-aware pre-sort ----
        legal_sorted = legal_ranker.rank(fused, qa)
        top50        = legal_sorted[:50]

        rrf_rank   = find_rank(fused,        exp_cids)
        legal_rank = find_rank(legal_sorted, exp_cids)
        print(f"\n  Pool: {pool_size}  |  RRF rank: {_rsp(rrf_rank, pool_size)}"
              f"  |  Legal-sorted rank: {_rsp(legal_rank, pool_size)}")

        # ---- Pipeline A: control (MS-MARCO) ----
        top50_a = copy.deepcopy(top50)   # deep copy so scores don't bleed between runs
        t0 = time.perf_counter()
        pipe_a = msmarco.rerank(query, top50_a, top_k=5)
        lat_a  = time.perf_counter() - t0
        latency_a.append(lat_a)
        annotate_legal(pipe_a, global_registry)

        # ---- Pipeline B: experiment (BGE) ----
        top50_b = copy.deepcopy(top50)
        t0 = time.perf_counter()
        pipe_b = bge.rerank(query, top50_b, top_k=5)
        lat_b  = time.perf_counter() - t0
        latency_b.append(lat_b)
        annotate_legal(pipe_b, global_registry)

        a_rank = find_rank(pipe_a, exp_cids)
        b_rank = find_rank(pipe_b, exp_cids)

        print(f"  MS-MARCO rank: {_rs(a_rank)}  |  BGE rank: {_rs(b_rank)}"
              f"  |  Latency A={lat_a:.2f}s  B={lat_b:.2f}s")

        # ---- Top-5 tables ----
        print_top5(pipe_a, "Pipeline A (MS-MARCO control), Top 5:", exp_cids)
        print_top5(pipe_b, "Pipeline B (BGE experiment), Top 5:", exp_cids)

        # ---- NDA diversity (Q2) ----
        if qspec["accept_multiple"]:
            for pname, pipe in [("A", pipe_a), ("B", pipe_b)]:
                nda_hits   = sum(1 for r in pipe if is_nda_relevant(r, nda_kw))
                titles     = [(r.get("document_title") or r.get("document_id") or "")[:30] for r in pipe]
                unique_docs = len(set(titles))
                print(f"  Q2 Pipeline {pname}: {nda_hits}/5 NDA-relevant, "
                      f"{unique_docs} unique docs")

        # ---- Q3 special check ----
        if label == "Q3":
            print(f"\n  Q3 CHECK — Form A vs Section 41:")
            for pname, pipe in [("A (MS-MARCO)", pipe_a), ("B (BGE)", pipe_b)]:
                form_ranks = [
                    i for i, r in enumerate(pipe, 1)
                    if ("form" in (r.get("content_type") or "").lower()
                        or "form" in (r.get("text") or "")[:100].lower())
                ]
                sec41_rank = find_rank(pipe, exp_cids)
                print(f"    Pipeline {pname}: Section 41={_rs(sec41_rank)}"
                      f"  Form/Admin content at ranks={form_ranks or 'none in top-5'}")

        # ---- Q4 special check ----
        if label == "Q4":
            print(f"\n  Q4 CHECK -- Section 59/54 vs Section 4/2:")
            for pname, pipe in [("A (MS-MARCO)", pipe_a), ("B (BGE)", pipe_b)]:
                sec59 = find_rank(pipe, ["chk_01051fb4680e"])
                sec54 = find_rank(pipe, ["chk_e83012ad27b7"])
                secs  = [str(r.get("section_number") or "") for r in pipe]
                print(f"    Pipeline {pname}: sec59={_rs(sec59)}  sec54={_rs(sec54)}"
                      f"  sections={secs}")

        # ---- Verdict per query ----
        verdict = "NEUTRAL"
        if exp_cids:
            if b_rank is not None and (a_rank is None or b_rank < a_rank):
                verdict = "IMPROVED"
            elif a_rank is not None and (b_rank is None or a_rank < b_rank):
                verdict = "HURT"
        elif qspec["accept_multiple"]:
            # Q2: compare NDA hits
            nda_a = sum(1 for r in pipe_a if is_nda_relevant(r, nda_kw))
            nda_b = sum(1 for r in pipe_b if is_nda_relevant(r, nda_kw))
            if nda_b > nda_a:
                verdict = "IMPROVED"
            elif nda_a > nda_b:
                verdict = "HURT"
        print(f"\n  {label} VERDICT: BGE vs MS-MARCO -> {verdict}")

        # ---- Metrics ----
        metrics: dict[str, dict] = {}
        if exp_cids:
            for pname, pipe in [("A_msmarco", pipe_a), ("B_bge", pipe_b)]:
                metrics[pname] = {
                    "R@5":  recall_at_k(pipe, exp_cids, 5),
                    "MRR":  mrr(pipe, exp_cids),
                    "NDCG": ndcg_at_k(pipe, exp_cids, 5),
                }

        summary_rows.append({
            "label":      label,
            "expected":   exp_desc,
            "rrf_rank":   _rsp(rrf_rank, pool_size),
            "legal_rank": _rsp(legal_rank, pool_size),
            "a_rank":     _rs(a_rank),
            "b_rank":     _rs(b_rank),
            "verdict":    verdict,
            "metrics":    metrics,
        })

    # ---------------------------------------------------------------- #
    # Summary table
    # ---------------------------------------------------------------- #
    print(f"\n{SEP}")
    print("EXPERIMENT 5 SUMMARY TABLE")
    print(SEP)
    print(f"\n  {'Q':<4} {'Expected Evidence':<42} {'RRF Pool':>9} {'Legal Pool':>11}"
          f" {'A (MSMARCO)':>12} {'B (BGE)':>8} {'Verdict':>10}")
    print("  " + "-" * 104)
    for row in summary_rows:
        print(f"  {row['label']:<4} {row['expected'][:40]:<42} {row['rrf_rank']:>9}"
              f" {row['legal_rank']:>11} {row['a_rank']:>12} {row['b_rank']:>8}"
              f" {row['verdict']:>10}")

    # ---------------------------------------------------------------- #
    # Metrics
    # ---------------------------------------------------------------- #
    print(f"\n  Per-query Retrieval Metrics (where single target defined):")
    print(f"  {'Q':<4} {'Pipeline':<15} {'R@5':>6} {'MRR':>6} {'NDCG@5':>8}")
    print("  " + "-" * 42)
    for row in summary_rows:
        for pname, m in row["metrics"].items():
            print(f"  {row['label']:<4} {pname:<15} {m['R@5']:>6.3f} {m['MRR']:>6.3f} {m['NDCG']:>8.3f}")

    # ---------------------------------------------------------------- #
    # Latency / model info
    # ---------------------------------------------------------------- #
    print(f"\n{SEP}")
    print("MODEL INFO & LATENCY")
    print(SEP)
    print(f"""
  Control:    cross-encoder/ms-marco-MiniLM-L-6-v2
    Load time:            {msmarco_load_s:.1f}s
    Per-query inference:  {sum(latency_a)/len(latency_a):.2f}s avg  (min={min(latency_a):.2f}s  max={max(latency_a):.2f}s)
    Parameters:           ~22M (MiniLM-L-6)
    Training domain:      MS MARCO web search QA

  Experiment: BAAI/bge-reranker-v2-m3
    Load time:            {bge_load_s:.1f}s
    Per-query inference:  {sum(latency_b)/len(latency_b):.2f}s avg  (min={min(latency_b):.2f}s  max={max(latency_b):.2f}s)
    Parameters:           ~568M (XLM-RoBERTa-Large)
    Training domain:      Multilingual retrieval (C-MTEB, BEIR, MIRACL)
""")

    # ---------------------------------------------------------------- #
    # Overall verdict
    # ---------------------------------------------------------------- #
    print(f"{SEP}")
    print("OVERALL RERANKER VERDICT")
    print(SEP)
    improved = [r["label"] for r in summary_rows if r["verdict"] == "IMPROVED"]
    hurt     = [r["label"] for r in summary_rows if r["verdict"] == "HURT"]
    neutral  = [r["label"] for r in summary_rows if r["verdict"] == "NEUTRAL"]
    print(f"""
  BGE IMPROVED: {improved}
  BGE HURT:     {hurt}
  NEUTRAL:      {neutral}
""")
    if len(improved) >= 2 and not hurt:
        print("  VERDICT: BGE-RERANKER BETTER")
    elif len(hurt) >= 2 and not improved:
        print("  VERDICT: MS-MARCO BETTER (or NO CLEAR WINNER)")
    elif improved and hurt:
        print("  VERDICT: NO CLEAR WINNER (mixed results)")
    elif improved:
        print("  VERDICT: BGE-RERANKER BETTER (marginal)")
    else:
        print("  VERDICT: NO CLEAR WINNER (all neutral)")

    print(f"""
  Qualitative analysis:
  - Q1: Does BGE preserve Section 73 from legal-sorted rank 2/3?
        MS-MARCO demotes it below rank 5 (known failure).
  - Q3: Does BGE prefer Section 41 (PRIMARY_ACT) over Section 1 / heading?
        Section 41 is at pool rank ~129. BGE needs to surface it in top 5.
  - Q4: Does BGE distinguish "breach warranty remedy" from "delivery conditions"?
        All candidates from 193003 (PRIMARY_ACT, same document).
        Content-type cannot discriminate. Only semantic reranking can.
  - Q2: Both models expected to return 5/5 NDA results.
""")

    print(f"{SEP}")
    print("RECOMMENDED EXPERIMENT 6")
    print(SEP)
    print("""
  If BGE reranker improves Q1 and Q3 but Q4 remains stuck:

  Experiment 6: Query Expansion for Implicit Section References
    - Q4 query has no act name, no section ref -> structured retrieval cannot inject
      Sale of Goods Act Section 59.
    - Add a lightweight legal-entity linker: if query intent=OBLIGATION_QUERY and
      category=vendor, and section_refs=[], map to Sale of Goods Act + default section
      range for breach/remedy (Sections 53-63).
    - This is deterministic (no LLM), uses existing metadata, and is reversible.
    - Evaluate with the same four queries.

  If BGE reranker does NOT improve Q3:

  Experiment 6 (alternative): BM25 Query Expansion for Section 41
    - "notice period" does not appear verbatim in Section 41 text.
    - Expand the BM25 query with legal synonyms: "notice period" -> also search
      "leave of absence", "termination notice", "notice of termination", "section 41".
    - Evaluate Q3 specifically.

  Do not implement Experiment 6 until this report is reviewed.
""")
    print(SEP)


if __name__ == "__main__":
    main()
