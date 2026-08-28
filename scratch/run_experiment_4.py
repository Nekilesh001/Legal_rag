"""
run_experiment_4.py — Legal-Aware Ranking Layer Evaluation.

PART 1: Verify Q1 canonical identity fix (Section 73 → structured rank 1)
PARTS 2-16: Three-pipeline comparison for Q1-Q4
  Pipeline A: RRF + policy (no reranker)
  Pipeline B: RRF + policy + MS-MARCO cross-encoder
  Pipeline C: RRF + policy + LegalAwareRanker + MS-MARCO cross-encoder

Final report: per-query rank table, top-5 tables, Q3/Q4 special checks,
              Recall@5/10, MRR, NDCG where feasible.
"""
from __future__ import annotations

import math
import os
import sys
import logging

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# Force UTF-8 output on Windows (default cp1252 cannot encode arrows/dashes)
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
    classify_content_type,
    get_source_authority,
)
from legal_rag.retrieval.structured import StructuredQueryRetriever
from legal_rag.retrieval.hybrid import CrossEncoderReranker, reciprocal_rank_fusion
from legal_rag.retrieval.policy import MetadataRetrievalPolicy

SEP = "=" * 80
THIN = "─" * 80

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
    },
    {
        "label": "Q2",
        "query": "What are the mandatory clauses in an NDA agreement?",
        "expected_chunks": [],   # multiple acceptable — check by document
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
    },
    {
        "label": "Q4",
        "query": "What happens if the seller breaches the contract?",
        "expected_chunks": ["chk_01051fb4680e", "chk_e83012ad27b7"],
        "expected_desc": "Sale of Goods Act Sections 59 / 54",
        "accept_multiple": False,
    },
]

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def load_stores(cfg):
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
    return embedder, qdrant, bm25


def find_rank(results: list[dict], chunk_ids: list[str]) -> int | None:
    """Return rank (1-indexed) of the FIRST matching chunk_id, or None."""
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
    ideal_hits = min(len(chunk_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def print_top5(results: list[dict], header: str, expected_chunks: list[str]):
    print(f"\n  {header}")
    hdr = f"  {'#':<4} {'Document':<32} {'Sec':<6} {'Pg':<4} {'ChunkID':<20} {'Authority':<18} {'ContentType':<22} {'Score':>8}  Snippet"
    print(hdr)
    print("  " + "─" * 145)
    for rank, r in enumerate(results[:5], 1):
        doc = (r.get("document_title") or r.get("document_id") or "")[:30]
        sec = str(r.get("section_number") or "")[:5]
        pg  = str(r.get("page_start") or "")[:3]
        cid = (r.get("chunk_id") or "")[:18]
        auth = (r.get("source_authority") or r.get("legal_breakdown", {}).get("authority_tier", "—"))[:16]
        ctype = (r.get("content_type") or r.get("legal_breakdown", {}).get("content_type", "—"))[:20]

        # Score: prefer reranker_score → legal_combined_score → adjusted_score → rrf_score
        score_val = (
            r.get("reranker_score")
            or r.get("legal_combined_score")
            or r.get("adjusted_score")
            or r.get("rrf_score")
            or 0.0
        )
        snippet = (r.get("text") or "")[:55].replace("\n", " ")
        marker = " ←" if r.get("chunk_id") in expected_chunks else ""
        print(f"  {rank:<4} {doc:<32} {sec:<6} {pg:<4} {cid:<20} {auth:<18} {ctype:<22} {score_val:>8.4f}  {snippet}{marker}")


def is_nda_relevant(r: dict, keywords: tuple) -> bool:
    title = (r.get("document_title") or r.get("document_id") or "").lower()
    return any(kw in title for kw in keywords)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    cfg = get_config()
    print(SEP)
    print("EXPERIMENT 4: Legal-Aware Ranking Layer Evaluation")
    print(SEP)

    print("\nLoading stores …")
    embedder, qdrant, bm25 = load_stores(cfg)
    print(f"  BM25 chunks: {len(bm25._chunk_metadata)}")

    # Bootstrap registry
    global_registry.bootstrap(bm25._chunk_metadata)
    all_docs = global_registry.all_documents()
    print(f"  Registry: {len(all_docs)} canonical documents")

    policy = MetadataRetrievalPolicy(cfg)
    structured_retriever = StructuredQueryRetriever(qdrant, bm25, global_registry)
    legal_ranker = LegalAwareRanker(
        weights=LegalRankerWeights(),
        registry=global_registry,
    )

    print("\n  Loading cross-encoder/ms-marco-MiniLM-L-6-v2 …")
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("  Reranker loaded.")

    # ---------------------------------------------------------------- #
    # PART 1 — Q1 canonical identity verification
    # ---------------------------------------------------------------- #
    print(f"\n{SEP}")
    print("PART 1 — Q1 CANONICAL IDENTITY VERIFICATION")
    print(SEP)

    q1_spec = QUERIES[0]
    qa1 = analyze_query(q1_spec["query"])
    print(f"\n  Query: {q1_spec['query']}")
    print(f"  QueryAnalyzer → act_names={qa1.act_names}, section_refs={qa1.section_refs}")

    resolved = global_registry.resolve_act_name("Indian Contract Act")
    print(f"  Registry 'Indian Contract Act' → {resolved}")
    if resolved:
        doc = global_registry.get_canonical(resolved[0])
        if doc:
            print(f"  Canonical title: {doc.canonical_title}")
            print(f"  Aliases: {doc.aliases[:5]}")

    q1_struct = structured_retriever.retrieve_structured_candidates(qa1)
    target_cid = q1_spec["expected_chunks"][0]
    print(f"\n  Structured candidates: {len(q1_struct)}")

    found = False
    for i, r in enumerate(q1_struct, 1):
        if r.get("chunk_id") == target_cid:
            found = True
            print(f"  ✓ Target {target_cid} FOUND at structured rank {i}")
            print(f"    doc_id={r.get('document_id')}, doc_title={r.get('document_title')}")
            print(f"    section_number={r.get('section_number')}")
            print(f"    retrieval_source={r.get('retrieval_source')}")
            print(f"    snippet: {(r.get('text') or '')[:120].replace(chr(10), ' ')}")
            break
    if not found:
        print(f"  ✗ Target {target_cid} NOT found in structured candidates")
        for i, r in enumerate(q1_struct[:5], 1):
            print(f"    #{i}: doc={r.get('document_id')} sec={r.get('section_number')} src={r.get('retrieval_source')}")

    # ---------------------------------------------------------------- #
    # PARTS 2-16 — Three-pipeline comparison
    # ---------------------------------------------------------------- #
    print(f"\n{SEP}")
    print("PARTS 2-16 — THREE-PIPELINE COMPARISON")
    print(SEP)

    summary_rows = []

    for qspec in QUERIES:
        label    = qspec["label"]
        query    = qspec["query"]
        exp_cids = qspec["expected_chunks"]
        exp_desc = qspec["expected_desc"]

        print(f"\n{THIN}")
        print(f"  {label}: {query}")
        print(f"  Expected: {exp_desc}")
        print(THIN)

        qa = analyze_query(query)
        print(f"\n  QueryAnalyzer → intent={qa.intent.value}, act_names={qa.act_names}, "
              f"section_refs={qa.section_refs}, cats={qa.category_hints}")

        # --- Structured retrieval ---
        struct_cands = structured_retriever.retrieve_structured_candidates(qa)
        print(f"  Structured candidates: {len(struct_cands)}")

        # --- Dense + sparse ---
        qvec = embedder.embed_query(query)
        dense_raw  = qdrant.search_children(qvec, 50, None)
        sparse_raw = bm25.search(query, 50)

        # --- RRF fusion ---
        fused = reciprocal_rank_fusion(dense_raw, sparse_raw, k=60)

        # Inject structured candidates not already present
        fused_ids = {item.get("chunk_id") for item in fused}
        for s in struct_cands:
            cid = s.get("chunk_id")
            if cid not in fused_ids:
                s_copy = s.copy()
                s_copy["rrf_score"] = 1.0 / (60 + 1)
                fused.append(s_copy)
                fused_ids.add(cid)

        # --- Policy adjustment (uses fixed registry) ---
        fused = policy.apply_policy(
            query, qa, fused, bm25_metadata=bm25._chunk_metadata
        )
        pool_size = len(fused)
        print(f"  Pool after RRF + policy: {pool_size} candidates")

        # Annotate every candidate with authority + content_type for display
        for item in fused:
            if "source_authority" not in item:
                auth = get_source_authority(item)
                ct   = classify_content_type(item, item.get("text", ""))
                item["source_authority"] = auth.name
                item["content_type"]     = ct.value

        # ---- PIPELINE A: RRF + policy, top 5 ----
        pipeline_a = fused[:5]

        # ---- PIPELINE B: RRF + policy + reranker, top 5 ----
        top50_b    = fused[:50]
        pipeline_b = reranker.rerank(query, top50_b, top_k=5)
        # Annotate after reranker (scores may overwrite)
        for r in pipeline_b:
            if "source_authority" not in r:
                r["source_authority"] = get_source_authority(r).name
                r["content_type"]     = classify_content_type(r, r.get("text","")).value

        # ---- PIPELINE C: RRF + policy + LegalAwareRanker + reranker, top 5 ----
        legal_sorted = legal_ranker.rank(fused, qa)          # full pool, legal-sorted
        top50_c      = legal_sorted[:50]
        pipeline_c   = reranker.rerank(query, top50_c, top_k=5)
        for r in pipeline_c:
            if "source_authority" not in r:
                r["source_authority"] = get_source_authority(r).name
                r["content_type"]     = classify_content_type(r, r.get("text","")).value

        # ---- Ranks ----
        rrf_rank   = find_rank(fused,       exp_cids)
        legal_rank = find_rank(legal_sorted, exp_cids)
        a_rank     = find_rank(pipeline_a,  exp_cids)
        b_rank     = find_rank(pipeline_b,  exp_cids)
        c_rank     = find_rank(pipeline_c,  exp_cids)

        def _rs(r): return str(r) if r else ">5"
        def _rsp(r): return str(r) if r else f">pool({pool_size})"

        print(f"\n  Target rank — RRF pool: {_rsp(rrf_rank)}"
              f"  Legal-sorted pool: {_rsp(legal_rank)}"
              f"  A={_rs(a_rank)}  B={_rs(b_rank)}  C={_rs(c_rank)}")

        # ---- Top-5 tables ----
        print_top5(pipeline_a, "Pipeline A — RRF + policy (no reranker), Top 5:", exp_cids)
        print_top5(pipeline_b, "Pipeline B — RRF + policy + MS-MARCO, Top 5:", exp_cids)
        print_top5(pipeline_c, "Pipeline C — RRF + policy + LegalRanker + MS-MARCO, Top 5:", exp_cids)

        # ---- Q2 special: NDA diversity ----
        if qspec.get("accept_multiple"):
            kw = qspec["nda_keywords"]
            for pname, pipe in [("A", pipeline_a), ("B", pipeline_b), ("C", pipeline_c)]:
                nda_hits = sum(1 for r in pipe if is_nda_relevant(r, kw))
                titles = [
                    (r.get("document_title") or r.get("document_id") or "")[:30]
                    for r in pipe
                ]
                unique_docs = len(set(titles))
                print(f"  Q2 Pipeline {pname}: {nda_hits}/5 NDA-relevant, {unique_docs} unique docs")

        # ---- Q3/Q4 special: form vs operative check ----
        if label == "Q3":
            print(f"\n  Q3 SPECIAL CHECK — Form A vs Section 41:")
            for pname, pipe in [("A", pipeline_a), ("B", pipeline_b), ("C", pipeline_c)]:
                form_a_rank = next(
                    (i for i, r in enumerate(pipe, 1)
                     if "form" in (r.get("content_type") or "").lower()
                     or "form" in (r.get("text") or "")[:100].lower()),
                    None
                )
                sec41_rank = find_rank(pipe, exp_cids)
                print(f"    Pipeline {pname}: Section 41 rank={_rs(sec41_rank)}  "
                      f"Form A rank={str(form_a_rank) if form_a_rank else 'not in top-5'}")

        if label == "Q4":
            print(f"\n  Q4 SPECIAL CHECK — Section 59/54 vs Section 4/2:")
            for pname, pipe in [("A", pipeline_a), ("B", pipeline_b), ("C", pipeline_c)]:
                sec59_rank = find_rank(pipe, ["chk_01051fb4680e"])
                sec54_rank = find_rank(pipe, ["chk_e83012ad27b7"])
                secs_in_top5 = [
                    str(r.get("section_number") or "") for r in pipe
                ]
                print(f"    Pipeline {pname}: sec59_rank={_rs(sec59_rank)}  "
                      f"sec54_rank={_rs(sec54_rank)}  "
                      f"sections={secs_in_top5}")

        # ---- Metrics ----
        metrics = {}
        if exp_cids:
            for pname, pipe in [("A", pipeline_a), ("B", pipeline_b), ("C", pipeline_c)]:
                metrics[pname] = {
                    "R@5": recall_at_k(pipe, exp_cids, 5),
                    "MRR": mrr(pipe, exp_cids),
                    "NDCG@5": ndcg_at_k(pipe, exp_cids, 5),
                }
        summary_rows.append({
            "label": label,
            "expected": exp_desc,
            "rrf_rank": _rsp(rrf_rank),
            "legal_pool_rank": _rsp(legal_rank),
            "a_rank": _rs(a_rank),
            "b_rank": _rs(b_rank),
            "c_rank": _rs(c_rank),
            "metrics": metrics,
        })

    # ---------------------------------------------------------------- #
    # Summary comparison table
    # ---------------------------------------------------------------- #
    print(f"\n{SEP}")
    print("PIPELINE COMPARISON SUMMARY")
    print(SEP)
    print(f"\n  {'Query':<6} {'Expected Evidence':<42} {'RRF Pool':>9} {'LegalPool':>10} "
          f"{'A(RRF)':>7} {'B(+Rerank)':>11} {'C(+Legal+Rerank)':>17}")
    print("  " + "─" * 108)
    for row in summary_rows:
        print(f"  {row['label']:<6} {row['expected'][:40]:<42} {row['rrf_rank']:>9} "
              f"{row['legal_pool_rank']:>10} {row['a_rank']:>7} {row['b_rank']:>11} "
              f"{row['c_rank']:>17}")

    # ---------------------------------------------------------------- #
    # Metrics table
    # ---------------------------------------------------------------- #
    print(f"\n  Retrieval Metrics (where single target chunk defined):")
    print(f"  {'Query':<6} {'Pipeline':<10} {'R@5':>6} {'MRR':>6} {'NDCG@5':>8}")
    print("  " + "─" * 40)
    for row in summary_rows:
        for pname, m in row["metrics"].items():
            print(f"  {row['label']:<6} {pname:<10} {m['R@5']:>6.3f} {m['MRR']:>6.3f} {m['NDCG@5']:>8.3f}")

    # ---------------------------------------------------------------- #
    # Weight documentation
    # ---------------------------------------------------------------- #
    print(f"\n{SEP}")
    print("LEGAL RANKER — CONFIGURED WEIGHTS")
    print(SEP)
    w = legal_ranker.weights
    print(f"""
  section_match              = {w.section_match}
  document_match             = {w.document_match}
  authority (per tier)       = {w.authority}
      PRIMARY_ACT(4)         → {4 * w.authority}
      RULES_REGULATIONS(3)   → {3 * w.authority}
      GOVERNMENT_ORDER(2)    → {2 * w.authority}
      RULEBOOK_PLAYBOOK(1)   → {1 * w.authority}
  category                   = {w.category}
  jurisdiction               = {w.jurisdiction}
  form_penalty               = {w.form_penalty}
  admin_notice_penalty       = {w.admin_notice_penalty}
  definition_penalty_on_oblig= {w.definition_penalty_on_obligation}
  legal_blend                = {w.legal_blend}
      (combined = rrf_adj_score + {w.legal_blend} × legal_score)
""")

    # ---------------------------------------------------------------- #
    # Q4 diagnostic: inspect legal scores for top candidates
    # ---------------------------------------------------------------- #
    print(f"\n{SEP}")
    print("Q4 LEGAL SCORE DIAGNOSTIC — Top 10 from Legal-Sorted Pool")
    print(SEP)
    # Re-run Q4 to get legal-sorted pool
    q4_spec = QUERIES[3]
    qa4 = analyze_query(q4_spec["query"])
    q4_struct = structured_retriever.retrieve_structured_candidates(qa4)
    qvec4 = embedder.embed_query(q4_spec["query"])
    d4 = qdrant.search_children(qvec4, 50, None)
    s4 = bm25.search(q4_spec["query"], 50)
    f4 = reciprocal_rank_fusion(d4, s4, k=60)
    fids4 = {x.get("chunk_id") for x in f4}
    for s in q4_struct:
        cid = s.get("chunk_id")
        if cid not in fids4:
            sc = s.copy(); sc["rrf_score"] = 1.0 / 61; f4.append(sc); fids4.add(cid)
    f4 = policy.apply_policy(q4_spec["query"], qa4, f4, bm25_metadata=bm25._chunk_metadata)
    legal4 = legal_ranker.rank(f4, qa4)

    print(f"\n  {'#':<4} {'Doc':<22} {'Sec':<6} {'Authority':<18} {'ContentType':<22}"
          f" {'LegalScore':>10} {'Combined':>10}  Snippet")
    print("  " + "─" * 115)
    for i, r in enumerate(legal4[:10], 1):
        doc   = (r.get("document_title") or r.get("document_id") or "")[:20]
        sec   = str(r.get("section_number") or "")[:5]
        auth  = (r.get("source_authority") or "")[:16]
        ct    = (r.get("content_type") or "")[:20]
        ls    = r.get("legal_score", 0.0)
        comb  = r.get("legal_combined_score", 0.0)
        snip  = (r.get("text") or "")[:50].replace("\n", " ")
        marker = " ←TARGET" if r.get("chunk_id") in q4_spec["expected_chunks"] else ""
        print(f"  {i:<4} {doc:<22} {sec:<6} {auth:<18} {ct:<22} {ls:>10.2f} {comb:>10.4f}  {snip}{marker}")

    print(f"\n{SEP}")
    print("REMAINING WEAKNESSES & EXPERIMENT 5 RECOMMENDATION")
    print(SEP)
    print("""
  See full report for analysis. Brief summary:

  Q1 — Depends on registry fix. If Section 73 is injected by structured retrieval,
        legal ranker boosts it via section_match=25 + document_match=10 + authority=12.
        Expected result: Section 73 at C-rank 1.

  Q2 — Already working. Legal ranker preserves NDA diversity (same category/authority
        for all NDA chunks). No regression expected.

  Q3 — PRIMARY HYPOTHESIS: Legal ranker should move Section 41 (Act, authority=PRIMARY_ACT)
        above Form A (Rules, authority=RULES_REGULATIONS) via authority differential + form penalty.
        Expected result: Section 41 at C-rank 1 or 2.

  Q4 — LIMITATION: All relevant chunks from 193003 (Sale of Goods Act) have identical
        authority (PRIMARY_ACT) and identical category (vendor). Section 59 ("breach of
        warranty") and Section 2 ("delivery conditions") both score the same legal_score.
        The legal ranker CANNOT differentiate them from metadata alone.
        Content-type classification cannot reliably distinguish "breach remedy" from
        "delivery obligation" without semantic understanding.
        Expected result: minimal improvement for Q4.

  RECOMMENDED EXPERIMENT 5:
    Replace cross-encoder/ms-marco-MiniLM-L-6-v2 with BAAI/bge-reranker-v2-m3.
    BAAI/bge-reranker-v2-m3 is trained on diverse multilingual retrieval data
    (including legal-style question-answer pairs) and has better domain generalization.
    Evaluation scope: same four queries, same Pipeline C candidate pool,
    compare only the reranker output.
    Do NOT change: embeddings, BM25, Qdrant, chunking, legal ranker, registry.
""")
    print(SEP)


if __name__ == "__main__":
    main()
