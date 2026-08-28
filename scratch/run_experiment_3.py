"""
run_experiment_3.py — Canonical Legal Identity + Reranker Evaluation.

Parts:
  A. Bootstrap LegalDocumentRegistry and report identity table
  B. Part C: Verify Q1 — exact Section 73 lookup via canonical identity
  C. Parts D-H: Reranker comparison (Pipeline A=RRF-only vs Pipeline B=RRF+reranker)
     for all four queries
"""
from __future__ import annotations

import os
import sys
import logging

# Suppress verbose HF/torch/tqdm output
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# Make src importable
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

from legal_rag.config import get_config
from legal_rag.indexing.bm25_store import BM25Store
from legal_rag.indexing.qdrant_store import QdrantVectorStore
from legal_rag.query.analyzer import analyze_query
from legal_rag.retrieval.legal_identity import LegalDocumentRegistry, registry as global_registry
from legal_rag.retrieval.structured import StructuredQueryRetriever
from legal_rag.retrieval.hybrid import reciprocal_rank_fusion
from legal_rag.retrieval.policy import MetadataRetrievalPolicy

SEP = "=" * 80

QUERIES = [
    {
        "label": "Q1",
        "query": "What does Section 73 of the Indian Contract Act say?",
        "expected_chunk": "chk_6c2b46f4b321",
        "expected_desc": "Indian Contract Act Section 73",
    },
    {
        "label": "Q2",
        "query": "What are the mandatory clauses in an NDA agreement?",
        "expected_chunk": None,
        "expected_desc": "NDA mandatory clauses (multiple acceptable)",
    },
    {
        "label": "Q3",
        "query": "What is the notice period under Tamil Nadu Shops Act?",
        "expected_chunk": "chk_56b1160532cc",
        "expected_desc": "Tamil Nadu Shops Act Section 41",
    },
    {
        "label": "Q4",
        "query": "What happens if the seller breaches the contract?",
        "expected_chunk": "chk_01051fb4680e",
        "expected_desc": "Sale of Goods Act Section 59",
    },
]


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


def print_top5(results: list[dict], header: str):
    print(f"\n  {header}")
    print(f"  {'#':<4} {'Document':<35} {'Sec':<8} {'Pg':<5} {'ChunkID':<20} {'Score':>8}  Snippet")
    print("  " + "-" * 110)
    for rank, r in enumerate(results[:5], 1):
        doc = (r.get("document_title") or r.get("document_id") or "")[:33]
        sec = str(r.get("section_number") or "")[:7]
        pg = str(r.get("page_start") or "")[:4]
        cid = (r.get("chunk_id") or "")[:18]
        score_val = r.get("reranker_score") or r.get("adjusted_score") or r.get("rrf_score") or 0.0
        snippet = (r.get("text") or "")[:60].replace("\n", " ")
        print(f"  {rank:<4} {doc:<35} {sec:<8} {pg:<5} {cid:<20} {score_val:>8.4f}  {snippet}")


def find_rank(results: list[dict], chunk_id: str | None) -> int | None:
    if not chunk_id:
        return None
    for i, r in enumerate(results, 1):
        if r.get("chunk_id") == chunk_id:
            return i
    return None


def main():
    cfg = get_config()
    print(SEP)
    print("EXPERIMENT 3: Canonical Legal Identity + Reranker Evaluation")
    print(SEP)

    # -----------------------------------------------------------------------
    # Load stores
    # -----------------------------------------------------------------------
    print("\nLoading embedding model, Qdrant, BM25 …")
    embedder, qdrant, bm25 = load_stores(cfg)
    print(f"  BM25 chunks loaded: {len(bm25._chunk_metadata)}")

    # -----------------------------------------------------------------------
    # PART A — Bootstrap LegalDocumentRegistry
    # -----------------------------------------------------------------------
    print(f"\n{SEP}")
    print("PART A — CANONICAL LEGAL DOCUMENT IDENTITY REGISTRY")
    print(SEP)

    global_registry.bootstrap(bm25._chunk_metadata)

    docs = global_registry.all_documents()
    print(f"\n  {len(docs)} canonical documents registered.\n")

    target_acts = [
        "Indian Contract Act",
        "Sale of Goods Act",
        "Transfer of Property Act",
        "Tamil Nadu Shops Act",
        "Arbitration and Conciliation Act",
    ]

    print(f"  {'Document ID':<22} {'Existing Title':<35} {'Canonical Title':<45} {'Aliases'}")
    print("  " + "-" * 150)
    for act_query in target_acts:
        resolved_ids = global_registry.resolve_act_name(act_query)
        if resolved_ids:
            for did in resolved_ids:
                doc = global_registry.get_canonical(did)
                if doc:
                    # Find an "existing title" from BM25 metadata
                    existing_title = did
                    for meta in bm25._chunk_metadata:
                        if (meta.get("document_id") or "").lower() == did.lower():
                            existing_title = meta.get("document_title") or did
                            break
                    aliases_str = ", ".join(doc.aliases[:4])
                    print(f"  {did:<22} {existing_title[:33]:<35} {doc.canonical_title[:43]:<45} {aliases_str}")
        else:
            print(f"  {'(not found)':<22} {act_query[:33]:<35} {'—':<45} —")

    # -----------------------------------------------------------------------
    # PART B/C — Verify Q1: exact Section 73 canonical lookup
    # -----------------------------------------------------------------------
    print(f"\n{SEP}")
    print("PART C — Q1 EXACT SECTION 73 LOOKUP VIA CANONICAL IDENTITY")
    print(SEP)

    q1 = QUERIES[0]
    qa1 = analyze_query(q1["query"])
    print(f"\n  Query: {q1['query']}")
    print(f"  QueryAnalyzer → act_names: {qa1.act_names}, section_refs: {qa1.section_refs}")

    resolved = global_registry.resolve_act_name("Indian Contract Act")
    print(f"  Registry resolution 'Indian Contract Act' → document_ids: {resolved}")

    structured_retriever = StructuredQueryRetriever(qdrant, bm25, global_registry)
    structured_candidates = structured_retriever.retrieve_structured_candidates(qa1)

    target_cid = q1["expected_chunk"]
    print(f"\n  Structured candidates returned: {len(structured_candidates)}")

    found_in_structured = False
    for r in structured_candidates:
        if r.get("chunk_id") == target_cid:
            found_in_structured = True
            print(f"  ✓ Target chunk {target_cid} FOUND in structured candidates!")
            print(f"    doc_id: {r.get('document_id')}")
            print(f"    doc_title: {r.get('document_title')}")
            print(f"    section_number: {r.get('section_number')}")
            print(f"    retrieval_source: {r.get('retrieval_source')}")
            print(f"    text snippet: {(r.get('text') or '')[:120].replace(chr(10), ' ')}")
            break

    if not found_in_structured:
        print(f"  ✗ Target chunk {target_cid} NOT found in structured candidates")
        # Show first 5 structured candidates to diagnose
        for i, r in enumerate(structured_candidates[:5], 1):
            print(f"    Struct #{i}: doc_id={r.get('document_id')} sec={r.get('section_number')} match={r.get('retrieval_source')}")

    # -----------------------------------------------------------------------
    # PARTS D-H — Build full candidate pool + run both pipelines for all queries
    # -----------------------------------------------------------------------
    print(f"\n{SEP}")
    print("PARTS D-H — RERANKER COMPARISON (Pipeline A: RRF-only vs Pipeline B: RRF+Reranker)")
    print(SEP)

    # Load reranker for Pipeline B
    from legal_rag.retrieval.hybrid import CrossEncoderReranker
    print("\n  Loading cross-encoder/ms-marco-MiniLM-L-6-v2 …")
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("  Reranker loaded.")

    policy = MetadataRetrievalPolicy(cfg)

    # Summary table rows
    comparison_rows = []

    for qspec in QUERIES:
        label = qspec["label"]
        query = qspec["query"]
        expected_cid = qspec["expected_chunk"]
        expected_desc = qspec["expected_desc"]

        print(f"\n{'—'*80}")
        print(f"  {label}: {query}")
        print(f"  Expected: {expected_desc}")
        print(f"{'—'*80}")

        # Query analysis
        qa = analyze_query(query)
        print(f"\n  QueryAnalyzer → act_names={qa.act_names}, section_refs={qa.section_refs}, "
              f"cats={qa.category_hints}")

        # Structured retrieval
        struct_cands = structured_retriever.retrieve_structured_candidates(qa)
        print(f"  Structured candidates: {len(struct_cands)}")

        # Dense retrieval
        qvec = embedder.embed_query(query)
        dense_raw = qdrant.search_children(qvec, 50, None)

        # BM25 retrieval
        sparse_raw = bm25.search(query, 50)

        # RRF fusion
        fused = reciprocal_rank_fusion(dense_raw, sparse_raw, k=60)

        # Add structured candidates not already in fused pool
        fused_ids = {item.get("chunk_id") for item in fused}
        for s_item in struct_cands:
            cid = s_item.get("chunk_id")
            if cid not in fused_ids:
                s_copy = s_item.copy()
                s_copy["rrf_score"] = 1.0 / (60 + 1)
                fused.append(s_copy)
                fused_ids.add(cid)

        # Apply metadata policy
        fused = policy.apply_policy(query, qa, fused, bm25_metadata=bm25._chunk_metadata)

        pool_size = len(fused)
        print(f"  Total candidate pool after policy: {pool_size}")

        # Find target rank in RRF pool (before reranking)
        rrf_rank = find_rank(fused, expected_cid)
        if rrf_rank:
            print(f"  Target {expected_cid} at RRF-pool rank: {rrf_rank}")
        elif expected_cid:
            print(f"  Target {expected_cid} NOT found in RRF pool (pool size={pool_size})")

        # PIPELINE A: RRF only (no reranker) — top 5
        pipeline_a = fused[:5]
        no_reranker_rank = find_rank(pipeline_a, expected_cid)

        print_top5(pipeline_a, "Pipeline A — RRF (no reranker), Top 5:")

        # PIPELINE B: RRF + reranker — top 5
        candidates_for_reranking = fused[:50] if len(fused) > 50 else fused
        pipeline_b = reranker.rerank(query, candidates_for_reranking, top_k=5)
        reranker_rank = find_rank(pipeline_b, expected_cid)

        print_top5(pipeline_b, "Pipeline B — RRF + Reranker, Top 5:")

        rrf_rank_str = str(rrf_rank) if rrf_rank else ">pool"
        a_rank_str = str(no_reranker_rank) if no_reranker_rank else ">5"
        b_rank_str = str(reranker_rank) if reranker_rank else ">5"

        comparison_rows.append({
            "label": label,
            "expected": expected_desc,
            "rrf_pool_rank": rrf_rank_str,
            "no_reranker_rank": a_rank_str,
            "reranker_rank": b_rank_str,
            "pipeline_a": pipeline_a,
            "pipeline_b": pipeline_b,
        })

        # PART H: Reranker error analysis for Q3 and Q4
        if label in ("Q3", "Q4"):
            print(f"\n  RERANKER ERROR ANALYSIS — {label}")
            print(f"  {'Rank':<6} {'Doc':<35} {'Sec':<8} {'Rerank Score':>12}  Label")
            print("  " + "-" * 85)

            def classify(r):
                title = (r.get("document_title") or r.get("document_id") or "").lower()
                sec = str(r.get("section_number") or "").lower()
                text = (r.get("text") or "").lower()
                if "rules" in title or "form" in text[:100]:
                    return "ADMIN/RULES"
                sec_int = 0
                try:
                    sec_int = int(sec.split(".")[0].split("(")[0])
                except:
                    pass
                if sec_int <= 4 and sec_int > 0:
                    return "DEFINITION/GENERAL"
                if any(kw in text for kw in ["definition", "shall mean", "interpretation"]):
                    return "DEFINITION/GENERAL"
                if any(kw in text for kw in ["breach", "remedy", "damages", "notice", "termination", "dismiss"]):
                    return "OPERATIVE PROVISION"
                return "OTHER"

            for rank, r in enumerate(pipeline_b[:5], 1):
                doc = (r.get("document_title") or r.get("document_id") or "")[:33]
                sec = str(r.get("section_number") or "")[:7]
                score = r.get("reranker_score", 0.0)
                label_type = classify(r)
                marker = " ← TARGET" if r.get("chunk_id") == expected_cid else ""
                print(f"  {rank:<6} {doc:<35} {sec:<8} {score:>12.4f}  {label_type}{marker}")

    # -----------------------------------------------------------------------
    # PART F — Comparison Summary Table
    # -----------------------------------------------------------------------
    print(f"\n{SEP}")
    print("PART F — RERANKER COMPARISON SUMMARY TABLE")
    print(SEP)
    print(f"\n  {'Query':<6} {'Expected Evidence':<45} {'RRF Rank':>10} {'No-Reranker':>13} {'Reranker':>10}")
    print("  " + "-" * 90)
    for row in comparison_rows:
        print(f"  {row['label']:<6} {row['expected'][:43]:<45} {row['rrf_pool_rank']:>10} "
              f"{row['no_reranker_rank']:>13} {row['reranker_rank']:>10}")

    # -----------------------------------------------------------------------
    # PART G — Verdict
    # -----------------------------------------------------------------------
    print(f"\n{SEP}")
    print("PART G — RERANKER VERDICT")
    print(SEP)

    verdicts = {}
    for row in comparison_rows:
        a = row["no_reranker_rank"]
        b = row["reranker_rank"]
        if a == ">5" and b != ">5":
            v = "IMPROVES"
        elif a != ">5" and b == ">5":
            v = "HURTS"
        elif a == ">5" and b == ">5":
            v = "NEUTRAL (both miss)"
        elif a != ">5" and b != ">5":
            a_int = int(a); b_int = int(b)
            if b_int < a_int:
                v = "IMPROVES"
            elif b_int > a_int:
                v = "HURTS"
            else:
                v = "NEUTRAL"
        else:
            v = "NEUTRAL"
        verdicts[row["label"]] = v
        print(f"  {row['label']}: Pipeline-A rank={a}  Pipeline-B rank={b}  → {v}")

    # Q2 special treatment: check if top-5 results are all NDA-relevant
    print(f"\n  Q2 NDA relevance check (Pipeline A vs B):")
    q2_row = next(r for r in comparison_rows if r["label"] == "Q2")
    for tag, pipe in [("Pipeline A", q2_row["pipeline_a"]), ("Pipeline B", q2_row["pipeline_b"])]:
        nda_hits = sum(
            1 for r in pipe
            if any(kw in (r.get("document_title") or "").lower()
                   for kw in ("nda", "mandatory", "playbook", "confidential"))
        )
        print(f"    {tag}: {nda_hits}/5 results are NDA-relevant")

    # Overall verdict
    hurts = sum(1 for v in verdicts.values() if v == "HURTS")
    improves = sum(1 for v in verdicts.values() if v == "IMPROVES")
    print(f"\n  OVERALL VERDICT: {improves} queries IMPROVED, {hurts} queries HURT by reranker.")
    if hurts > improves:
        print("  → Current reranker HURTS legal retrieval overall.")
    elif improves > hurts:
        print("  → Current reranker IMPROVES legal retrieval overall.")
    else:
        print("  → Current reranker is NEUTRAL overall.")

    # -----------------------------------------------------------------------
    # Remaining problems + recommendation
    # -----------------------------------------------------------------------
    print(f"\n{SEP}")
    print("SUMMARY: REMAINING RETRIEVAL PROBLEMS")
    print(SEP)
    print("""
  1. [Q1 — FIXED or TBD]: Canonical identity resolves 'Indian Contract Act' → 'A187209'.
     Section 73 should now enter the structured candidate pool.
     Confirm in Part C output above.

  2. [Q3/Q4 — RERANKER MISRANKING]: If reranker HURTS Q3/Q4, MS-MARCO MiniLM is
     promoting administrative form text and general definitions above operative
     breach/notice/remedy provisions. Root cause: MS-MARCO trained on web QA, not
     statutory legal text.

  3. [Q3 — TAMIL NADU SHOPS RULES vs ACT]: 'Tamil Nadu Shops and Establishments Rules, 1948'
     (Form A administrative notices) and the Act itself share overlapping terms.
     MS-MARCO cross-encoder favours the Rules document.

  4. [Q4 — SALE OF GOODS ACT SECTION RANKING]: Section 2 (Definitions) and Section 4
     (Contract of sale) are semantically close to 'seller breaches contract' in MS-MARCO
     embedding space but are not operative remedies. Section 59 (Remedy for breach of
     warranty) and Section 54 (Suit for price) are the correct targets.
""")

    print(f"\n{SEP}")
    print("RECOMMENDED NEXT EXPERIMENT (Experiment 4)")
    print(SEP)
    print("""
  ONLY IF reranker HURTS or is NEUTRAL for Q3/Q4:

  Replace the generic MS-MARCO MiniLM cross-encoder with a domain-adapted legal reranker:

  Option A (preferred): 
    BAAI/bge-reranker-v2-m3  (multilingual, legal-aware cross-encoder)
    — Evaluate same four queries, identical candidate pools, identical top-k.

  Option B (lightweight):
    Statutory Priority Scoring rule layer:
    — Boost chunks from authoritative Act sections over commentary/Rules documents.
    — Apply section-number-specific boost for small integers (Section 2, 4) = -penalty.
    — Zero LLM calls, no model download.

  Do NOT change:
    - embedding model
    - chunking
    - BM25
    - candidate generation pipeline
    - Qdrant corpus

  Stop after running the four queries and comparing reranker rank with Experiment 3.
""")
    print(SEP)


if __name__ == "__main__":
    main()
