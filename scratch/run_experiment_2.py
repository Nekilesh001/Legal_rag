import sys
import json
from pathlib import Path
from collections import defaultdict

# Add project src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legal_rag.config import get_config
from legal_rag.engine import LegalRagEngine
from legal_rag.query.analyzer import analyze_query
from legal_rag.retrieval.hybrid import reciprocal_rank_fusion

cfg = get_config()
cfg.rag_metadata_aware_retrieval = True
engine = LegalRagEngine(cfg)
retriever = engine._get_retriever()

print("="*80)
print("EXPERIMENT 2: CANDIDATE GENERATION & METADATA RETRIEVAL EVALUATION")
print("="*80)

test_queries = [
    ("Q1", "What does Section 73 of the Indian Contract Act say?", ["chk_6c2b46f4b321"]), # ICA Sec 73
    ("Q2", "What are the mandatory clauses in an NDA agreement?", [m["chunk_id"] for m in engine.bm25._chunk_metadata if m.get("document_id") in ["Mandatory Clauses", "Negotiation Playbook"] and "nda" in (m.get("document_title","") + " " + m.get("section_title","") + " " + engine.bm25._chunk_texts[engine.bm25._chunk_ids.index(m["chunk_id"])]).lower()]),
    ("Q3", "What is the notice period under Tamil Nadu Shops Act?", ["chk_56b1160532cc"]), # TN Shops Act Sec 41
    ("Q4", "What happens if the seller breaches the contract?", ["chk_01051fb4680e", "chk_e83012ad27b7", "chk_6790f90cb6cc"]), # Sale of Goods Act
]

summary_rows = []

for qid, q_text, target_ids in test_queries:
    print("\n" + "="*80)
    print(f"{qid}: '{q_text}'")
    print("="*80)
    
    analysis = analyze_query(q_text)
    print(f"Query Analysis: intent={analysis.intent.value} | sec={analysis.section_refs} | act={analysis.act_names} | cat={analysis.category_hints} | jur={analysis.jurisdictions}")
    
    # 1. Structured candidate lookup
    structured_candidates = retriever.structured_retriever.retrieve_structured_candidates(analysis)
    print(f"\n1. Structured Metadata Candidates: {len(structured_candidates)}")
    for s_idx, sc in enumerate(structured_candidates[:5], 1):
        print(f"   {s_idx}. ID={sc.get('chunk_id')} | Doc={sc.get('document_title')} | Sec={sc.get('section_number')} | Source={sc.get('retrieval_source')}")

    # 2. Dense candidates
    query_vector = engine.embedder.embed_query(q_text)
    dense_candidates = engine.qdrant.search_children(query_vector, top_k=50)
    print(f"\n2. Dense Candidates (Top 5):")
    for d_idx, dc in enumerate(dense_candidates[:5], 1):
        p = dc.get("payload", {})
        print(f"   {d_idx}. ID={p.get('chunk_id')} | Doc={p.get('document_title')} | Sec={p.get('section_number')} | Score={dc['score']:.4f}")

    # 3. BM25 candidates
    bm25_candidates = engine.bm25.search(q_text, top_k=50)
    print(f"\n3. BM25 Candidates (Top 5):")
    for b_idx, bc in enumerate(bm25_candidates[:5], 1):
        print(f"   {b_idx}. ID={bc.get('chunk_id')} | Doc={bc.get('document_title')} | Sec={bc.get('section_number')} | Score={bc['score']:.4f}")

    # 4. Combined Candidate Union
    combined_map = {}
    for sc in structured_candidates:
        cid = sc["chunk_id"]
        sc_copy = sc.copy()
        sc_copy["retrieval_sources"] = [sc.get("retrieval_source", "structured")]
        combined_map[cid] = sc_copy
        
    for r_idx, dc in enumerate(dense_candidates, 1):
        cid = dc["payload"]["chunk_id"]
        if cid not in combined_map:
            c_copy = dc["payload"].copy()
            c_copy["retrieval_sources"] = ["dense"]
            combined_map[cid] = c_copy
        else:
            combined_map[cid]["retrieval_sources"].append("dense")
            
    for r_idx, bc in enumerate(bm25_candidates, 1):
        cid = bc["chunk_id"]
        if cid not in combined_map:
            c_copy = bc.copy()
            c_copy["retrieval_sources"] = ["bm25"]
            combined_map[cid] = c_copy
        else:
            combined_map[cid]["retrieval_sources"].append("bm25")

    combined_candidates = list(combined_map.values())
    print(f"\n4. Combined Candidate Pool Size: {len(combined_candidates)} deduplicated chunks")

    # 5. RRF Fusion & Policy Boosting
    fused = reciprocal_rank_fusion(dense_candidates, bm25_candidates, k=retriever.rrf_k)
    fused_ids = {item.get("chunk_id") for item in fused}
    for s_item in structured_candidates:
        cid = s_item.get("chunk_id")
        if cid not in fused_ids:
            s_copy = s_item.copy()
            s_copy["rrf_score"] = 1.0 / (retriever.rrf_k + 1)
            fused.append(s_copy)
            fused_ids.add(cid)

    boosted = retriever.policy.apply_policy(q_text, analysis, fused)
    
    # 6. Cross-Encoder Reranking
    reranked = retriever.reranker.rerank(q_text, boosted, top_k=10)
    
    print("\n--- FINAL RERANKED TOP 5 RESULTS ---")
    for r_idx, item in enumerate(reranked[:5], 1):
        doc = item.get("document_title", "N/A")
        sec = item.get("section_number") or item.get("section_title") or "N/A"
        score = item.get("reranker_score", 0.0)
        cid = item.get("chunk_id")
        source = item.get("retrieval_source", "fused")
        snippet = (item.get("text") or "").replace("\n", " ")[:100]
        print(f"  Rank {r_idx}: Score={score:.4f} | Doc={doc[:30]:30s} | Sec={str(sec)[:15]:15s} | Source={source} | ID={cid}")
        print(f"          Snippet: {snippet}...")

    # Trace expected evidence target ranks across all stages
    target = target_ids[0] if target_ids else None
    in_pool = "No"
    pool_rank = ">100"
    rrf_rank = ">100"
    rerank_rank = ">10"

    if target:
        for p_idx, c in enumerate(combined_candidates, 1):
            if c.get("chunk_id") == target:
                in_pool = "Yes"
                pool_rank = str(p_idx)
                break
                
        for r_idx, c in enumerate(boosted, 1):
            if c.get("chunk_id") == target:
                rrf_rank = str(r_idx)
                break
                
        for rk_idx, c in enumerate(reranked, 1):
            if c.get("chunk_id") == target:
                rerank_rank = str(rk_idx)
                break

    summary_rows.append({
        "qid": qid,
        "query": q_text,
        "target": target,
        "in_pool": in_pool,
        "pool_rank": pool_rank,
        "rrf_rank": rrf_rank,
        "rerank_rank": rerank_rank,
    })

print("\n" + "="*80)
print("EXPERIMENT 2 RANK COMPARISON TABLE")
print("="*80)

print(f"{'Query':<6} | {'Target Evidence ID':<16} | {'In Pool?':<8} | {'Pool Rank':<10} | {'RRF Rank':<9} | {'Rerank Rank':<11}")
print("-" * 75)
for row in summary_rows:
    print(f"{row['qid']:<6} | {str(row['target']):<16} | {row['in_pool']:<8} | {row['pool_rank']:<10} | {row['rrf_rank']:<9} | {row['rerank_rank']:<11}")

print("\n" + "="*80)
print("Q2 BROAD QUERY NDA CHUNKS EVALUATION")
print("="*80)
q2_results = retriever.retrieve("What are the mandatory clauses in an NDA agreement?")
print(f"Top 5 Reranked Clauses for Q2:")
for idx, r in enumerate(q2_results[:5], 1):
    print(f"  {idx}. Doc: {r.document_title} | Section/Clause: {r.section_title or r.section_number} | RerankerScore: {r.reranker_score:.4f}")
