import sys
import json
import math
import time
from pathlib import Path
from collections import defaultdict, Counter

# Add project src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legal_rag.config import get_config
from legal_rag.engine import LegalRagEngine
from legal_rag.query.analyzer import analyze_query
from legal_rag.retrieval.policy import build_structural_retrieval_text
from qdrant_client.http import models as qm

cfg = get_config()
engine = LegalRagEngine(cfg)

print("="*80)
print("PART A: QDRANT BASELINE CLEANUP")
print("="*80)

client = engine.qdrant._client
child_coll = cfg.rag_qdrant_collection + "_children"

# 1. Scroll all points
all_points = []
offset = None
while True:
    res, offset = client.scroll(
        collection_name=child_coll,
        limit=1000,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    all_points.extend(res)
    if offset is None:
        break

print(f"Total points before cleanup: {len(all_points)}")

canonical_bm25_ids = set(engine.bm25._chunk_ids)
print(f"Canonical BM25 chunk IDs count: {len(canonical_bm25_ids)}")

# Find legacy point IDs to delete
point_ids_to_delete = []
seen_canonical_ids = set()

for p in all_points:
    payload = p.payload or {}
    cid = payload.get("chunk_id")
    
    if cid not in canonical_bm25_ids:
        # Not in canonical BM25 set -> Legacy dry run point
        point_ids_to_delete.append(p.id)
    elif cid in seen_canonical_ids:
        # Duplicate point for canonical chunk -> Delete duplicate point
        point_ids_to_delete.append(p.id)
    else:
        seen_canonical_ids.add(cid)

print(f"Points identified for deletion: {len(point_ids_to_delete)}")

if point_ids_to_delete:
    client.delete(
        collection_name=child_coll,
        points_selector=qm.PointIdsList(points=point_ids_to_delete),
    )
    print("Deleted legacy points successfully.")

# Verify cleanup
verify_points = []
offset = None
while True:
    res, offset = client.scroll(
        collection_name=child_coll,
        limit=1000,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    verify_points.extend(res)
    if offset is None:
        break

print(f"Total points after cleanup: {len(verify_points)}")

qdrant_chunk_ids = {p.payload.get("chunk_id") for p in verify_points if p.payload}
chunk_types = {p.payload.get("chunk_type") for p in verify_points if p.payload}

print(f"Qdrant Points == Canonical BM25 Count: {len(verify_points) == len(canonical_bm25_ids)}")
print(f"Canonical BM25 IDs == Qdrant IDs: {canonical_bm25_ids == qdrant_chunk_ids}")
print(f"Missing canonical chunks: {len(canonical_bm25_ids - qdrant_chunk_ids)}")
print(f"Chunk types present: {chunk_types}")


print("\n" + "="*80)
print("PART B: STRUCTURAL RETRIEVAL TEXT EXAMPLE")
print("="*80)

# Find Indian Contract Act Sec 73 metadata & raw text
sec73_meta = None
sec73_raw_text = None
for i, m in enumerate(engine.bm25._chunk_metadata):
    if m.get("document_id") == "A187209" and str(m.get("section_number")) == "73":
        sec73_meta = m
        sec73_raw_text = engine.bm25._chunk_texts[i]
        break

if sec73_meta:
    struct_text = build_structural_retrieval_text(sec73_meta, sec73_raw_text)
    print("--- BEFORE (RAW CHUNK TEXT) ---")
    print(sec73_raw_text[:200] + "...")
    print("\n--- AFTER (DETERMINISTIC STRUCTURAL CONTEXT) ---")
    print(struct_text[:350] + "...")


print("\n" + "="*80)
print("EVALUATION & METRICS: BASELINE VS METADATA-AWARE RETRIEVAL")
print("="*80)

queries_info = [
    ("Q1", "What does Section 73 of the Indian Contract Act say?", ["chk_6c2b46f4b321"]),
    ("Q2", "What are the mandatory clauses in an NDA agreement?", [m["chunk_id"] for m in engine.bm25._chunk_metadata if m.get("document_id") in ["Mandatory Clauses", "Negotiation Playbook"] and "nda" in (m.get("document_title","") + " " + m.get("section_title","") + " " + engine.bm25._chunk_texts[engine.bm25._chunk_ids.index(m["chunk_id"])]).lower()]),
    ("Q3", "What is the notice period under Tamil Nadu Shops Act?", ["chk_56b1160532cc"]),
    ("Q4", "What happens if the seller breaches the contract?", ["chk_01051fb4680e", "chk_e83012ad27b7", "chk_6790f90cb6cc"]),
]

retriever = engine._get_retriever()

# Helper metric calculators
def calc_recip_rank(targets, ranked_cids):
    for r, cid in enumerate(ranked_cids, 1):
        if cid in targets:
            return 1.0 / r
    return 0.0

def calc_recall(targets, ranked_cids, k):
    retrieved_k = set(ranked_cids[:k])
    hits = sum(1 for t in targets if t in retrieved_k)
    return hits / max(1, len(targets))

def calc_ndcg(targets, ranked_cids, k):
    dcg = 0.0
    for r, cid in enumerate(ranked_cids[:k], 1):
        if cid in targets:
            dcg += 1.0 / math.log2(r + 1)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(targets), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0

for mode in ["BASELINE", "METADATA_AWARE"]:
    print(f"\n" + "#"*80)
    print(f"EVALUATION MODE: {mode}")
    print("#"*80)
    
    cfg.rag_metadata_aware_retrieval = (mode == "METADATA_AWARE")
    retriever.policy.config = cfg
    
    mrr_list = []
    r5_list = []
    r10_list = []
    ndcg10_list = []
    
    for qid, q_text, targets in queries_info:
        results = retriever.retrieve(q_text)
        ranked_cids = [r.chunk_id for r in results]
        
        # Rank of target
        target_rank = ">20"
        for r, cid in enumerate(ranked_cids, 1):
            if cid in targets:
                target_rank = str(r)
                break
                
        rr = calc_recip_rank(targets, ranked_cids)
        r5 = calc_recall(targets, ranked_cids, 5)
        r10 = calc_recall(targets, ranked_cids, 10)
        ndcg10 = calc_ndcg(targets, ranked_cids, 10)
        
        mrr_list.append(rr)
        r5_list.append(r5)
        r10_list.append(r10)
        ndcg10_list.append(ndcg10)
        
        print(f"\n{qid}: '{q_text}'")
        print(f"  Target Rank: {target_rank}")
        print(f"  Recall@5: {r5:.4f} | Recall@10: {r10:.4f} | MRR: {rr:.4f} | NDCG@10: {ndcg10:.4f}")
        print(f"  Top 3 Retrieved:")
        for r_idx, r_item in enumerate(results[:3], 1):
            print(f"    Rank {r_idx}: Doc={r_item.document_title} | Sec={r_item.section_number or r_item.section_title} | Page={r_item.page_start} | RerankerScore={r_item.reranker_score:.4f}")

    mean_mrr = sum(mrr_list) / len(mrr_list)
    mean_r5 = sum(r5_list) / len(r5_list)
    mean_r10 = sum(r10_list) / len(r10_list)
    mean_ndcg = sum(ndcg10_list) / len(ndcg10_list)
    
    print(f"\n=== OVERALL METRICS ({mode}) ===")
    print(f"  Mean Recall@5:  {mean_r5:.4f}")
    print(f"  Mean Recall@10: {mean_r10:.4f}")
    print(f"  Mean MRR:       {mean_mrr:.4f}")
    print(f"  Mean NDCG@10:   {mean_ndcg:.4f}")


print("\n" + "="*80)
print("PART J: Q2 BROAD-QUERY ANALYSIS (NDA CHUNKS IN RETRIEVAL)")
print("="*80)

q2_query = "What are the mandatory clauses in an NDA agreement?"
q2_targets = set(queries_info[1][2])

cfg.rag_metadata_aware_retrieval = True
retriever.policy.config = cfg
q2_results = retriever.retrieve(q2_query)
q2_cids = [r.chunk_id for r in q2_results]

nda_in_top5 = sum(1 for cid in q2_cids[:5] if cid in q2_targets)
nda_in_top10 = sum(1 for cid in q2_cids[:10] if cid in q2_targets)

print(f"Relevant NDA Rulebook Chunks retrieved:")
print(f"  In Top-5:  {nda_in_top5} / 5")
print(f"  In Top-10: {nda_in_top10} / 10")
print("Retrieved Clause Breakdown:")
for idx, r in enumerate(q2_results[:5], 1):
    print(f"  {idx}. Doc: {r.document_title} | Clause: {r.section_title or r.section_number} | RerankerScore: {r.reranker_score:.4f}")
