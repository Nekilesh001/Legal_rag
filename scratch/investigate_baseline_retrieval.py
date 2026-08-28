import sys
import json
import time
from pathlib import Path
from collections import Counter, defaultdict

# Add project src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legal_rag.config import get_config
from legal_rag.engine import LegalRagEngine
from legal_rag.query.analyzer import analyze_query

cfg = get_config()
engine = LegalRagEngine(cfg)

print("="*80)
print("PART 1: QDRANT POINT COUNT & COLLECTION AUDIT")
print("="*80)

client = engine.qdrant._client
child_coll = cfg.rag_qdrant_collection + "_children"
parent_coll = cfg.rag_qdrant_collection + "_parents"

# Scroll all points from child collection
points = []
offset = None
while True:
    res, offset = client.scroll(
        collection_name=child_coll,
        limit=1000,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    points.extend(res)
    if offset is None:
        break

print(f"Total scroll count from '{child_coll}': {len(points)}")

# Classify points by payload attributes
point_types = Counter()
doc_counts = Counter()
chunk_type_counts = Counter()
chunk_ids = defaultdict(list)
parent_ids_in_points = set()

for p in points:
    payload = p.payload or {}
    c_type = payload.get("chunk_type", "unknown")
    doc_id = payload.get("document_id", "unknown")
    cid = payload.get("chunk_id", str(p.id))
    pid = payload.get("parent_id")
    
    chunk_type_counts[c_type] += 1
    doc_counts[doc_id] += 1
    chunk_ids[cid].append(p.id)
    if pid:
        parent_ids_in_points.add(pid)

# Count duplicate chunk_ids
dup_chunk_id_count = sum(1 for cid, pids in chunk_ids.items() if len(pids) > 1)
total_dup_occurrences = sum(len(pids) for cid, pids in chunk_ids.items() if len(pids) > 1)

print("\nPayload Classifications:")
print(f"  chunk_type counts: {dict(chunk_type_counts)}")
print(f"  Unique chunk_id count: {len(chunk_ids)}")
print(f"  Duplicate chunk_ids (occurring > 1 time): {dup_chunk_id_count}")
print(f"  Total points belonging to duplicate chunk_ids: {total_dup_occurrences}")

# Check BM25 chunks
bm25_chunk_ids = set(engine.bm25._chunk_ids)
print(f"\nBM25 Unique Chunk IDs: {len(bm25_chunk_ids)}")

# Check overlap between BM25 and Qdrant
qdrant_unique_cids = set(chunk_ids.keys())
in_qdrant_not_bm25 = qdrant_unique_cids - bm25_chunk_ids
in_bm25_not_qdrant = bm25_chunk_ids - qdrant_unique_cids

print(f"Unique Chunk IDs in Qdrant but NOT BM25: {len(in_qdrant_not_bm25)}")
print(f"Unique Chunk IDs in BM25 but NOT Qdrant: {len(in_bm25_not_qdrant)}")

# Inspect the points in Qdrant not in BM25
print("\nExtra points distribution in Qdrant (not in BM25):")
extra_docs = Counter()
for cid in in_qdrant_not_bm25:
    for p in points:
        if p.payload and p.payload.get("chunk_id") == cid:
            extra_docs[p.payload.get("document_title", "unknown")] += 1
            break
print(f"  Extra docs distribution in Qdrant: {dict(extra_docs)}")

print("\nAudit Summary Table:")
print(f"  Child vectors (expected from BM25): {len(bm25_chunk_ids)}")
print(f"  Parent records in child collection: {chunk_type_counts.get('parent', 0)}")
print(f"  Duplicate/Legacy points: {len(points) - len(bm25_chunk_ids)}")
print(f"  Total Qdrant Points in child collection: {len(points)}")


print("\n" + "="*80)
print("PART 4: QUERY ANALYZER OUTPUTS")
print("="*80)

queries = [
    "What does Section 73 of the Indian Contract Act say?",
    "What are the mandatory clauses in an NDA agreement?",
    "What is the notice period under Tamil Nadu Shops Act?",
    "What happens if the seller breaches the contract?",
]

for idx, q in enumerate(queries, 1):
    analyzed = analyze_query(q)
    print(f"\nQUERY {idx}: '{q}'")
    print(f"  intent: {analyzed.intent.value}")
    print(f"  section_refs: {analyzed.section_refs}")
    print(f"  act_names: {analyzed.act_names}")
    print(f"  category_hints: {analyzed.category_hints}")
    print(f"  jurisdictions: {analyzed.jurisdictions}")


print("\n" + "="*80)
print("PART 5: EXPECTED EVIDENCE AVAILABILITY IN INDEX")
print("="*80)

def search_index_by_text_or_meta(title_substr=None, section_substr=None, text_substr=None):
    matches = []
    for i, meta in enumerate(engine.bm25._chunk_metadata):
        doc_title = meta.get("document_title", "")
        sec_num = str(meta.get("section_number") or "")
        sec_title = str(meta.get("section_title") or "")
        text = engine.bm25._chunk_texts[i]
        
        match_title = (not title_substr) or (title_substr.lower() in doc_title.lower())
        match_sec = (not section_substr) or (section_substr.lower() in sec_num.lower() or section_substr.lower() in sec_title.lower())
        match_text = (not text_substr) or (text_substr.lower() in text.lower())
        
        if match_title and match_sec and match_text:
            matches.append({
                "chunk_id": meta.get("chunk_id"),
                "document_id": meta.get("document_id"),
                "parent_id": meta.get("parent_id"),
                "document_title": doc_title,
                "section_number": sec_num,
                "section_title": sec_title,
                "page_start": meta.get("page_start"),
                "text_snippet": text.replace("\n", " ")[:200]
            })
    return matches

print("\n1. Indian Contract Act - Section 73:")
ev1 = search_index_by_text_or_meta(title_substr="Contract", section_substr="73")
if not ev1:
    ev1 = search_index_by_text_or_meta(text_substr="73. Compensation for loss or damage caused by breach of contract")
if not ev1:
    ev1 = search_index_by_text_or_meta(text_substr="Section 73")
print(f"  Status: {'EXISTS' if ev1 else 'NOT FOUND'} (Found {len(ev1)} matching chunks)")
for m in ev1[:5]:
    print(f"    - ID={m['chunk_id']} | Doc={m['document_title']} | Sec={m['section_number']} ({m['section_title']}) | Page={m['page_start']}")
    print(f"      Snippet: {m['text_snippet']}...")

print("\n2. NDA Mandatory Clauses Content:")
ev2 = search_index_by_text_or_meta(title_substr="NDA", text_substr="mandatory")
if not ev2:
    ev2 = search_index_by_text_or_meta(text_substr="Non-Disclosure Agreement")
if not ev2:
    ev2 = search_index_by_text_or_meta(text_substr="nda")
print(f"  Status: {'EXISTS' if ev2 else 'NOT FOUND'} (Found {len(ev2)} matching chunks)")
for m in ev2[:5]:
    print(f"    - ID={m['chunk_id']} | Doc={m['document_title']} | Sec={m['section_number']} ({m['section_title']}) | Page={m['page_start']}")
    print(f"      Snippet: {m['text_snippet']}...")

print("\n3. Tamil Nadu Shops Notice Period Provision:")
ev3 = search_index_by_text_or_meta(title_substr="Shops", text_substr="notice")
if not ev3:
    ev3 = search_index_by_text_or_meta(title_substr="Shops", section_substr="41")
print(f"  Status: {'EXISTS' if ev3 else 'NOT FOUND'} (Found {len(ev3)} matching chunks)")
for m in ev3[:5]:
    print(f"    - ID={m['chunk_id']} | Doc={m['document_title']} | Sec={m['section_number']} ({m['section_title']}) | Page={m['page_start']}")
    print(f"      Snippet: {m['text_snippet']}...")

print("\n4. Seller Breach / Remedy Provision (Sale of Goods Act):")
ev4 = search_index_by_text_or_meta(title_substr="Sale of Goods", text_substr="breach")
if not ev4:
    ev4 = search_index_by_text_or_meta(title_substr="193003", text_substr="seller")
if not ev4:
    ev4 = search_index_by_text_or_meta(text_substr="seller")
print(f"  Status: {'EXISTS' if ev4 else 'NOT FOUND'} (Found {len(ev4)} matching chunks)")
for m in ev4[:5]:
    print(f"    - ID={m['chunk_id']} | Doc={m['document_title']} | Sec={m['section_number']} ({m['section_title']}) | Page={m['page_start']}")
    print(f"      Snippet: {m['text_snippet']}...")


print("\n" + "="*80)
print("PART 2: TOP 20 CANDIDATES FOR ALL 4 STAGES ACROSS ALL 4 QUERIES")
print("="*80)

retriever = engine._get_retriever()

for idx, q in enumerate(queries, 1):
    print("\n" + "#"*80)
    print(f"QUERY {idx}: '{q}'")
    print("#"*80)
    
    query_vector = engine.embedder.embed_query(q)
    
    # A. Dense Top 20
    dense_hits = engine.qdrant.search_children(query_vector, top_k=20)
    print(f"\n--- A. DENSE TOP 20 (BGE-M3) ---")
    for r_idx, hit in enumerate(dense_hits, 1):
        payload = hit.get("payload", {})
        doc = payload.get("document_title", "N/A")
        sec = payload.get("section_number") or payload.get("section_title") or "N/A"
        page = payload.get("page_start", "N/A")
        cid = payload.get("chunk_id", "N/A")
        snippet = (payload.get("text") or "").replace("\n", " ")[:120]
        print(f"  Rank {r_idx:2d} | Score={hit['score']:.4f} | Doc={doc[:30]:30s} | Sec={str(sec)[:15]:15s} | Page={str(page):4s} | ID={cid} | Snippet: {snippet}...")
        
    # B. BM25 Top 20
    bm25_hits = engine.bm25.search(q, top_k=20)
    print(f"\n--- B. BM25 TOP 20 ---")
    for r_idx, hit in enumerate(bm25_hits, 1):
        doc = hit.get("document_title", "N/A")
        sec = hit.get("section_number") or hit.get("section_title") or "N/A"
        page = hit.get("page_start", "N/A")
        cid = hit.get("chunk_id", "N/A")
        snippet = (hit.get("text") or "").replace("\n", " ")[:120]
        print(f"  Rank {r_idx:2d} | Score={hit['score']:.4f} | Doc={doc[:30]:30s} | Sec={str(sec)[:15]:15s} | Page={str(page):4s} | ID={cid} | Snippet: {snippet}...")

    # C. RRF Fused Top 20 (Un-reranked)
    dense_candidates = engine.qdrant.search_children(query_vector, top_k=50)
    bm25_candidates = engine.bm25.search(q, top_k=50)
    
    rrf_scores = defaultdict(float)
    chunk_map = {}
    
    for r_idx, hit in enumerate(dense_candidates, 1):
        cid = hit["payload"]["chunk_id"]
        rrf_scores[cid] += 1.0 / (60 + r_idx)
        chunk_map[cid] = hit["payload"]
        
    for r_idx, hit in enumerate(bm25_candidates, 1):
        cid = hit["chunk_id"]
        rrf_scores[cid] += 1.0 / (60 + r_idx)
        if cid not in chunk_map:
            chunk_map[cid] = hit
            
    rrf_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    print(f"\n--- C. RRF FUSED TOP 20 ---")
    for r_idx, (cid, rrf_score) in enumerate(rrf_sorted, 1):
        payload = chunk_map[cid]
        doc = payload.get("document_title", "N/A")
        sec = payload.get("section_number") or payload.get("section_title") or "N/A"
        page = payload.get("page_start", "N/A")
        snippet = (payload.get("text") or "").replace("\n", " ")[:120]
        print(f"  Rank {r_idx:2d} | RRFScore={rrf_score:.6f} | Doc={doc[:30]:30s} | Sec={str(sec)[:15]:15s} | Page={str(page):4s} | ID={cid} | Snippet: {snippet}...")

    # D. Reranked Top 20
    reranked_hits = retriever.retrieve(q)
    print(f"\n--- D. RERANKED TOP 20 ---")
    for r_idx, hit in enumerate(reranked_hits[:20], 1):
        doc = hit.document_title or "N/A"
        sec = hit.section_number or hit.section_title or "N/A"
        page = hit.page_start or "N/A"
        cid = hit.chunk_id
        score = hit.reranker_score if hit.reranker_score is not None else 0.0
        snippet = hit.text.replace("\n", " ")[:120]
        print(f"  Rank {r_idx:2d} | RerankerScore={score:.4f} | Doc={doc[:30]:30s} | Sec={str(sec)[:15]:15s} | Page={str(page):4s} | ID={cid} | Snippet: {snippet}...")
