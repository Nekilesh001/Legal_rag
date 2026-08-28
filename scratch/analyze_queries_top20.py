import sys
from pathlib import Path
from collections import defaultdict

# Add project src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legal_rag.config import get_config
from legal_rag.engine import LegalRagEngine

cfg = get_config()
engine = LegalRagEngine(cfg)
retriever = engine._get_retriever()

print("="*80)
print("FINDING EXACT TARGET EVIDENCE CHUNKS IN BM25 INDEX")
print("="*80)

# Helper function to find chunks by text/title matching
def find_chunks(title_kw="", text_kw=""):
    results = []
    for i, meta in enumerate(engine.bm25._chunk_metadata):
        title = meta.get("document_title", "") + " " + meta.get("document_id", "")
        text = engine.bm25._chunk_texts[i]
        
        if title_kw.lower() in title.lower() and text_kw.lower() in text.lower():
            results.append({
                "index": i,
                "chunk_id": meta.get("chunk_id"),
                "document_title": meta.get("document_title"),
                "document_id": meta.get("document_id"),
                "section_number": meta.get("section_number"),
                "section_title": meta.get("section_title"),
                "page_start": meta.get("page_start"),
                "snippet": text.replace("\n", " ")[:150]
            })
    return results

# 1. Indian Contract Act Sec 73
q1_targets = find_chunks(title_kw="1872", text_kw="Compensation for loss or damage caused by breach of contract")
if not q1_targets:
    q1_targets = find_chunks(title_kw="A187209", text_kw="Section 73")
if not q1_targets:
    q1_targets = find_chunks(text_kw="73. Compensation for loss or damage")

print(f"\n1. Indian Contract Act Section 73 ({len(q1_targets)} chunks found):")
for t in q1_targets[:3]:
    print(f"   ID: {t['chunk_id']} | Doc: {t['document_title']} | Sec: {t['section_number']} | Snippet: {t['snippet']}...")

# 2. NDA Mandatory Clauses Rulebook
q2_targets = find_chunks(title_kw="Mandatory Clauses", text_kw="NDA")
if not q2_targets:
    q2_targets = find_chunks(title_kw="Playbook", text_kw="NDA")
if not q2_targets:
    q2_targets = find_chunks(text_kw="Non-Disclosure Agreement")

print(f"\n2. NDA Mandatory Clauses Content ({len(q2_targets)} chunks found):")
for t in q2_targets[:3]:
    print(f"   ID: {t['chunk_id']} | Doc: {t['document_title']} | Sec: {t['section_number']} | Snippet: {t['snippet']}...")

# 3. Tamil Nadu Shops Notice Period
q3_targets = find_chunks(title_kw="Shops", text_kw="Notice of dismissal")
if not q3_targets:
    q3_targets = find_chunks(title_kw="Shops", text_kw="one month")
if not q3_targets:
    q3_targets = find_chunks(title_kw="Shops", text_kw="notice")

print(f"\n3. Tamil Nadu Shops Notice Period ({len(q3_targets)} chunks found):")
for t in q3_targets[:3]:
    print(f"   ID: {t['chunk_id']} | Doc: {t['document_title']} | Sec: {t['section_number']} | Snippet: {t['snippet']}...")

# 4. Sale of Goods Act Breach of Contract
q4_targets = find_chunks(title_kw="193003", text_kw="breach")
if not q4_targets:
    q4_targets = find_chunks(title_kw="Sale of Goods", text_kw="remedy")
if not q4_targets:
    q4_targets = find_chunks(title_kw="193003", text_kw="seller")

print(f"\n4. Sale of Goods Act Seller Breach/Remedy ({len(q4_targets)} chunks found):")
for t in q4_targets[:3]:
    print(f"   ID: {t['chunk_id']} | Doc: {t['document_title']} | Sec: {t['section_number']} | Snippet: {t['snippet']}...")


print("\n" + "="*80)
print("STAGE-BY-STAGE RANK ANALYSIS FOR ALL 4 QUERIES")
print("="*80)

queries = [
    ("Query 1", "What does Section 73 of the Indian Contract Act say?", q1_targets),
    ("Query 2", "What are the mandatory clauses in an NDA agreement?", q2_targets),
    ("Query 3", "What is the notice period under Tamil Nadu Shops Act?", q3_targets),
    ("Query 4", "What happens if the seller breaches the contract?", q4_targets),
]

for label, q, target_list in queries:
    print(f"\n{label}: '{q}'")
    query_vector = engine.embedder.embed_query(q)
    
    dense_hits = engine.qdrant.search_children(query_vector, top_k=100)
    bm25_hits = engine.bm25.search(q, top_k=100)
    
    # RRF
    rrf_scores = defaultdict(float)
    for r_idx, hit in enumerate(dense_hits, 1):
        rrf_scores[hit["payload"]["chunk_id"]] += 1.0 / (60 + r_idx)
    for r_idx, hit in enumerate(bm25_hits, 1):
        rrf_scores[hit["chunk_id"]] += 1.0 / (60 + r_idx)
    rrf_sorted = [cid for cid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)]
    
    # Reranked
    reranked_hits = retriever.retrieve(q)
    reranked_cids = [h.chunk_id for h in reranked_hits]
    
    dense_cids = [h["payload"]["chunk_id"] for h in dense_hits]
    bm25_cids = [h["chunk_id"] for h in bm25_hits]
    
    if not target_list:
        print("  No target chunks matched search criteria.")
        continue
        
    for t in target_list[:5]:
        t_cid = t["chunk_id"]
        d_rank = dense_cids.index(t_cid) + 1 if t_cid in dense_cids else ">100"
        b_rank = bm25_cids.index(t_cid) + 1 if t_cid in bm25_cids else ">100"
        r_rank = rrf_sorted.index(t_cid) + 1 if t_cid in rrf_sorted else ">100"
        rk_rank = reranked_cids.index(t_cid) + 1 if t_cid in reranked_cids else ">100"
        
        print(f"  Target Chunk {t_cid} | Doc: {t['document_title']} | Sec: {t['section_number']}:")
        print(f"    Dense Rank: {str(d_rank):>4s} | BM25 Rank: {str(b_rank):>4s} | RRF Rank: {str(r_rank):>4s} | Reranker Rank: {str(rk_rank):>4s}")
