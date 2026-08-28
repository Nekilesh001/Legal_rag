import sys
import time
from pathlib import Path

# Add project src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legal_rag.config import get_config
from legal_rag.engine import LegalRagEngine

cfg = get_config()
engine = LegalRagEngine(cfg)

# Verify Qdrant vector store
info = engine.qdrant._client.get_collection(cfg.rag_qdrant_collection + "_children")
print("=== QDRANT VECTOR STORE METRICS ===")
print(f"Collection Name: {cfg.rag_qdrant_collection}_children")
print(f"Points Count (Child Chunks): {info.points_count}")
print(f"Status: {info.status}")

# Verify BM25 index
print("\n=== BM25 INDEX METRICS ===")
print(f"BM25 Indexed Chunks: {len(engine.bm25._chunk_ids)}")

queries = [
    "What does Section 73 of the Indian Contract Act say?",
    "What are the mandatory clauses in an NDA agreement?",
    "What is the notice period under Tamil Nadu Shops Act?",
    "What happens if the seller breaches the contract?",
]

print("\n=== PURE RETRIEVAL SANITY CHECKS (NO LLM GENERATION) ===")
retriever = engine._get_retriever()

for idx, q in enumerate(queries, 1):
    print(f"\n--- QUERY {idx}: '{q}' ---")
    query_vector = engine.embedder.embed_query(q)

    # 1. Dense top 1
    dense_results = engine.qdrant.search_children(query_vector, top_k=1)
    dense_top = dense_results[0] if dense_results else None
    if dense_top:
        dp = dense_top["payload"]
        print(f"  [DENSE TOP-1] Score={dense_top['score']:.4f} | Doc={dp.get('document_title')} | Sec={dp.get('section_number') or dp.get('section_title') or 'N/A'} | Page={dp.get('page_start')} | ID={dp.get('chunk_id')}")

    # 2. BM25 top 1
    bm25_results = engine.bm25.search(q, top_k=1)
    bm25_top = bm25_results[0] if bm25_results else None
    if bm25_top:
        print(f"  [BM25 TOP-1] Score={bm25_top['score']:.4f} | Doc={bm25_top.get('document_title')} | Sec={bm25_top.get('section_number') or bm25_top.get('section_title') or 'N/A'} | Page={bm25_top.get('page_start')} | ID={bm25_top.get('chunk_id')}")

    # 3. Hybrid RRF + Reranked top 1
    fused_results = retriever.retrieve(q)
    fused_top = fused_results[0] if fused_results else None
    if fused_top:
        print(f"  [FUSED TOP-1] RerankerScore={fused_top.reranker_score:.4f} | Doc={fused_top.document_title} | Sec={fused_top.section_number or fused_top.section_title or 'N/A'} | Page={fused_top.page_start} | ID={fused_top.chunk_id}")
        snippet = fused_top.text.replace("\n", " ")[:150]
        print(f"  [FUSED TOP-1 SNIPPET] {snippet}...")
