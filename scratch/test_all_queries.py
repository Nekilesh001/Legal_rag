import time
from legal_rag.config import get_config
from legal_rag.engine import LegalRagEngine

test_queries = [
    ("Test A (Exact Legal)", "What does Section 73 of the Indian Contract Act say?"),
    ("Test B (Threshold Notice)", "What is the notice period under the Tamil Nadu Shops Act?"),
    ("Test C (NDA Clauses)", "What are the mandatory clauses in an NDA agreement?"),
    ("Test D (Breach Remedies)", "What happens if the seller breaches the contract?"),
    ("Test E (Out of Corpus)", "What are the speed limit regulations under the Tokyo Traffic Law?"),
]

cfg = get_config()
engine = LegalRagEngine(cfg)
print("=== LEGAL RAG ENGINE INTEGRATION TEST ===")

for name, query in test_queries:
    print(f"\n--- {name} ---")
    print(f"Query: {query}")
    t0 = time.time()
    try:
        res = engine.query(query)
        elapsed = time.time() - t0
        print(f"Latency: {elapsed:.2f}s")
        print(f"Evidence Status: {res.evidence_status}")
        print(f"Confidence: {res.confidence}")
        print(f"Citations count: {len(res.citations)}")
        print(f"Answer snippet: {res.answer[:150]}...")
    except Exception as e:
        print(f"FAILED with error: {e}")

print("\n=== ALL INTEGRATION TESTS COMPLETE ===")
