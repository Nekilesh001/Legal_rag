import traceback
from legal_rag.config import get_config
from legal_rag.engine import LegalRagEngine

try:
    cfg = get_config()
    engine = LegalRagEngine(cfg)
    print("Engine initialized.")
    query = "What does Section 73 of the Indian Contract Act say?"
    print(f"Querying: {query}")
    res = engine.query(query)
    print("SUCCESS!")
    print("Evidence status:", res.evidence_status)
    print("Answer:", res.answer[:200])
except Exception as e:
    print("EXCEPTIONAL ERROR:")
    traceback.print_exc()
