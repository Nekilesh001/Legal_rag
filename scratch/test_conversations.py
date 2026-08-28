import time
import json
from legal_rag.config import get_config
from legal_rag.engine import LegalRagEngine

cfg = get_config()
engine = LegalRagEngine(cfg)

print("=== MULTI-TURN CONVERSATION & ROUTING BENCHMARK ===")

# Conversation 1: No New Retrieval (Type A Contextual Follow-up)
print("\n--------------------------------------------------")
print("CONVERSATION 1: Contextual Follow-up (Type A — No New Retrieval)")
print("--------------------------------------------------")

q1_1 = "What does Section 73 of the Indian Contract Act say?"
print(f"Turn 1 Query: '{q1_1}'")
t0 = time.perf_counter()
res1_1 = engine.query(q1_1, model_mode="fast")
t1_1 = (time.perf_counter() - t0) * 1000
print(f"Turn 1 Time: {t1_1:.2f} ms ({t1_1/1000:.2f} s) | Status: {res1_1.evidence_status.value} | Citations: {len(res1_1.citations)}")

context_c1 = [
    {"role": "user", "content": q1_1},
    {
        "role": "assistant",
        "content": res1_1.answer,
        "response": {
            "citations": [c.model_dump() for c in res1_1.citations],
            "supporting_chunks": [c.model_dump() for c in res1_1.supporting_chunks],
        }
    }
]

q1_2 = "Explain that in simple terms."
print(f"Turn 2 Follow-up: '{q1_2}'")
t0 = time.perf_counter()
res1_2 = engine.query(q1_2, conversation_context=context_c1, model_mode="fast")
t1_2 = (time.perf_counter() - t0) * 1000
print(f"Turn 2 Time: {t1_2:.2f} ms ({t1_2/1000:.2f} s) | Status: {res1_2.evidence_status.value} | Citations: {len(res1_2.citations)}")
print(f"Turn 2 Answer: {res1_2.answer[:150].encode('ascii', errors='ignore').decode('ascii')}...")


# Conversation 2: Retrieval Required Follow-up (Type B)
print("\n--------------------------------------------------")
print("CONVERSATION 2: Retrieval Follow-up (Type B — Subject Linked Retrieval)")
print("--------------------------------------------------")

q2_1 = "What does Section 73 of the Indian Contract Act say?"
q2_2 = "Does that apply to a seller who breaches a sale contract?"
print(f"Turn 1 Query: '{q2_1}'")
res2_1 = engine.query(q2_1, model_mode="fast")

context_c2 = [
    {"role": "user", "content": q2_1},
    {
        "role": "assistant",
        "content": res2_1.answer,
        "response": {
            "citations": [c.model_dump() for c in res2_1.citations],
            "supporting_chunks": [c.model_dump() for c in res2_1.supporting_chunks],
        }
    }
]

print(f"Turn 2 Follow-up: '{q2_2}'")
t0 = time.perf_counter()
res2_2 = engine.query(q2_2, conversation_context=context_c2, model_mode="fast")
t2_2 = (time.perf_counter() - t0) * 1000
print(f"Turn 2 Time: {t2_2:.2f} ms ({t2_2/1000:.2f} s) | Status: {res2_2.evidence_status.value} | Citations: {len(res2_2.citations)}")
print(f"Turn 2 Answer: {res2_2.answer[:150].encode('ascii', errors='ignore').decode('ascii')}...")


# Conversation 3: Completely New Query (Type C)
print("\n--------------------------------------------------")
print("CONVERSATION 3: New Query (Type C)")
print("--------------------------------------------------")

q3_1 = "What is the notice period under the Tamil Nadu Shops Act?"
print(f"Turn 1 Query: '{q3_1}'")
t0 = time.perf_counter()
res3_1 = engine.query(q3_1, model_mode="fast")
t3_1 = (time.perf_counter() - t0) * 1000
print(f"Turn 1 Time: {t3_1:.2f} ms ({t3_1/1000:.2f} s) | Status: {res3_1.evidence_status.value} | Citations: {len(res3_1.citations)}")


# Conversation 4: Out of Corpus Abstention
print("\n--------------------------------------------------")
print("CONVERSATION 4: Out of Corpus Abstention")
print("--------------------------------------------------")

q4_1 = "What are the speed limit regulations under the Tokyo Traffic Law?"
print(f"Turn 1 Query: '{q4_1}'")
t0 = time.perf_counter()
res4_1 = engine.query(q4_1, model_mode="fast")
t4_1 = (time.perf_counter() - t0) * 1000
print(f"Turn 1 Time: {t4_1:.2f} ms ({t4_1/1000:.2f} s) | Status: {res4_1.evidence_status.value} | Citations: {len(res4_1.citations)}")

print("\n=== CONVERSATION BENCHMARKS COMPLETE ===")
