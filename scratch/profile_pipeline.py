import time
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from legal_rag.config import get_config
from legal_rag.engine import LegalRagEngine

def profile_query(engine: LegalRagEngine, query_text: str, name: str):
    print(f"\n==================================================")
    print(f"PROFILING {name}: '{query_text}'")
    print(f"==================================================")
    
    t_start = time.perf_counter()
    res = engine.query(query_text)
    t_total = (time.perf_counter() - t_start) * 1000
    
    print(f"Total Time: {t_total:.2f} ms ({t_total/1000:.2f} s)")
    print(f"Evidence Status: {res.evidence_status.value}")
    print(f"Confidence: {res.confidence.value}")
    print(f"Citations count: {len(res.citations)}")
    print(f"Answer snippet: {res.answer[:120]}...")
    return res, t_total

cfg = get_config()

print("--- COLD START INITIALIZATION ---")
t0_init = time.perf_counter()
engine = LegalRagEngine(cfg)
t_init = (time.perf_counter() - t0_init) * 1000
print(f"Engine Initialized in {t_init:.2f} ms ({t_init/1000:.2f} s)")

exact_q = "What does Section 73 of the Indian Contract Act, 1872 say?"
normal_q = "What is the notice period under the Tamil Nadu Shops Act?"
broad_q = "What happens if the seller breaches a contract?"

print("\n=================== COLD START RUNS ===================")
_, t_exact_cold = profile_query(engine, exact_q, "Exact Query (Cold)")
_, t_normal_cold = profile_query(engine, normal_q, "Normal Query (Cold)")
_, t_broad_cold = profile_query(engine, broad_q, "Broad Query (Cold)")

print("\n=================== WARM START RUNS ===================")
_, t_exact_warm = profile_query(engine, exact_q, "Exact Query (Warm)")
_, t_normal_warm = profile_query(engine, normal_q, "Normal Query (Warm)")
_, t_broad_warm = profile_query(engine, broad_q, "Broad Query (Warm)")

print("\n==================================================")
print("LATENCY SUMMARY TABLE")
print("==================================================")
print(f"{'Query Type':<15} | {'Cold Start':<12} | {'Warm Start':<12} | {'Target':<10}")
print("-" * 55)
print(f"{'Exact':<15} | {t_exact_cold/1000:>10.2f} s | {t_exact_warm/1000:>10.2f} s | <= 10.0 s")
print(f"{'Normal':<15} | {t_normal_cold/1000:>10.2f} s | {t_normal_warm/1000:>10.2f} s | <= 10.0 s")
print(f"{'Broad':<15} | {t_broad_cold/1000:>10.2f} s | {t_broad_warm/1000:>10.2f} s | <= 10.0 s")
