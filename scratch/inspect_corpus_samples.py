"""
inspect_corpus_samples.py — Inspect BM25 metadata & registry documents to ground evaluation dataset.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legal_rag.indexing.bm25_store import BM25Store
from legal_rag.config import get_config
from legal_rag.retrieval.legal_identity import registry

cfg = get_config()
bm25 = BM25Store(cfg.bm25_dir)
bm25.load()
registry.bootstrap(bm25._chunk_metadata)

print("Total BM25 Chunks:", len(bm25._chunk_metadata))
print("Total Documents:", len(registry.all_documents()))

docs = registry.all_documents()
for d in docs[:15]:
    print(f"Canon: {d.canonical_title:<40} Jurisdiction: {d.jurisdiction:<12} Aliases: {d.aliases[:2]}")

