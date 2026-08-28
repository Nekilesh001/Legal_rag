"""
BM25 sparse retrieval index backed by rank_bm25.
Serialized to disk for reuse across runs.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from legal_rag.models.chunk import ChildChunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercasing tokenizer for BM25."""
    return text.lower().split()


class BM25Store:
    """
    Maintains a BM25 index over child chunk texts.
    Serializes index and metadata to disk.
    """

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []
        self._chunk_metadata: list[dict[str, Any]] = []

    @property
    def _index_path(self) -> Path:
        return self.index_dir / "bm25_index.pkl"

    @property
    def _meta_path(self) -> Path:
        return self.index_dir / "bm25_meta.json"

    def build(self, chunks: list[ChildChunk]) -> None:
        """Build BM25 index from a list of child chunks."""
        if not chunks:
            logger.warning("BM25: no chunks to index")
            return

        self._chunk_ids = [c.chunk_id for c in chunks]
        self._chunk_texts = [c.text for c in chunks]
        self._chunk_metadata = [
            {
                "chunk_id": c.chunk_id,
                "parent_id": c.parent_id,
                "document_id": c.document_id,
                "document_title": c.document_title,
                "category": c.category,
                "section_number": c.section_number,
                "section_title": c.section_title,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "text": c.text,
            }
            for c in chunks
        ]

        tokenized = [_tokenize(t) for t in self._chunk_texts]
        self._index = BM25Okapi(tokenized)
        logger.info("BM25: indexed %d child chunks", len(chunks))
        self.save()

    def save(self) -> None:
        """Persist the BM25 index and metadata to disk."""
        with open(self._index_path, "wb") as f:
            pickle.dump(
                {
                    "index": self._index,
                    "chunk_ids": self._chunk_ids,
                    "chunk_texts": self._chunk_texts,
                },
                f,
            )
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self._chunk_metadata, f, ensure_ascii=False, indent=2)
        logger.info("BM25: index saved to %s", self.index_dir)

    def load(self) -> bool:
        """Load existing index from disk. Returns True if successful."""
        if not self._index_path.exists():
            logger.info("BM25: no existing index found at %s", self._index_path)
            return False
        try:
            with open(self._index_path, "rb") as f:
                data = pickle.load(f)
            self._index = data["index"]
            self._chunk_ids = data["chunk_ids"]
            self._chunk_texts = data["chunk_texts"]
            with open(self._meta_path, "r", encoding="utf-8") as f:
                self._chunk_metadata = json.load(f)
            logger.info("BM25: loaded index with %d chunks", len(self._chunk_ids))
            return True
        except Exception as e:
            logger.error("BM25: failed to load index: %s", e)
            return False

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """
        Perform BM25 keyword search.
        Returns list of dicts with 'score' and metadata fields.
        """
        if self._index is None:
            logger.error("BM25: index not built or loaded")
            return []

        tokenized_query = _tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Get top_k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            # BM25Okapi can produce negative scores with very small corpora;
            # include all non-zero results to avoid empty returns in tests.
            if scores[idx] != 0:
                meta = self._chunk_metadata[idx].copy()
                meta["score"] = float(scores[idx])
                results.append(meta)

        return results
