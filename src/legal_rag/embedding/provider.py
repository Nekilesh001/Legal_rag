"""
Embedding provider abstraction + BGE-M3 local implementation.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract interface for all embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class BGEm3EmbeddingProvider(EmbeddingProvider):
    """
    Local BGE-M3 embedding via sentence_transformers.
    No API cost. Requires ~2.5 GB model (auto-downloaded on first use).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", model_name)
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self._dim: int = self._model.get_sentence_embedding_dimension() or 1024
        logger.info("Embedding model loaded. Dimension: %d", self._dim)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents in sub-batches with progress logging."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        sub_batch_size = 128

        for i in range(0, len(texts), sub_batch_size):
            batch = texts[i : i + sub_batch_size]
            embeddings = self._model.encode(
                batch,
                batch_size=self._batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            all_embeddings.extend(embeddings.tolist())
            if (i + sub_batch_size) % 1024 == 0 or (i + len(batch)) == len(texts):
                logger.info("Embedded %d / %d chunks...", min(i + len(batch), len(texts)), len(texts))

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        emb = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb[0].tolist()


def get_embedding_provider(
    provider: str = "local_bge",
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 32,
) -> EmbeddingProvider:
    """Factory for embedding providers."""
    if provider == "local_bge":
        return BGEm3EmbeddingProvider(model_name=model_name, batch_size=batch_size)
    raise ValueError(f"Unknown embedding provider: {provider}")
