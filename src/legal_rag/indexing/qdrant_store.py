"""
Qdrant vector store abstraction.
Centralizes all Qdrant operations — application code never calls qdrant_client directly.
"""
from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from legal_rag.models.chunk import ChildChunk, ParentChunk

logger = logging.getLogger(__name__)

# Separate collections for parent and child chunks
CHILD_COLLECTION_SUFFIX = "_children"
PARENT_COLLECTION_SUFFIX = "_parents"


class QdrantVectorStore:
    """
    Wraps Qdrant for collection management, upsert, search, and deletion.
    One instance per base collection name.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_dim: int,
        url: str = "http://localhost:6333",
        in_memory: bool = False,
        embedding_model: str = "",
    ) -> None:
        self.base_name = collection_name
        self.child_collection = collection_name + CHILD_COLLECTION_SUFFIX
        self.parent_collection = collection_name + PARENT_COLLECTION_SUFFIX
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model

        if in_memory:
            self._client = QdrantClient(":memory:")
            logger.info("Qdrant: using in-memory mode")
        else:
            try:
                self._client = QdrantClient(url=url, timeout=2.0)
                # Test connection
                self._client.get_collections()
                logger.info("Qdrant: connected to server at %s", url)
            except Exception as e:
                # Fall back to persistent local disk storage
                from legal_rag.config import get_config
                q_dir = get_config().rag_data_dir / "qdrant_db"
                q_dir.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(q_dir))
                logger.info(
                    "Qdrant: server unavailable at %s (%s). Using persistent local disk storage at %s",
                    url, e, q_dir
                )

    # ------------------------------------------------------------------ #
    # Collection management
    # ------------------------------------------------------------------ #

    def ensure_collections(self) -> None:
        """Create child and parent collections if they don't exist."""
        for coll in [self.child_collection, self.parent_collection]:
            existing = [c.name for c in self._client.get_collections().collections]
            if coll not in existing:
                self._client.create_collection(
                    collection_name=coll,
                    vectors_config=qm.VectorParams(
                        size=self.embedding_dim,
                        distance=qm.Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection: %s", coll)
            else:
                logger.debug("Collection exists: %s", coll)

    def delete_collection(self, collection: str) -> None:
        self._client.delete_collection(collection)
        logger.info("Deleted Qdrant collection: %s", collection)

    # ------------------------------------------------------------------ #
    # Upsert
    # ------------------------------------------------------------------ #

    def upsert_children(
        self, chunks: list[ChildChunk], embeddings: list[list[float]]
    ) -> None:
        """Upsert child chunks with their embeddings."""
        if not chunks:
            return
        points = [
            qm.PointStruct(
                id=abs(hash(c.chunk_id)) % (2**63),
                vector=emb,
                payload=self._child_payload(c),
            )
            for c, emb in zip(chunks, embeddings)
        ]
        self._client.upsert(collection_name=self.child_collection, points=points)
        logger.info("Upserted %d child chunks to Qdrant", len(points))

    def upsert_parents(
        self, chunks: list[ParentChunk], embeddings: list[list[float]]
    ) -> None:
        """Upsert parent chunks (for context expansion retrieval)."""
        if not chunks:
            return
        points = [
            qm.PointStruct(
                id=abs(hash(c.chunk_id)) % (2**63),
                vector=emb,
                payload=self._parent_payload(c),
            )
            for c, emb in zip(chunks, embeddings)
        ]
        self._client.upsert(collection_name=self.parent_collection, points=points)
        logger.info("Upserted %d parent chunks to Qdrant", len(points))

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def search_children(
        self,
        query_vector: list[float],
        top_k: int = 20,
        filter_: qm.Filter | None = None,
    ) -> list[dict[str, Any]]:
        """Dense vector search over child chunks."""
        if hasattr(self._client, "query_points"):
            res = self._client.query_points(
                collection_name=self.child_collection,
                query=query_vector,
                limit=top_k,
                query_filter=filter_,
                with_payload=True,
                with_vectors=False,
            )
            points = res.points
        else:
            points = self._client.search(
                collection_name=self.child_collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filter_,
                with_payload=True,
                with_vectors=False,
            )
        return [{"score": r.score, "payload": r.payload, "id": r.id} for r in points]

    def get_parent_by_chunk_id(self, chunk_id: str) -> dict[str, Any] | None:
        """Retrieve a parent chunk by its chunk_id from the payload."""
        results = self._client.scroll(
            collection_name=self.parent_collection,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="chunk_id", match=qm.MatchValue(value=chunk_id))]
            ),
            limit=1,
            with_payload=True,
        )
        points, _ = results
        return points[0].payload if points else None

    def get_child_by_chunk_id(self, chunk_id: str) -> dict[str, Any] | None:
        """Retrieve a child chunk by its chunk_id."""
        results = self._client.scroll(
            collection_name=self.child_collection,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="chunk_id", match=qm.MatchValue(value=chunk_id))]
            ),
            limit=1,
            with_payload=True,
        )
        points, _ = results
        return points[0].payload if points else None

    # ------------------------------------------------------------------ #
    # Payload builders
    # ------------------------------------------------------------------ #

    def _child_payload(self, c: ChildChunk) -> dict[str, Any]:
        return {
            "chunk_id": c.chunk_id,
            "parent_id": c.parent_id,
            "document_id": c.document_id,
            "document_version_id": c.document_version_id,
            "document_title": c.document_title,
            "source_type": c.source_type,
            "category": c.category,
            "section_number": c.section_number,
            "section_title": c.section_title,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "extraction_method": c.extraction_method.value,
            "chunk_type": "child",
            "embedding_model": c.embedding_model,
            "text": c.text,
            "token_count": c.token_count,
            "cross_references": c.cross_references,
        }

    def _parent_payload(self, c: ParentChunk) -> dict[str, Any]:
        return {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "document_version_id": c.document_version_id,
            "document_title": c.document_title,
            "source_type": c.source_type,
            "category": c.category,
            "section_number": c.section_number,
            "section_title": c.section_title,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "extraction_method": c.extraction_method.value,
            "chunk_type": "parent",
            "embedding_model": c.embedding_model,
            "text": c.text,
            "token_count": c.token_count,
            "child_ids": c.child_ids,
        }
