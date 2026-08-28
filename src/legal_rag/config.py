"""
Central typed configuration for the Legal RAG Engine.
All values come from environment variables / .env file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RagConfig(BaseSettings):
    """Single authoritative configuration object for the entire Legal RAG Engine."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # NVIDIA / LLM
    # ------------------------------------------------------------------ #
    nvidia_api_key: str = Field(..., alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(
        "https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL"
    )

    rag_llm_provider: Literal["nvidia"] = Field("nvidia", alias="RAG_LLM_PROVIDER")
    rag_llm_model: str = Field(
        "nvidia/nemotron-3-super-120b-a12b", alias="RAG_LLM_MODEL"
    )
    rag_fast_llm_model: str = Field(
        "openai/gpt-oss-120b", alias="RAG_FAST_LLM_MODEL"
    )
    rag_llm_max_tokens: int = Field(4096, alias="RAG_LLM_MAX_TOKENS")

    rag_llm_temperature: float = Field(0.1, alias="RAG_LLM_TEMPERATURE")
    rag_llm_timeout: int = Field(120, alias="RAG_LLM_TIMEOUT")
    rag_llm_max_retries: int = Field(3, alias="RAG_LLM_MAX_RETRIES")

    # ------------------------------------------------------------------ #
    # Corpus / Filesystem
    # ------------------------------------------------------------------ #
    rag_corpus_path: Path = Field(
        Path("D:/ONE_DATA/Anthropic/Excercise"), alias="RAG_CORPUS_PATH"
    )
    rag_data_dir: Path = Field(
        Path("D:/ONE_DATA/Anthropic/Excercise/data"), alias="RAG_DATA_DIR"
    )

    @field_validator("rag_corpus_path", "rag_data_dir", mode="before")
    @classmethod
    def expand_path(cls, v: object) -> Path:
        return Path(str(v)).expanduser().resolve()

    # ------------------------------------------------------------------ #
    # Qdrant
    # ------------------------------------------------------------------ #
    rag_qdrant_url: str = Field("http://localhost:6333", alias="RAG_QDRANT_URL")
    rag_qdrant_collection: str = Field("legal_rag_v1", alias="RAG_QDRANT_COLLECTION")
    rag_qdrant_in_memory: bool = Field(False, alias="RAG_QDRANT_IN_MEMORY")

    # ------------------------------------------------------------------ #
    # Embedding
    # ------------------------------------------------------------------ #
    rag_embedding_provider: Literal["local_bge", "nvidia_api"] = Field(
        "local_bge", alias="RAG_EMBEDDING_PROVIDER"
    )
    rag_embedding_model: str = Field("BAAI/bge-m3", alias="RAG_EMBEDDING_MODEL")
    rag_embedding_batch_size: int = Field(32, alias="RAG_EMBEDDING_BATCH_SIZE")

    # ------------------------------------------------------------------ #
    # Chunking
    # ------------------------------------------------------------------ #
    rag_parent_max_tokens: int = Field(1800, alias="RAG_PARENT_MAX_TOKENS")
    rag_child_max_tokens: int = Field(350, alias="RAG_CHILD_MAX_TOKENS")
    rag_chunk_overlap_tokens: int = Field(50, alias="RAG_CHUNK_OVERLAP_TOKENS")
    rag_semantic_refinement: bool = Field(False, alias="RAG_SEMANTIC_REFINEMENT")

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    rag_dense_top_k: int = Field(20, alias="RAG_DENSE_TOP_K")
    rag_sparse_top_k: int = Field(20, alias="RAG_SPARSE_TOP_K")
    rag_rerank_top_k: int = Field(5, alias="RAG_RERANK_TOP_K")
    rag_rrf_k: int = Field(60, alias="RAG_RRF_K")  # RRF constant
    rag_reranker_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="RAG_RERANKER_MODEL",
    )
    """
    Cross-encoder reranker model.
    Switch to BAAI/bge-reranker-v2-m3 for domain-adapted legal reranking.
    Both models use the CrossEncoderReranker interface — no other code change required.
    """

    # ------------------------------------------------------------------ #
    # Confidence / Retry / Abstention
    # ------------------------------------------------------------------ #
    rag_confidence_high_threshold: float = Field(
        0.75, alias="RAG_CONFIDENCE_HIGH_THRESHOLD"
    )
    rag_confidence_low_threshold: float = Field(
        0.40, alias="RAG_CONFIDENCE_LOW_THRESHOLD"
    )
    rag_max_retry_attempts: int = Field(2, alias="RAG_MAX_RETRY_ATTEMPTS")

    # ------------------------------------------------------------------ #
    # Metadata-Aware Retrieval (Experiment 1)
    # ------------------------------------------------------------------ #
    rag_metadata_aware_retrieval: bool = Field(
        True, alias="RAG_METADATA_AWARE_RETRIEVAL"
    )
    rag_exact_section_boost: float = Field(10.0, alias="RAG_EXACT_SECTION_BOOST")
    rag_document_act_boost: float = Field(5.0, alias="RAG_DOCUMENT_ACT_BOOST")
    rag_category_boost: float = Field(3.0, alias="RAG_CATEGORY_BOOST")
    rag_jurisdiction_boost: float = Field(2.0, alias="RAG_JURISDICTION_BOOST")

    # ------------------------------------------------------------------ #
    # Optional features
    # ------------------------------------------------------------------ #
    rag_contextual_enrichment: bool = Field(False, alias="RAG_CONTEXTUAL_ENRICHMENT")
    rag_xref_expansion: bool = Field(True, alias="RAG_XREF_EXPANSION")
    rag_xref_max_depth: int = Field(2, alias="RAG_XREF_MAX_DEPTH")
    rag_xref_max_extra: int = Field(5, alias="RAG_XREF_MAX_EXTRA")

    # ------------------------------------------------------------------ #
    # OCR
    # ------------------------------------------------------------------ #
    rag_ocr_provider: Literal["tesseract"] = Field(
        "tesseract", alias="RAG_OCR_PROVIDER"
    )
    rag_tesseract_cmd: str = Field(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        alias="RAG_TESSERACT_CMD",
    )
    rag_ocr_confidence_threshold: float = Field(
        60.0, alias="RAG_OCR_CONFIDENCE_THRESHOLD"
    )
    rag_ocr_words_per_page_threshold: int = Field(
        20, alias="RAG_OCR_WORDS_PER_PAGE_THRESHOLD"
    )

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    rag_api_host: str = Field("0.0.0.0", alias="RAG_API_HOST")
    rag_api_port: int = Field(8000, alias="RAG_API_PORT")
    rag_api_reload: bool = Field(True, alias="RAG_API_RELOAD")

    # ------------------------------------------------------------------ #
    # Derived paths (computed, not env vars)
    # ------------------------------------------------------------------ #
    @property
    def artifacts_dir(self) -> Path:
        return self.rag_data_dir / "artifacts"

    @property
    def bm25_dir(self) -> Path:
        return self.rag_data_dir / "bm25_index"

    @property
    def reports_dir(self) -> Path:
        return self.rag_data_dir / "ingestion_reports"

    def ensure_data_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        for p in [self.artifacts_dir, self.bm25_dir, self.reports_dir]:
            p.mkdir(parents=True, exist_ok=True)


# Module-level singleton — import this everywhere
_config: RagConfig | None = None


def get_config() -> RagConfig:
    global _config
    if _config is None:
        _config = RagConfig()
    return _config
