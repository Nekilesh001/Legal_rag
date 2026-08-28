"""
File validation and SHA-256 hashing + deduplication for ingestion.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from legal_rag.models.document import DocumentStatus

logger = logging.getLogger(__name__)

MIN_FILE_SIZE_BYTES = 1  # zero-byte files are corrupt


def validate_file(path: Path) -> tuple[DocumentStatus, str]:
    """
    Validate a single file before parsing.
    Returns (status, notes).
    """
    if not path.exists():
        return DocumentStatus.INVALID, "File does not exist"

    if not path.is_file():
        return DocumentStatus.INVALID, "Path is not a file"

    try:
        size = path.stat().st_size
    except OSError as e:
        return DocumentStatus.INVALID, f"Cannot stat file: {e}"

    if size < MIN_FILE_SIZE_BYTES:
        return DocumentStatus.CORRUPT, f"File is empty (0 bytes): {path.name}"

    ext = path.suffix.lower()
    if ext not in {".pdf", ".md", ".markdown"}:
        return DocumentStatus.UNSUPPORTED, f"Unsupported extension: {ext}"

    # Check readability
    try:
        with open(path, "rb") as f:
            f.read(64)  # just peek
    except (OSError, PermissionError) as e:
        return DocumentStatus.INVALID, f"Cannot read file: {e}"

    return DocumentStatus.SUCCESS, ""


def hash_file(path: Path) -> str:
    """
    Compute SHA-256 hash of a file's full contents.
    Raises OSError on read failure.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ #
# Deduplication
# ------------------------------------------------------------------ #

class CanonicalDocument:
    """
    Tracks one canonical record per unique content hash.
    Multiple source paths can map to the same content.
    """
    def __init__(
        self,
        canonical_id: str,
        content_hash: str,
        first_path: Path,
        first_category: str,
    ) -> None:
        self.canonical_id = canonical_id
        self.content_hash = content_hash
        self.source_paths: list[str] = [str(first_path)]
        self.source_categories: list[str] = [first_category]
        self.source_file_names: list[str] = [first_path.name]

    def add_duplicate(self, path: Path, category: str) -> None:
        p = str(path)
        if p not in self.source_paths:
            self.source_paths.append(p)
        if category not in self.source_categories:
            self.source_categories.append(category)
        if path.name not in self.source_file_names:
            self.source_file_names.append(path.name)


class DeduplicationRegistry:
    """
    Registry mapping content_hash -> CanonicalDocument.
    Deterministic: first file seen for a hash becomes the canonical.
    """
    def __init__(self) -> None:
        self._by_hash: dict[str, CanonicalDocument] = {}
        self._counter: int = 0

    def register(
        self, path: Path, content_hash: str, category: str
    ) -> tuple[CanonicalDocument, bool]:
        """
        Register a file.
        Returns (canonical_doc, is_new).
        is_new=True means first time this hash was seen.
        is_new=False means it is a duplicate of an existing canonical.
        """
        if content_hash in self._by_hash:
            canonical = self._by_hash[content_hash]
            canonical.add_duplicate(path, category)
            logger.info(
                "Duplicate detected: %s is identical to %s (hash=%s)",
                path.name,
                canonical.source_paths[0],
                content_hash[:12],
            )
            return canonical, False
        else:
            self._counter += 1
            canonical_id = f"cdoc_{self._counter:05d}"
            canonical = CanonicalDocument(canonical_id, content_hash, path, category)
            self._by_hash[content_hash] = canonical
            logger.debug(
                "New canonical document: %s (hash=%s)", path.name, content_hash[:12]
            )
            return canonical, True

    def __len__(self) -> int:
        return len(self._by_hash)

    def all_canonical(self) -> list[CanonicalDocument]:
        return list(self._by_hash.values())
