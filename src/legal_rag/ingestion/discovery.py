"""
File discovery for the Legal RAG ingestion pipeline.
Recursively scans the corpus root and classifies files.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Files/dirs to always ignore
_IGNORE_NAMES: frozenset[str] = frozenset({
    ".git", ".gitkeep", ".env", ".env.example", "__pycache__",
    "node_modules", ".venv", "venv", ".DS_Store", "Thumbs.db",
    "pyproject.toml", "requirements.txt", "docker-compose.yml",
    "nvidia_llm.py",
})

_IGNORE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".pyc", ".pyo", ".pyd", ".so", ".dll",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".cfg",
    ".ini", ".log", ".lock", ".gitignore", ".gitkeep",
    ".example", ".txt", ".sh", ".bat", ".ps1",
    ".zip", ".tar", ".gz", ".bz2", ".7z",
    ".exe", ".msi", ".pkg", ".deb",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
})

# Formats we actually ingest
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".md", ".markdown"})


def discover_files(corpus_root: Path) -> list[Path]:
    """
    Recursively discover all ingestable files under corpus_root.
    Skips system files, virtual envs, and unsupported formats.
    Returns a sorted list of absolute Paths.
    """
    discovered: list[Path] = []

    if not corpus_root.exists():
        logger.error("Corpus root does not exist: %s", corpus_root)
        return []

    if not corpus_root.is_dir():
        logger.error("Corpus root is not a directory: %s", corpus_root)
        return []

    for root, dirs, files in os.walk(corpus_root):
        root_path = Path(root)

        # Prune ignored directories in-place so os.walk doesn't recurse into them
        dirs[:] = [
            d for d in dirs
            if d not in _IGNORE_NAMES and not d.startswith(".")
        ]

        for filename in sorted(files):
            if filename in _IGNORE_NAMES or filename.startswith("."):
                logger.debug("Skipping ignored file: %s", filename)
                continue

            ext = Path(filename).suffix.lower()

            if ext in _IGNORE_EXTENSIONS:
                logger.debug("Skipping unsupported extension %s: %s", ext, filename)
                continue

            if ext not in SUPPORTED_EXTENSIONS:
                logger.debug("Unknown extension %s for file: %s", ext, filename)
                continue

            full_path = root_path / filename
            discovered.append(full_path)
            logger.debug("Discovered: %s", full_path)

    logger.info(
        "Discovery complete: found %d ingestable files under %s",
        len(discovered),
        corpus_root,
    )
    return discovered


def get_category_from_path(file_path: Path, corpus_root: Path) -> str:
    """
    Derive the source category from the file's path relative to corpus_root.
    Examples:
        legal_documents/Finance/A1993-51.pdf  -> "legal_documents/Finance"
        contract_rules/employment/mandatory_clauses.md -> "contract_rules/employment"
    """
    try:
        rel = file_path.relative_to(corpus_root)
        parts = rel.parts
        if len(parts) >= 2:
            return "/".join(parts[:-1])
        return parts[0] if parts else "unknown"
    except ValueError:
        return "unknown"


def get_source_domain(file_path: Path, corpus_root: Path) -> str:
    """
    Classify into top-level domain: 'contract_rules' or 'legal_documents'.
    """
    try:
        rel = file_path.relative_to(corpus_root)
        return rel.parts[0] if rel.parts else "unknown"
    except ValueError:
        return "unknown"
