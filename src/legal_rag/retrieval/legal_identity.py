"""
Canonical Legal Document Identity — Experiment 3.

Provides a reusable registry mapping raw document_ids and filenames
to canonical legal act titles and searchable aliases.

Design principles:
- Bootstrapped from actual BM25 metadata at runtime (no static hardcoding required).
- A small seed registry covers well-known acts whose filenames are opaque.
- At runtime any document whose title is already descriptive (e.g.
  "The Tamil Nadu Shops and Establishments Act, 1947") generates its own
  aliases automatically.
- Lookup is bidirectional: act name → document_id, document_id → canonical title.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed registry for opaque filenames (numeric IDs, cryptic codes, etc.)
# Each entry maps a known document_id pattern to canonical legal identity.
# ---------------------------------------------------------------------------
_SEED_REGISTRY: list[dict[str, Any]] = [
    {
        "document_id_pattern": "a187209",   # The Indian Contract Act, 1872 (Act 9 of 1872)
        "canonical_title": "Indian Contract Act, 1872",
        "year": 1872,
        "jurisdiction": "India",
        "aliases": [
            "Indian Contract Act",
            "Contract Act 1872",
            "ICA 1872",
            "Act 9 of 1872",
        ],
    },
    {
        "document_id_pattern": "193003",    # The Sale of Goods Act, 1930 (Act 3 of 1930)
        "canonical_title": "Sale of Goods Act, 1930",
        "year": 1930,
        "jurisdiction": "India",
        "aliases": [
            "Sale of Goods Act",
            "Goods Act 1930",
            "SGA 1930",
            "Act 3 of 1930",
        ],
    },
    {
        "document_id_pattern": "a1882-04",  # Transfer of Property Act, 1882
        "canonical_title": "Transfer of Property Act, 1882",
        "year": 1882,
        "jurisdiction": "India",
        "aliases": [
            "Transfer of Property Act",
            "TPA 1882",
            "TP Act",
            "Act 4 of 1882",
        ],
    },
    {
        "document_id_pattern": "a1963-36",  # Limitation Act, 1963
        "canonical_title": "Limitation Act, 1963",
        "year": 1963,
        "jurisdiction": "India",
        "aliases": [
            "Limitation Act",
            "Act 36 of 1963",
        ],
    },
    {
        "document_id_pattern": "a1970-39",  # Patents Act, 1970
        "canonical_title": "Patents Act, 1970",
        "year": 1970,
        "jurisdiction": "India",
        "aliases": [
            "Patents Act",
            "Act 39 of 1970",
        ],
    },
    {
        "document_id_pattern": "a2002-54",  # Competition Act, 2002
        "canonical_title": "Competition Act, 2002",
        "year": 2002,
        "jurisdiction": "India",
        "aliases": [
            "Competition Act",
            "Act 12 of 2003",
        ],
    },
    {
        "document_id_pattern": "a2017-12",  # Central Goods and Services Tax Act, 2017
        "canonical_title": "Central Goods and Services Tax Act, 2017",
        "year": 2017,
        "jurisdiction": "India",
        "aliases": [
            "CGST Act",
            "GST Act 2017",
            "Central GST Act",
            "Act 12 of 2017",
        ],
    },
    {
        "document_id_pattern": "1637820824",  # Tamil Nadu Shops and Establishments Act, 1947
        "canonical_title": "Tamil Nadu Shops and Establishments Act, 1947",
        "year": 1947,
        "jurisdiction": "Tamil Nadu",
        "aliases": [
            "Tamil Nadu Shops Act",
            "Tamil Nadu Shops and Establishments Act",
            "TN Shops Act",
            "TN Shops & Establishments Act",
            "Tamil Nadu Shops Act 1947",
        ],
    },
    {
        "document_id_pattern": "1637820826",  # Tamil Nadu Shops and Establishments Rules, 1948
        "canonical_title": "Tamil Nadu Shops and Establishments Rules, 1948",
        "year": 1948,
        "jurisdiction": "Tamil Nadu",
        "aliases": [
            "Tamil Nadu Shops Rules",
            "TN Shops Rules 1948",
        ],
    },
]


@dataclass
class CanonicalDocument:
    """Represents the canonical identity of one legal document."""
    document_id: str
    canonical_title: str
    year: int | None = None
    jurisdiction: str = "India"
    aliases: list[str] = field(default_factory=list)

    def all_searchable_names(self) -> list[str]:
        """All names that should match this document when queried."""
        names = [self.canonical_title, self.document_id]
        names.extend(self.aliases)
        return [n.lower() for n in names]


class LegalDocumentRegistry:
    """
    Bidirectional registry: document_id ↔ CanonicalDocument.

    Bootstrapped lazily from BM25 metadata + seed registry.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, CanonicalDocument] = {}   # document_id → CanonicalDocument
        self._alias_map: dict[str, str] = {}              # lowercase alias/title → document_id
        self._loaded = False

    def bootstrap(self, bm25_metadata: list[dict[str, Any]]) -> None:
        """
        Build the registry from BM25 chunk metadata + seed entries.
        Safe to call multiple times (idempotent).
        """
        if self._loaded:
            return

        # 1. Seed entries first (covers opaque filenames)
        for seed in _SEED_REGISTRY:
            pattern = seed["document_id_pattern"].lower()
            # Match against BOTH document_id AND document_title
            # (document_id may be a hash; document_title retains the original filename)
            matched_ids: set[str] = set()
            for meta in bm25_metadata:
                doc_id = (meta.get("document_id") or "").lower()
                doc_title = (meta.get("document_title") or "").lower()
                if pattern in doc_id or pattern in doc_title:
                    matched_ids.add(meta["document_id"])

            for did in matched_ids:
                # Look up the actual stored title from metadata for this document
                stored_title = did  # fallback
                for meta in bm25_metadata:
                    if meta.get("document_id") == did:
                        stored_title = meta.get("document_title") or did
                        break
                doc = CanonicalDocument(
                    document_id=did,
                    canonical_title=seed["canonical_title"],
                    year=seed.get("year"),
                    jurisdiction=seed.get("jurisdiction", "India"),
                    aliases=list(seed.get("aliases", [])) + [stored_title],
                )
                self._register(doc)

        # 2. Auto-register remaining documents whose titles are already descriptive
        seen_ids: set[str] = {d.lower() for d in self._by_id}
        for meta in bm25_metadata:
            did = meta.get("document_id") or ""
            title = meta.get("document_title") or ""
            if not did or did.lower() in seen_ids:
                continue
            # Only auto-register if title is meaningfully different from the bare ID
            if title and title.lower() != did.lower():
                doc = CanonicalDocument(
                    document_id=did,
                    canonical_title=title,
                    aliases=self._generate_auto_aliases(title),
                )
                self._register(doc)
                seen_ids.add(did.lower())

        self._loaded = True
        logger.info(
            "LegalDocumentRegistry bootstrapped: %d canonical documents, %d aliases",
            len(self._by_id),
            len(self._alias_map),
        )

    def _register(self, doc: CanonicalDocument) -> None:
        self._by_id[doc.document_id] = doc
        for name in doc.all_searchable_names():
            self._alias_map[name] = doc.document_id

    def _generate_auto_aliases(self, title: str) -> list[str]:
        """Generate sensible aliases from a descriptive title."""
        aliases: list[str] = []
        # Drop "The " prefix
        stripped = re.sub(r"^the\s+", "", title, flags=re.IGNORECASE).strip()
        if stripped.lower() != title.lower():
            aliases.append(stripped)
        # Year-stripped version
        year_stripped = re.sub(r",?\s*\d{4}$", "", stripped).strip()
        if year_stripped and year_stripped.lower() != stripped.lower():
            aliases.append(year_stripped)
        # NOTE: Short abbreviations (e.g. "RA", "ACA") are intentionally NOT generated
        # here — they are too short to be safe for partial-substring matching and cause
        # false positives (e.g. "RA" substring of "cont**ra**ct").
        return aliases

    def resolve_act_name(self, query_act_name: str) -> list[str]:
        """
        Given a query act name (e.g. "Indian Contract Act"), return matching document_ids.
        Uses exact alias lookup first; falls back to partial substring matching only for
        aliases of length >= 5 to prevent short abbreviations from causing false matches.
        """
        q = query_act_name.lower().strip()
        matches: list[str] = []

        # Exact alias lookup
        if q in self._alias_map:
            matches.append(self._alias_map[q])
            return matches

        # Partial match: only for aliases with length >= 5 characters
        # This prevents short abbreviations like "RA" from matching "contract"
        MIN_ALIAS_LEN = 5
        for alias, did in self._alias_map.items():
            if len(alias) < MIN_ALIAS_LEN:
                continue
            if q in alias or alias in q:
                if did not in matches:
                    matches.append(did)

        return matches

    def get_canonical(self, document_id: str) -> CanonicalDocument | None:
        return self._by_id.get(document_id)

    def all_documents(self) -> list[CanonicalDocument]:
        return list(self._by_id.values())


# Module-level singleton
registry = LegalDocumentRegistry()
