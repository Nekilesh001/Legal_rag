"""
Legal-Aware Ranking Layer — Experiment 4.

A deterministic, query-aware scoring layer applied AFTER RRF fusion
and BEFORE the cross-encoder reranker.

Design:
  legal_score = section_match
              + document_match
              + authority_score          (per-tier weight × tier_level)
              + category_match
              + jurisdiction_match
              - form_penalty             (query-intent-gated)
              - admin_notice_penalty     (query-intent-gated)
              - definition_penalty       (OBLIGATION_QUERY only)

  combined_score = adjusted_rrf_score + legal_blend × legal_score

Candidates are re-sorted by combined_score before the cross-encoder
sees them, so the cross-encoder operates on a legally-informed set.

Key invariant: authority and form penalties are SIGNALS, not hard filters.
A Rules document that is explicitly requested can still outrank an Act.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

from legal_rag.models.retrieval import QueryAnalysis, QueryIntent
from legal_rag.retrieval.legal_identity import LegalDocumentRegistry

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Source-authority taxonomy
# ------------------------------------------------------------------ #

class SourceAuthority(IntEnum):
    """
    Hierarchical authority of a legal source document.
    Higher = more authoritative for purposes of operative provisions.
    """
    COMMENTARY_OTHER = 0
    RULEBOOK_PLAYBOOK = 1   # Contract playbooks, NDA mandatory-clause guides
    GOVERNMENT_ORDER = 2    # G.O., Notifications
    RULES_REGULATIONS = 3   # Statutory rules, subordinate legislation
    PRIMARY_ACT = 4         # Parliament/Assembly Act, Statute, Ordinance


# ------------------------------------------------------------------ #
# Content-type taxonomy
# ------------------------------------------------------------------ #

class ContentType(str, Enum):
    OPERATIVE_PROVISION = "operative_provision"
    DEFINITION = "definition"
    HEADING = "heading"
    FORM_TEMPLATE = "form_template"
    ADMINISTRATIVE_NOTICE = "administrative_notice"
    RULEBOOK_CLAUSE = "rulebook_clause"
    UNKNOWN = "unknown"


# ------------------------------------------------------------------ #
# Pattern banks
# ------------------------------------------------------------------ #

# Form / administrative template indicators
_FORM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bFORM\s+[A-Z]\b"),
    re.compile(r"\bForm\s+[A-Z]\b"),
    re.compile(r"\[See\s+Rule", re.IGNORECASE),        # Form preamble
    re.compile(r"\[Name\s+of", re.IGNORECASE),
    re.compile(r"\[to be filled", re.IGNORECASE),
    re.compile(r"\[Signature\b", re.IGNORECASE),
]

# Administrative notice (broader — does not require a blank form)
_ADMIN_NOTICE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"NOTICE\s+IS\s+HEREBY\s+GIVEN", re.IGNORECASE),
    re.compile(r"hereby\s+notif", re.IGNORECASE),
]

# Definition section indicators
_DEFINITION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bin this (act|chapter|section|part)\b", re.IGNORECASE),
    re.compile(r'unless the context (otherwise )?requires', re.IGNORECASE),
    re.compile(r'"\w[\w\s]+" (?:means|shall mean)\b', re.IGNORECASE),
    re.compile(r"\bshall have the meaning\b", re.IGNORECASE),
]

# Chapter/Part heading
_HEADING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*(?:CHAPTER|PART|SCHEDULE)\s+[IVXLCDM\d]+", re.IGNORECASE | re.MULTILINE),
]


# ------------------------------------------------------------------ #
# Classification functions
# ------------------------------------------------------------------ #

def get_source_authority(
    meta: dict[str, Any],
    registry: "LegalDocumentRegistry | None" = None,
) -> SourceAuthority:
    """
    Derive source-authority tier from chunk metadata.

    Resolution order (stops at first match):
      1. Canonical registry lookup — seeded documents (e.g. '193003' = Sale of Goods Act)
         are classified as PRIMARY_ACT regardless of their opaque filename.
      2. Title/category keyword heuristics for non-seeded documents.

    Args:
        meta: chunk metadata dict (must contain 'document_id' and/or 'document_title').
        registry: optional LegalDocumentRegistry; when provided, seeded documents receive
                  their canonical authority tier before keyword fallbacks.
    """
    doc_id = (meta.get("document_id") or "").lower()
    title  = (meta.get("document_title") or doc_id).lower()
    category = (meta.get("category") or "").lower()

    # ---- 1. Registry-first lookup (highest priority) ----
    # Seeded documents have opaque filenames (e.g. '193003', 'A187209') that
    # contain no keyword signals.  The registry knows their true canonical title
    # and they should always be classified as PRIMARY_ACT.
    if registry is not None and registry._loaded:
        canonical = registry.get_canonical(meta.get("document_id") or "")
        if canonical is not None:
            # Seeded/registered document — derive authority from canonical title
            ctitle = canonical.canonical_title.lower()
            if any(kw in ctitle for kw in ("playbook", "rulebook", "mandatory clause", "nda")):
                return SourceAuthority.RULEBOOK_PLAYBOOK
            if re.match(r"g\.o\b", ctitle) or any(
                kw in ctitle for kw in ("g.o.", "government order", "notification", "gazette")
            ):
                return SourceAuthority.GOVERNMENT_ORDER
            if any(kw in ctitle for kw in ("rules", "regulations", "bye-laws", "bylaws")):
                return SourceAuthority.RULES_REGULATIONS
            if any(kw in ctitle for kw in ("act", "code", "ordinance", "statute")):
                return SourceAuthority.PRIMARY_ACT
            # Registered but no keyword match — default to PRIMARY_ACT for seeded entries
            return SourceAuthority.PRIMARY_ACT

    # ---- 2. Keyword heuristics on raw title / category ----
    # Contract playbooks / NDA mandatory clause guides
    if any(kw in title or kw in category
           for kw in ("playbook", "rulebook", "mandatory clause", "mandatory_clause", "nda")):
        return SourceAuthority.RULEBOOK_PLAYBOOK

    # Government orders / gazette notifications
    if re.match(r"g\.o\b", title) or any(
        kw in title for kw in ("g.o.", "government order", "notification", "gazette")
    ):
        return SourceAuthority.GOVERNMENT_ORDER

    # Rules / regulations — check BEFORE "act" to handle documents whose title
    # references a parent act (e.g. "Tamil Nadu Shops and Establishments Rules")
    if any(kw in title for kw in ("rules", "regulations", "bye-laws", "bylaws")):
        return SourceAuthority.RULES_REGULATIONS

    # Primary Acts / Statutes
    if any(kw in title for kw in ("act", "code", "ordinance", "statute")):
        return SourceAuthority.PRIMARY_ACT

    # Category-based fallback for known statutory domains
    if category in ("vendor", "employment", "lease", "dispute", "finance", "ip_acts"):
        return SourceAuthority.PRIMARY_ACT

    return SourceAuthority.COMMENTARY_OTHER


def classify_content_type(meta: dict[str, Any], text: str) -> ContentType:
    """
    Classify content type from metadata and first 400 characters of chunk text.
    Deterministic — no LLM.
    """
    title = (meta.get("document_title") or "").lower()
    category = (meta.get("category") or "").lower()
    sec_title = (meta.get("section_title") or "").lower()
    sample = text[:500] if text else ""

    # Contract playbook clauses
    if any(kw in title or kw in category
           for kw in ("playbook", "rulebook", "mandatory clause", "nda")):
        return ContentType.RULEBOOK_CLAUSE
    if sample and sample[:50].lower().startswith("clause:"):
        return ContentType.RULEBOOK_CLAUSE

    # Form templates (highest-priority admin classification)
    for pat in _FORM_PATTERNS:
        if pat.search(sample):
            return ContentType.FORM_TEMPLATE

    # Chapter / Part headings
    for pat in _HEADING_PATTERNS:
        if pat.match(sample):
            return ContentType.HEADING

    # Administrative notices (after form check so FORM A is caught first)
    for pat in _ADMIN_NOTICE_PATTERNS:
        if pat.search(sample):
            return ContentType.ADMINISTRATIVE_NOTICE

    # Definitions
    for pat in _DEFINITION_PATTERNS:
        if pat.search(sample):
            return ContentType.DEFINITION
    if any(kw in sec_title for kw in (
        "definition", "interpretation", "meaning", "preliminary", "short title"
    )):
        return ContentType.DEFINITION

    # Default — treat as operative provision
    return ContentType.OPERATIVE_PROVISION


# ------------------------------------------------------------------ #
# Configurable weights
# ------------------------------------------------------------------ #

@dataclass
class LegalRankerWeights:
    """
    Weights for each scoring component of the legal-aware ranker.
    Document these clearly so they can be tuned systematically.

    Rationale for values:
      section_match   25.0  — explicit section lookup is the strongest signal
      document_match  10.0  — queried Act = correct authoritative source
      authority        3.0  — per-tier multiplier; Act(4)=12, Rules(3)=9, GO(2)=6
      category         2.0  — query-domain alignment
      jurisdiction     2.0  — geographic alignment
      form_penalty    15.0  — clear anti-signal for operative-provision queries
      admin_penalty   10.0  — softer than form — "notice" text is broad
      def_penalty      5.0  — mild penalty for definitions on obligation queries
      intent_content_pref 8.0 — intent-aware content-type preference
      concept_match    4.0  — alignment per matched legal concept
      legal_blend      0.001 — scales legal_score to RRF score magnitude
    """
    section_match: float = 25.0
    document_match: float = 10.0
    authority: float = 3.0
    category: float = 2.0
    jurisdiction: float = 2.0
    form_penalty: float = 15.0
    admin_notice_penalty: float = 10.0
    definition_penalty_on_obligation: float = 5.0
    intent_content_pref: float = 8.0
    concept_match: float = 4.0
    legal_blend: float = 0.001


# Query intents for which form/admin/definition penalties apply
_OPERATIVE_INTENTS: frozenset[QueryIntent] = frozenset({
    QueryIntent.SPECIFIC_CLAUSE_LOOKUP,
    QueryIntent.THRESHOLD_PARAMETER,
    QueryIntent.OBLIGATION_QUERY,
})

_LEGAL_CONCEPT_KEYWORDS: dict[str, list[str]] = {
    "breach": ["breach", "breaches", "default", "violation"],
    "remedy": ["remedy", "remedies", "relief", "rectification", "suit for"],
    "warranty": ["warranty", "warranties", "guarantee"],
    "damages": ["damages", "compensation", "loss", "indemnity", "diminution"],
    "notice": ["notice", "notice period", "dismissal", "discharge", "dispense with services"],
    "termination": ["termination", "cancellation", "rescission"],
    "obligation": ["obligation", "mandatory", "shall", "must", "duty"],
    "delivery": ["delivery", "delivers", "conveyance", "shipment"],
    "price": ["price", "payment", "consideration", "cost"],
}


def extract_concepts_from_text(text: str) -> set[str]:
    """Extract recognized legal concepts from text using keyword dictionary."""
    t_lower = text.lower()
    found: set[str] = set()
    for concept, kws in _LEGAL_CONCEPT_KEYWORDS.items():
        if any(kw in t_lower for kw in kws):
            found.add(concept)
    return found


def compute_intent_content_preference(
    intent: QueryIntent,
    content_type: ContentType,
    pref_weight: float = 8.0,
) -> float:
    """
    Computes intent-aware content-type preference score.
    Relates content-type directly to query intent (never a global override).
    """
    if intent == QueryIntent.THRESHOLD_PARAMETER:
        if content_type == ContentType.OPERATIVE_PROVISION:
            return pref_weight
        elif content_type in (ContentType.HEADING, ContentType.FORM_TEMPLATE):
            return -pref_weight
        elif content_type == ContentType.DEFINITION:
            return -6.0

    elif intent == QueryIntent.OBLIGATION_QUERY:
        if content_type in (ContentType.OPERATIVE_PROVISION, ContentType.RULEBOOK_CLAUSE):
            return pref_weight * 0.75
        elif content_type in (ContentType.HEADING, ContentType.ADMINISTRATIVE_NOTICE):
            return -pref_weight * 0.75
        elif content_type == ContentType.DEFINITION:
            return -5.0
        elif content_type == ContentType.FORM_TEMPLATE:
            return -15.0

    elif intent == QueryIntent.DEFINITION_INQUIRY:
        if content_type == ContentType.DEFINITION:
            return pref_weight * 1.25

    elif intent == QueryIntent.SPECIFIC_CLAUSE_LOOKUP:
        if content_type == ContentType.OPERATIVE_PROVISION:
            return pref_weight * 0.5
        elif content_type == ContentType.HEADING:
            return -pref_weight * 0.5

    return 0.0


# ------------------------------------------------------------------ #
# Main ranker class
# ------------------------------------------------------------------ #

class LegalAwareRanker:
    """
    Deterministic legal-aware ranking layer.

    Scores each candidate on legal-structural signals and produces a
    combined score that is used to re-sort the candidate pool BEFORE
    the cross-encoder sees it.  The cross-encoder then operates on a
    legally-informed top-N, not just on raw RRF ordering.
    """

    def __init__(
        self,
        weights: LegalRankerWeights | None = None,
        registry: LegalDocumentRegistry | None = None,
    ) -> None:
        self.weights = weights or LegalRankerWeights()
        self.registry = registry

    # ---------------------------------------------------------------- #
    # Per-candidate scoring
    # ---------------------------------------------------------------- #

    def score_candidate(
        self,
        candidate: dict[str, Any],
        query_analysis: QueryAnalysis,
        resolved_doc_ids: set[str],
    ) -> tuple[float, dict[str, Any]]:
        """
        Compute (legal_score, breakdown) for one candidate.
        breakdown contains individual component scores for debugging.
        """
        w = self.weights
        meta = candidate.get("payload", candidate)
        text = candidate.get("text", "")

        doc_id = (meta.get("document_id") or "").lower()
        sec_num = str(meta.get("section_number") or "").lower()
        sec_title = (meta.get("section_title") or "").lower()
        category = (meta.get("category") or "").lower()
        doc_title = (meta.get("document_title") or "").lower()

        # Classify structural attributes — pass registry for registry-first authority lookup
        authority = get_source_authority(meta, registry=self.registry)
        content_type = classify_content_type(meta, text)

        score = 0.0
        breakdown: dict[str, Any] = {
            "authority_tier": authority.name,
            "content_type": content_type.value,
        }

        # --- 1. Section match ---
        sec_match = 0.0
        sec_refs = [
            s.lower().replace("section", "").strip()
            for s in query_analysis.section_refs
        ]
        for sref in sec_refs:
            if not sref:
                continue
            if sref == sec_num:
                sec_match = w.section_match
                break
            # Word-boundary match (73 matches "73(a)" but not "173")
            if re.search(r"(?<!\d)" + re.escape(sref) + r"(?!\d)", sec_num):
                sec_match = w.section_match * 0.9
                break
            if sref in sec_title:
                sec_match = w.section_match * 0.7
                break
        score += sec_match
        breakdown["section_match"] = sec_match

        # --- 2. Document match (canonical identity) ---
        doc_match = 0.0
        if resolved_doc_ids and doc_id in resolved_doc_ids:
            doc_match = w.document_match
        score += doc_match
        breakdown["document_match"] = doc_match

        # --- 3. Source authority ---
        auth_score = authority.value * w.authority
        score += auth_score
        breakdown["authority_score"] = auth_score

        # --- 4. Category match ---
        cat_score = 0.0
        for cat_hint in query_analysis.category_hints:
            if (
                cat_hint == category
                or cat_hint in doc_title
                or (cat_hint == "nda" and (
                    "nda" in doc_title
                    or "mandatory" in doc_title
                    or "playbook" in doc_title
                ))
            ):
                cat_score = w.category
                break
        score += cat_score
        breakdown["category_score"] = cat_score

        # --- 5. Jurisdiction match ---
        jur_score = 0.0
        for jur in query_analysis.jurisdictions:
            j = jur.lower()
            if j in doc_title or j in doc_id:
                jur_score = w.jurisdiction
                break
        score += jur_score
        breakdown["jurisdiction_score"] = jur_score

        # --- 6. Intent-Aware Content-Type Preference ---
        intent_pref_score = compute_intent_content_preference(
            query_analysis.intent, content_type, pref_weight=w.intent_content_pref
        )
        score += intent_pref_score
        breakdown["intent_content_pref_score"] = intent_pref_score

        # --- 7. Legal-Concept Alignment ---
        q_concepts = extract_concepts_from_text(
            query_analysis.original_query + " " + " ".join(query_analysis.category_hints)
        )
        c_concepts = extract_concepts_from_text(
            f"{text} {sec_title} {doc_title}"
        )
        matched_concepts = q_concepts.intersection(c_concepts)
        concept_score = len(matched_concepts) * w.concept_match
        score += concept_score
        breakdown["matched_concepts"] = list(matched_concepts)
        breakdown["concept_score"] = concept_score

        # --- 8. Query-aware form / admin / definition penalties ---
        penalty = 0.0
        if query_analysis.intent in _OPERATIVE_INTENTS:
            if content_type == ContentType.FORM_TEMPLATE:
                penalty = w.form_penalty
            elif content_type == ContentType.ADMINISTRATIVE_NOTICE:
                penalty = w.admin_notice_penalty
            elif (
                content_type == ContentType.DEFINITION
                and query_analysis.intent == QueryIntent.OBLIGATION_QUERY
            ):
                penalty = w.definition_penalty_on_obligation
        score -= penalty
        breakdown["penalty"] = -penalty

        breakdown["legal_score"] = score
        return score, breakdown

    # ---------------------------------------------------------------- #
    # Pool-level ranking
    # ---------------------------------------------------------------- #

    def rank(
        self,
        candidates: list[dict[str, Any]],
        query_analysis: QueryAnalysis,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Score all candidates, compute combined_score, re-sort pool.

        combined_score = adjusted_rrf_score + legal_blend × legal_score

        Returns the full pool re-sorted by combined_score (or top_k items).
        """
        # Resolve canonical document IDs for queried act names
        resolved_doc_ids: set[str] = set()
        if self.registry is not None:
            for act in query_analysis.act_names:
                for did in self.registry.resolve_act_name(act):
                    resolved_doc_ids.add(did.lower())

        results: list[dict[str, Any]] = []
        for cand in candidates:
            legal_score, breakdown = self.score_candidate(
                cand, query_analysis, resolved_doc_ids
            )
            item = cand.copy()
            # Use adjusted_score (post-policy) as base if available, else rrf_score
            base = item.get("adjusted_score", item.get("rrf_score", item.get("score", 0.0)))
            combined = base + self.weights.legal_blend * legal_score

            item["legal_score"] = legal_score
            item["legal_breakdown"] = breakdown
            item["legal_combined_score"] = combined
            item["source_authority"] = breakdown["authority_tier"]
            item["content_type"] = breakdown["content_type"]
            results.append(item)

        results.sort(key=lambda x: x["legal_combined_score"], reverse=True)

        logger.info(
            "LegalAwareRanker: scored %d candidates → top resolved_doc_ids=%s",
            len(results),
            list(resolved_doc_ids)[:4],
        )
        return results[:top_k] if top_k is not None else results

