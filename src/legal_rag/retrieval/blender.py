"""
Score Normalization, Blending, and Evidence Protection module — Experiment 7.

Provides:
1. Batch Min-Max Normalization for BGE cross-encoder and LegalAwareRanker scores.
2. Controlled Score Blending: blended_score = norm_bge + lambda * norm_legal.
3. Deterministic Evidence Protection: assigns Tier 1 / Tier 2 protection boosts
   to structured legal candidates to prevent generic semantic candidates from
   completely displacing strong legal evidence.
"""
from __future__ import annotations

import logging
from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field

from legal_rag.models.retrieval import QueryAnalysis

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Protection Tier Taxonomy
# ------------------------------------------------------------------ #

class ProtectionTier(IntEnum):
    NONE = 0
    TIER_2_STRONG_LINKED = 1   # Source linked + strong legal score / category
    TIER_1_EXACT_EVIDENCE = 2  # Exact document match AND exact section match


# ------------------------------------------------------------------ #
# Score Normalizer & Blender
# ------------------------------------------------------------------ #

def min_max_normalize(scores: list[float], eps: float = 1e-8) -> list[float]:
    """
    Min-Max normalize a list of float scores to [0.0, 1.0].
    If all scores are identical, returns 0.5 for all elements.
    """
    if not scores:
        return []
    min_val = min(scores)
    max_val = max(scores)
    rng = max_val - min_val
    if rng < eps:
        return [0.5] * len(scores)
    return [(s - min_val) / rng for s in scores]


class ScoreBlender:
    """
    Normalizes and blends BGE reranker scores with LegalAwareRanker scores.
    """

    def __init__(self, lambda_weight: float = 0.25) -> None:
        self.lambda_weight = lambda_weight

    def blend_batch(
        self,
        candidates: list[dict[str, Any]],
        lambda_weight: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Calculates normalized_bge_score, normalized_legal_score, and blended_score
        for a batch of candidates (in-place modification / returning new list).
        """
        lam = lambda_weight if lambda_weight is not None else self.lambda_weight
        if not candidates:
            return []

        raw_bge = [c.get("reranker_score", 0.0) for c in candidates]
        raw_legal = [c.get("legal_score", 0.0) for c in candidates]

        norm_bge = min_max_normalize(raw_bge)
        norm_legal = min_max_normalize(raw_legal)

        results = []
        for c, nb, nl in zip(candidates, norm_bge, norm_legal):
            item = c.copy()
            item["normalized_bge_score"] = nb
            item["normalized_legal_score"] = nl
            item["blended_score"] = nb + lam * nl
            results.append(item)

        return results


# ------------------------------------------------------------------ #
# Protection Handler
# ------------------------------------------------------------------ #

class ProtectedEvidenceHandler:
    """
    Determines protection tiers for structured candidates and applies
    controlled eligibility boosts.
    """

    def __init__(
        self,
        tier_1_boost: float = 0.35,
        tier_2_boost: float = 0.20,
    ) -> None:
        self.tier_1_boost = tier_1_boost
        self.tier_2_boost = tier_2_boost

    def evaluate_protection_tier(
        self,
        candidate: dict[str, Any],
        query_analysis: QueryAnalysis,
    ) -> ProtectionTier:
        """
        Classifies a candidate into Tier 1, Tier 2, or NONE based on metadata signals.
        """
        breakdown = candidate.get("legal_breakdown", {})
        sec_match = breakdown.get("section_match", 0.0)
        doc_match = breakdown.get("document_match", 0.0)
        retrieval_src = candidate.get("retrieval_source", "")
        legal_score = candidate.get("legal_score", 0.0)
        authority_score = breakdown.get("authority_score", 0.0)

        # Tier 1 — Exact evidence: exact doc match AND exact section match
        if (doc_match > 0 and sec_match > 0) or "exact_section_lookup" in retrieval_src:
            return ProtectionTier.TIER_1_EXACT_EVIDENCE

        # Tier 2 — Strong linked evidence: doc match OR high authority + category match
        if (doc_match > 0 and (sec_match > 0 or legal_score >= 12.0)) or (
            authority_score >= 12.0 and legal_score >= 14.0
        ):
            return ProtectionTier.TIER_2_STRONG_LINKED

        return ProtectionTier.NONE

    def apply_protection(
        self,
        candidates: list[dict[str, Any]],
        query_analysis: QueryAnalysis,
        score_key: str = "blended_score",
    ) -> list[dict[str, Any]]:
        """
        Applies protection tier boosts to candidates and sorts by protected_score.
        """
        results = []
        for c in candidates:
            tier = self.evaluate_protection_tier(c, query_analysis)
            boost = 0.0
            if tier == ProtectionTier.TIER_1_EXACT_EVIDENCE:
                boost = self.tier_1_boost
            elif tier == ProtectionTier.TIER_2_STRONG_LINKED:
                boost = self.tier_2_boost

            base_score = c.get(score_key, c.get("normalized_bge_score", c.get("reranker_score", 0.0)))
            item = c.copy()
            item["protection_tier"] = tier.name
            item["protection_boost"] = boost
            item["protected_score"] = base_score + boost
            results.append(item)

        results.sort(key=lambda x: x["protected_score"], reverse=True)
        return results
