"""
Conversation-Aware Query Router — Follow-Up Intelligence
Classifies user queries in multi-turn conversations into:
1. TYPE_A_CONTEXTUAL: Clarification, simplification, summary of previous answer. (NO NEW RETRIEVAL)
2. TYPE_B_RETRIEVAL_FOLLOWUP: Subject-linked follow-up requiring new legal evidence lookup.
3. TYPE_C_NEW_QUERY: Standalone new legal research query.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TurnType(str, Enum):
    CONTEXTUAL_FOLLOWUP = "contextual_followup"  # Type A: No retrieval
    RETRIEVAL_FOLLOWUP = "retrieval_followup"    # Type B: Context resolution + retrieval
    NEW_QUERY = "new_query"                      # Type C: Fresh retrieval


@dataclass
class ConversationRoutingResult:
    turn_type: TurnType
    resolved_query: str
    previous_answer: str | None = None
    previous_evidence: list[Any] | None = None
    previous_citations: list[Any] | None = None


# Patterns for Type A (pure explanation / simplification / summary of existing evidence)
_TYPE_A_PATTERNS = [
    r"^(can you\s+)?explain\s+(that|this|the previous answer)\s*(in\s+simple\s+terms|simply|more)?",
    r"^(what\s+does\s+that\s+mean|summarize\s+that|give\s+an?\s+example|clarify\s+that)",
    r"^(in\s+simple\s+words|simplify\s+this|break\s+that\s+down|rephrase\s+that)",
    r"^(why\s+is\s+that|explain\s+further|tell\s+me\s+more\s+about\s+that)",
]

# Words indicating referral to previous context
_REFERRAL_PRONOUNS = {"that", "this", "it", "those", "these", "the above", "said section", "such breach"}


def classify_conversation_turn(
    user_query: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> ConversationRoutingResult:
    """
    Classify turn type deterministically without extra LLM overhead.
    """
    clean_q = user_query.strip().lower()

    if not conversation_history:
        return ConversationRoutingResult(
            turn_type=TurnType.NEW_QUERY,
            resolved_query=user_query,
        )

    # Find last assistant message and user message
    last_assistant_msg = None
    last_user_msg = None

    for msg in reversed(conversation_history):
        role = msg.get("role", "")
        if role == "assistant" and not last_assistant_msg:
            last_assistant_msg = msg
        elif role == "user" and not last_user_msg:
            last_user_msg = msg
        if last_assistant_msg and last_user_msg:
            break

    if not last_assistant_msg:
        return ConversationRoutingResult(
            turn_type=TurnType.NEW_QUERY,
            resolved_query=user_query,
        )

    prev_ans = last_assistant_msg.get("content", "")
    prev_resp = last_assistant_msg.get("response", {})
    prev_citations = prev_resp.get("citations", []) if isinstance(prev_resp, dict) else getattr(prev_resp, "citations", [])
    prev_evidence = prev_resp.get("supporting_chunks", []) if isinstance(prev_resp, dict) else getattr(prev_resp, "supporting_chunks", [])

    # Check for Type A (pure explanation / simplification)
    for pat in _TYPE_A_PATTERNS:
        if re.search(pat, clean_q):
            logger.info("Conversation Router: TYPE A (Contextual Follow-up) detected — skipping retrieval")
            return ConversationRoutingResult(
                turn_type=TurnType.CONTEXTUAL_FOLLOWUP,
                resolved_query=user_query,
                previous_answer=prev_ans,
                previous_evidence=prev_evidence,
                previous_citations=prev_citations,
            )

    # Check for Type B (referral pronouns indicating context-linked new question)
    words = set(re.findall(r"\b\w+\b", clean_q))
    has_referral = bool(words & _REFERRAL_PRONOUNS)

    if has_referral and last_user_msg:
        prev_user_q = last_user_msg.get("content", "")
        resolved_q = f"{prev_user_q} — {user_query}"
        logger.info("Conversation Router: TYPE B (Retrieval Follow-up) detected — resolved: '%s'", resolved_q[:80])
        return ConversationRoutingResult(
            turn_type=TurnType.RETRIEVAL_FOLLOWUP,
            resolved_query=resolved_q,
            previous_answer=prev_ans,
            previous_evidence=prev_evidence,
            previous_citations=prev_citations,
        )

    logger.info("Conversation Router: TYPE C (New Standalone Query)")
    return ConversationRoutingResult(
        turn_type=TurnType.NEW_QUERY,
        resolved_query=user_query,
    )
