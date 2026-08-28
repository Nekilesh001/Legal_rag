"""
Token counting utility.
Uses tiktoken if available; falls back to word-count approximation.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_encoder = None
_USE_TIKTOKEN = False

try:
    import tiktoken
    _encoder = tiktoken.get_encoding("cl100k_base")
    _USE_TIKTOKEN = True
    logger.debug("Token counting: using tiktoken cl100k_base")
except Exception:
    logger.debug("tiktoken not available — using word-count proxy (×1.3 factor)")


def count_tokens(text: str) -> int:
    """
    Count tokens in a string.
    Uses tiktoken cl100k_base if available, otherwise approximates as words × 1.3.
    """
    if not text:
        return 0
    if _USE_TIKTOKEN and _encoder is not None:
        return len(_encoder.encode(text))
    # Approximation: average English word ≈ 1.3 tokens
    return int(len(text.split()) * 1.3)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately max_tokens.
    Prefers sentence boundaries when possible.
    """
    if count_tokens(text) <= max_tokens:
        return text

    if _USE_TIKTOKEN and _encoder is not None:
        tokens = _encoder.encode(text)
        truncated_tokens = tokens[:max_tokens]
        return _encoder.decode(truncated_tokens)
    else:
        # Word-based approximation
        words = text.split()
        max_words = int(max_tokens / 1.3)
        return " ".join(words[:max_words])
