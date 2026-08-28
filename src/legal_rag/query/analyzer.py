"""
Query understanding: extract structured hints from a user query.
Deterministic regex + keyword matching — no LLM used here.
"""
from __future__ import annotations

import logging
import re

from legal_rag.models.retrieval import QueryAnalysis, QueryIntent

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Patterns
# ------------------------------------------------------------------ #

_SECTION_RE = re.compile(
    r"\b(?:Section|Sec\.|S\.)\s*(\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))*)",
    re.IGNORECASE,
)
_ARTICLE_RE = re.compile(r"\bArticle\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
_SCHEDULE_RE = re.compile(r"\bSchedule\s+([IVXLCDM]+|\d+[A-Z]?)\b", re.IGNORECASE)
# Known legal documents / Acts in corpus
_KNOWN_ACTS: list[str] = [
    "Indian Contract Act",
    "Sale of Goods Act",
    "Tamil Nadu Shops and Establishments Act",
    "Tamil Nadu Shops Act",
    "Code of Civil Procedure",
    "Arbitration and Conciliation Act",
    "Negotiable Instruments Act",
    "Transfer of Property Act",
    "Minimum Wages Act",
    "Payment of Gratuity Act",
    "Payment of Wages Act",
    "Maternity Benefit Act",
    "Trade Marks Act",
    "Information Technology Act",
    "Indian Stamp Act",
    "Registration Act",
    "Competition Act",
    "MSME Act",
]

_ACT_RE = re.compile(
    r"\b(?:[A-Z][a-z0-9]+\s+)+(?:Act|Code|Rules?|Regulations?|Ordinance)\b"
    r"(?:\s*,\s*\d{4}|\s+\d{4})?"
)

_DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b")
_NUMBER_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(%|days?|months?|years?|rupees?|percent|INR|Rs\.?)\b",
    re.IGNORECASE,
)
_JURISDICTION_RE = re.compile(
    r"\b(India|Tamil Nadu|Karnataka|Maharashtra|Delhi|Andhra Pradesh|Telangana|"
    r"Kerala|West Bengal|Gujarat|Rajasthan|Punjab|Haryana)\b",
    re.IGNORECASE,
)

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "employment": ["employment", "employee", "employer", "salary", "wages", "maternity",
                   "gratuity", "provident", "shops", "labor", "labour", "notice period"],
    "lease": ["lease", "rent", "landlord", "tenant", "property", "stamp duty", "registration"],
    "nda": ["non-disclosure", "nda", "confidential", "trade secret"],
    "finance": ["finance", "bank", "loan", "debt", "negotiable instrument",
                "cheque", "dishonour", "recovery", "sarfaesi"],
    "ip_acts": ["patent", "trademark", "copyright", "intellectual property", "it act", "cyber"],
    "vendor": ["vendor", "supply", "contract", "competition", "msme", "gst", "goods", "seller", "breach"],
    "dispute": ["arbitration", "dispute", "civil procedure", "limitation", "mediation"],
    "case_law": ["judgment", "court", "supreme court", "high court", "case", "appeal"],
}

_INTENT_SIGNALS: dict[QueryIntent, list[str]] = {
    QueryIntent.SPECIFIC_CLAUSE_LOOKUP: [
        "what does section", "section says", "text of section", "what is section", "section"
    ],
    QueryIntent.THRESHOLD_PARAMETER: [
        "notice period", "how many days", "percentage", "maximum", "minimum", "within how"
    ],
    QueryIntent.OBLIGATION_QUERY: [
        "mandatory clauses", "mandatory clause", "obligation", "must", "shall", "required to",
        "liable", "duty of", "responsible", "breaches", "breach"
    ],
    QueryIntent.DEFINITION_INQUIRY: [
        "defined", "definition", "meaning of", "shall mean"
    ],
    QueryIntent.CROSS_CONTRACT_COMPARISON: [
        "compare", "difference between", "across", "all contracts", "both agreements"
    ],
    QueryIntent.COMPLIANCE_AUDIT: [
        "allowed", "permissible", "valid", "enforceable", "is it legal", "comply", "violation"
    ],
}


def analyze_query(query: str) -> QueryAnalysis:
    """
    Extract structured information from a user's legal query.
    Purely deterministic — no LLM invoked.
    """
    section_refs = [m.group(0) for m in _SECTION_RE.finditer(query)]
    section_refs += [m.group(0) for m in _ARTICLE_RE.finditer(query)]
    section_refs += [m.group(0) for m in _SCHEDULE_RE.finditer(query)]

    # Extract Acts safely
    act_names: list[str] = []
    # 1. Regex title-cased matches
    for m in _ACT_RE.finditer(query):
        act_str = m.group(0).strip()
        # Clean question prefixes if any
        act_clean = re.sub(r"^(What|How|Which|Why|Is|Are|Under|Where)\s+", "", act_str, flags=re.I).strip()
        if len(act_clean.split()) >= 2 and len(act_clean) < 60:
            act_names.append(act_clean)

    # 2. Known Acts match
    query_lower = query.lower()
    for known in _KNOWN_ACTS:
        if known.lower() in query_lower and known not in act_names:
            act_names.append(known)

    dates = [m.group(1) for m in _DATE_RE.finditer(query)]
    numbers = [m.group(0) for m in _NUMBER_RE.finditer(query)]
    jurisdictions = list(set(m.group(1) for m in _JURISDICTION_RE.finditer(query)))

    category_hints = [
        cat for cat, kws in _CATEGORY_KEYWORDS.items()
        if any(kw in query_lower for kw in kws)
    ]

    # Determine intent
    intent = QueryIntent.UNKNOWN
    for candidate_intent, signals in _INTENT_SIGNALS.items():
        if any(sig in query_lower for sig in signals):
            intent = candidate_intent
            break
    if intent == QueryIntent.UNKNOWN and section_refs:
        intent = QueryIntent.SPECIFIC_CLAUSE_LOOKUP

    analysis = QueryAnalysis(
        original_query=query,
        normalized_query=query.strip(),
        intent=intent,
        section_refs=section_refs,
        act_names=act_names,
        jurisdictions=jurisdictions,
        category_hints=category_hints,
        dates=dates,
        numbers=numbers,
    )
    logger.debug(
        "Query analysis: intent=%s sections=%s acts=%s cats=%s",
        intent.value, section_refs[:3], act_names[:2], category_hints[:3],
    )
    return analysis
