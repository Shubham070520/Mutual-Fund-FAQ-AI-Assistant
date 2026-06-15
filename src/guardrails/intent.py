"""
Intent classifier — determines whether a user query is factual, advisory,
or out of scope. Uses a two-layer approach:

Layer 1: Fast keyword-based detection for obvious cases
Layer 2: LLM-based classification for ambiguous queries
"""

import logging
import re

from config.settings import GROQ_API_KEY, LLM_MODEL, ADVISORY_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# --- Intent Labels ---
INTENT_FACTUAL = "FACTUAL"
INTENT_ADVISORY = "ADVISORY"
INTENT_OUT_OF_SCOPE = "OUT_OF_SCOPE"

# --- Layer 1: Keyword-Based Detection ---

# Phrases that signal advisory / recommendation-seeking intent
ADVISORY_PHRASES: list[str] = [
    "should i invest",
    "should i buy",
    "should i sell",
    "which is better",
    "which fund",
    "which is best",
    "recommend",
    "suggest",
    "best fund",
    "best scheme",
    "good investment",
    "good fund",
    "good returns",
    "will it grow",
    "will it go up",
    "will returns",
    "give good returns",
    "give returns",
    "compare funds",
    "compare scheme",
    "better returns",
    "highest returns",
    "top performing",
    "worth investing",
    "is it safe",
    "risk free",
    "guaranteed returns",
    "how much should i invest",
    "how much to invest",
    "when to invest",
    "when to buy",
    "when to sell",
    "what should i invest",
]

# Patterns that indicate out-of-scope queries
OUT_OF_SCOPE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(weather|temperature|rain|forecast)\b", re.IGNORECASE),
    re.compile(r"\b(stock market|share price|equity tip)\b", re.IGNORECASE),
    re.compile(r"\b(credit card|loan|insurance|fd|fixed deposit|rd)\b", re.IGNORECASE),
    re.compile(r"\b(politics|election|government policy)\b", re.IGNORECASE),
    re.compile(r"\b(sports|cricket|football|movie|entertainment)\b", re.IGNORECASE),
    re.compile(r"\b(how to (?:cook|drive|travel|learn|study))\b", re.IGNORECASE),
    re.compile(r"\b(tell me (?:a joke|about yourself|your name))\b", re.IGNORECASE),
    re.compile(r"\b(what is your|who are you|who made you)\b", re.IGNORECASE),
]

# Patterns for mutual fund domain keywords (to detect in-scope)
MF_DOMAIN_KEYWORDS: list[str] = [
    "expense ratio", "nav", "sip", "lump sum", "minimum",
    "benchmark", "fund manager", "category", "aum",
    "exit load", "lock-in", "elss", "folio",
    "redemption", "dividend", "growth", "direct",
    "regular", "scheme", "fund", "portfolio",
    "returns", "performance", "rating", "risk",
    "hdfc", "mid cap", "small cap", "large cap",
    "etf", "fof", "gold", "silver", "defence",
]


def is_advisory_keyword(query: str) -> bool:
    """
    Layer 1: Fast keyword-based advisory detection.

    Checks if the query contains known advisory phrases like
    "should I invest", "which is better", "recommend", etc.

    Args:
        query: User input text

    Returns:
        True if advisory keywords detected
    """
    query_lower = query.lower().strip()

    for phrase in ADVISORY_PHRASES:
        if phrase in query_lower:
            logger.info("Advisory keyword detected: '%s' in query", phrase)
            return True

    return False


def is_out_of_scope(query: str) -> bool:
    """
    Check if the query is clearly out of the mutual fund domain.

    Uses regex patterns for common out-of-scope topics like weather,
    politics, sports, and other financial products (FD, insurance, etc.).

    Args:
        query: User input text

    Returns:
        True if the query is out of scope
    """
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if pattern.search(query):
            logger.info("Out-of-scope pattern matched: %s", pattern.pattern)
            return True

    return False


def is_mutual_fund_domain(query: str) -> bool:
    """
    Check if the query is related to mutual funds.

    Args:
        query: User input text

    Returns:
        True if mutual fund domain keywords found
    """
    query_lower = query.lower()
    for keyword in MF_DOMAIN_KEYWORDS:
        if keyword in query_lower:
            return True
    return False


# --- Layer 2: LLM-Based Classification ---

_LLM_CLASSIFIER_PROMPT = """You are a query intent classifier for a mutual fund FAQ assistant.

Classify the following user query into exactly ONE category:
- FACTUAL: The user is asking for objective, verifiable facts about a mutual fund scheme (expense ratio, NAV, SIP amount, benchmark, fund manager, etc.)
- ADVISORY: The user is seeking investment advice, recommendations, comparisons, or opinions (should I invest, which is better, will it grow, etc.)
- OUT_OF_SCOPE: The query is not related to mutual funds at all (weather, sports, cooking, other financial products, etc.)

Respond with ONLY the category name: FACTUAL, ADVISORY, or OUT_OF_SCOPE

Query: "{query}"
Category:"""


def classify_intent_llm(query: str) -> str:
    """
    Layer 2: LLM-based intent classification for ambiguous queries.

    Uses the Groq LLM to classify the query when keyword-based detection
    is inconclusive.

    Args:
        query: User input text

    Returns:
        One of: FACTUAL, ADVISORY, OUT_OF_SCOPE
    """
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — falling back to keyword classification")
        return _keyword_fallback(query)

    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)

        prompt = _LLM_CLASSIFIER_PROMPT.format(query=query)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a query intent classifier. Respond with only FACTUAL, ADVISORY, or OUT_OF_SCOPE."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )

        label = response.choices[0].message.content.strip().upper()

        # Validate label
        if label not in (INTENT_FACTUAL, INTENT_ADVISORY, INTENT_OUT_OF_SCOPE):
            logger.warning("LLM returned unexpected label: '%s' — using fallback", label)
            return _keyword_fallback(query)

        logger.info("LLM classified query as: %s", label)
        return label

    except Exception as e:
        logger.error("LLM classification failed: %s — using fallback", e)
        return _keyword_fallback(query)


def _keyword_fallback(query: str) -> str:
    """
    Keyword-based fallback when LLM classification is unavailable.

    Args:
        query: User input text

    Returns:
        Best-guess intent classification
    """
    if is_out_of_scope(query):
        return INTENT_OUT_OF_SCOPE

    if is_advisory_keyword(query):
        return INTENT_ADVISORY

    if is_mutual_fund_domain(query):
        return INTENT_FACTUAL

    # Default: assume factual for short queries, out-of-scope for non-MF
    logger.info("Keyword fallback: no clear signal — defaulting to OUT_OF_SCOPE")
    return INTENT_OUT_OF_SCOPE


def classify_intent(query: str) -> str:
    """
    Classify user query intent using a two-layer approach.

    Pipeline:
    1. Check for obvious out-of-scope patterns (fast)
    2. Check for advisory keywords (fast)
    3. Check for mutual fund domain keywords (fast)
    4. If ambiguous, fall back to LLM classification

    Args:
        query: User input text (should be pre-sanitized for PII)

    Returns:
        One of: "FACTUAL", "ADVISORY", "OUT_OF_SCOPE"
    """
    if not query or not query.strip():
        return INTENT_OUT_OF_SCOPE

    # Layer 1: Fast keyword checks
    if is_out_of_scope(query):
        logger.info("Intent: OUT_OF_SCOPE (keyword)")
        return INTENT_OUT_OF_SCOPE

    if is_advisory_keyword(query):
        logger.info("Intent: ADVISORY (keyword)")
        return INTENT_ADVISORY

    if is_mutual_fund_domain(query):
        logger.info("Intent: FACTUAL (keyword)")
        return INTENT_FACTUAL

    # Layer 2: LLM classification for ambiguous cases
    logger.info("Keywords inconclusive — using LLM classification")
    intent = classify_intent_llm(query)
    logger.info("Intent: %s (LLM)", intent)
    return intent
