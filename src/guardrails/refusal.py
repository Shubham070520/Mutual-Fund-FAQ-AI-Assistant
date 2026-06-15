"""
Refusal response generator — produces polite, informative refusal messages
when queries are classified as advisory or out-of-scope.

Every refusal includes:
1. Clear explanation that the bot only provides facts
2. An educational link (AMFI or SEBI)
3. A timestamp footer
"""

import logging
from datetime import datetime, timezone

from config.settings import EDUCATIONAL_LINKS

logger = logging.getLogger(__name__)

# --- Refusal Templates ---

ADVISORY_REFUSAL = (
    "I can only provide factual information about mutual fund schemes. "
    "I cannot offer investment advice, recommendations, or comparisons.\n\n"
    "For guidance on making investment decisions, please visit:\n"
    "{educational_link}\n\n"
    "Last updated from sources: {date}"
)

OUT_OF_SCOPE_REFUSAL = (
    "I'm designed to answer factual questions about HDFC Mutual Fund schemes only. "
    "Your question appears to be outside my scope.\n\n"
    "For general investor education, please visit:\n"
    "{educational_link}\n\n"
    "Last updated from sources: {date}"
)

EMPTY_QUERY_REFUSAL = (
    "Please ask a specific question about HDFC Mutual Fund schemes. "
    "For example: 'What is the expense ratio of HDFC Mid Cap Fund?'\n\n"
    "Last updated from sources: {date}"
)

NO_CONTEXT_REFUSAL = (
    "I don't have this information in my current sources. "
    "Please try rephrasing your question or ask about one of the supported schemes.\n\n"
    "Last updated from sources: {date}"
)

# --- Educational Link Selection ---

_ADVISORY_LINKS = {
    "default": EDUCATIONAL_LINKS.get("default", "https://www.amfiindia.com/investor-education"),
    "sebi": EDUCATIONAL_LINKS.get("sebi", "https://www.sebi.gov.in/investor-education"),
}


def _get_educational_link(intent: str) -> str:
    """
    Select the appropriate educational link based on intent.

    Args:
        intent: Classification intent (ADVISORY or OUT_OF_SCOPE)

    Returns:
        URL string for investor education
    """
    if intent == "OUT_OF_SCOPE":
        return _ADVISORY_LINKS["sebi"]
    return _ADVISORY_LINKS["default"]


def _get_current_date() -> str:
    """Return the current date in a human-readable format."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d")


def generate_refusal(intent: str, query: str = "") -> str:
    """
    Generate a polite refusal response based on the classified intent.

    Handles three scenarios:
    - ADVISORY: User seeks advice/recommendations → polite decline + AMFI link
    - OUT_OF_SCOPE: Query not about mutual funds → scope explanation + SEBI link
    - Empty/blank query → prompt with example question

    Args:
        intent: One of "FACTUAL", "ADVISORY", "OUT_OF_SCOPE"
        query: Original user query (used for logging)

    Returns:
        Formatted refusal string with educational link and date footer
    """
    date = _get_current_date()

    if not query or not query.strip():
        logger.info("Generating refusal for empty query")
        return EMPTY_QUERY_REFUSAL.format(date=date)

    if intent == "ADVISORY":
        link = _get_educational_link("ADVISORY")
        logger.info("Generating advisory refusal (link=%s)", link)
        return ADVISORY_REFUSAL.format(
            educational_link=link,
            date=date,
        )

    if intent == "OUT_OF_SCOPE":
        link = _get_educational_link("OUT_OF_SCOPE")
        logger.info("Generating out-of-scope refusal (link=%s)", link)
        return OUT_OF_SCOPE_REFUSAL.format(
            educational_link=link,
            date=date,
        )

    # Fallback: if intent is FACTUAL but no context was found
    logger.info("Generating no-context refusal")
    return NO_CONTEXT_REFUSAL.format(date=date)


def is_refusal_response(response: str) -> bool:
    """
    Check if a response string is a refusal (contains refusal markers).

    Useful for post-processing validation to ensure factual responses
    don't accidentally contain refusal language.

    Args:
        response: Response text to check

    Returns:
        True if the response appears to be a refusal
    """
    refusal_markers = [
        "I can only provide factual information",
        "I cannot offer investment advice",
        "outside my scope",
        "I don't have this information",
        "Please ask a specific question",
    ]

    response_lower = response.lower()
    return any(marker.lower() in response_lower for marker in refusal_markers)


def format_factual_footer(date: str | None = None) -> str:
    """
    Generate the standard footer for factual responses.

    Args:
        date: Optional date string; defaults to today

    Returns:
        Footer string like "Last updated from sources: 2026-06-09"
    """
    if date is None:
        date = _get_current_date()
    return f"Last updated from sources: {date}"
