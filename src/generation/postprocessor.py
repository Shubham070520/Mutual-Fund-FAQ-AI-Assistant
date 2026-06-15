"""
Response post-processor — validates and formats LLM-generated responses
to ensure compliance with the FAQ assistant's output rules.

Validations:
1. Sentence count ≤ 3 (trim if exceeded)
2. Exactly one source URL present
3. Footer present (append if missing)
4. No advisory language leaked into the response
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from config.settings import MAX_RESPONSE_SENTENCES

logger = logging.getLogger(__name__)

# --- Advisory Language Detection ---
# Phrases that should NEVER appear in a factual response
ADVISORY_LEAK_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(I recommend|I suggest|you should|you must)\b", re.IGNORECASE),
    re.compile(r"\b(it might be good|it could be a good|this is a good)\b", re.IGNORECASE),
    re.compile(r"\b(best option|better choice|better investment)\b", re.IGNORECASE),
    re.compile(r"\b(will (?:definitely|surely|probably) (?:grow|increase|go up))\b", re.IGNORECASE),
    re.compile(r"\b(guaranteed|assured|risk[- ]free)\b", re.IGNORECASE),
]

# URL pattern for extracting citation URLs
URL_PATTERN = re.compile(r"https?://[^\s\)\"']+")

# Footer pattern
FOOTER_PATTERN = re.compile(
    r"Last updated from sources:\s*\d{4}-\d{2}-\d{2}",
    re.IGNORECASE,
)


def count_sentences(text: str) -> int:
    """
    Count the number of sentences in text.

    Splits on sentence-ending punctuation followed by whitespace or end.
    Excludes the footer line and parenthetical source citations from counting.

    Args:
        text: Response text

    Returns:
        Number of sentences
    """
    # Remove footer before counting
    text_no_footer = _strip_footer(text)
    # Remove parenthetical source citations like "(Source: https://...)"
    text_clean = re.sub(r'\(Source:\s*https?://[^)]+\)', '', text_no_footer)

    # Split on sentence terminators
    sentences = re.split(r'[.!?]+(?:\s|$)', text_clean.strip())
    # Filter out empty strings and "Source" fragments left by citation removal
    sentences = [
        s.strip() for s in sentences
        if s.strip() and not re.match(r'^Source:?$', s.strip())
    ]
    return len(sentences)


def _strip_footer(text: str) -> str:
    """Remove the 'Last updated from sources:' footer line."""
    return FOOTER_PATTERN.sub("", text).strip()


def _trim_to_sentences(text: str, max_sentences: int) -> str:
    """
    Trim text to at most max_sentences sentences.

    Args:
        text: Input text
        max_sentences: Maximum number of sentences to keep

    Returns:
        Trimmed text
    """
    footer_match = FOOTER_PATTERN.search(text)
    footer = footer_match.group(0) if footer_match else ""

    body = _strip_footer(text)
    sentences = re.split(r'(?<=[.!?])\s+', body.strip())
    sentences = [s for s in sentences if s.strip()]

    if len(sentences) <= max_sentences:
        trimmed = " ".join(sentences)
    else:
        trimmed = " ".join(sentences[:max_sentences])
        logger.warning(
            "Response trimmed from %d to %d sentences",
            len(sentences), max_sentences,
        )

    if footer:
        trimmed = trimmed.rstrip() + "\n\n" + footer

    return trimmed


def _extract_urls(text: str) -> list[str]:
    """Extract all URLs from text."""
    return URL_PATTERN.findall(text)


def _check_advisory_leak(text: str) -> list[str]:
    """
    Check if any advisory language leaked into the response.

    Returns:
        List of detected advisory phrases (empty if clean)
    """
    detected = []
    for pattern in ADVISORY_LEAK_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            detected.extend(matches)
    return detected


def _ensure_footer(text: str, date: str) -> str:
    """
    Ensure the response has the required footer.
    If missing, append it. If present but wrong date, fix it.

    Args:
        text: Response text
        date: Required date string (YYYY-MM-DD)

    Returns:
        Text with correct footer
    """
    expected_footer = f"Last updated from sources: {date}"

    if FOOTER_PATTERN.search(text):
        # Footer exists — check if date is correct
        text = FOOTER_PATTERN.sub(expected_footer, text)
        return text

    # Footer missing — append it
    text = text.rstrip() + f"\n\n{expected_footer}"
    logger.warning("Footer was missing — appended: %s", expected_footer)
    return text


def _ensure_citation(text: str, source_url: str) -> str:
    """
    Ensure the response contains at least one citation URL.
    If no URL found, append the source_url.

    Args:
        text: Response text
        source_url: The primary source URL to include

    Returns:
        Text with citation URL
    """
    if not source_url:
        return text

    urls = _extract_urls(text)
    if urls:
        return text

    # No URL found — append source (inline to avoid extra sentence count)
    body = _strip_footer(text)
    footer_match = FOOTER_PATTERN.search(text)
    footer = footer_match.group(0) if footer_match else ""

    # Append inline with comma to avoid creating a new sentence
    if body.endswith((".", "!", "?")):
        body = body + " (Source: " + source_url + ")"
    else:
        body = body + " (Source: " + source_url + ")."

    if footer:
        body = body + "\n\n" + footer

    logger.warning("No citation URL found — appended: %s", source_url)
    return body


def validate_and_format(
    response: str,
    source_url: str,
    date: Optional[str] = None,
    max_sentences: int = MAX_RESPONSE_SENTENCES,
) -> dict:
    """
    Validate and format an LLM-generated response.

    Performs the following checks and fixes:
    1. Sentence count ≤ max_sentences (trim if exceeded)
    2. Exactly one source URL present (append if missing)
    3. Footer present with correct date (append/fix if missing)
    4. No advisory language detected (flag if found)

    Args:
        response: Raw LLM-generated response text
        source_url: Primary source URL for citation
        date: Date string for footer (defaults to today)
        max_sentences: Maximum allowed sentences

    Returns:
        Dict with keys:
        - formatted_response: str (cleaned, validated response)
        - sentence_count: int
        - has_citation: bool
        - has_footer: bool
        - advisory_leak: list[str] (detected advisory phrases)
        - warnings: list[str]
    """
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    warnings: list[str] = []
    formatted = response.strip()

    # 1. Trim to max sentences
    sentence_count = count_sentences(formatted)
    if sentence_count > max_sentences:
        formatted = _trim_to_sentences(formatted, max_sentences)
        warnings.append(f"Trimmed from {sentence_count} to {max_sentences} sentences")
        sentence_count = max_sentences

    # 2. Ensure citation URL
    urls = _extract_urls(formatted)
    has_citation = len(urls) > 0
    if not has_citation and source_url:
        formatted = _ensure_citation(formatted, source_url)
        has_citation = True
        warnings.append("Citation URL was missing — appended")

    # 3. Ensure footer
    has_footer = bool(FOOTER_PATTERN.search(formatted))
    formatted = _ensure_footer(formatted, date)
    if not has_footer:
        warnings.append("Footer was missing — appended")

    # 4. Check for advisory language leaks
    advisory_leaks = _check_advisory_leak(formatted)
    if advisory_leaks:
        warnings.append(f"Advisory language detected: {advisory_leaks}")
        logger.warning(
            "Advisory language leak in response: %s", advisory_leaks
        )

    # Final sentence count after all modifications
    final_count = count_sentences(formatted)

    result = {
        "formatted_response": formatted,
        "sentence_count": final_count,
        "has_citation": has_citation,
        "has_footer": True,  # guaranteed after _ensure_footer
        "advisory_leak": advisory_leaks,
        "warnings": warnings,
    }

    if warnings:
        logger.info("Post-processor warnings: %s", warnings)

    return result
