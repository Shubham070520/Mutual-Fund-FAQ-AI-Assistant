"""
PII (Personally Identifiable Information) sanitizer.
Detects and redacts sensitive personal data from user queries
before they reach the retrieval or generation pipeline.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SanitizationResult:
    """Result of PII sanitization."""
    sanitized_query: str
    pii_detected: list[str]
    was_modified: bool


# Indian PII patterns — ordered by specificity to avoid partial matches
PII_PATTERNS: dict[str, re.Pattern] = {
    "pan": re.compile(
        r"[A-Z]{5}[0-9]{4}[A-Z]",
        re.IGNORECASE,
    ),
    "aadhaar": re.compile(
        r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
    ),
    "phone": re.compile(
        r"(?:\+91[\s\-]?)?(?:\()?\d{3,5}(?:\))?[\s\-]?\d{3,4}[\s\-]?\d{3,4}\b"
    ),
    "email": re.compile(
        r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
        re.IGNORECASE,
    ),
}

# Validation helpers — stricter checks for ambiguous patterns
_PAN_STRICT = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
_AADHAAR_STRIP = re.compile(r"[\s\-]")


def _validate_pan(match_text: str) -> bool:
    """PAN must be exactly 10 chars, uppercase alpha pattern."""
    cleaned = match_text.upper().strip()
    return bool(_PAN_STRICT.match(cleaned))


def _validate_aadhaar(match_text: str) -> bool:
    """Aadhaar must be exactly 12 digits after stripping separators."""
    digits = _AADHAAR_STRIP.sub("", match_text)
    return len(digits) == 12 and digits.isdigit()


def _validate_phone(match_text: str) -> bool:
    """Phone number must have at least 10 digits."""
    digits = re.sub(r"\D", "", match_text)
    # Indian phone: 10 digits (or 12 with +91 country code)
    return len(digits) >= 10


def sanitize(query: str) -> str:
    """
    Strip PII patterns from user input.

    Replaces detected PII with [REDACTED] to prevent sensitive data
    from reaching the retrieval or generation layers.

    Args:
        query: Raw user input

    Returns:
        Sanitized query with PII replaced by [REDACTED]
    """
    result = sanitize_detailed(query)
    return result.sanitized_query


def sanitize_detailed(query: str) -> SanitizationResult:
    """
    Sanitize PII from user input with detailed detection report.

    Detection pipeline:
    1. PAN card number (5 alpha + 4 digits + 1 alpha)
    2. Aadhaar number (12 digits, possibly separated)
    3. Phone number (Indian format, with or without +91)
    4. Email address

    Each pattern has a validation step to reduce false positives.

    Args:
        query: Raw user input

    Returns:
        SanitizationResult with sanitized text, detected PII types,
        and whether the query was modified
    """
    if not query or not query.strip():
        return SanitizationResult(
            sanitized_query=query,
            pii_detected=[],
            was_modified=False,
        )

    sanitized = query
    detected: list[str] = []

    # PAN detection
    for match in PII_PATTERNS["pan"].finditer(sanitized):
        if _validate_pan(match.group()):
            detected.append("pan")
            sanitized = sanitized[:match.start()] + "[REDACTED]" + sanitized[match.end():]
            logger.info("PII detected: PAN card number")
            break  # one redaction is enough

    # Aadhaar detection
    for match in PII_PATTERNS["aadhaar"].finditer(sanitized):
        if _validate_aadhaar(match.group()):
            detected.append("aadhaar")
            sanitized = sanitized[:match.start()] + "[REDACTED]" + sanitized[match.end():]
            logger.info("PII detected: Aadhaar number")
            break

    # Phone detection
    for match in PII_PATTERNS["phone"].finditer(sanitized):
        if _validate_phone(match.group()):
            detected.append("phone")
            sanitized = sanitized[:match.start()] + "[REDACTED]" + sanitized[match.end():]
            logger.info("PII detected: Phone number")
            break

    # Email detection
    for match in PII_PATTERNS["email"].finditer(sanitized):
        detected.append("email")
        sanitized = sanitized[:match.start()] + "[REDACTED]" + sanitized[match.end():]
        logger.info("PII detected: Email address")
        break

    was_modified = sanitized != query

    if detected:
        logger.warning(
            "PII sanitization: detected %s in query, redacted %d field(s)",
            detected, len(detected),
        )

    return SanitizationResult(
        sanitized_query=sanitized,
        pii_detected=detected,
        was_modified=was_modified,
    )
