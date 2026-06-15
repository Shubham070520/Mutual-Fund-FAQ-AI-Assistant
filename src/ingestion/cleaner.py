"""
Text cleaner module for normalizing and cleaning scraped content.
Removes noise, normalizes whitespace/unicode, and preserves section headings.
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Common boilerplate patterns to remove
BOILERPLATE_PATTERNS = [
    r"(?i)cookie\s*(policy|notice|consent|settings).*",
    r"(?i)(sign\s*in|log\s*in|register|subscribe)\s*(now|here|today)?",
    r"(?i)(share|tweet|post)\s*(on|to|via)\s*(facebook|twitter|linkedin|whatsapp).*",
    r"(?i)download\s*(our|the)\s*(app|mobile\s*app).*",
    r"(?i)follow\s*us\s*(on|:).*",
    r"(?i)copyright\s*©.*",
    r"(?i)all\s*rights\s*reserved.*",
    r"(?i)terms\s*(and|&)\s*conditions.*",
    r"(?i)privacy\s*policy.*",
    r"(?i)disclaimer.*?(?=\n\n|$)",
]

# Legal disclaimer blocks commonly found in fund documents
LEGAL_BOILERPLATE = [
    r"(?i)mutual\s*fund\s*investments\s*are\s*subject\s*to\s*market\s*risks.*",
    r"(?i)past\s*performance\s*(is|does)\s*not\s*(guarantee|indicate).*",
    r"(?i)please\s*read\s*(the|all)\s*(scheme|fund)\s*(related|information)\s*documents.*",
    r"(?i)this\s*(document|material|information)\s*(is|does\s*not)\s*(for\s*information|constitute).*",
]


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to their canonical form."""
    # NFKC normalization: compatible decomposition + canonical composition
    text = unicodedata.normalize("NFKC", text)
    # Replace common special chars
    replacements = {
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u2026": "...",  # ellipsis
        "\u00a0": " ",   # non-breaking space
        "\u200b": "",    # zero-width space
        "\ufeff": "",    # BOM
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph structure."""
    # Replace tabs with spaces
    text = text.replace("\t", " ")
    # Collapse multiple spaces on same line
    text = re.sub(r" +", " ", text)
    # Collapse 3+ newlines into double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def remove_boilerplate(text: str) -> str:
    """Remove common web boilerplate and legal disclaimer text."""
    for pattern in BOILERPLATE_PATTERNS + LEGAL_BOILERPLATE:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    return text


def remove_urls_and_emails(text: str) -> str:
    """Remove URLs and email addresses from text."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(r"[\w.-]+@[\w.-]+\.\w+", "", text)
    return text


def clean_section_headings(text: str) -> str:
    """
    Normalize section headings for better chunking.
    Ensures headings are on their own line followed by a newline.
    Preserves structured fund data lines (Key Fund Metrics, Performance, FAQs).
    """
    # Pattern: line that is ALL CAPS or Title Case and short (< 80 chars)
    lines = text.split("\n")
    result = []

    # Lines that are part of structured fund data - don't convert to headings
    structured_prefixes = (
        "NAV:", "Fund Size", "Exit Load:", "Expense Ratio:",
        "Minimum SIP", "Min SIP", "Minimum Lumpsum", "Min Lumpsum",
        "Fund Manager:", "Benchmark", "Category:", "AMC",
        "Launch", "Lock-in", "Returns:", "Scheme:",
        "Q:", "A:",
    )

    for line in lines:
        stripped = line.strip()

        # Skip structured data lines - preserve them as-is
        if any(stripped.startswith(prefix) for prefix in structured_prefixes):
            result.append(line)
            continue

        # Detect likely headings: short, no period at end, mostly uppercase or title case
        if (
            stripped
            and len(stripped) < 80
            and not stripped.endswith(".")
            and (stripped.isupper() or stripped.istitle())
            and not stripped.startswith("-")
            and not stripped.startswith("*")
        ):
            result.append(f"\n## {stripped}\n")
        else:
            result.append(line)
    return "\n".join(result)


def clean_text(text: str, preserve_headings: bool = True) -> str:
    """
    Full cleaning pipeline for scraped content.

    Steps:
        1. Normalize unicode characters
        2. Remove boilerplate text
        3. Clean section headings (optional)
        4. Normalize whitespace

    Args:
        text: Raw extracted text
        preserve_headings: Whether to detect and format section headings

    Returns:
        Cleaned text
    """
    if not text or not text.strip():
        logger.warning("Empty text received for cleaning")
        return ""

    logger.debug("Cleaning text (%d chars)", len(text))

    text = normalize_unicode(text)
    text = remove_boilerplate(text)

    if preserve_headings:
        text = clean_section_headings(text)

    text = normalize_whitespace(text)

    logger.debug("Cleaned text (%d chars)", len(text))
    return text
