"""
Groww-specific scheme page parser.
Extracts structured fund data (NAV, AUM, Exit Load, SIP, Fund Manager, etc.)
from Groww's HTML page structure and produces clean text optimized for RAG chunking.

Groww pages render key metrics in cards/tables which the generic scraper
flattens into unstructured text. This parser extracts those fields explicitly.
"""

import logging
import re
from typing import Optional

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# --- Key metric labels found on Groww scheme pages ---
# These are the labels used in Groww's key metrics strip and overview table.
# We match them case-insensitively to extract corresponding values.

NAV_LABELS = [
    "nav", "net asset value", "nav as on",
]
AUM_LABELS = [
    "aum", "fund size", "assets under management", "corpus", "scheme size",
]
EXIT_LOAD_LABELS = [
    "exit load", "exit penalty",
]
EXPENSE_RATIO_LABELS = [
    "expense ratio", "total expense", "ter",
]
SIP_MIN_LABELS = [
    "min sip", "minimum sip", "sip minimum", "min. sip",
]
LUMPSUM_MIN_LABELS = [
    "min lumpsum", "minimum lumpsum", "min. lumpsum", "min investment",
    "minimum investment",
]
FUND_MANAGER_LABELS = [
    "fund manager", "fund mgr", "portfolio manager",
]
BENCHMARK_LABELS = [
    "benchmark", "benchmark index",
]
CATEGORY_LABELS = [
    "category", "fund category", "scheme category",
]
AMC_LABELS = [
    "amc", "asset management company", "fund house",
]
LAUNCH_LABELS = [
    "launch date", "inception date", "scheme launch",
]
LOCK_IN_LABELS = [
    "lock in", "lock-in period", "lock in period",
]


# Navigation/boilerplate keywords that should NOT appear in metric values
_NAVIGATION_KEYWORDS = [
    "invest in", "stocks", "mutual funds", "ipo", "calculator", "nfo",
    "track all", "compare", "portfolio", "watchlist", "demat", "sip calculator",
    "download", "help & support", "terms and conditions", "privacy policy",
    "about us", "careers", "blog", "investor relations",
]


def _is_valid_metric_value(value: str, field_name: str = "") -> bool:
    """
    Validate that a metric value looks like actual data, not navigation boilerplate.

    Returns False if the value:
    - Is too long (>150 chars)
    - Contains navigation keywords
    - Looks like a menu item or link list
    """
    if not value:
        return False

    val_lower = value.lower()

    # Reject overly long values (likely navigation dumps)
    if len(value) > 150:
        return False

    # Reject values with navigation keywords
    for kw in _NAVIGATION_KEYWORDS:
        if kw in val_lower:
            return False

    # Reject values that are mostly words (no numbers for numeric fields)
    numeric_fields = {"nav", "aum", "expense_ratio", "sip_minimum", "lumpsum_minimum", "exit_load"}
    if field_name in numeric_fields:
        # Should contain at least one digit
        if not any(c.isdigit() for c in value):
            return False

    # Reject values with too many commas (likely link lists)
    if value.count(",") > 5:
        return False

    return True


def _normalize_label(text: str) -> str:
    """Normalize text for label matching: lowercase, strip, collapse spaces."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _find_value_for_label(soup: BeautifulSoup, labels: list[str]) -> Optional[str]:
    """
    Search the page for a value associated with any of the given labels.
    Groww renders metrics as label-value pairs in various layouts:
    - dt/dd pairs
    - Adjacent spans/divs
    - Table rows (td/th)
    - Key-value cards
    """
    for label in labels:
        normalized_label = _normalize_label(label)

        # Strategy 1: Look for elements whose text matches the label
        for element in soup.find_all(string=re.compile(re.escape(normalized_label), re.IGNORECASE)):
            parent = element.find_parent()
            if not parent:
                continue

            # Check next sibling element
            next_el = parent.find_next_sibling()
            if next_el:
                val = next_el.get_text(strip=True)
                if val and _normalize_label(val) != normalized_label:
                    return val

            # Check next element in DOM (not sibling)
            next_el = parent.find_next()
            if next_el and isinstance(next_el, Tag):
                val = next_el.get_text(strip=True)
                if val and _normalize_label(val) != normalized_label and len(val) < 200:
                    return val

            # Check parent's parent for adjacent value
            grandparent = parent.find_parent()
            if grandparent:
                # Look for value in a sibling of parent's container
                gp_next = grandparent.find_next_sibling()
                if gp_next:
                    val = gp_next.get_text(strip=True)
                    if val and _normalize_label(val) != normalized_label:
                        return val

        # Strategy 2: Look in table rows (th/td pairs)
        for th in soup.find_all(["th", "td"]):
            th_text = _normalize_label(th.get_text())
            if normalized_label in th_text:
                td = th.find_next_sibling(["td", "th"])
                if td:
                    val = td.get_text(strip=True)
                    if val:
                        return val
                # Also check next row
                tr = th.find_parent("tr")
                if tr:
                    next_tr = tr.find_next_sibling("tr")
                    if next_tr:
                        val = next_tr.get_text(strip=True)
                        if val:
                            return val

        # Strategy 3: Look for dt/dd pairs
        for dt in soup.find_all("dt"):
            dt_text = _normalize_label(dt.get_text())
            if normalized_label in dt_text:
                dd = dt.find_next_sibling("dd")
                if dd:
                    val = dd.get_text(strip=True)
                    if val:
                        return val

    return None


def _extract_returns_table(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract performance/returns table data from Groww pages.
    Returns are typically shown as 1M, 3M, 6M, 1Y, 3Y, 5Y columns.
    """
    # Look for return period headers
    return_periods = re.compile(r"^\s*(1\s*[MmWw]|3\s*[Mm]|6\s*[Mm]|1\s*[Yy]|3\s*[Yy]|5\s*[Yy]|Max)\s*$")

    for element in soup.find_all(string=return_periods):
        # Found a return period header - look for the parent table/container
        container = element.find_parent(["table", "div", "section"])
        if container:
            # Get all text from this container
            text = container.get_text(separator=" | ", strip=True)
            if len(text) > 10 and "%" in text:
                return text

    return None


def _extract_faq_section(soup: BeautifulSoup) -> list[dict]:
    """
    Extract FAQ section from the page.
    Groww renders FAQs as expandable accordion items.
    """
    faqs = []

    # Look for FAQ section heading
    for heading in soup.find_all(string=re.compile(r"(FAQ|Frequently Asked)", re.IGNORECASE)):
        container = heading.find_parent(["section", "div"])
        if not container:
            continue

        # Look for question-answer pairs
        # Common patterns: button/h3 for question, div/p for answer
        questions = container.find_all(["h3", "h4", "button", "summary"])
        for q in questions:
            question = q.get_text(strip=True)
            if not question or len(question) < 5:
                continue

            # Find the answer (next sibling or next element)
            answer_el = q.find_next_sibling()
            if not answer_el:
                answer_el = q.find_next(["p", "div", "span"])

            answer = answer_el.get_text(strip=True) if answer_el else ""
            if question and answer:
                faqs.append({"question": question, "answer": answer})

    return faqs


def parse_groww_scheme(html: str, source_url: str) -> dict:
    """
    Parse a Groww mutual fund scheme page and extract structured data.

    Args:
        html: Raw HTML content of the page
        source_url: URL of the page

    Returns:
        dict with:
            - structured_data: dict of key-value pairs (NAV, AUM, etc.)
            - text: Combined clean text optimized for RAG chunking
            - faqs: list of FAQ question-answer pairs
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "noscript", "svg"]):
        tag.decompose()

    structured = {}

    # Extract key fund metrics
    metric_fields = {
        "nav": NAV_LABELS,
        "aum": AUM_LABELS,
        "exit_load": EXIT_LOAD_LABELS,
        "expense_ratio": EXPENSE_RATIO_LABELS,
        "sip_minimum": SIP_MIN_LABELS,
        "lumpsum_minimum": LUMPSUM_MIN_LABELS,
        "fund_manager": FUND_MANAGER_LABELS,
        "benchmark": BENCHMARK_LABELS,
        "category": CATEGORY_LABELS,
        "amc": AMC_LABELS,
        "launch_date": LAUNCH_LABELS,
        "lock_in": LOCK_IN_LABELS,
    }

    for field_name, labels in metric_fields.items():
        value = _find_value_for_label(soup, labels)
        if value and _is_valid_metric_value(value, field_name):
            structured[field_name] = value

    # Extract returns table
    returns = _extract_returns_table(soup)
    if returns:
        structured["returns"] = returns

    # Extract FAQs
    faqs = _extract_faq_section(soup)

    # Build structured text block for RAG
    text_parts = []

    # Section: Key Metrics
    metrics_lines = []
    field_display = {
        "nav": "NAV",
        "aum": "Fund Size (AUM)",
        "exit_load": "Exit Load",
        "expense_ratio": "Expense Ratio",
        "sip_minimum": "Minimum SIP Amount",
        "lumpsum_minimum": "Minimum Lumpsum Investment",
        "fund_manager": "Fund Manager",
        "benchmark": "Benchmark Index",
        "category": "Category",
        "amc": "AMC (Fund House)",
        "launch_date": "Launch / Inception Date",
        "lock_in": "Lock-in Period",
    }

    for field, display_name in field_display.items():
        if field in structured:
            metrics_lines.append(f"{display_name}: {structured[field]}")

    if metrics_lines:
        text_parts.append("## Key Fund Metrics\n")
        text_parts.append("\n".join(metrics_lines))

    # Section: Returns
    if "returns" in structured:
        text_parts.append("\n## Performance Returns\n")
        text_parts.append(structured["returns"])

    # Section: FAQs
    if faqs:
        text_parts.append("\n## Frequently Asked Questions\n")
        for faq in faqs:
            text_parts.append(f"Q: {faq['question']}")
            text_parts.append(f"A: {faq['answer']}\n")

    # Combine structured text
    structured_text = "\n\n".join(text_parts)

    logger.info(
        "Parsed Groww scheme page: %d metrics, %d FAQs from %s",
        len([v for k, v in structured.items() if k != "returns"]),
        len(faqs),
        source_url,
    )

    return {
        "structured_data": structured,
        "text": structured_text,
        "faqs": faqs,
    }


def build_scheme_facts_block(structured_data: dict, scheme_name: str) -> str:
    """
    Build a dense facts block from structured data that can be prepended
    to chunked text for better retrieval context.

    This ensures every chunk carries the essential fund facts.
    """
    lines = [f"Scheme: {scheme_name}"]

    field_display = {
        "nav": "NAV",
        "aum": "Fund Size (AUM)",
        "exit_load": "Exit Load",
        "expense_ratio": "Expense Ratio",
        "sip_minimum": "Min SIP",
        "lumpsum_minimum": "Min Lumpsum",
        "fund_manager": "Fund Manager",
        "benchmark": "Benchmark",
        "category": "Category",
        "amc": "AMC",
        "launch_date": "Launch Date",
        "lock_in": "Lock-in",
    }

    for field, display in field_display.items():
        if field in structured_data:
            lines.append(f"{display}: {structured_data[field]}")

    if "returns" in structured_data:
        lines.append(f"Returns: {structured_data['returns']}")

    return " | ".join(lines)


def enrich_chunk_with_facts(chunk: dict, facts_block: str) -> dict:
    """
    Enrich a chunk's text with the scheme facts block.
    Only prepends if the chunk doesn't already contain key metrics.
    """
    text = chunk.get("text", "")

    # Check if chunk already has key metrics
    has_metrics = any(
        kw in text.lower()
        for kw in ["nav", "fund size", "aum", "exit load", "sip", "fund manager"]
    )

    if not has_metrics and facts_block:
        chunk["text"] = f"{facts_block}\n\n{text}"
        chunk["metadata"]["has_facts_block"] = True

    return chunk
