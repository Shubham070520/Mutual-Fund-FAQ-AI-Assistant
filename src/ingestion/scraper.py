"""
Web scraper module for extracting content from HTML pages and PDFs.
Handles Groww scheme pages, AMC pages, AMFI/SEBI guidance, and PDF documents.
"""

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup, Tag

from src.ingestion.groww_parser import parse_groww_scheme, build_scheme_facts_block

logger = logging.getLogger(__name__)

# --- Constants ---
REQUEST_TIMEOUT = 30  # seconds
REQUEST_DELAY = 2  # seconds between requests (rate limiting)
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # exponential backoff multiplier

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _compute_hash(text: str) -> str:
    """Compute SHA-256 hash of content for change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fetch_with_retry(url: str, stream: bool = False) -> Optional[requests.Response]:
    """Fetch a URL with exponential backoff retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
                stream=stream,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            wait = RETRY_BACKOFF ** attempt
            logger.warning(
                "Attempt %d/%d failed for %s: %s. Retrying in %ds...",
                attempt + 1, MAX_RETRIES, url, e, wait,
            )
            time.sleep(wait)
    logger.error("All %d attempts failed for %s", MAX_RETRIES, url)
    return None


def scrape_html(url: str) -> dict:
    """
    Scrape an HTML page and return extracted text with metadata.
    For Groww scheme pages, also extracts structured fund data.

    Returns:
        dict with keys: text, metadata (source_url, document_type, scheme_name, etc.)
    """
    logger.info("Scraping HTML: %s", url)

    response = _fetch_with_retry(url)
    if response is None:
        return {"text": "", "metadata": {"source_url": url, "error": "fetch_failed"}}

    html_content = response.text
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove non-content elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    # Extract page title
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Extract main content
    # Try common content containers first, then fall back to body
    content = ""
    for selector in [
        "main",
        "article",
        '[role="main"]',
        ".content",
        "#content",
        ".page-content",
    ]:
        element = soup.select_one(selector)
        if element:
            content = element.get_text(separator="\n", strip=True)
            break

    if not content:
        body = soup.find("body")
        content = body.get_text(separator="\n", strip=True) if body else ""

    scraped_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    metadata = {
        "source_url": url,
        "document_type": "scheme_page",
        "title": title,
        "scraped_date": scraped_date,
        "content_hash": _compute_hash(content),
        "format": "html",
    }

    # For Groww scheme pages, run structured extraction
    structured_data = {}
    if "groww.in" in url:
        try:
            groww_result = parse_groww_scheme(html_content, url)
            structured_data = groww_result.get("structured_data", {})
            groww_text = groww_result.get("text", "")
            faqs = groww_result.get("faqs", [])

            # Prepend structured text to generic content for richer chunks
            if groww_text:
                content = f"{groww_text}\n\n---\n\n{content}"

            if faqs:
                metadata["faq_count"] = len(faqs)

            logger.info(
                "Groww structured extraction: %d metrics, %d FAQs",
                len(structured_data), len(faqs),
            )
        except Exception as e:
            logger.warning("Groww parser failed for %s: %s", url, e)

    if structured_data:
        metadata["structured_data"] = structured_data

    return {
        "text": content,
        "metadata": metadata,
    }


def extract_pdf(url: str) -> dict:
    """
    Download and extract text from a PDF document.

    Returns:
        dict with keys: text, metadata
    """
    logger.info("Extracting PDF: %s", url)

    response = _fetch_with_retry(url, stream=True)
    if response is None:
        return {"text": "", "metadata": {"source_url": url, "error": "fetch_failed"}}

    try:
        doc = fitz.open(stream=response.content, filetype="pdf")
    except Exception as e:
        logger.error("Failed to open PDF %s: %s", url, e)
        return {"text": "", "metadata": {"source_url": url, "error": f"pdf_open_failed: {e}"}}

    pages_text = []
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()

    full_text = "\n\n".join(pages_text)
    scraped_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return {
        "text": full_text,
        "metadata": {
            "source_url": url,
            "document_type": _guess_pdf_type(url),
            "title": urlparse(url).path.split("/")[-1],
            "scraped_date": scraped_date,
            "content_hash": _compute_hash(full_text),
            "format": "pdf",
            "page_count": len(pages_text),
        },
    }


def _guess_pdf_type(url: str) -> str:
    """Guess the PDF document type from the URL."""
    url_lower = url.lower()
    if "factsheet" in url_lower:
        return "factsheet"
    elif "kim" in url_lower or "key-information" in url_lower:
        return "kim"
    elif "sid" in url_lower or "scheme-information" in url_lower:
        return "sid"
    return "pdf_document"


def scrape_url(url_entry: dict) -> dict:
    """
    Route a URL entry to the appropriate extractor based on format.

    Args:
        url_entry: dict with keys: url, type, scheme, format

    Returns:
        dict with keys: text, metadata (enriched with url_entry fields)
    """
    url = url_entry["url"]
    fmt = url_entry.get("format", "html")

    # Apply rate limiting delay
    time.sleep(REQUEST_DELAY)

    # Route to extractor
    if fmt == "pdf":
        result = extract_pdf(url)
    else:
        result = scrape_html(url)

    # Enrich metadata with url_entry fields
    result["metadata"]["scheme_name"] = url_entry.get("scheme") or ""
    result["metadata"]["document_type"] = url_entry.get("type", result["metadata"].get("document_type", "unknown"))

    return result
