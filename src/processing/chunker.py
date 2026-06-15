"""
Text chunker module using LangChain's RecursiveCharacterTextSplitter.
Splits cleaned documents into overlapping chunks while preserving metadata.
"""

import logging
import uuid
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS

logger = logging.getLogger(__name__)


def _get_section_heading(text: str, position: int) -> Optional[str]:
    """
    Find the nearest section heading before a given position in text.
    Looks for markdown-style headings (## Heading) or ALL-CAPS lines.
    """
    preceding = text[:position]
    lines = preceding.split("\n")

    # Walk backwards to find the closest heading
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("## "):
            return stripped.lstrip("#").strip()
        if stripped and stripped.isupper() and len(stripped) < 80:
            return stripped
    return None


def create_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: list[str] = None,
) -> RecursiveCharacterTextSplitter:
    """Create and return a configured text splitter instance."""
    if separators is None:
        separators = CHUNK_SEPARATORS

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )


def _classify_chunk_content(text: str) -> list[str]:
    """
    Classify what type of content a chunk contains.
    Used for metadata tagging to improve retrieval relevance.
    """
    text_lower = text.lower()
    content_types = []

    # Structured fund metrics
    if any(kw in text_lower for kw in ["nav:", "nav as on", "nav of", "net asset value", "the nav"]):
        content_types.append("nav")
    if any(kw in text_lower for kw in ["exit load", "exit penalty"]):
        content_types.append("exit_load")
    if any(kw in text_lower for kw in ["aum", "fund size", "assets under management", "corpus"]):
        content_types.append("aum")
    if any(kw in text_lower for kw in ["sip", "systematic investment"]):
        content_types.append("sip")
    if any(kw in text_lower for kw in ["fund manager", "portfolio manager", "managed by", "fund mgr"]):
        content_types.append("fund_manager")
    if any(kw in text_lower for kw in ["expense ratio", "ter"]):
        content_types.append("expense_ratio")
    if any(kw in text_lower for kw in ["benchmark", "index"]):
        content_types.append("benchmark")
    if any(kw in text_lower for kw in ["return", "cagr", "performance"]):
        content_types.append("performance")
    if "q:" in text_lower and "a:" in text_lower:
        content_types.append("faq")

    return content_types


def chunk_text(
    text: str,
    metadata: dict,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split text into overlapping chunks, each enriched with metadata.

    Args:
        text: Cleaned text content to chunk
        metadata: Base metadata dict (source_url, scheme_name, document_type, etc.)
        chunk_size: Target size per chunk in characters
        chunk_overlap: Overlap between consecutive chunks in characters

    Returns:
        List of chunk dicts, each containing:
            - chunk_id: unique UUID
            - text: chunk content
            - metadata: enriched metadata with source info + section heading
    """
    if not text or not text.strip():
        logger.warning("Empty text received, skipping chunking")
        return []

    splitter = create_splitter(chunk_size, chunk_overlap)
    raw_chunks = splitter.split_text(text)

    if not raw_chunks:
        logger.warning("No chunks produced from text (%d chars)", len(text))
        return []

    chunks = []
    search_start = 0

    for i, chunk_text in enumerate(raw_chunks):
        # Find the position of this chunk in the original text
        position = text.find(chunk_text[:50], search_start)
        if position == -1:
            position = search_start
        search_start = position + len(chunk_text) // 2

        # Detect section heading near this chunk
        section_heading = _get_section_heading(text, position)

        chunk_id = str(uuid.uuid4())

        # Classify chunk content for metadata tagging
        content_types = _classify_chunk_content(chunk_text)

        chunk_metadata = {
            **metadata,
            "chunk_index": i,
            "chunk_total": len(raw_chunks),
            "section_heading": section_heading or "",
            "char_count": len(chunk_text),
            "content_types": content_types,
            "has_structured_data": len(content_types) > 0,
        }

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "metadata": chunk_metadata,
        })

    logger.info(
        "Chunked document '%s' into %d chunks (size=%d, overlap=%d)",
        metadata.get("source_url", "unknown"),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk a list of scraped documents.

    Args:
        documents: List of dicts with 'text' and 'metadata' keys (from scraper)

    Returns:
        Flat list of all chunks across all documents
    """
    all_chunks = []
    for doc in documents:
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})

        if not text:
            logger.warning(
                "Skipping empty document: %s", metadata.get("source_url", "unknown")
            )
            continue

        doc_chunks = chunk_text(text, metadata)
        all_chunks.extend(doc_chunks)

    logger.info(
        "Total chunks produced: %d from %d documents",
        len(all_chunks), len(documents),
    )
    return all_chunks
