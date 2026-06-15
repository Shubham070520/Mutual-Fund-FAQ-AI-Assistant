"""
Ingestion pipeline orchestrator.
Coordinates scraping, cleaning, chunking, and saving to disk.
Designed to run both locally and via GitHub Actions scheduler.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path for module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import CONFIG_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.ingestion.scraper import scrape_url
from src.ingestion.cleaner import clean_text
from src.ingestion.groww_parser import build_scheme_facts_block, enrich_chunk_with_facts
from src.processing.chunker import chunk_documents

logger = logging.getLogger(__name__)


def load_url_corpus() -> list[dict]:
    """Load the URL corpus from config/urls.json."""
    urls_path = CONFIG_DIR / "urls.json"
    if not urls_path.exists():
        logger.error("URL corpus not found at %s", urls_path)
        return []

    with open(urls_path, "r", encoding="utf-8") as f:
        urls = json.load(f)

    logger.info("Loaded %d URLs from corpus", len(urls))
    return urls


def save_raw_documents(documents: list[dict]) -> None:
    """Save raw scraped documents to data/raw/ as JSON."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    output_path = RAW_DATA_DIR / f"raw_documents_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d raw documents to %s", len(documents), output_path)


def save_processed_chunks(chunks: list[dict]) -> None:
    """Save processed chunks to data/processed/ as JSON."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    output_path = PROCESSED_DATA_DIR / f"chunks_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d chunks to %s", len(chunks), output_path)


def run_pipeline() -> dict:
    """
    Execute the full ingestion pipeline.

    Steps:
        1. Load URL corpus from config/urls.json
        2. Scrape each URL (HTML or PDF)
        3. Clean extracted text
        4. Chunk documents into overlapping segments
        5. Save raw documents and processed chunks to disk

    Returns:
        Summary dict with counts and status
    """
    logger.info("=" * 60)
    logger.info("Starting ingestion pipeline")
    logger.info("=" * 60)

    start_time = datetime.now(timezone.utc)
    summary = {
        "start_time": start_time.isoformat(),
        "urls_total": 0,
        "urls_scraped": 0,
        "urls_failed": 0,
        "documents_cleaned": 0,
        "chunks_produced": 0,
        "status": "success",
        "errors": [],
    }

    # Step 1: Load URLs
    url_entries = load_url_corpus()
    summary["urls_total"] = len(url_entries)

    if not url_entries:
        logger.error("No URLs to process. Aborting pipeline.")
        summary["status"] = "failed"
        summary["errors"].append("Empty URL corpus")
        return summary

    # Step 2: Scrape each URL
    logger.info("Phase: Scraping %d URLs...", len(url_entries))
    raw_documents = []

    for i, entry in enumerate(url_entries, 1):
        url = entry.get("url", "")
        logger.info("[%d/%d] Scraping: %s", i, len(url_entries), url)

        try:
            result = scrape_url(entry)

            if result.get("metadata", {}).get("error"):
                logger.warning("Scrape failed for %s: %s", url, result["metadata"]["error"])
                summary["urls_failed"] += 1
                summary["errors"].append(f"{url}: {result['metadata']['error']}")
                continue

            if not result.get("text", "").strip():
                logger.warning("Empty content from %s", url)
                summary["urls_failed"] += 1
                summary["errors"].append(f"{url}: empty_content")
                continue

            raw_documents.append(result)
            summary["urls_scraped"] += 1

        except Exception as e:
            logger.error("Unexpected error scraping %s: %s", url, e)
            summary["urls_failed"] += 1
            summary["errors"].append(f"{url}: {str(e)}")

    logger.info(
        "Scraping complete: %d/%d succeeded",
        summary["urls_scraped"], summary["urls_total"],
    )

    # Save raw documents
    if raw_documents:
        save_raw_documents(raw_documents)

    # Step 3: Clean extracted text
    logger.info("Phase: Cleaning %d documents...", len(raw_documents))
    cleaned_documents = []

    for doc in raw_documents:
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})

        cleaned = clean_text(text, preserve_headings=True)
        if cleaned:
            cleaned_documents.append({"text": cleaned, "metadata": metadata})
            summary["documents_cleaned"] += 1
        else:
            logger.warning(
                "Document empty after cleaning: %s",
                metadata.get("source_url", "unknown"),
            )

    logger.info("Cleaning complete: %d documents retained", len(cleaned_documents))

    # Step 4: Chunk documents
    logger.info("Phase: Chunking documents...")
    chunks = chunk_documents(cleaned_documents)

    # Step 4b: Enrich chunks with structured fund facts
    enriched_count = 0
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        structured_data = metadata.get("structured_data", {})
        scheme_name = metadata.get("scheme_name", "")

        if structured_data:
            facts_block = build_scheme_facts_block(structured_data, scheme_name)
            enrich_chunk_with_facts(chunk, facts_block)
            enriched_count += 1

    logger.info("Enriched %d/%d chunks with structured fund facts", enriched_count, len(chunks))
    summary["chunks_produced"] = len(chunks)

    # Step 5: Save processed chunks
    if chunks:
        save_processed_chunks(chunks)

    # Final summary
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    summary["end_time"] = end_time.isoformat()
    summary["duration_seconds"] = round(duration, 2)

    if summary["urls_failed"] == summary["urls_total"]:
        summary["status"] = "failed"
    elif summary["urls_failed"] > 0:
        summary["status"] = "partial_success"

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("  URLs scraped:   %d/%d", summary["urls_scraped"], summary["urls_total"])
    logger.info("  Docs cleaned:   %d", summary["documents_cleaned"])
    logger.info("  Chunks produced: %d", summary["chunks_produced"])
    logger.info("  Duration:        %.1fs", duration)
    logger.info("  Status:          %s", summary["status"])
    logger.info("=" * 60)

    # Save pipeline summary
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = PROCESSED_DATA_DIR / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    result = run_pipeline()
    sys.exit(0 if result["status"] != "failed" else 1)
