"""
Indexer module — loads processed chunks from disk, embeds them,
and stores them in ChromaDB for retrieval.
Designed to run as part of the ingestion pipeline or standalone.
"""

import glob
import json
import logging
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR
from src.processing.embeddings import embed_batch, get_dimensions
from src.processing.vectorstore import (
    add_vectors,
    delete_collection,
    get_collection,
    get_collection_count,
)

logger = logging.getLogger(__name__)


def load_latest_chunks() -> list[dict]:
    """
    Load the most recent processed chunks file from data/processed/.
    Files are named: chunks_YYYYMMDD_HHMMSS.json

    Returns:
        List of chunk dicts with 'chunk_id', 'text', and 'metadata'
    """
    pattern = str(PROCESSED_DATA_DIR / "chunks_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        logger.error("No chunk files found in %s", PROCESSED_DATA_DIR)
        return []

    latest = files[-1]
    logger.info("Loading chunks from: %s", latest)

    with open(latest, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info("Loaded %d chunks from %s", len(chunks), Path(latest).name)
    return chunks


def index_chunks(chunks: list[dict], reset_collection: bool = True) -> dict:
    """
    Embed and store chunks into ChromaDB.

    Args:
        chunks: List of chunk dicts (from chunker or load_latest_chunks)
        reset_collection: If True, wipe existing data before indexing

    Returns:
        Summary dict with counts and status
    """
    summary = {
        "chunks_loaded": len(chunks),
        "chunks_indexed": 0,
        "embedding_dimensions": 0,
        "collection_count_after": 0,
        "status": "success",
    }

    if not chunks:
        logger.error("No chunks to index")
        summary["status"] = "failed"
        return summary

    # Reset collection if requested (default for pipeline runs)
    if reset_collection:
        logger.info("Resetting collection before indexing")
        delete_collection()

    collection = get_collection()

    # Extract data for embedding
    ids = [chunk["chunk_id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # Generate embeddings
    logger.info("Generating embeddings for %d chunks...", len(texts))
    embeddings = embed_batch(texts)
    summary["embedding_dimensions"] = len(embeddings[0]) if embeddings else 0

    # Store in ChromaDB
    logger.info("Storing %d vectors in ChromaDB...", len(ids))
    add_vectors(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    summary["chunks_indexed"] = len(ids)

    # Verify
    count = get_collection_count()
    summary["collection_count_after"] = count

    if count >= len(ids):
        logger.info("Indexing verified: collection has %d vectors", count)
    else:
        logger.warning(
            "Index count mismatch: expected ≥%d, got %d", len(ids), count
        )
        summary["status"] = "warning"

    return summary


def run_indexer(reset_collection: bool = True) -> dict:
    """
    Full indexer pipeline: load chunks → embed → store.

    Args:
        reset_collection: Wipe existing vectors before indexing

    Returns:
        Summary dict
    """
    logger.info("=" * 50)
    logger.info("Starting indexer")
    logger.info("=" * 50)

    # Load chunks
    chunks = load_latest_chunks()
    if not chunks:
        return {"status": "failed", "error": "No chunks found"}

    # Index
    summary = index_chunks(chunks, reset_collection=reset_collection)

    logger.info("Indexer complete: %d chunks indexed (dim=%d)",
                summary["chunks_indexed"], summary["embedding_dimensions"])
    logger.info("=" * 50)

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    result = run_indexer()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] != "failed" else 1)
