"""
ChromaDB vector store wrapper.
Manages persistent vector storage for the mutual fund FAQ corpus.
"""

import logging
from typing import Optional

import chromadb
import chromadb.errors
from chromadb.api.models.Collection import Collection

from config.settings import (
    VECTORDB_COLLECTION_NAME,
    VECTORDB_DISTANCE_METRIC,
    VECTORDB_DIR,
)

logger = logging.getLogger(__name__)

# Module-level singleton
_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[Collection] = None


def get_client() -> chromadb.PersistentClient:
    """Get or create the persistent ChromaDB client."""
    global _client

    if _client is not None:
        return _client

    logger.info("Initializing ChromaDB client at: %s", VECTORDB_DIR)
    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    logger.info("ChromaDB client initialized")
    return _client


def get_collection(
    name: str = VECTORDB_COLLECTION_NAME,
    reset: bool = False,
) -> Collection:
    """
    Get or create the vector collection.

    Args:
        name: Collection name (default: mf_faq_corpus)
        reset: If True, delete and recreate the collection

    Returns:
        ChromaDB Collection instance
    """
    global _collection

    client = get_client()

    if reset:
        logger.warning("Resetting collection '%s' — all existing data will be lost", name)
        try:
            client.delete_collection(name)
        except (ValueError, chromadb.errors.NotFoundError):
            pass  # Collection doesn't exist yet
        _collection = None

    if _collection is not None:
        return _collection

    logger.info("Getting or creating collection: %s (metric=%s)", name, VECTORDB_DISTANCE_METRIC)
    _collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": VECTORDB_DISTANCE_METRIC},
    )

    count = _collection.count()
    logger.info("Collection ready: %d existing vectors", count)
    return _collection


def add_vectors(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """
    Add vectors to the collection in batches.

    ChromaDB has a batch size limit, so we split large additions
    into chunks of 500.

    Args:
        ids: Unique IDs for each vector
        embeddings: Embedding vectors (same length as ids)
        documents: Original text for each vector
        metadatas: Metadata dicts for each vector
    """
    collection = get_collection()
    batch_size = 500
    total = len(ids)

    # Sanitize metadata values — ChromaDB requires strings/ints/floats
    clean_metadatas = []
    for meta in metadatas:
        clean = {}
        for k, v in meta.items():
            if v is None:
                clean[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        clean_metadatas.append(clean)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        logger.info("Adding vectors %d–%d of %d", start + 1, end, total)

        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=clean_metadatas[start:end],
        )

    logger.info("Successfully added %d vectors to collection", total)


def query_vectors(
    query_embedding: list[float],
    n_results: int = 5,
    where: Optional[dict] = None,
) -> dict:
    """
    Query the collection for similar vectors.

    Args:
        query_embedding: Query vector
        n_results: Number of results to return
        where: Optional metadata filter (e.g., {"scheme_name": "HDFC Mid Cap Fund"})

    Returns:
        Dict with keys: ids, documents, metadatas, distances
    """
    collection = get_collection()

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }

    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    # Flatten results (ChromaDB returns nested lists for batch queries)
    return {
        "ids": results["ids"][0] if results["ids"] else [],
        "documents": results["documents"][0] if results["documents"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        "distances": results["distances"][0] if results["distances"] else [],
    }


def get_collection_count() -> int:
    """Return the number of vectors in the collection."""
    collection = get_collection()
    return collection.count()


def delete_collection(name: str = VECTORDB_COLLECTION_NAME) -> None:
    """Delete a collection entirely."""
    global _collection
    client = get_client()
    try:
        client.delete_collection(name)
        _collection = None
        logger.info("Deleted collection: %s", name)
    except (ValueError, chromadb.errors.NotFoundError):
        logger.warning("Collection '%s' does not exist", name)
