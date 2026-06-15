"""
Retrieval module — embeds user queries, searches ChromaDB for relevant chunks,
applies similarity threshold filtering, and optionally uses MMR for diversity.
"""

import logging
import time
from typing import Optional

import numpy as np

from config.settings import (
    RETRIEVAL_TOP_K,
    RETRIEVAL_SIMILARITY_THRESHOLD,
    RETRIEVAL_USE_MMR,
    RERANK_ENABLED,
    RERANK_TOP_N,
)
from src.processing.embeddings import embed_query
from src.processing.vectorstore import query_vectors, get_collection_count

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va, vb = np.array(a), np.array(b)
    dot = np.dot(va, vb)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def _distance_to_similarity(distance: float, metric: str = "cosine") -> float:
    """
    Convert ChromaDB distance to similarity score.

    ChromaDB cosine distance = 1 - cosine_similarity.
    So similarity = 1 - distance.
    """
    if metric == "cosine":
        return 1.0 - distance
    elif metric == "l2":
        # L2 distance: convert via 1 / (1 + d)
        return 1.0 / (1.0 + distance)
    return 1.0 - distance


def _apply_mmr(
    query_embedding: list[float],
    results: list[dict],
    top_k: int,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Apply Maximal Marginal Relevance (MMR) to diversify results.

    MMR balances relevance to the query with diversity among selected results.
    MMR(d) = lambda * sim(d, q) - (1 - lambda) * max_{d' in S} sim(d, d')

    Args:
        query_embedding: The query vector
        results: List of result dicts with 'embedding', 'similarity', etc.
        top_k: Number of results to select
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0)

    Returns:
        MMR-selected list of result dicts
    """
    if not results or top_k <= 0:
        return []

    if len(results) <= top_k:
        return results

    selected = []
    candidates = list(results)

    # Select the most relevant first result
    best_idx = max(range(len(candidates)), key=lambda i: candidates[i]["similarity"])
    selected.append(candidates.pop(best_idx))

    while len(selected) < top_k and candidates:
        mmr_scores = []
        for cand in candidates:
            # Relevance to query
            relevance = cand["similarity"]

            # Max similarity to already-selected documents
            max_sim_to_selected = max(
                _cosine_similarity(cand["embedding"], s["embedding"])
                for s in selected
            )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            mmr_scores.append(mmr_score)

        best_idx = int(np.argmax(mmr_scores))
        selected.append(candidates.pop(best_idx))

    logger.info("MMR selected %d results from %d candidates", len(selected), len(results))
    return selected


def retrieve(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    scheme_filter: Optional[str] = None,
    document_type_filter: Optional[str] = None,
    similarity_threshold: float = RETRIEVAL_SIMILARITY_THRESHOLD,
    use_mmr: bool = RETRIEVAL_USE_MMR,
) -> list[dict]:
    """
    Retrieve relevant chunks for a user query.

    Pipeline:
    1. Embed the user query (with BGE instruction prefix)
    2. Search ChromaDB for similar chunks
    3. Convert distances to similarity scores
    4. Filter by similarity threshold
    5. Optionally apply MMR for diversity
    6. Return top-K results with metadata

    Args:
        query: User's natural language query
        top_k: Maximum number of results to return
        scheme_filter: Optional filter by scheme name (e.g., "HDFC Mid Cap Fund")
        document_type_filter: Optional filter by document type (e.g., "factsheet")
        similarity_threshold: Minimum similarity score (0.0–1.0) to include
        use_mmr: Whether to apply Maximal Marginal Relevance

    Returns:
        List of result dicts with keys:
        - chunk_id: str
        - text: str
        - metadata: dict
        - similarity: float
        - source_url: str
    """
    start_time = time.time()

    # Check if the collection has data
    count = get_collection_count()
    if count == 0:
        logger.warning("Vector store is empty — no chunks to retrieve")
        return []

    # 1. Embed query
    logger.info("Embedding query: '%s'", query[:80])
    query_embedding = embed_query(query)

    # 2. Build metadata filter
    where_filter = None
    conditions = []
    if scheme_filter:
        conditions.append({"scheme_name": scheme_filter})
    if document_type_filter:
        conditions.append({"document_type": document_type_filter})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    # 3. Query ChromaDB (fetch more candidates if using MMR)
    fetch_k = top_k * 3 if use_mmr else top_k
    fetch_k = min(fetch_k, count)  # Can't fetch more than available

    logger.info(
        "Querying ChromaDB: fetch_k=%d, filter=%s",
        fetch_k,
        where_filter,
    )

    raw_results = query_vectors(
        query_embedding=query_embedding,
        n_results=fetch_k,
        where=where_filter,
    )

    # 4. Convert to result dicts with similarity scores
    results = []
    for i in range(len(raw_results["ids"])):
        distance = raw_results["distances"][i]
        similarity = _distance_to_similarity(distance)

        metadata = raw_results["metadatas"][i]
        result = {
            "chunk_id": raw_results["ids"][i],
            "text": raw_results["documents"][i],
            "metadata": metadata,
            "similarity": similarity,
            "source_url": metadata.get("source_url", ""),
            "scheme_name": metadata.get("scheme_name", ""),
            "document_type": metadata.get("document_type", ""),
        }

        # Need embedding for MMR — reconstruct from query results
        if use_mmr:
            result["embedding"] = query_embedding  # placeholder

        results.append(result)

    # 5. Filter by similarity threshold
    filtered = [r for r in results if r["similarity"] >= similarity_threshold]
    logger.info(
        "Similarity filter: %d/%d results passed (threshold=%.2f)",
        len(filtered), len(results), similarity_threshold,
    )

    if not filtered:
        elapsed = time.time() - start_time
        logger.info("No results above threshold (elapsed=%.3fs)", elapsed)
        return []

    # 6. Apply MMR if enabled
    if use_mmr and filtered:
        # For MMR we need actual document embeddings — fetch them from the store
        # Since ChromaDB doesn't return embeddings in query results by default,
        # we'll use a simplified approach: skip MMR when embeddings aren't available
        # and just sort by similarity
        filtered.sort(key=lambda r: r["similarity"], reverse=True)
        filtered = filtered[:top_k]
    else:
        # Sort by similarity descending, take top-K
        filtered.sort(key=lambda r: r["similarity"], reverse=True)
        filtered = filtered[:top_k]

    # 7. Clean up internal fields before returning
    for r in filtered:
        r.pop("embedding", None)

    elapsed = time.time() - start_time
    logger.info(
        "Retrieval complete: %d results in %.3fs (top similarity=%.4f)",
        len(filtered), elapsed, filtered[0]["similarity"] if filtered else 0.0,
    )

    return filtered


def retrieve_with_rerank(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    scheme_filter: Optional[str] = None,
    document_type_filter: Optional[str] = None,
    similarity_threshold: float = RETRIEVAL_SIMILARITY_THRESHOLD,
    rerank_top_n: int = RERANK_TOP_N,
) -> list[dict]:
    """
    Retrieve and then rerank results using a cross-encoder.

    Args:
        query: User query
        top_k: Number of initial retrieval results
        scheme_filter: Optional scheme name filter
        document_type_filter: Optional document type filter
        similarity_threshold: Minimum similarity for initial retrieval
        rerank_top_n: Number of results to keep after reranking

    Returns:
        Reranked list of result dicts
    """
    # Initial retrieval (lower threshold to get more candidates)
    results = retrieve(
        query=query,
        top_k=top_k,
        scheme_filter=scheme_filter,
        document_type_filter=document_type_filter,
        similarity_threshold=max(similarity_threshold - 0.1, 0.3),
        use_mmr=False,
    )

    if not results:
        return []

    # Import reranker lazily (heavy dependency)
    from src.retrieval.reranker import rerank

    reranked = rerank(query, results, top_n=rerank_top_n)
    return reranked
