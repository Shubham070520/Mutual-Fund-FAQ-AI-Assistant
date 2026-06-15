"""
Reranker module — uses a cross-encoder model to rerank retrieved chunks
for improved relevance. Cross-encoders jointly encode query-document pairs
and produce more accurate relevance scores than bi-encoders.
"""

import logging
from typing import Optional

from config.settings import RERANK_TOP_N

logger = logging.getLogger(__name__)

# Module-level singleton
_reranker_model = None
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_reranker_model():
    """
    Load and cache the cross-encoder reranker model (singleton pattern).

    The ms-marco-MiniLM-L-6-v2 model is:
    - Lightweight (~80MB)
    - Fast inference
    - Trained on MS MARCO passage ranking
    - Well-suited for short-text reranking

    Returns:
        CrossEncoder instance
    """
    global _reranker_model

    if _reranker_model is not None:
        return _reranker_model

    from sentence_transformers import CrossEncoder

    logger.info("Loading reranker model: %s", RERANKER_MODEL_NAME)
    _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
    logger.info("Reranker model loaded successfully")
    return _reranker_model


def rerank(
    query: str,
    chunks: list[dict],
    top_n: int = RERANK_TOP_N,
) -> list[dict]:
    """
    Rerank a list of retrieved chunks using a cross-encoder.

    For each chunk, the cross-encoder computes a relevance score by jointly
    encoding the (query, chunk_text) pair. This produces more accurate
    relevance ranking than cosine similarity alone.

    Args:
        query: The user's query string
        chunks: List of result dicts from the retriever, each containing
                at least 'text' and 'similarity' keys
        top_n: Number of top results to return after reranking

    Returns:
        Reranked list of result dicts, sorted by cross-encoder score.
        Each dict gains an additional 'rerank_score' key.
    """
    if not chunks:
        return []

    if len(chunks) <= 1:
        return chunks

    model = _get_reranker_model()

    # Build (query, document) pairs for the cross-encoder
    pairs = [(query, chunk["text"]) for chunk in chunks]

    logger.info("Reranking %d chunks for query: '%s'", len(chunks), query[:60])

    # Get relevance scores from cross-encoder
    scores = model.predict(pairs)

    # Attach scores to chunks
    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        reranked_chunk = dict(chunk)  # shallow copy
        reranked_chunk["rerank_score"] = float(score)
        scored_chunks.append(reranked_chunk)

    # Sort by rerank score descending
    scored_chunks.sort(key=lambda c: c["rerank_score"], reverse=True)

    # Take top-N
    result = scored_chunks[:top_n]

    logger.info(
        "Reranking complete: top score=%.4f, returning %d results",
        result[0]["rerank_score"] if result else 0.0,
        len(result),
    )

    # Log ranking changes
    for i, chunk in enumerate(result):
        original_pos = next(
            (j for j, c in enumerate(chunks) if c["chunk_id"] == chunk["chunk_id"]),
            -1,
        )
        if original_pos != i:
            logger.debug(
                "Chunk '%s' moved from position %d to %d (score=%.4f)",
                chunk["chunk_id"][:20], original_pos, i, chunk["rerank_score"],
            )

    return result


def compute_relevance_score(query: str, text: str) -> float:
    """
    Compute a single relevance score for a query-text pair.

    Useful for standalone relevance evaluation without a full retrieval pipeline.

    Args:
        query: Query string
        text: Document/chunk text

    Returns:
        Relevance score (higher is more relevant)
    """
    model = _get_reranker_model()
    score = model.predict([(query, text)])[0]
    return float(score)


def is_relevant(
    query: str,
    text: str,
    threshold: float = 0.0,
) -> bool:
    """
    Determine if a text is relevant to a query based on cross-encoder score.

    The ms-marco-MiniLM model outputs scores typically in range [-10, 10],
    where positive scores indicate relevance.

    Args:
        query: Query string
        text: Document text
        threshold: Minimum score to consider relevant (default 0.0)

    Returns:
        True if the cross-encoder score >= threshold
    """
    score = compute_relevance_score(query, text)
    return score >= threshold
