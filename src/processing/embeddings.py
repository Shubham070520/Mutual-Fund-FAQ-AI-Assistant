"""
Embedding model wrapper using BGE (BAAI General Embedding).
Provides a consistent interface for encoding text into vector embeddings.
"""

import logging
from functools import lru_cache
from typing import Union

from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once, reused everywhere
_model_instance: SentenceTransformer | None = None


def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Load and cache the embedding model (singleton pattern).

    The model is downloaded on first use and cached locally by HuggingFace.

    Args:
        model_name: HuggingFace model identifier.
                    Defaults to BAAI/bge-small-en-v1.5 (384 dims).

    Returns:
        Loaded SentenceTransformer instance
    """
    global _model_instance

    if _model_instance is not None:
        return _model_instance

    logger.info("Loading embedding model: %s", model_name)
    _model_instance = SentenceTransformer(model_name)
    logger.info(
        "Model loaded successfully (dimensions=%d)",
        _model_instance.get_embedding_dimension(),
    )
    return _model_instance


def embed_text(text: str) -> list[float]:
    """
    Encode a single text string into an embedding vector.

    BGE models expect queries to be prefixed with "Represent this sentence
    for searching relevant passages:" for asymmetric retrieval.
    For indexing documents, use the raw text directly.

    Args:
        text: Input text to embed

    Returns:
        List of floats (embedding vector)
    """
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_query(query: str) -> list[float]:
    """
    Encode a user query into an embedding vector.

    For BGE models, prepends the instruction prefix for better
    asymmetric retrieval performance.

    Args:
        query: User query text

    Returns:
        List of floats (embedding vector)
    """
    model = get_embedding_model()
    # BGE instruction prefix for query embeddings
    instruction = "Represent this sentence for searching relevant passages: "
    embedding = model.encode(instruction + query, normalize_embeddings=True)
    return embedding.tolist()


def embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Encode a batch of texts into embedding vectors.

    Args:
        texts: List of text strings to embed
        batch_size: Number of texts to process in parallel

    Returns:
        List of embedding vectors (one per input text)
    """
    if not texts:
        return []

    model = get_embedding_model()
    logger.info("Embedding batch of %d texts (batch_size=%d)", len(texts), batch_size)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
    )

    result = [emb.tolist() for emb in embeddings]
    logger.info("Batch embedding complete: %d vectors of dim %d", len(result), len(result[0]) if result else 0)
    return result


def get_dimensions() -> int:
    """Return the embedding dimensionality of the current model."""
    model = get_embedding_model()
    return model.get_embedding_dimension()
