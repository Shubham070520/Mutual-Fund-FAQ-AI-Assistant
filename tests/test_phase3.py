"""Verification test for Phase 3 modules — embeddings, vectorstore, indexer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.processing.embeddings import embed_text, embed_query, embed_batch, get_dimensions
from src.processing.vectorstore import (
    get_collection,
    add_vectors,
    query_vectors,
    get_collection_count,
    delete_collection,
)
from src.processing.indexer import index_chunks


def test_embeddings():
    """Test single, query, and batch embedding."""
    # Single text
    vec = embed_text("HDFC Mid Cap Fund expense ratio is 0.80%")
    assert isinstance(vec, list), "embed_text should return a list"
    assert len(vec) > 0, "Embedding should not be empty"
    dim = len(vec)
    print(f"[PASS] embed_text: returned vector of dim {dim}")

    # Query embedding (with BGE instruction prefix)
    qvec = embed_query("What is the expense ratio?")
    assert isinstance(qvec, list), "embed_query should return a list"
    assert len(qvec) == dim, "Query embedding dim should match text embedding dim"
    print(f"[PASS] embed_query: returned vector of dim {dim}")

    # Batch embedding
    texts = [
        "HDFC Small Cap Fund has a minimum SIP of ₹500.",
        "HDFC Defence Fund tracks the Nifty India Defence Index.",
        "HDFC Gold ETF FoF invests in gold ETF units.",
    ]
    batch = embed_batch(texts)
    assert len(batch) == 3, f"Expected 3 vectors, got {len(batch)}"
    assert all(len(v) == dim for v in batch), "All batch vectors should have same dim"
    print(f"[PASS] embed_batch: returned {len(batch)} vectors of dim {dim}")

    # get_dimensions helper
    assert get_dimensions() == dim, "get_dimensions() mismatch"
    print(f"[PASS] get_dimensions: {dim}")

    return dim


def test_vectorstore(dim: int):
    """Test ChromaDB collection operations."""
    # Reset for clean test
    delete_collection()
    collection = get_collection()
    assert collection.count() == 0, "Collection should be empty after reset"
    print("[PASS] Collection created/reset successfully")

    # Add vectors
    ids = ["test_001", "test_002", "test_003"]
    texts = [
        "HDFC Mid Cap Fund direct plan has an expense ratio of 0.80%.",
        "The minimum SIP amount for HDFC Small Cap Fund is ₹500.",
        "HDFC Defence Fund benchmark is Nifty India Defence Index.",
    ]
    embeddings = embed_batch(texts)
    metadatas = [
        {"scheme_name": "HDFC Mid Cap Fund", "document_type": "factsheet", "source_url": "https://groww.in/test1"},
        {"scheme_name": "HDFC Small Cap Fund", "document_type": "faq", "source_url": "https://groww.in/test2"},
        {"scheme_name": "HDFC Defence Fund", "document_type": "sid", "source_url": "https://groww.in/test3"},
    ]

    add_vectors(ids, embeddings, texts, metadatas)
    count = get_collection_count()
    assert count == 3, f"Expected 3 vectors, got {count}"
    print(f"[PASS] add_vectors: {count} vectors stored")

    # Query vectors
    qvec = embed_query("What is the expense ratio of HDFC Mid Cap Fund?")
    results = query_vectors(qvec, n_results=2)
    assert len(results["ids"]) == 2, f"Expected 2 results, got {len(results['ids'])}"
    assert "documents" in results, "Results should contain documents"
    assert "metadatas" in results, "Results should contain metadatas"
    assert "distances" in results, "Results should contain distances"
    print(f"[PASS] query_vectors: top result = '{results['documents'][0][:60]}...' (distance={results['distances'][0]:.4f})")

    # Verify top result is the most relevant
    top_doc = results["documents"][0]
    assert "expense ratio" in top_doc.lower(), f"Top result should mention expense ratio, got: {top_doc[:80]}"
    print("[PASS] Top result is semantically relevant")

    return True


def test_indexer():
    """Test the indexer pipeline with synthetic chunks."""
    delete_collection()  # Clean slate

    chunks = [
        {
            "chunk_id": "idx_001",
            "text": "HDFC Gold ETF FoF invests primarily in units of gold exchange traded funds.",
            "metadata": {"scheme_name": "HDFC Gold ETF Fund of Fund", "document_type": "factsheet", "source_url": "https://groww.in/test_gold"},
        },
        {
            "chunk_id": "idx_002",
            "text": "The minimum lump sum investment in HDFC Silver ETF FoF is ₹1,000.",
            "metadata": {"scheme_name": "HDFC Silver ETF FoF", "document_type": "faq", "source_url": "https://groww.in/test_silver"},
        },
        {
            "chunk_id": "idx_003",
            "text": "HDFC Mid Cap Fund aims to generate returns by investing in mid-cap companies.",
            "metadata": {"scheme_name": "HDFC Mid Cap Fund", "document_type": "sid", "source_url": "https://groww.in/test_midcap"},
        },
    ]

    summary = index_chunks(chunks, reset_collection=True)
    assert summary["status"] != "failed", f"Indexer failed: {summary}"
    assert summary["chunks_indexed"] == 3, f"Expected 3 indexed, got {summary['chunks_indexed']}"
    assert summary["collection_count_after"] == 3, f"Collection count mismatch"
    print(f"[PASS] index_chunks: {summary['chunks_indexed']} chunks indexed, dim={summary['embedding_dimensions']}")

    # Verify queryability after indexing
    qvec = embed_query("What does HDFC Gold ETF FoF invest in?")
    results = query_vectors(qvec, n_results=1)
    assert "gold" in results["documents"][0].lower(), "Top result should be about gold"
    print(f"[PASS] Post-index query: '{results['documents'][0][:60]}...'")

    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3 Verification Tests")
    print("=" * 60)

    print("\n--- Test 1: Embeddings ---")
    dim = test_embeddings()

    print("\n--- Test 2: Vector Store ---")
    test_vectorstore(dim)

    print("\n--- Test 3: Indexer Pipeline ---")
    summary = test_indexer()

    print("\n" + "=" * 60)
    print("ALL PHASE 3 TESTS PASSED")
    print(f"Embedding dim: {dim}")
    print(f"Indexed chunks: {summary['chunks_indexed']}")
    print("=" * 60)
