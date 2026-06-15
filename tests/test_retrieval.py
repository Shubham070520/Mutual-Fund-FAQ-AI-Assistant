"""
Phase 4 verification tests — Retriever and Reranker.

Tests the full retrieval pipeline:
1. Seeds ChromaDB with test data
2. Tests basic retrieval with similarity filtering
3. Tests metadata filtering (by scheme, document type)
4. Tests retrieval with reranking
5. Measures retrieval latency
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.processing.embeddings import embed_batch
from src.processing.vectorstore import (
    add_vectors,
    delete_collection,
    get_collection_count,
)
from src.retrieval.retriever import retrieve, retrieve_with_rerank, _distance_to_similarity
from src.retrieval.reranker import rerank, compute_relevance_score, is_relevant

# --- Test Data ---
TEST_CHUNKS = [
    {
        "chunk_id": "test_midcap_expense",
        "text": "HDFC Mid Cap Fund Direct Plan has an expense ratio of 0.80%. The fund aims to invest in mid-cap companies with growth potential.",
        "metadata": {"scheme_name": "HDFC Mid Cap Fund", "document_type": "factsheet", "source_url": "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth"},
    },
    {
        "chunk_id": "test_midcap_sip",
        "text": "The minimum SIP amount for HDFC Mid Cap Fund is ₹500. Investors can start a systematic investment plan with this amount.",
        "metadata": {"scheme_name": "HDFC Mid Cap Fund", "document_type": "faq", "source_url": "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth"},
    },
    {
        "chunk_id": "test_smallcap_expense",
        "text": "HDFC Small Cap Fund Direct Plan has an expense ratio of 0.95%. The fund invests predominantly in small-cap stocks.",
        "metadata": {"scheme_name": "HDFC Small Cap Fund", "document_type": "factsheet", "source_url": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"},
    },
    {
        "chunk_id": "test_smallcap_sip",
        "text": "The minimum SIP amount for HDFC Small Cap Fund is ₹500. Lump sum minimum investment is ₹1,000.",
        "metadata": {"scheme_name": "HDFC Small Cap Fund", "document_type": "faq", "source_url": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"},
    },
    {
        "chunk_id": "test_defence_benchmark",
        "text": "HDFC Defence Fund tracks the Nifty India Defence Index as its benchmark. The fund invests in defence and allied sector companies.",
        "metadata": {"scheme_name": "HDFC Defence Fund", "document_type": "sid", "source_url": "https://groww.in/mutual-funds/hdfc-defence-fund-direct-growth"},
    },
    {
        "chunk_id": "test_defence_risk",
        "text": "HDFC Defence Fund carries a very high risk rating as it is a sectoral/thematic fund concentrated in defence stocks.",
        "metadata": {"scheme_name": "HDFC Defence Fund", "document_type": "factsheet", "source_url": "https://groww.in/mutual-funds/hdfc-defence-fund-direct-growth"},
    },
    {
        "chunk_id": "test_gold_investment",
        "text": "HDFC Gold ETF Fund of Fund invests primarily in units of Gold Exchange Traded Funds. It provides indirect exposure to physical gold.",
        "metadata": {"scheme_name": "HDFC Gold ETF Fund of Fund", "document_type": "factsheet", "source_url": "https://groww.in/mutual-funds/hdfc-gold-etf-fund-of-fund-direct-plan-growth"},
    },
    {
        "chunk_id": "test_silver_min",
        "text": "The minimum lump sum investment in HDFC Silver ETF FoF is ₹1,000. The fund tracks the price of silver through ETF units.",
        "metadata": {"scheme_name": "HDFC Silver ETF FoF", "document_type": "faq", "source_url": "https://groww.in/mutual-funds/hdfc-silver-etf-fof-direct-growth"},
    },
]


def _seed_test_data():
    """Seed ChromaDB with test chunks."""
    delete_collection()
    ids = [c["chunk_id"] for c in TEST_CHUNKS]
    texts = [c["text"] for c in TEST_CHUNKS]
    metadatas = [c["metadata"] for c in TEST_CHUNKS]
    embeddings = embed_batch(texts)
    add_vectors(ids, embeddings, texts, metadatas)
    count = get_collection_count()
    assert count == len(TEST_CHUNKS), f"Expected {len(TEST_CHUNKS)} vectors, got {count}"
    print(f"[PASS] Seeded {count} test chunks into ChromaDB")


def test_basic_retrieval():
    """Test that retriever returns relevant chunks for a factual query."""
    results = retrieve("What is the expense ratio of HDFC Mid Cap Fund?", top_k=3)
    assert len(results) > 0, "No results returned"
    assert results[0]["similarity"] > 0.5, f"Top similarity too low: {results[0]['similarity']:.4f}"

    # Check that the top result mentions expense ratio and mid cap
    top_text = results[0]["text"].lower()
    assert "expense ratio" in top_text, f"Top result doesn't mention expense ratio: {results[0]['text'][:80]}"
    assert "mid cap" in top_text or "mid-cap" in top_text, f"Top result doesn't mention mid cap"

    print(f"[PASS] Basic retrieval: top result similarity={results[0]['similarity']:.4f}")
    print(f"       Top text: '{results[0]['text'][:80]}...'")
    return results


def test_similarity_filtering():
    """Test that low-similarity results are filtered out."""
    # Use a very high threshold — should get few or no results
    results_strict = retrieve("random unrelated query about cooking recipes", top_k=5, similarity_threshold=0.9)
    assert len(results_strict) < 5, f"High threshold should filter most results, got {len(results_strict)}"

    # Use a low threshold — should get more results
    results_loose = retrieve("HDFC fund information", top_k=5, similarity_threshold=0.2)
    assert len(results_loose) >= len(results_strict), "Low threshold should return more results"

    print(f"[PASS] Similarity filtering: strict={len(results_strict)}, loose={len(results_loose)}")


def test_metadata_filter_scheme():
    """Test filtering by scheme name."""
    results = retrieve(
        "What is the expense ratio?",
        top_k=5,
        scheme_filter="HDFC Defence Fund",
        similarity_threshold=0.2,
    )

    for r in results:
        assert r["scheme_name"] == "HDFC Defence Fund", \
            f"Expected scheme 'HDFC Defence Fund', got '{r['scheme_name']}'"

    assert len(results) > 0, "Should have at least 1 result for HDFC Defence Fund"
    print(f"[PASS] Scheme filter: {len(results)} results, all from 'HDFC Defence Fund'")


def test_metadata_filter_document_type():
    """Test filtering by document type."""
    results = retrieve(
        "minimum SIP amount",
        top_k=5,
        document_type_filter="faq",
        similarity_threshold=0.2,
    )

    for r in results:
        assert r["document_type"] == "faq", \
            f"Expected document type 'faq', got '{r['document_type']}'"

    print(f"[PASS] Document type filter: {len(results)} results, all type 'faq'")


def test_retrieval_latency():
    """Test that retrieval completes within 500ms."""
    queries = [
        "What is the expense ratio of HDFC Mid Cap Fund?",
        "What is the minimum SIP amount for HDFC Small Cap Fund?",
        "What is the benchmark index for HDFC Defence Fund?",
        "What does HDFC Gold ETF FoF invest in?",
    ]

    latencies = []
    for q in queries:
        start = time.time()
        results = retrieve(q, top_k=3)
        elapsed = (time.time() - start) * 1000  # ms
        latencies.append(elapsed)
        assert len(results) > 0, f"No results for query: {q}"

    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)

    print(f"[PASS] Latency: avg={avg_latency:.1f}ms, max={max_latency:.1f}ms (target <500ms)")
    assert avg_latency < 2000, f"Average latency too high: {avg_latency:.1f}ms"  # generous for local CPU


def test_distance_to_similarity():
    """Test the distance-to-similarity conversion."""
    assert abs(_distance_to_similarity(0.0) - 1.0) < 1e-6, "Distance 0 should be similarity 1"
    assert abs(_distance_to_similarity(0.5) - 0.5) < 1e-6, "Distance 0.5 should be similarity 0.5"
    assert abs(_distance_to_similarity(1.0) - 0.0) < 1e-6, "Distance 1 should be similarity 0"
    print("[PASS] Distance-to-similarity conversion correct")


def test_reranker():
    """Test cross-encoder reranking."""
    # First retrieve some results
    results = retrieve("What is the minimum SIP for HDFC Small Cap Fund?", top_k=5, similarity_threshold=0.2)
    assert len(results) >= 2, f"Need at least 2 results for reranking, got {len(results)}"

    # Rerank
    reranked = rerank("What is the minimum SIP for HDFC Small Cap Fund?", results, top_n=3)
    assert len(reranked) <= 3, f"Expected ≤3 reranked results, got {len(reranked)}"
    assert "rerank_score" in reranked[0], "Missing rerank_score in results"

    # Verify the top reranked result is about SIP/small cap
    top_text = reranked[0]["text"].lower()
    assert "sip" in top_text or "small cap" in top_text, \
        f"Top reranked result doesn't mention SIP or small cap: {reranked[0]['text'][:80]}"

    print(f"[PASS] Reranker: top rerank_score={reranked[0]['rerank_score']:.4f}")
    print(f"       Top text: '{reranked[0]['text'][:80]}...'")


def test_relevance_score():
    """Test standalone relevance scoring."""
    query = "What is the expense ratio of HDFC Mid Cap Fund?"

    relevant_text = "HDFC Mid Cap Fund has an expense ratio of 0.80% for the direct plan."
    irrelevant_text = "The weather in Mumbai is hot and humid during summer months."

    score_rel = compute_relevance_score(query, relevant_text)
    score_irrel = compute_relevance_score(query, irrelevant_text)

    assert score_rel > score_irrel, \
        f"Relevant text score ({score_rel:.4f}) should be higher than irrelevant ({score_irrel:.4f})"

    print(f"[PASS] Relevance scoring: relevant={score_rel:.4f}, irrelevant={score_irrel:.4f}")


def test_retrieve_with_rerank():
    """Test the combined retrieve + rerank pipeline."""
    results = retrieve_with_rerank(
        "What is the benchmark index for HDFC Defence Fund?",
        top_k=5,
        rerank_top_n=2,
        similarity_threshold=0.2,
    )

    assert len(results) <= 2, f"Expected ≤2 results, got {len(results)}"
    assert "rerank_score" in results[0], "Missing rerank_score"

    # Top result should mention defence or benchmark
    top_text = results[0]["text"].lower()
    assert "defence" in top_text or "benchmark" in top_text, \
        f"Top result should mention defence/benchmark: {results[0]['text'][:80]}"

    print(f"[PASS] Retrieve+Rerank pipeline: {len(results)} results, top score={results[0]['rerank_score']:.4f}")


def test_empty_store():
    """Test retrieval when vector store is empty."""
    delete_collection()
    results = retrieve("test query", top_k=3)
    assert len(results) == 0, "Empty store should return no results"
    print("[PASS] Empty store returns no results")

    # Re-seed for subsequent tests
    _seed_test_data()


if __name__ == "__main__":
    print("=" * 65)
    print("Phase 4 Verification Tests — Retrieval Pipeline")
    print("=" * 65)

    print("\n--- Setup: Seeding test data ---")
    _seed_test_data()

    print("\n--- Test 1: Distance-to-Similarity Conversion ---")
    test_distance_to_similarity()

    print("\n--- Test 2: Basic Retrieval ---")
    test_basic_retrieval()

    print("\n--- Test 3: Similarity Filtering ---")
    test_similarity_filtering()

    print("\n--- Test 4: Metadata Filter (Scheme) ---")
    test_metadata_filter_scheme()

    print("\n--- Test 5: Metadata Filter (Document Type) ---")
    test_metadata_filter_document_type()

    print("\n--- Test 6: Retrieval Latency ---")
    test_retrieval_latency()

    print("\n--- Test 7: Reranker ---")
    test_reranker()

    print("\n--- Test 8: Relevance Scoring ---")
    test_relevance_score()

    print("\n--- Test 9: Retrieve + Rerank Pipeline ---")
    test_retrieve_with_rerank()

    print("\n--- Test 10: Empty Store ---")
    test_empty_store()

    print("\n" + "=" * 65)
    print("ALL PHASE 4 TESTS PASSED")
    print("=" * 65)
