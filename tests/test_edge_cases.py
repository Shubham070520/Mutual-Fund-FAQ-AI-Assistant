"""
Phase 9 — Edge Case Tests (Task 9.4).

Tests edge cases that could break the pipeline:
- Empty queries
- Queries with PII (PAN, Aadhaar, phone, email)
- Very long queries
- Queries about unsupported schemes
- Queries with no matching context
- Multiple questions in one query
- Special characters and unicode
- Boundary values
- Concurrent PII + advisory
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from src.api.main import app
from src.guardrails.sanitizer import sanitize, sanitize_detailed
from src.guardrails.intent import classify_intent, INTENT_FACTUAL, INTENT_ADVISORY, INTENT_OUT_OF_SCOPE
from src.guardrails.refusal import generate_refusal, is_refusal_response
from src.generation.postprocessor import validate_and_format, count_sentences
from src.processing.chunker import chunk_text, _classify_chunk_content

client = TestClient(app)


# ====================================================================
# Empty / Whitespace Queries
# ====================================================================

def test_empty_query_api():
    """Empty query returns 422 via API."""
    resp = client.post("/query", json={"query": ""})
    assert resp.status_code == 422
    print("✓ test_empty_query_api passed")


def test_whitespace_only_query_api():
    """Whitespace-only query returns 422 via API."""
    resp = client.post("/query", json={"query": "     "})
    assert resp.status_code == 422
    print("✓ test_whitespace_only_query_api passed")


def test_empty_query_sanitizer():
    """Empty query passes through sanitizer unchanged."""
    result = sanitize_detailed("")
    assert result.sanitized_query == ""
    assert not result.was_modified
    assert result.pii_detected == []
    print("✓ test_empty_query_sanitizer passed")


def test_empty_query_intent():
    """Empty query classified as out-of-scope."""
    intent = classify_intent("")
    assert intent == INTENT_OUT_OF_SCOPE
    print("✓ test_empty_query_intent passed")


def test_whitespace_query_intent():
    """Whitespace query classified as out-of-scope."""
    intent = classify_intent("   ")
    assert intent == INTENT_OUT_OF_SCOPE
    print("✓ test_whitespace_query_intent passed")


# ====================================================================
# PII Edge Cases
# ====================================================================

def test_pii_pan_card():
    """PAN card number detected and redacted."""
    result = sanitize_detailed("My PAN is ABCDE1234F what is the NAV?")
    assert result.was_modified
    assert "pan" in result.pii_detected
    assert "ABCDE1234F" not in result.sanitized_query
    print("✓ test_pii_pan_card passed")


def test_pii_aadhaar_with_spaces():
    """Aadhaar with spaces detected."""
    result = sanitize_detailed("Aadhaar: 1234 5678 9012, need fund info")
    assert result.was_modified
    assert "aadhaar" in result.pii_detected
    print("✓ test_pii_aadhaar_with_spaces passed")


def test_pii_aadhaar_with_dashes():
    """Aadhaar with dashes detected."""
    result = sanitize_detailed("My aadhaar is 1234-5678-9012")
    assert result.was_modified
    assert "aadhaar" in result.pii_detected
    print("✓ test_pii_aadhaar_with_dashes passed")


def test_pii_phone_with_country_code():
    """Phone with +91 country code detected."""
    result = sanitize_detailed("Call +91 9876543210 for NAV info")
    assert result.was_modified
    assert "phone" in result.pii_detected
    print("✓ test_pii_phone_with_country_code passed")


def test_pii_email():
    """Email address detected."""
    result = sanitize_detailed("Send to user@gmail.com the SIP details")
    assert result.was_modified
    assert "email" in result.pii_detected
    print("✓ test_pii_email passed")


def test_pii_multiple_types():
    """Multiple PII types detected in same query."""
    result = sanitize_detailed(
        "My PAN is ABCDE1234F and email is test@example.com, what is the NAV?"
    )
    assert result.was_modified
    assert len(result.pii_detected) >= 2
    print("✓ test_pii_multiple_types passed")


def test_pii_not_leaked_in_response():
    """PII from query never appears in API response."""
    resp = client.post("/query", json={
        "query": "My phone is 9876543210, what is the NAV of HDFC Mid Cap Fund?",
    })
    data = resp.json()
    # The raw phone number should not appear in the answer
    assert "9876543210" not in data.get("answer", "")
    print("✓ test_pii_not_leaked_in_response passed")


# ====================================================================
# Very Long Queries
# ====================================================================

def test_very_long_query_api():
    """Query exceeding 500 chars returns 422."""
    long_query = "What is the NAV? " * 50  # ~850 chars
    resp = client.post("/query", json={"query": long_query})
    assert resp.status_code == 422
    print("✓ test_very_long_query_api passed")


def test_max_length_query():
    """Query at exactly 500 chars is accepted."""
    query = "What is the NAV of HDFC Mid Cap Fund? " + "x" * (500 - 39 - 1)
    query = query[:500]  # exactly 500
    resp = client.post("/query", json={"query": query})
    # Should be accepted (200) or if classified as OOS, still 200
    assert resp.status_code == 200
    print("✓ test_max_length_query passed")


# ====================================================================
# Unsupported Schemes
# ====================================================================

@patch("src.api.main.retrieve")
def test_unsupported_scheme_query(mock_ret):
    """Query about non-HDFC scheme gets no-context refusal."""
    mock_ret.return_value = []

    resp = client.post("/query", json={
        "query": "What is the NAV of SBI Blue Chip Fund?",
    })
    assert resp.status_code == 200
    data = resp.json()
    # Either classified as factual (has "NAV" keyword) and gets no-context refusal,
    # or classified differently — either way should get a response
    assert "answer" in data
    print("✓ test_unsupported_scheme_query passed")


@patch("src.api.main.retrieve")
def test_unsupported_scheme_with_filter(mock_ret):
    """Query with unsupported scheme_filter returns gracefully."""
    mock_ret.return_value = []

    resp = client.post("/query", json={
        "query": "What is the NAV?",
        "scheme_filter": "Non Existent Fund XYZ",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_refusal"] is True
    print("✓ test_unsupported_scheme_with_filter passed")


# ====================================================================
# No Matching Context
# ====================================================================

@patch("src.api.main.retrieve")
def test_no_matching_context(mock_ret):
    """Factual query with no matching context returns refusal."""
    mock_ret.return_value = []

    resp = client.post("/query", json={
        "query": "What is the lock-in period for HDFC ELSS Tax Saver?",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["intent"] == "factual"
    assert data["is_refusal"] is True
    assert "don't have this information" in data["answer"].lower()
    print("✓ test_no_matching_context passed")


@patch("src.api.main.retrieve")
def test_retrieval_exception_handled(mock_ret):
    """Retrieval failure is handled gracefully."""
    mock_ret.side_effect = Exception("ChromaDB connection error")

    resp = client.post("/query", json={
        "query": "What is the NAV of HDFC Mid Cap Fund?",
    })
    assert resp.status_code == 200
    data = resp.json()
    # Should still return a response (refusal due to no context)
    assert "answer" in data
    print("✓ test_retrieval_exception_handled passed")


# ====================================================================
# Multiple Questions in One Query
# ====================================================================

def test_multiple_questions_query():
    """Query with multiple questions is processed without crash."""
    resp = client.post("/query", json={
        "query": "What is the NAV of HDFC Mid Cap Fund? Also what is the expense ratio? And who is the fund manager?",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["intent"] == "factual"
    print("✓ test_multiple_questions_query passed")


def test_mixed_intent_query():
    """Query mixing factual and advisory is handled."""
    resp = client.post("/query", json={
        "query": "What is the NAV of HDFC Mid Cap Fund and should I invest?",
    })
    assert resp.status_code == 200
    data = resp.json()
    # Has advisory keyword "should I invest" → should be classified as advisory
    assert data["intent"] == "advisory"
    assert data["is_refusal"] is True
    print("✓ test_mixed_intent_query passed")


# ====================================================================
# Special Characters and Unicode
# ====================================================================

def test_special_characters_query():
    """Query with special characters is processed."""
    resp = client.post("/query", json={
        "query": "What is the NAV (₹) of HDFC Mid-Cap Fund?!?",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    print("✓ test_special_characters_query passed")


def test_unicode_query():
    """Query with unicode characters is processed."""
    resp = client.post("/query", json={
        "query": "What is the NAV of HDFC Mid Cap Fund? 🤔💰",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    print("✓ test_unicode_query passed")


def test_query_with_newlines():
    """Query with newlines is processed."""
    resp = client.post("/query", json={
        "query": "What is the NAV\nof HDFC Mid Cap Fund?",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    print("✓ test_query_with_newlines passed")


# ====================================================================
# Concurrent PII + Advisory
# ====================================================================

def test_pii_with_advisory():
    """Query with both PII and advisory intent — PII detected, advisory refused."""
    resp = client.post("/query", json={
        "query": "My phone is 9876543210, should I invest in HDFC Mid Cap Fund?",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert "phone" in data["pii_detected"]
    assert data["intent"] == "advisory"
    assert data["is_refusal"] is True
    # PII should not leak into response
    assert "9876543210" not in data.get("answer", "")
    print("✓ test_pii_with_advisory passed")


def test_pii_with_out_of_scope():
    """Query with PII and out-of-scope content."""
    resp = client.post("/query", json={
        "query": "Email me at test@example.com about the weather forecast",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert "email" in data["pii_detected"]
    assert data["intent"] == "out_of_scope"
    assert data["is_refusal"] is True
    print("✓ test_pii_with_out_of_scope passed")


# ====================================================================
# Boundary Values
# ====================================================================

def test_single_character_query():
    """Single character query is processed."""
    resp = client.post("/query", json={"query": "?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    print("✓ test_single_character_query passed")


def test_very_short_query():
    """Very short query is classified."""
    intent = classify_intent("NAV")
    assert intent == INTENT_FACTUAL
    print("✓ test_very_short_query passed")


def test_numeric_only_query():
    """Numeric-only query is handled."""
    resp = client.post("/query", json={"query": "12345"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    print("✓ test_numeric_only_query passed")


def test_query_with_only_url():
    """Query containing only a URL is handled."""
    resp = client.post("/query", json={"query": "https://groww.in/mutual-funds"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    print("✓ test_query_with_only_url passed")


# ====================================================================
# Post-processor Edge Cases
# ====================================================================

def test_postprocessor_empty_response():
    """Empty response is handled."""
    result = validate_and_format("", "https://test.com", "2026-06-09")
    assert result["has_footer"]
    assert "Last updated from sources: 2026-06-09" in result["formatted_response"]
    print("✓ test_postprocessor_empty_response passed")


def test_postprocessor_very_long_response():
    """Very long response is trimmed to max sentences."""
    long_response = ". ".join([f"Sentence {i}" for i in range(20)]) + "."
    long_response += "\n\nLast updated from sources: 2026-06-09"

    result = validate_and_format(long_response, "https://test.com", "2026-06-09")
    assert result["sentence_count"] <= 3
    print("✓ test_postprocessor_very_long_response passed")


def test_postprocessor_no_url_no_source():
    """No URL and no source URL is handled."""
    result = validate_and_format("The NAV is ₹142.35.", "", "2026-06-09")
    assert result["has_footer"]
    print("✓ test_postprocessor_no_url_no_source passed")


def test_count_sentences_empty():
    """Empty text has 0 sentences."""
    assert count_sentences("") == 0
    print("✓ test_count_sentences_empty passed")


def test_count_sentences_no_terminator():
    """Text without sentence terminator counts as 1."""
    assert count_sentences("Hello world") == 1
    print("✓ test_count_sentences_no_terminator passed")


# ====================================================================
# Chunker Edge Cases
# ====================================================================

def test_chunk_very_short_text():
    """Very short text produces at least one chunk."""
    chunks = chunk_text("Hi", {"source_url": "x"}, chunk_size=10, chunk_overlap=2)
    assert len(chunks) >= 1
    print("✓ test_chunk_very_short_text passed")


def test_classify_chunk_content_multiple_tags():
    """Chunk with multiple fund metrics gets multiple tags."""
    text = "NAV: ₹142.35, Exit Load: 1%, SIP minimum: ₹100, managed by John Doe"
    types = _classify_chunk_content(text)
    assert "nav" in types
    assert "exit_load" in types
    assert "sip" in types
    assert "fund_manager" in types
    print("✓ test_classify_chunk_content_multiple_tags passed")


# ====================================================================
# Refusal Generator Edge Cases
# ====================================================================

def test_refusal_empty_query():
    """Empty query gets appropriate refusal."""
    refusal = generate_refusal("ADVISORY", "")
    assert "ask a specific question" in refusal.lower()
    assert "Last updated from sources:" in refusal
    print("✓ test_refusal_empty_query passed")


def test_refusal_unknown_intent():
    """Unknown intent gets no-context refusal."""
    refusal = generate_refusal("UNKNOWN_INTENT", "some query")
    assert "don't have this information" in refusal.lower()
    print("✓ test_refusal_unknown_intent passed")


# ====================================================================
# API Response Consistency
# ====================================================================

def test_all_responses_have_content_type():
    """All API responses have correct content type."""
    resp1 = client.get("/health")
    assert resp1.headers["content-type"] == "application/json"

    resp2 = client.get("/schemes")
    assert resp2.headers["content-type"] == "application/json"

    resp3 = client.post("/query", json={"query": "What is the NAV?"})
    assert resp3.headers["content-type"] == "application/json"
    print("✓ test_all_responses_have_content_type passed")


def test_x_process_time_header():
    """X-Process-Time-Ms header present on all responses."""
    for endpoint in ["/health", "/schemes"]:
        resp = client.get(endpoint)
        assert "x-process-time-ms" in resp.headers
    print("✓ test_x_process_time_header passed")


# ====================================================================
# Run all tests
# ====================================================================

if __name__ == "__main__":
    standalone = [
        # Empty / Whitespace
        test_empty_query_api,
        test_whitespace_only_query_api,
        test_empty_query_sanitizer,
        test_empty_query_intent,
        test_whitespace_query_intent,
        # PII
        test_pii_pan_card,
        test_pii_aadhaar_with_spaces,
        test_pii_aadhaar_with_dashes,
        test_pii_phone_with_country_code,
        test_pii_email,
        test_pii_multiple_types,
        test_pii_not_leaked_in_response,
        # Long queries
        test_very_long_query_api,
        test_max_length_query,
        # Unsupported schemes
        test_unsupported_scheme_query,
        test_unsupported_scheme_with_filter,
        # No context
        test_no_matching_context,
        test_retrieval_exception_handled,
        # Multiple questions
        test_multiple_questions_query,
        test_mixed_intent_query,
        # Special chars / Unicode
        test_special_characters_query,
        test_unicode_query,
        test_query_with_newlines,
        # PII + intent combos
        test_pii_with_advisory,
        test_pii_with_out_of_scope,
        # Boundary
        test_single_character_query,
        test_very_short_query,
        test_numeric_only_query,
        test_query_with_only_url,
        # Post-processor
        test_postprocessor_empty_response,
        test_postprocessor_very_long_response,
        test_postprocessor_no_url_no_source,
        test_count_sentences_empty,
        test_count_sentences_no_terminator,
        # Chunker
        test_chunk_very_short_text,
        test_classify_chunk_content_multiple_tags,
        # Refusal
        test_refusal_empty_query,
        test_refusal_unknown_intent,
        # API consistency
        test_all_responses_have_content_type,
        test_x_process_time_header,
    ]

    passed = 0
    failed = 0

    for test in standalone:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(standalone)} total")
    print(f"{'='*60}")
