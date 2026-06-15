"""
Phase 9 — End-to-End Integration Tests.

Tests the full pipeline flow:
  UI → API → Sanitizer → Intent Classifier → Refusal OR (Retriever → Generator → Post-processor) → API → UI

Uses FastAPI TestClient with mocked heavy dependencies (retrieval, LLM)
to validate the entire request lifecycle.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

# ====================================================================
# Helpers — Reusable mock factories
# ====================================================================

def _mock_retrieve_result(text, url, scheme, similarity=0.88):
    """Build a single mock retrieval chunk."""
    return {
        "chunk_id": "e2e-chunk-1",
        "text": text,
        "metadata": {
            "source_url": url,
            "scheme_name": scheme,
            "document_type": "scheme_page",
        },
        "similarity": similarity,
        "source_url": url,
        "scheme_name": scheme,
    }


def _mock_generate_result(response, source_url, status="success", context_used=1):
    """Build a mock generation result dict."""
    return {
        "response": response,
        "source_url": source_url,
        "context_used": context_used,
        "latency_ms": 50.0,
        "status": status,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


SAMPLE_NAV_RESPONSE = (
    "The NAV of HDFC Mid Cap Fund is ₹142.35 as of June 2026. "
    "(Source: https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth)\n\n"
    "Last updated from sources: {date}"
).format(date=datetime.now(timezone.utc).strftime("%Y-%m-%d"))


# ====================================================================
# E2E Flow: Factual Queries (Full Pipeline)
# ====================================================================

@patch("src.api.main.retrieve")
@patch("src.api.main.generate_response")
def test_e2e_factual_nav_query(mock_gen, mock_ret):
    """Full pipeline for a factual NAV query."""
    mock_ret.return_value = [
        _mock_retrieve_result(
            "The NAV of HDFC Mid Cap Fund is ₹142.35 as on June 2026.",
            "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
            "HDFC Mid Cap Fund",
        )
    ]
    mock_gen.return_value = _mock_generate_result(
        SAMPLE_NAV_RESPONSE,
        "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
    )

    resp = client.post("/query", json={"query": "What is the NAV of HDFC Mid Cap Fund?"})
    assert resp.status_code == 200

    data = resp.json()
    # Full pipeline assertions
    assert data["intent"] == "factual"
    assert data["is_refusal"] is False
    assert "142.35" in data["answer"]
    assert data["source_url"] == "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth"
    assert data["scheme"] == "HDFC Mid Cap Fund"
    assert data["context_used"] == 1
    assert isinstance(data["latency_ms"], (int, float))
    assert data["latency_ms"] >= 0
    assert data["pii_detected"] == []
    assert data["warnings"] == []
    assert data["last_updated"] == datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print("✓ test_e2e_factual_nav_query passed")


@patch("src.api.main.retrieve")
@patch("src.api.main.generate_response")
def test_e2e_factual_expense_ratio(mock_gen, mock_ret):
    """Full pipeline for expense ratio query with scheme filter."""
    mock_ret.return_value = [
        _mock_retrieve_result(
            "The expense ratio of HDFC Mid Cap Fund is 0.74%.",
            "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
            "HDFC Mid Cap Fund",
        )
    ]
    mock_gen.return_value = _mock_generate_result(
        "The expense ratio of HDFC Mid Cap Fund is 0.74%. "
        "(Source: https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth)\n\n"
        f"Last updated from sources: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
    )

    resp = client.post("/query", json={
        "query": "What is the expense ratio of HDFC Mid Cap Fund?",
        "scheme_filter": "HDFC Mid Cap Fund",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_refusal"] is False
    assert "0.74" in data["answer"]
    assert data["scheme"] == "HDFC Mid Cap Fund"

    # Verify scheme_filter was passed to retriever
    call_kwargs = mock_ret.call_args.kwargs
    assert call_kwargs.get("scheme_filter") == "HDFC Mid Cap Fund"
    print("✓ test_e2e_factual_expense_ratio passed")


@patch("src.api.main.retrieve")
@patch("src.api.main.generate_response")
def test_e2e_factual_sip_amount(mock_gen, mock_ret):
    """Full pipeline for SIP minimum amount query."""
    mock_ret.return_value = [
        _mock_retrieve_result(
            "The minimum SIP amount for HDFC Small Cap Fund is ₹100.",
            "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
            "HDFC Small Cap Fund",
            similarity=0.92,
        )
    ]
    mock_gen.return_value = _mock_generate_result(
        "The minimum SIP amount for HDFC Small Cap Fund is ₹100. "
        "(Source: https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth)\n\n"
        f"Last updated from sources: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    )

    resp = client.post("/query", json={
        "query": "What is the minimum SIP amount for HDFC Small Cap Fund?",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_refusal"] is False
    assert "100" in data["answer"]
    assert data["scheme"] == "HDFC Small Cap Fund"
    print("✓ test_e2e_factual_sip_amount passed")


@patch("src.api.main.retrieve")
@patch("src.api.main.generate_response")
def test_e2e_factual_benchmark(mock_gen, mock_ret):
    """Full pipeline for benchmark index query."""
    mock_ret.return_value = [
        _mock_retrieve_result(
            "The benchmark for HDFC Defence Fund is NIFTY India Defence Index.",
            "https://groww.in/mutual-funds/hdfc-defence-fund-direct-growth",
            "HDFC Defence Fund",
        )
    ]
    mock_gen.return_value = _mock_generate_result(
        "The benchmark index for HDFC Defence Fund is NIFTY India Defence Index. "
        "(Source: https://groww.in/mutual-funds/hdfc-defence-fund-direct-growth)\n\n"
        f"Last updated from sources: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "https://groww.in/mutual-funds/hdfc-defence-fund-direct-growth",
    )

    resp = client.post("/query", json={
        "query": "What is the benchmark index for HDFC Defence Fund?",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_refusal"] is False
    assert "NIFTY" in data["answer"] or "Defence" in data["answer"]
    print("✓ test_e2e_factual_benchmark passed")


# ====================================================================
# E2E Flow: Advisory Queries (Refusal Path)
# ====================================================================

def test_e2e_advisory_should_i_invest():
    """Full pipeline: advisory 'should I invest' → refusal with AMFI link."""
    resp = client.post("/query", json={
        "query": "Should I invest in HDFC Mid Cap Fund?",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["intent"] == "advisory"
    assert data["is_refusal"] is True
    assert "only provide factual" in data["answer"].lower() or \
           "cannot offer investment advice" in data["answer"].lower()
    assert "amfiindia.com" in (data.get("educational_link") or "")
    assert data["context_used"] == 0  # no retrieval for advisory
    assert data["source_url"] is None
    assert data["last_updated"] != ""
    print("✓ test_e2e_advisory_should_i_invest passed")


def test_e2e_advisory_comparison():
    """Full pipeline: comparison query → refusal."""
    resp = client.post("/query", json={
        "query": "Which fund is better - HDFC Small Cap or HDFC Mid Cap?",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_refusal"] is True
    assert data["intent"] == "advisory"
    assert data["context_used"] == 0
    print("✓ test_e2e_advisory_comparison passed")


def test_e2e_advisory_good_returns():
    """Full pipeline: 'good returns' query → refusal."""
    resp = client.post("/query", json={
        "query": "Will HDFC Defence Fund give good returns?",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_refusal"] is True
    assert data["intent"] == "advisory"
    print("✓ test_e2e_advisory_good_returns passed")


def test_e2e_advisory_recommend():
    """Full pipeline: 'recommend' query → refusal."""
    resp = client.post("/query", json={
        "query": "Can you recommend the best HDFC fund for me?",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_refusal"] is True
    assert data["intent"] == "advisory"
    print("✓ test_e2e_advisory_recommend passed")


# ====================================================================
# E2E Flow: Out-of-Scope Queries (Refusal Path)
# ====================================================================

def test_e2e_out_of_scope_weather():
    """Full pipeline: weather query → out-of-scope refusal with SEBI link."""
    resp = client.post("/query", json={
        "query": "What is the weather today?",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["intent"] == "out_of_scope"
    assert data["is_refusal"] is True
    assert "outside my scope" in data["answer"].lower()
    assert "sebi.gov.in" in (data.get("educational_link") or "")
    assert data["context_used"] == 0
    print("✓ test_e2e_out_of_scope_weather passed")


def test_e2e_out_of_scope_sports():
    """Full pipeline: sports query → out-of-scope refusal."""
    resp = client.post("/query", json={
        "query": "Who won the cricket match yesterday?",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["intent"] == "out_of_scope"
    assert data["is_refusal"] is True
    print("✓ test_e2e_out_of_scope_sports passed")


def test_e2e_out_of_scope_stocks():
    """Full pipeline: stock tips query → out-of-scope refusal."""
    resp = client.post("/query", json={
        "query": "Can you give me share price tips for Reliance?",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["intent"] == "out_of_scope"
    assert data["is_refusal"] is True
    print("✓ test_e2e_out_of_scope_stocks passed")


# ====================================================================
# E2E Flow: PII Sanitization Through Pipeline
# ====================================================================

def test_e2e_pii_phone_detected_and_sanitized():
    """PII (phone) detected and query still processed."""
    resp = client.post("/query", json={
        "query": "My phone is 9876543210, what is the NAV of HDFC Mid Cap Fund?",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert "phone" in data["pii_detected"]
    # The query should still be processed (classified as factual since it has MF keywords)
    assert data["intent"] == "factual"
    print("✓ test_e2e_pii_phone_detected_and_sanitized passed")


def test_e2e_pii_email_detected():
    """PII (email) detected in query."""
    resp = client.post("/query", json={
        "query": "Email me at user@example.com about HDFC Mid Cap Fund NAV",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert "email" in data["pii_detected"]
    print("✓ test_e2e_pii_email_detected passed")


# ====================================================================
# E2E Flow: Advisory Leak Detection
# ====================================================================

@patch("src.api.main.retrieve")
@patch("src.api.main.generate_response")
def test_e2e_advisory_leak_replaced_with_refusal(mock_gen, mock_ret):
    """LLM returning advisory language → replaced with refusal."""
    mock_ret.return_value = [
        _mock_retrieve_result(
            "HDFC Defence Fund info.",
            "https://groww.in/mutual-funds/hdfc-defence-fund-direct-growth",
            "HDFC Defence Fund",
        )
    ]
    mock_gen.return_value = _mock_generate_result(
        "I recommend investing in HDFC Defence Fund as it will definitely grow.\n\n"
        f"Last updated from sources: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "https://groww.in/mutual-funds/hdfc-defence-fund-direct-growth",
    )

    resp = client.post("/query", json={
        "query": "What is the benchmark for HDFC Defence Fund?",
    })
    assert resp.status_code == 200
    data = resp.json()

    # Advisory leak should be caught and replaced
    assert data["is_refusal"] is True
    assert "recommend" not in data["answer"].lower() or "cannot" in data["answer"].lower()
    assert "only provide factual" in data["answer"].lower() or \
           "cannot offer investment advice" in data["answer"].lower()
    print("✓ test_e2e_advisory_leak_replaced_with_refusal passed")


# ====================================================================
# E2E Flow: No Context Fallback
# ====================================================================

@patch("src.api.main.retrieve")
def test_e2e_no_context_returns_refusal(mock_ret):
    """Factual query with no matching context → refusal."""
    mock_ret.return_value = []

    resp = client.post("/query", json={
        "query": "What is the lock-in period for HDFC ELSS fund?",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["intent"] == "factual"
    assert data["is_refusal"] is True
    assert "don't have this information" in data["answer"].lower()
    assert data["context_used"] == 0
    print("✓ test_e2e_no_context_returns_refusal passed")


# ====================================================================
# E2E Flow: Response Envelope Integrity
# ====================================================================

def test_e2e_envelope_all_fields_present():
    """Every response has all required envelope fields."""
    resp = client.post("/query", json={
        "query": "Should I invest in HDFC Mid Cap Fund?",
    })
    data = resp.json()

    required_fields = [
        "answer", "last_updated", "intent", "is_refusal",
        "context_used", "latency_ms", "warnings", "pii_detected",
    ]
    for field in required_fields:
        assert field in data, f"Missing field '{field}'"

    # Optional fields should be present (can be None)
    assert "source_url" in data
    assert "educational_link" in data
    assert "scheme" in data
    print("✓ test_e2e_envelope_all_fields_present passed")


def test_e2e_latency_tracking():
    """Latency is measured end-to-end through the full pipeline."""
    resp = client.post("/query", json={
        "query": "Should I invest in HDFC Small Cap Fund?",
    })
    data = resp.json()

    assert isinstance(data["latency_ms"], (int, float))
    assert data["latency_ms"] >= 0
    print("✓ test_e2e_latency_tracking passed")


def test_e2e_date_footer_format():
    """Date footer is in YYYY-MM-DD format."""
    resp = client.post("/query", json={
        "query": "Should I invest in HDFC Mid Cap Fund?",
    })
    data = resp.json()

    date_str = data["last_updated"]
    assert len(date_str) == 10
    datetime.strptime(date_str, "%Y-%m-%d")
    print("✓ test_e2e_date_footer_format passed")


# ====================================================================
# E2E Flow: Multiple Sequential Queries (Chat Simulation)
# ====================================================================

def test_e2e_sequential_queries():
    """Simulate a chat session with multiple queries."""
    queries = [
        ("Should I invest in HDFC Mid Cap?", True, "advisory"),
        ("What is the weather today?", True, "out_of_scope"),
        ("Which fund is best?", True, "advisory"),
    ]

    for query, expected_refusal, expected_intent in queries:
        resp = client.post("/query", json={"query": query})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_refusal"] == expected_refusal, f"Failed for query: {query}"
        assert data["intent"] == expected_intent, f"Failed for query: {query}"

    print("✓ test_e2e_sequential_queries passed")


# ====================================================================
# E2E Flow: Endpoint Cross-Checks
# ====================================================================

def test_e2e_health_and_schemes_consistency():
    """Health and schemes endpoints return consistent data."""
    health = client.get("/health").json()
    schemes = client.get("/schemes").json()

    assert health["status"] == "ok"
    assert schemes["scheme_count"] == 5
    assert schemes["amc_name"] == "HDFC Mutual Fund"
    print("✓ test_e2e_health_and_schemes_consistency passed")


# ====================================================================
# Run all tests
# ====================================================================

if __name__ == "__main__":
    # Tests that DON'T need mocking
    standalone_tests = [
        test_e2e_advisory_should_i_invest,
        test_e2e_advisory_comparison,
        test_e2e_advisory_good_returns,
        test_e2e_advisory_recommend,
        test_e2e_out_of_scope_weather,
        test_e2e_out_of_scope_sports,
        test_e2e_out_of_scope_stocks,
        test_e2e_pii_phone_detected_and_sanitized,
        test_e2e_pii_email_detected,
        test_e2e_envelope_all_fields_present,
        test_e2e_latency_tracking,
        test_e2e_date_footer_format,
        test_e2e_sequential_queries,
        test_e2e_health_and_schemes_consistency,
    ]

    # Tests that need mocking (patches applied via decorators)
    mocked_tests = [
        test_e2e_factual_nav_query,
        test_e2e_factual_expense_ratio,
        test_e2e_factual_sip_amount,
        test_e2e_factual_benchmark,
        test_e2e_advisory_leak_replaced_with_refusal,
        test_e2e_no_context_returns_refusal,
    ]

    passed = 0
    failed = 0

    for test in standalone_tests + mocked_tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(standalone_tests) + len(mocked_tests)} total")
    print(f"{'='*60}")
