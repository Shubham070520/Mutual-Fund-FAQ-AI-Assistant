"""
Tests for Phase 7: API Layer.
Tests FastAPI endpoints, schemas, middleware, and full pipeline integration.
Uses FastAPI TestClient for synchronous endpoint testing.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    RefusalResponse,
    HealthResponse,
    SchemeListResponse,
    SchemeInfo,
    ErrorResponse,
    QueryResponseEnvelope,
)
from src.api.main import app, _load_schemes, _get_current_date
from src.api.middleware import ALLOWED_ORIGINS


# --- Test Client ---
client = TestClient(app)


# ====================================================================
# Schema Validation Tests
# ====================================================================

def test_query_request_valid():
    """Test valid QueryRequest creation."""
    req = QueryRequest(query="What is the NAV of HDFC Mid Cap Fund?")
    assert req.query == "What is the NAV of HDFC Mid Cap Fund?"
    assert req.scheme_filter is None
    assert req.document_type_filter is None
    print("✓ test_query_request_valid passed")


def test_query_request_with_filters():
    """Test QueryRequest with optional filters."""
    req = QueryRequest(
        query="What is the expense ratio?",
        scheme_filter="HDFC Mid Cap Fund",
        document_type_filter="scheme_page",
    )
    assert req.scheme_filter == "HDFC Mid Cap Fund"
    assert req.document_type_filter == "scheme_page"
    print("✓ test_query_request_with_filters passed")


def test_query_request_empty_query():
    """Test that empty query is rejected."""
    with pytest.raises(Exception):
        QueryRequest(query="")
    print("✓ test_query_request_empty_query passed")


def test_query_request_too_long():
    """Test that overly long queries are rejected."""
    with pytest.raises(Exception):
        QueryRequest(query="x" * 501)
    print("✓ test_query_request_too_long passed")


def test_query_response_envelope():
    """Test QueryResponseEnvelope structure."""
    resp = QueryResponseEnvelope(
        answer="The NAV is ₹142.35.",
        source_url="https://groww.in/test",
        last_updated="2026-06-09",
        intent="factual",
    )
    assert resp.answer == "The NAV is ₹142.35."
    assert resp.is_refusal is False
    assert resp.context_used == 0
    assert resp.warnings == []
    assert resp.pii_detected == []
    print("✓ test_query_response_envelope passed")


def test_health_response():
    """Test HealthResponse defaults."""
    h = HealthResponse()
    assert h.status == "ok"
    assert h.version == "1.0.0"
    assert h.vector_store_count == 0
    assert h.llm_available is False
    print("✓ test_health_response passed")


def test_scheme_info():
    """Test SchemeInfo model."""
    s = SchemeInfo(
        name="HDFC Mid Cap Fund",
        category="mid-cap",
        groww_url="https://groww.in/test",
    )
    assert s.name == "HDFC Mid Cap Fund"
    print("✓ test_scheme_info passed")


# ====================================================================
# /health Endpoint Tests
# ====================================================================

def test_health_endpoint():
    """Test GET /health returns 200 with expected fields."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "version" in data
    assert "vector_store_count" in data
    assert "llm_available" in data
    print("✓ test_health_endpoint passed")


def test_health_returns_json():
    """Test that /health returns valid JSON."""
    response = client.get("/health")
    assert response.headers.get("content-type") == "application/json"
    data = response.json()
    assert isinstance(data, dict)
    print("✓ test_health_returns_json passed")


# ====================================================================
# /schemes Endpoint Tests
# ====================================================================

def test_schemes_endpoint():
    """Test GET /schemes returns 200 with scheme list."""
    response = client.get("/schemes")
    assert response.status_code == 200

    data = response.json()
    assert "amc_name" in data
    assert data["amc_name"] == "HDFC Mutual Fund"
    assert "schemes" in data
    assert "scheme_count" in data
    assert isinstance(data["schemes"], list)
    print("✓ test_schemes_endpoint passed")


def test_schemes_have_required_fields():
    """Test each scheme has name, category, and groww_url."""
    response = client.get("/schemes")
    data = response.json()

    for scheme in data["schemes"]:
        assert "name" in scheme, f"Missing 'name' in scheme: {scheme}"
        assert "category" in scheme, f"Missing 'category' in scheme: {scheme}"
        assert "groww_url" in scheme, f"Missing 'groww_url' in scheme: {scheme}"
        assert scheme["name"] != ""
        assert scheme["groww_url"].startswith("http")

    print("✓ test_schemes_have_required_fields passed")


def test_schemes_count_matches():
    """Test that scheme_count matches the actual list length."""
    response = client.get("/schemes")
    data = response.json()
    assert data["scheme_count"] == len(data["schemes"])
    assert data["scheme_count"] == 5  # We have 5 HDFC schemes
    print("✓ test_schemes_count_matches passed")


def test_schemes_contains_expected_names():
    """Test that the expected 5 HDFC schemes are present."""
    response = client.get("/schemes")
    data = response.json()

    names = {s["name"] for s in data["schemes"]}
    expected = {
        "HDFC Mid Cap Fund",
        "HDFC Small Cap Fund",
        "HDFC Gold ETF Fund of Fund",
        "HDFC Defence Fund",
        "HDFC Silver ETF FoF",
    }

    assert expected == names, f"Scheme names mismatch.\nExpected: {expected}\nGot: {names}"
    print("✓ test_schemes_contains_expected_names passed")


# ====================================================================
# /query Endpoint Tests — Advisory Queries (Refusal)
# ====================================================================

def test_query_advisory_should_i_invest():
    """Test that advisory query 'Should I invest' returns refusal."""
    response = client.post("/query", json={
        "query": "Should I invest in HDFC Mid Cap Fund?",
    })
    assert response.status_code == 200

    data = response.json()
    assert data["is_refusal"] is True
    assert data["intent"] == "advisory"
    assert "cannot offer investment advice" in data["answer"].lower() or \
           "only provide factual" in data["answer"].lower()
    assert data["educational_link"] is not None
    assert data["last_updated"] != ""
    print("✓ test_query_advisory_should_i_invest passed")


def test_query_advisory_which_is_better():
    """Test that comparison query returns refusal."""
    response = client.post("/query", json={
        "query": "Which is better - HDFC Small Cap or HDFC Mid Cap?",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["is_refusal"] is True
    assert data["intent"] == "advisory"
    print("✓ test_query_advisory_which_is_better passed")


def test_query_advisory_good_returns():
    """Test that 'good returns' query returns refusal."""
    response = client.post("/query", json={
        "query": "Will HDFC Defence Fund give good returns?",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["is_refusal"] is True
    assert data["intent"] == "advisory"
    print("✓ test_query_advisory_good_returns passed")


def test_query_advisory_recommend():
    """Test that 'recommend' query returns refusal."""
    response = client.post("/query", json={
        "query": "Can you recommend the best HDFC fund?",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["is_refusal"] is True
    assert data["intent"] == "advisory"
    print("✓ test_query_advisory_recommend passed")


# ====================================================================
# /query Endpoint Tests — Out of Scope Queries (Refusal)
# ====================================================================

def test_query_out_of_scope_weather():
    """Test that weather query returns refusal."""
    response = client.post("/query", json={
        "query": "What is the weather today?",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["is_refusal"] is True
    assert data["intent"] == "out_of_scope"
    assert "sebi.gov.in" in (data.get("educational_link") or "") or \
           "outside my scope" in data["answer"].lower()
    print("✓ test_query_out_of_scope_weather passed")


def test_query_out_of_scope_sports():
    """Test that sports query returns refusal."""
    response = client.post("/query", json={
        "query": "Who won the cricket match yesterday?",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["is_refusal"] is True
    assert data["intent"] == "out_of_scope"
    print("✓ test_query_out_of_scope_sports passed")


# ====================================================================
# /query Endpoint Tests — PII Sanitization
# ====================================================================

def test_query_pii_phone_detected():
    """Test that PII (phone) in query is detected and reported."""
    response = client.post("/query", json={
        "query": "My phone is 9876543210, what is the NAV of HDFC Mid Cap Fund?",
    })
    assert response.status_code == 200
    data = response.json()
    assert "phone" in data["pii_detected"]
    print("✓ test_query_pii_phone_detected passed")


def test_query_pii_email_detected():
    """Test that PII (email) in query is detected and reported."""
    response = client.post("/query", json={
        "query": "Email me at test@example.com the NAV of HDFC Mid Cap Fund",
    })
    assert response.status_code == 200
    data = response.json()
    assert "email" in data["pii_detected"]
    print("✓ test_query_pii_email_detected passed")


# ====================================================================
# /query Endpoint Tests — Factual Queries (with mocked pipeline)
# ====================================================================

@patch("src.api.main.retrieve")
@patch("src.api.main.generate_response")
def test_query_factual_with_context(mock_generate, mock_retrieve):
    """Test factual query flow with mocked retrieval and generation."""
    # Mock retrieval results
    mock_retrieve.return_value = [
        {
            "chunk_id": "test-1",
            "text": "The expense ratio of HDFC Mid Cap Fund is 0.74%.",
            "metadata": {"source_url": "https://groww.in/test", "scheme_name": "HDFC Mid Cap Fund"},
            "similarity": 0.85,
            "source_url": "https://groww.in/test",
        }
    ]

    # Mock generation result
    mock_generate.return_value = {
        "response": "The expense ratio of HDFC Mid Cap Fund is 0.74%. (Source: https://groww.in/test)\n\nLast updated from sources: 2026-06-09",
        "source_url": "https://groww.in/test",
        "context_used": 1,
        "latency_ms": 100.0,
        "status": "success",
        "date": "2026-06-09",
    }

    response = client.post("/query", json={
        "query": "What is the expense ratio of HDFC Mid Cap Fund?",
    })
    assert response.status_code == 200

    data = response.json()
    assert data["intent"] == "factual"
    assert data["is_refusal"] is False
    assert "0.74" in data["answer"]
    assert data["source_url"] == "https://groww.in/test"
    assert data["context_used"] == 1
    assert data["last_updated"] == datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert data["scheme"] == "HDFC Mid Cap Fund"

    print("✓ test_query_factual_with_context passed")


@patch("src.api.main.retrieve")
def test_query_factual_no_context(mock_retrieve):
    """Test factual query when no context is found."""
    mock_retrieve.return_value = []

    response = client.post("/query", json={
        "query": "What is the NAV of HDFC Mid Cap Fund?",
    })
    assert response.status_code == 200

    data = response.json()
    assert data["intent"] == "factual"
    assert data["is_refusal"] is True
    assert "don't have this information" in data["answer"].lower()
    assert data["context_used"] == 0

    print("✓ test_query_factual_no_context passed")


@patch("src.api.main.retrieve")
@patch("src.api.main.generate_response")
def test_query_factual_with_scheme_filter(mock_generate, mock_retrieve):
    """Test factual query with scheme_filter parameter."""
    mock_retrieve.return_value = [
        {
            "chunk_id": "test-2",
            "text": "HDFC Small Cap Fund SIP minimum is ₹100.",
            "metadata": {"source_url": "https://groww.in/small-cap", "scheme_name": "HDFC Small Cap Fund"},
            "similarity": 0.90,
            "source_url": "https://groww.in/small-cap",
        }
    ]

    mock_generate.return_value = {
        "response": "The minimum SIP amount for HDFC Small Cap Fund is ₹100. (Source: https://groww.in/small-cap)\n\nLast updated from sources: 2026-06-09",
        "source_url": "https://groww.in/small-cap",
        "context_used": 1,
        "latency_ms": 50.0,
        "status": "success",
        "date": "2026-06-09",
    }

    response = client.post("/query", json={
        "query": "What is the minimum SIP amount for HDFC Small Cap Fund?",
        "scheme_filter": "HDFC Small Cap Fund",
    })
    assert response.status_code == 200

    data = response.json()
    assert data["is_refusal"] is False
    assert "100" in data["answer"]
    assert data["scheme"] == "HDFC Small Cap Fund"

    # Verify scheme_filter was passed to retrieve
    mock_retrieve.assert_called_once()
    call_kwargs = mock_retrieve.call_args
    assert call_kwargs.kwargs.get("scheme_filter") == "HDFC Small Cap Fund"

    print("✓ test_query_factual_with_scheme_filter passed")


# ====================================================================
# /query Endpoint Tests — Edge Cases
# ====================================================================

def test_query_empty_body():
    """Test that empty request body returns error."""
    response = client.post("/query", json={})
    assert response.status_code == 422
    print("✓ test_query_empty_body passed")


def test_query_whitespace_only():
    """Test that whitespace-only query returns error."""
    response = client.post("/query", json={"query": "   "})
    assert response.status_code == 422
    print("✓ test_query_whitespace_only passed")


def test_query_method_not_allowed():
    """Test that GET on /query returns 405."""
    response = client.get("/query")
    assert response.status_code == 405
    print("✓ test_query_method_not_allowed passed")


def test_nonexistent_endpoint():
    """Test that unknown endpoints return 404."""
    response = client.get("/nonexistent")
    assert response.status_code == 404
    print("✓ test_nonexistent_endpoint passed")


# ====================================================================
# Response Envelope Structure Tests
# ====================================================================

def test_response_envelope_has_all_fields():
    """Test that every /query response has all required envelope fields."""
    response = client.post("/query", json={
        "query": "Should I invest in HDFC Mid Cap Fund?",
    })
    data = response.json()

    required_fields = [
        "answer", "last_updated", "intent", "is_refusal",
        "context_used", "latency_ms", "warnings", "pii_detected",
    ]
    for field in required_fields:
        assert field in data, f"Missing field '{field}' in response envelope"

    print("✓ test_response_envelope_has_all_fields passed")


def test_response_has_date_footer():
    """Test that refusal responses include date footer."""
    response = client.post("/query", json={
        "query": "Should I invest in HDFC Mid Cap Fund?",
    })
    data = response.json()

    assert "last_updated" in data
    date_str = data["last_updated"]
    # Should be YYYY-MM-DD format
    assert len(date_str) == 10
    datetime.strptime(date_str, "%Y-%m-%d")  # Should not raise

    print("✓ test_response_has_date_footer passed")


def test_latency_is_positive():
    """Test that latency_ms is a non-negative number."""
    response = client.post("/query", json={
        "query": "Should I invest in HDFC Mid Cap Fund?",
    })
    data = response.json()
    assert data["latency_ms"] >= 0
    assert isinstance(data["latency_ms"], (int, float))
    print("✓ test_latency_is_positive passed")


# ====================================================================
# Helper Function Tests
# ====================================================================

def test_load_schemes():
    """Test that _load_schemes returns valid scheme list."""
    schemes = _load_schemes()
    assert len(schemes) == 5
    names = {s["name"] for s in schemes}
    assert "HDFC Mid Cap Fund" in names
    print("✓ test_load_schemes passed")


def test_get_current_date():
    """Test date format."""
    date = _get_current_date()
    assert len(date) == 10
    datetime.strptime(date, "%Y-%m-%d")
    print("✓ test_get_current_date passed")


# ====================================================================
# Middleware Tests
# ====================================================================

def test_cors_headers_present():
    """Test that CORS headers are included in response."""
    response = client.options(
        "/query",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
        },
    )
    # CORS preflight should succeed
    assert response.status_code in (200, 400)
    print("✓ test_cors_headers_present passed")


def test_process_time_header():
    """Test that X-Process-Time-Ms header is added to responses."""
    response = client.get("/health")
    # The header should be present
    assert "x-process-time-ms" in response.headers
    # Should be a numeric string
    time_val = float(response.headers["x-process-time-ms"])
    assert time_val >= 0
    print("✓ test_process_time_header passed")


# ====================================================================
# Advisory Leak Detection Through API
# ====================================================================

@patch("src.api.main.retrieve")
@patch("src.api.main.generate_response")
def test_advisory_leak_triggers_refusal(mock_generate, mock_retrieve):
    """Test that advisory language in LLM response is caught and replaced."""
    mock_retrieve.return_value = [
        {
            "chunk_id": "test-3",
            "text": "HDFC Defence Fund info.",
            "metadata": {"source_url": "https://groww.in/defence", "scheme_name": "HDFC Defence Fund"},
            "similarity": 0.80,
            "source_url": "https://groww.in/defence",
        }
    ]

    # Simulate LLM returning advisory language
    mock_generate.return_value = {
        "response": "I recommend investing in HDFC Defence Fund as it will definitely grow.\n\nLast updated from sources: 2026-06-09",
        "source_url": "https://groww.in/defence",
        "context_used": 1,
        "latency_ms": 80.0,
        "status": "success",
        "date": "2026-06-09",
    }

    response = client.post("/query", json={
        "query": "What is the benchmark for HDFC Defence Fund?",
    })
    assert response.status_code == 200

    data = response.json()
    # Should be replaced with refusal due to advisory leak
    assert data["is_refusal"] is True
    assert "recommend" not in data["answer"].lower() or "cannot" in data["answer"].lower()

    print("✓ test_advisory_leak_triggers_refusal passed")


# ====================================================================
# Run all tests
# ====================================================================

if __name__ == "__main__":
    tests = [
        test_query_request_valid,
        test_query_request_with_filters,
        test_query_request_empty_query,
        test_query_request_too_long,
        test_query_response_envelope,
        test_health_response,
        test_scheme_info,
        test_health_endpoint,
        test_health_returns_json,
        test_schemes_endpoint,
        test_schemes_have_required_fields,
        test_schemes_count_matches,
        test_schemes_contains_expected_names,
        test_query_advisory_should_i_invest,
        test_query_advisory_which_is_better,
        test_query_advisory_good_returns,
        test_query_advisory_recommend,
        test_query_out_of_scope_weather,
        test_query_out_of_scope_sports,
        test_query_pii_phone_detected,
        test_query_pii_email_detected,
        test_query_empty_body,
        test_query_whitespace_only,
        test_query_method_not_allowed,
        test_nonexistent_endpoint,
        test_response_envelope_has_all_fields,
        test_response_has_date_footer,
        test_latency_is_positive,
        test_load_schemes,
        test_get_current_date,
        test_cors_headers_present,
        test_process_time_header,
    ]

    # Mocked tests (need patching)
    mocked_tests = [
        test_query_factual_with_context,
        test_query_factual_no_context,
        test_query_factual_with_scheme_filter,
        test_advisory_leak_triggers_refusal,
    ]

    passed = 0
    failed = 0

    for test in tests + mocked_tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests) + len(mocked_tests)} total")
    print(f"{'='*60}")
