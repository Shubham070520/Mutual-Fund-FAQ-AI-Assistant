"""
Tests for Phase 8: User Interface (Streamlit + Gradio).
Tests API client functions, chat logic, configuration, and response formatting.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from config.settings import API_BASE_URL, AMC_NAME


# ====================================================================
# Configuration Tests
# ====================================================================

def test_api_base_url_configured():
    """Test that API_BASE_URL is set and valid."""
    assert API_BASE_URL is not None
    assert API_BASE_URL.startswith("http")
    assert "localhost" in API_BASE_URL or "127.0.0.1" in API_BASE_URL
    print("✓ test_api_base_url_configured passed")


def test_amc_name_configured():
    """Test that AMC_NAME is set."""
    assert AMC_NAME == "HDFC Mutual Fund"
    print("✓ test_amc_name_configured passed")


# ====================================================================
# Streamlit Constants Tests
# ====================================================================

def test_streamlit_constants():
    """Test Streamlit UI constants are properly defined."""
    from src.ui.app import APP_TITLE, APP_ICON, DISCLAIMER, EXAMPLE_QUESTIONS, CATEGORY_LABELS

    assert APP_TITLE == "Mutual Fund FAQ Assistant"
    assert APP_ICON == "📊"
    assert "Facts-only" in DISCLAIMER
    assert "No investment advice" in DISCLAIMER
    assert len(EXAMPLE_QUESTIONS) == 3
    assert all(isinstance(q, str) for q in EXAMPLE_QUESTIONS)
    assert all("HDFC" in q for q in EXAMPLE_QUESTIONS)
    assert len(CATEGORY_LABELS) == 5

    print("✓ test_streamlit_constants passed")


def test_streamlit_example_questions_quality():
    """Test that example questions cover different fact types."""
    from src.ui.app import EXAMPLE_QUESTIONS

    # Should cover expense ratio, SIP, and fund manager
    texts = " ".join(EXAMPLE_QUESTIONS).lower()
    assert "expense ratio" in texts, "Missing expense ratio example"
    assert "sip" in texts, "Missing SIP example"
    assert "fund manager" in texts, "Missing fund manager example"

    print("✓ test_streamlit_example_questions_quality passed")


# ====================================================================
# Streamlit API Client Tests
# ====================================================================

@patch("src.ui.app.requests.post")
def test_call_api_post_success(mock_post):
    """Test successful POST call to API."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"answer": "test response", "is_refusal": False}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    from src.ui.app import call_api

    result = call_api("/query", method="POST", json_body={"query": "test"})

    assert result is not None
    assert result["answer"] == "test response"
    mock_post.assert_called_once()

    print("✓ test_call_api_post_success passed")


@patch("src.ui.app.requests.get")
def test_call_api_get_success(mock_get):
    """Test successful GET call to API."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    from src.ui.app import call_api

    result = call_api("/health", method="GET")

    assert result is not None
    assert result["status"] == "ok"

    print("✓ test_call_api_get_success passed")


@patch("src.ui.app.requests.post")
def test_call_api_connection_error(mock_post):
    """Test API client handles connection errors gracefully."""
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    from src.ui.app import call_api

    result = call_api("/query", method="POST", json_body={"query": "test"})

    assert result is not None
    assert "error" in result
    assert "Cannot connect" in result["error"]

    print("✓ test_call_api_connection_error passed")


@patch("src.ui.app.requests.post")
def test_call_api_timeout(mock_post):
    """Test API client handles timeouts gracefully."""
    mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

    from src.ui.app import call_api

    result = call_api("/query", method="POST", json_body={"query": "test"})

    assert result is not None
    assert "error" in result
    assert "timed out" in result["error"].lower()

    print("✓ test_call_api_timeout passed")


@patch("src.ui.app.requests.post")
def test_call_api_http_error(mock_post):
    """Test API client handles HTTP errors gracefully."""
    mock_response = MagicMock()
    mock_response.status_code = 422
    mock_response.text = "Validation error"
    mock_post.side_effect = requests.exceptions.HTTPError(response=mock_response)

    from src.ui.app import call_api

    result = call_api("/query", method="POST", json_body={"query": ""})

    assert result is not None
    assert "error" in result

    print("✓ test_call_api_http_error passed")


# ====================================================================
# Streamlit send_query Tests
# ====================================================================

@patch("src.ui.app.call_api")
def test_send_query_factual(mock_call):
    """Test send_query with a factual response."""
    mock_call.return_value = {
        "answer": "The NAV is ₹142.35.",
        "source_url": "https://groww.in/test",
        "is_refusal": False,
        "intent": "factual",
        "scheme": "HDFC Mid Cap Fund",
        "last_updated": "2026-06-09",
        "latency_ms": 150.0,
        "context_used": 2,
        "educational_link": None,
        "pii_detected": [],
    }

    from src.ui.app import send_query

    result = send_query("What is the NAV of HDFC Mid Cap Fund?")

    assert result["answer"] == "The NAV is ₹142.35."
    assert result["source_url"] == "https://groww.in/test"
    assert result["is_refusal"] is False
    assert result["intent"] == "factual"

    print("✓ test_send_query_factual passed")


@patch("src.ui.app.call_api")
def test_send_query_refusal(mock_call):
    """Test send_query with an advisory refusal."""
    mock_call.return_value = {
        "answer": "I can only provide factual information...",
        "is_refusal": True,
        "intent": "advisory",
        "educational_link": "https://www.amfiindia.com/investor-education",
        "last_updated": "2026-06-09",
        "source_url": None,
        "scheme": None,
        "context_used": 0,
        "latency_ms": 10.0,
        "pii_detected": [],
    }

    from src.ui.app import send_query

    result = send_query("Should I invest in HDFC Mid Cap Fund?")

    assert result["is_refusal"] is True
    assert result["intent"] == "advisory"
    assert "amfiindia.com" in result["educational_link"]

    print("✓ test_send_query_refusal passed")


@patch("src.ui.app.call_api")
def test_send_query_with_scheme_filter(mock_call):
    """Test send_query passes scheme_filter correctly."""
    mock_call.return_value = {
        "answer": "Test response",
        "is_refusal": False,
        "intent": "factual",
    }

    from src.ui.app import send_query

    send_query("What is the NAV?", scheme_filter="HDFC Mid Cap Fund")

    mock_call.assert_called_once()
    call_args = mock_call.call_args
    body = call_args.kwargs.get("json_body") or call_args[1].get("json_body")
    assert body["scheme_filter"] == "HDFC Mid Cap Fund"

    print("✓ test_send_query_with_scheme_filter passed")


@patch("src.ui.app.call_api")
def test_send_query_api_error(mock_call):
    """Test send_query handles API errors gracefully."""
    mock_call.return_value = {"error": "Cannot connect to the API server."}

    from src.ui.app import send_query

    result = send_query("What is the NAV?")

    assert result["is_refusal"] is True
    assert "Cannot connect" in result["answer"]
    assert result["intent"] == "error"

    print("✓ test_send_query_api_error passed")


# ====================================================================
# Gradio Tests
# ====================================================================

def test_gradio_constants():
    """Test Gradio UI constants."""
    from src.ui.gradio_app import APP_TITLE, APP_DESCRIPTION, EXAMPLE_QUESTIONS

    assert APP_TITLE == "Mutual Fund FAQ Assistant"
    assert "Facts-only" in APP_DESCRIPTION
    assert len(EXAMPLE_QUESTIONS) >= 3
    assert all(len(q) == 1 for q in EXAMPLE_QUESTIONS)  # Each example is a [question] list

    print("✓ test_gradio_constants passed")


@patch("src.ui.gradio_app.call_query_api")
def test_gradio_chat_fn_factual(mock_api):
    """Test Gradio chat function with factual response."""
    mock_api.return_value = {
        "answer": "The expense ratio is 0.74%.",
        "source_url": "https://groww.in/test",
        "scheme": "HDFC Mid Cap Fund",
        "last_updated": "2026-06-09",
        "latency_ms": 100.0,
        "is_refusal": False,
        "intent": "factual",
        "pii_detected": [],
    }

    from src.ui.gradio_app import chat_fn

    result = chat_fn("What is the expense ratio?", [])

    assert "0.74%" in result
    assert "groww.in/test" in result
    assert "HDFC Mid Cap Fund" in result

    print("✓ test_gradio_chat_fn_factual passed")


@patch("src.ui.gradio_app.call_query_api")
def test_gradio_chat_fn_advisory(mock_api):
    """Test Gradio chat function with advisory refusal."""
    mock_api.return_value = {
        "answer": "I can only provide factual information.",
        "is_refusal": True,
        "intent": "advisory",
        "source_url": None,
        "scheme": None,
        "last_updated": "2026-06-09",
        "latency_ms": 5.0,
        "pii_detected": [],
    }

    from src.ui.gradio_app import chat_fn

    result = chat_fn("Should I invest?", [])

    assert "factual" in result.lower() or "facts-only" in result.lower()
    assert "🚫" in result  # Refusal indicator

    print("✓ test_gradio_chat_fn_advisory passed")


@patch("src.ui.gradio_app.call_query_api")
def test_gradio_chat_fn_pii(mock_api):
    """Test Gradio chat function reports PII detection."""
    mock_api.return_value = {
        "answer": "The NAV is ₹142.35.",
        "is_refusal": False,
        "intent": "factual",
        "source_url": "https://groww.in/test",
        "pii_detected": ["phone"],
    }

    from src.ui.gradio_app import chat_fn

    result = chat_fn("My phone is 9876543210, what is the NAV?", [])

    assert "phone" in result
    assert "⚠️" in result  # PII warning

    print("✓ test_gradio_chat_fn_pii passed")


@patch("src.ui.gradio_app.call_query_api")
def test_gradio_chat_fn_error(mock_api):
    """Test Gradio chat function handles API errors."""
    mock_api.return_value = {"error": "Connection refused"}

    from src.ui.gradio_app import chat_fn

    result = chat_fn("What is the NAV?", [])

    assert "⚠️" in result
    assert "Connection refused" in result

    print("✓ test_gradio_chat_fn_error passed")


def test_gradio_chat_fn_empty_message():
    """Test Gradio chat function with empty message."""
    from src.ui.gradio_app import chat_fn

    result = chat_fn("", [])

    assert "ask a specific question" in result.lower() or "please" in result.lower()

    print("✓ test_gradio_chat_fn_empty_message passed")


@patch("src.ui.gradio_app.call_query_api")
def test_gradio_chat_fn_educational_link(mock_api):
    """Test Gradio chat function shows educational link for refusals."""
    mock_api.return_value = {
        "answer": "I can only provide factual information.",
        "is_refusal": True,
        "intent": "advisory",
        "educational_link": "https://www.amfiindia.com/investor-education",
        "source_url": None,
    }

    from src.ui.gradio_app import chat_fn

    result = chat_fn("Which fund is best?", [])

    assert "amfiindia.com" in result or "Investor Education" in result

    print("✓ test_gradio_chat_fn_educational_link passed")


# ====================================================================
# Gradio API Client Tests
# ====================================================================

@patch("src.ui.gradio_app.requests.post")
def test_gradio_api_client_success(mock_post):
    """Test Gradio API client with successful response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"answer": "test"}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    from src.ui.gradio_app import call_query_api

    result = call_query_api("What is the NAV?")

    assert result["answer"] == "test"

    print("✓ test_gradio_api_client_success passed")


@patch("src.ui.gradio_app.requests.post")
def test_gradio_api_client_connection_error(mock_post):
    """Test Gradio API client with connection error."""
    mock_post.side_effect = requests.exceptions.ConnectionError()

    from src.ui.gradio_app import call_query_api

    result = call_query_api("test")

    assert "error" in result
    assert "Cannot connect" in result["error"]

    print("✓ test_gradio_api_client_connection_error passed")


@patch("src.ui.gradio_app.requests.post")
def test_gradio_api_client_timeout(mock_post):
    """Test Gradio API client with timeout."""
    mock_post.side_effect = requests.exceptions.Timeout()

    from src.ui.gradio_app import call_query_api

    result = call_query_api("test")

    assert "error" in result
    assert "timed out" in result["error"].lower()

    print("✓ test_gradio_api_client_timeout passed")


# ====================================================================
# Category Labels Test
# ====================================================================

def test_category_labels_complete():
    """Test that all scheme categories have display labels."""
    from src.ui.app import CATEGORY_LABELS

    expected_categories = [
        "mid-cap", "small-cap", "gold-etf-fof",
        "sectoral-defence", "silver-etf-fof",
    ]

    for cat in expected_categories:
        assert cat in CATEGORY_LABELS, f"Missing label for category: {cat}"
        assert len(CATEGORY_LABELS[cat]) > 0

    print("✓ test_category_labels_complete passed")


# ====================================================================
# Run all tests
# ====================================================================

if __name__ == "__main__":
    tests = [
        test_api_base_url_configured,
        test_amc_name_configured,
        test_streamlit_constants,
        test_streamlit_example_questions_quality,
        test_call_api_post_success,
        test_call_api_get_success,
        test_call_api_connection_error,
        test_call_api_timeout,
        test_call_api_http_error,
        test_send_query_factual,
        test_send_query_refusal,
        test_send_query_with_scheme_filter,
        test_send_query_api_error,
        test_gradio_constants,
        test_gradio_chat_fn_factual,
        test_gradio_chat_fn_advisory,
        test_gradio_chat_fn_pii,
        test_gradio_chat_fn_error,
        test_gradio_chat_fn_empty_message,
        test_gradio_chat_fn_educational_link,
        test_gradio_api_client_success,
        test_gradio_api_client_connection_error,
        test_gradio_api_client_timeout,
        test_category_labels_complete,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*60}")
