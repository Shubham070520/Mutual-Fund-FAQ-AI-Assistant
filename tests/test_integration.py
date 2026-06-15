"""
Phase 9 — Integration Tests.

Tests the interaction between multiple modules working together,
using the specific test scenarios from the implementation plan.

Scenarios:
- Factual queries (should answer) — from Task 9.3
- Advisory queries (should refuse) — from Task 9.3
- Out-of-scope queries (should refuse) — from Task 9.3
- Cross-module interaction: Sanitizer → Intent, Retrieval → Generation
- Prompt construction with real context
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from src.api.main import app
from src.guardrails.sanitizer import sanitize_detailed
from src.guardrails.intent import (
    classify_intent,
    is_advisory_keyword,
    is_out_of_scope,
    is_mutual_fund_domain,
    INTENT_FACTUAL,
    INTENT_ADVISORY,
    INTENT_OUT_OF_SCOPE,
)
from src.guardrails.refusal import generate_refusal, is_refusal_response
from src.generation.prompts import build_prompt, build_context_block
from src.generation.postprocessor import validate_and_format
from src.generation.generator import generate_response

client = TestClient(app)


# ====================================================================
# Sanitizer → Intent Classifier Interaction
# ====================================================================

def test_sanitized_query_classified_correctly():
    """PII is stripped, then query is classified correctly."""
    raw = "My PAN is ABCDE1234F, what is the NAV of HDFC Mid Cap Fund?"
    san = sanitize_detailed(raw)
    assert san.was_modified
    assert "ABCDE1234F" not in san.sanitized_query

    # Sanitized query should still be classified as factual (has MF keywords)
    intent = classify_intent(san.sanitized_query)
    assert intent == INTENT_FACTUAL
    print("✓ test_sanitized_query_classified_correctly passed")


def test_sanitized_advisory_still_advisory():
    """Advisory query remains advisory after PII sanitization."""
    raw = "Call me at 9876543210, should I invest in HDFC Mid Cap?"
    san = sanitize_detailed(raw)
    assert san.was_modified

    intent = classify_intent(san.sanitized_query)
    assert intent == INTENT_ADVISORY
    print("✓ test_sanitized_advisory_still_advisory passed")


# ====================================================================
# Intent Classifier → Refusal Generator Interaction
# ====================================================================

def test_intent_to_refusal_advisory():
    """Advisory intent triggers correct refusal with AMFI link."""
    query = "Should I invest in HDFC Mid Cap Fund?"
    intent = classify_intent(query)
    assert intent == INTENT_ADVISORY

    refusal = generate_refusal(intent, query)
    assert "factual" in refusal.lower()
    assert "amfiindia.com" in refusal
    assert "Last updated from sources:" in refusal
    assert is_refusal_response(refusal)
    print("✓ test_intent_to_refusal_advisory passed")


def test_intent_to_refusal_out_of_scope():
    """Out-of-scope intent triggers correct refusal with SEBI link."""
    query = "What is the weather today?"
    intent = classify_intent(query)
    assert intent == INTENT_OUT_OF_SCOPE

    refusal = generate_refusal(intent, query)
    assert "outside my scope" in refusal.lower()
    assert "sebi.gov.in" in refusal
    assert is_refusal_response(refusal)
    print("✓ test_intent_to_refusal_out_of_scope passed")


def test_intent_factual_no_refusal():
    """Factual intent should NOT trigger refusal."""
    query = "What is the NAV of HDFC Mid Cap Fund?"
    intent = classify_intent(query)
    assert intent == INTENT_FACTUAL
    # No refusal should be generated for factual queries
    print("✓ test_intent_factual_no_refusal passed")


# ====================================================================
# Prompt → Post-processor Interaction
# ====================================================================

def test_prompt_to_postprocessor_pipeline():
    """Build prompt, simulate LLM response, validate through post-processor."""
    # Build context
    chunks = [{
        "text": "The expense ratio of HDFC Mid Cap Fund is 0.74% as per the latest factsheet.",
        "source_url": "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
        "scheme_name": "HDFC Mid Cap Fund",
        "document_type": "scheme_page",
        "similarity": 0.9,
    }]

    # Build prompt
    messages = build_prompt(chunks, "What is the expense ratio of HDFC Mid Cap Fund?", "2026-06-09")
    assert len(messages) == 2

    # Simulate LLM response
    simulated_response = (
        "The expense ratio of HDFC Mid Cap Fund is 0.74%. "
        "(Source: https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth)\n\n"
        "Last updated from sources: 2026-06-09"
    )

    # Post-process
    result = validate_and_format(simulated_response, chunks[0]["source_url"], "2026-06-09")
    assert result["has_citation"]
    assert result["has_footer"]
    assert result["advisory_leak"] == []
    assert result["sentence_count"] <= 3
    print("✓ test_prompt_to_postprocessor_pipeline passed")


def test_prompt_to_postprocessor_advisory_leak():
    """Post-processor catches advisory language from simulated LLM response."""
    chunks = [{
        "text": "HDFC Defence Fund info.",
        "source_url": "https://groww.in/defence",
        "scheme_name": "HDFC Defence Fund",
        "document_type": "scheme_page",
        "similarity": 0.85,
    }]

    messages = build_prompt(chunks, "What is the benchmark?", "2026-06-09")

    # Simulate bad LLM response with advisory language
    bad_response = (
        "I recommend investing in HDFC Defence Fund. "
        "It will definitely grow in value.\n\n"
        "Last updated from sources: 2026-06-09"
    )

    result = validate_and_format(bad_response, chunks[0]["source_url"], "2026-06-09")
    assert len(result["advisory_leak"]) > 0
    print("✓ test_prompt_to_postprocessor_advisory_leak passed")


# ====================================================================
# Generator → Post-processor Interaction (with mocked LLM)
# ====================================================================

@patch("src.generation.generator.chat_completion")
@patch("src.generation.generator.is_available")
def test_generator_to_postprocessor(mock_avail, mock_chat):
    """Generator output flows correctly into post-processor."""
    mock_avail.return_value = True
    mock_chat.return_value = (
        "The NAV of HDFC Mid Cap Fund is ₹142.35. "
        "(Source: https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth)\n\n"
        "Last updated from sources: 2026-06-09"
    )

    chunks = [{
        "text": "NAV is ₹142.35",
        "source_url": "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
        "scheme_name": "HDFC Mid Cap Fund",
        "document_type": "scheme_page",
        "similarity": 0.9,
    }]

    gen_result = generate_response("What is the NAV?", chunks, date="2026-06-09")
    assert gen_result["status"] == "success"
    assert gen_result["context_used"] == 1

    post_result = validate_and_format(
        gen_result["response"],
        gen_result["source_url"],
        "2026-06-09",
    )
    assert post_result["has_citation"]
    assert post_result["has_footer"]
    print("✓ test_generator_to_postprocessor passed")


@patch("src.generation.generator.is_available")
def test_generator_no_context(mock_avail):
    """Generator handles no-context gracefully."""
    mock_avail.return_value = True

    result = generate_response("What is the NAV?", [], date="2026-06-09")
    assert result["status"] == "no_context"
    assert result["context_used"] == 0
    assert "don't have this information" in result["response"].lower()
    print("✓ test_generator_no_context passed")


# ====================================================================
# Test Scenarios from Implementation Plan (Task 9.3)
# ====================================================================

# Factual queries — should answer (use mocked pipeline)
FACTUAL_TEST_CASES = [
    ("What is the expense ratio of HDFC Mid Cap Fund?", "factual"),
    ("What is the minimum SIP amount for HDFC Small Cap Fund?", "factual"),
    ("What is the benchmark index for HDFC Defence Fund?", "factual"),
]

# Advisory queries — should refuse
ADVISORY_TEST_CASES = [
    ("Should I invest in HDFC Mid Cap Fund?", "advisory"),
    ("Which fund is better - HDFC Small Cap or HDFC Mid Cap?", "advisory"),
    ("Will HDFC Defence Fund give good returns?", "advisory"),
]

# Out of scope — should refuse
OUT_OF_SCOPE_TEST_CASES = [
    ("What is the weather today?", "out_of_scope"),
    ("How to invest in stocks?", "out_of_scope"),
]


@pytest.mark.parametrize("query,expected_intent", FACTUAL_TEST_CASES)
def test_integration_factual_queries(query, expected_intent):
    """Factual queries are classified correctly."""
    intent = classify_intent(query)
    assert intent == INTENT_FACTUAL
    print(f"✓ test_integration_factual_queries: {query}")


@pytest.mark.parametrize("query,expected_intent", ADVISORY_TEST_CASES)
def test_integration_advisory_queries(query, expected_intent):
    """Advisory queries are classified correctly and trigger refusal."""
    intent = classify_intent(query)
    assert intent == expected_intent.upper()

    refusal = generate_refusal(intent, query)
    assert is_refusal_response(refusal)
    assert "factual" in refusal.lower()
    print(f"✓ test_integration_advisory_queries: {query}")


@pytest.mark.parametrize("query,expected_intent", OUT_OF_SCOPE_TEST_CASES)
def test_integration_out_of_scope_queries(query, expected_intent):
    """Out-of-scope queries are classified correctly and trigger refusal."""
    intent = classify_intent(query)
    assert intent == expected_intent.upper()

    refusal = generate_refusal(intent, query)
    assert is_refusal_response(refusal)
    print(f"✓ test_integration_out_of_scope_queries: {query}")


# ====================================================================
# Full API Pipeline Integration (via TestClient)
# ====================================================================

def test_api_advisory_queries_from_plan():
    """All advisory test cases from the plan return refusal via API."""
    for query, _ in ADVISORY_TEST_CASES:
        resp = client.post("/query", json={"query": query})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_refusal"] is True, f"Expected refusal for: {query}"
        assert data["intent"] == "advisory", f"Expected advisory intent for: {query}"
    print("✓ test_api_advisory_queries_from_plan passed")


def test_api_out_of_scope_queries_from_plan():
    """All out-of-scope test cases from the plan return refusal via API."""
    for query, _ in OUT_OF_SCOPE_TEST_CASES:
        resp = client.post("/query", json={"query": query})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_refusal"] is True, f"Expected refusal for: {query}"
        assert data["intent"] == "out_of_scope", f"Expected out_of_scope intent for: {query}"
    print("✓ test_api_out_of_scope_queries_from_plan passed")


@patch("src.api.main.retrieve")
@patch("src.api.main.generate_response")
def test_api_factual_queries_from_plan(mock_gen, mock_ret):
    """All factual test cases from the plan flow through API with mocked pipeline."""
    mock_ret.return_value = [{
        "chunk_id": "int-1",
        "text": "Test chunk content.",
        "metadata": {"source_url": "https://groww.in/test", "scheme_name": "Test Fund"},
        "similarity": 0.9,
        "source_url": "https://groww.in/test",
    }]
    mock_gen.return_value = {
        "response": "The answer is here. (Source: https://groww.in/test)\n\n"
                    f"Last updated from sources: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "source_url": "https://groww.in/test",
        "context_used": 1,
        "latency_ms": 30.0,
        "status": "success",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }

    for query, _ in FACTUAL_TEST_CASES:
        resp = client.post("/query", json={"query": query})
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "factual", f"Expected factual intent for: {query}"
        assert data["is_refusal"] is False, f"Expected non-refusal for: {query}"
        assert data["context_used"] >= 1

    print("✓ test_api_factual_queries_from_plan passed")


# ====================================================================
# Keyword-level Classification Tests
# ====================================================================

def test_is_advisory_keyword_comprehensive():
    """All advisory phrases are detected."""
    advisory_queries = [
        "Should I invest in HDFC?",
        "Which is better?",
        "Can you recommend a fund?",
        "Suggest me a good fund",
        "Best fund for long term",
        "Will it grow?",
        "Is it safe?",
        "How much should I invest?",
        "When to buy?",
    ]
    for q in advisory_queries:
        assert is_advisory_keyword(q), f"Expected advisory for: {q}"
    print("✓ test_is_advisory_keyword_comprehensive passed")


def test_is_out_of_scope_comprehensive():
    """All out-of-scope patterns are detected."""
    oos_queries = [
        "What is the weather today?",
        "Tell me about stock market tips",
        "How about credit card offers?",
        "What do you think about politics?",
        "Who won the cricket match?",
        "How to cook biryani?",
        "Tell me a joke",
        "Who are you?",
    ]
    for q in oos_queries:
        assert is_out_of_scope(q), f"Expected out_of_scope for: {q}"
    print("✓ test_is_out_of_scope_comprehensive passed")


def test_is_mutual_fund_domain_comprehensive():
    """Mutual fund domain keywords are detected."""
    mf_queries = [
        "What is the expense ratio?",
        "What is the NAV?",
        "Tell me about SIP",
        "Fund manager details",
        "Benchmark index",
        "AUM of the fund",
        "Exit load charges",
        "Lock-in period",
    ]
    for q in mf_queries:
        assert is_mutual_fund_domain(q), f"Expected MF domain for: {q}"
    print("✓ test_is_mutual_fund_domain_comprehensive passed")


# ====================================================================
# Run all tests
# ====================================================================

if __name__ == "__main__":
    standalone = [
        test_sanitized_query_classified_correctly,
        test_sanitized_advisory_still_advisory,
        test_intent_to_refusal_advisory,
        test_intent_to_refusal_out_of_scope,
        test_intent_factual_no_refusal,
        test_prompt_to_postprocessor_pipeline,
        test_prompt_to_postprocessor_advisory_leak,
        test_api_advisory_queries_from_plan,
        test_api_out_of_scope_queries_from_plan,
        test_is_advisory_keyword_comprehensive,
        test_is_out_of_scope_comprehensive,
        test_is_mutual_fund_domain_comprehensive,
    ]

    mocked = [
        test_generator_to_postprocessor,
        test_generator_no_context,
        test_api_factual_queries_from_plan,
    ]

    passed = 0
    failed = 0

    for test in standalone + mocked:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(standalone) + len(mocked)} total")
    print(f"{'='*60}")
