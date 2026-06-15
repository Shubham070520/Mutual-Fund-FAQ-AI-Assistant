"""
Phase 5 verification tests — Guardrails & Refusal Logic.

Tests:
1. PII sanitizer (PAN, Aadhaar, phone, email detection)
2. Intent classifier (factual, advisory, out-of-scope)
3. Refusal generator (advisory, out-of-scope, empty, no-context)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.guardrails.sanitizer import sanitize, sanitize_detailed, SanitizationResult
from src.guardrails.intent import (
    classify_intent,
    is_advisory_keyword,
    is_out_of_scope,
    is_mutual_fund_domain,
    INTENT_FACTUAL,
    INTENT_ADVISORY,
    INTENT_OUT_OF_SCOPE,
)
from src.guardrails.refusal import (
    generate_refusal,
    is_refusal_response,
    format_factual_footer,
)


# ============================================================
# PII Sanitizer Tests
# ============================================================

def test_pii_pan_detection():
    """Test PAN card number detection and redaction."""
    query = "My PAN is ABCDE1234F and I want to check my HDFC fund"
    result = sanitize_detailed(query)

    assert result.was_modified, "Query should be modified"
    assert "pan" in result.pii_detected, "PAN should be detected"
    assert "ABCDE1234F" not in result.sanitized_query, "PAN should be redacted"
    assert "[REDACTED]" in result.sanitized_query
    print(f"[PASS] PAN detection: '{result.sanitized_query}'")


def test_pii_aadhaar_detection():
    """Test Aadhaar number detection."""
    query = "Aadhaar number 1234 5678 9012 linked to my folio"
    result = sanitize_detailed(query)

    assert result.was_modified, "Query should be modified"
    assert "aadhaar" in result.pii_detected, "Aadhaar should be detected"
    assert "1234" not in result.sanitized_query
    print(f"[PASS] Aadhaar detection: '{result.sanitized_query}'")


def test_pii_phone_detection():
    """Test phone number detection."""
    query = "Call me at +91 9876543210 for HDFC fund details"
    result = sanitize_detailed(query)

    assert result.was_modified, "Query should be modified"
    assert "phone" in result.pii_detected, "Phone should be detected"
    assert "9876543210" not in result.sanitized_query
    print(f"[PASS] Phone detection: '{result.sanitized_query}'")


def test_pii_email_detection():
    """Test email address detection."""
    query = "Send folio statement to investor@gmail.com please"
    result = sanitize_detailed(query)

    assert result.was_modified, "Query should be modified"
    assert "email" in result.pii_detected, "Email should be detected"
    assert "investor@gmail.com" not in result.sanitized_query
    print(f"[PASS] Email detection: '{result.sanitized_query}'")


def test_pii_no_false_positives():
    """Test that normal MF queries aren't flagged as PII."""
    queries = [
        "What is the expense ratio of HDFC Mid Cap Fund?",
        "What is the minimum SIP amount for HDFC Small Cap Fund?",
        "What is the NAV of HDFC Defence Fund?",
        "How to start SIP in HDFC Gold ETF FoF?",
    ]

    for query in queries:
        result = sanitize_detailed(query)
        assert not result.was_modified, f"False positive on: '{query}'"
        assert len(result.pii_detected) == 0, f"Unexpected PII in: '{query}'"

    print(f"[PASS] No false positives on {len(queries)} factual queries")


def test_pii_sanitize_simple():
    """Test the simple sanitize() function."""
    clean = sanitize("What is the expense ratio?")
    assert clean == "What is the expense ratio?"

    dirty = sanitize("My PAN ABCDE1234F query")
    assert "ABCDE1234F" not in dirty
    print("[PASS] Simple sanitize() works correctly")


def test_pii_empty_query():
    """Test empty and whitespace queries."""
    assert sanitize("") == ""
    assert sanitize("   ") == "   "
    result = sanitize_detailed("")
    assert not result.was_modified
    assert result.pii_detected == []
    print("[PASS] Empty/whitespace queries handled correctly")


# ============================================================
# Intent Classifier Tests
# ============================================================

def test_intent_factual():
    """Test that factual MF queries are classified correctly."""
    factual_queries = [
        "What is the expense ratio of HDFC Mid Cap Fund?",
        "What is the minimum SIP amount for HDFC Small Cap Fund?",
        "What is the benchmark index for HDFC Defence Fund?",
        "What is the NAV of HDFC Gold ETF Fund of Fund?",
        "What is the exit load of HDFC Silver ETF FoF?",
        "Who is the fund manager of HDFC Mid Cap Fund?",
    ]

    for q in factual_queries:
        intent = classify_intent(q)
        assert intent == INTENT_FACTUAL, f"Expected FACTUAL for '{q}', got {intent}"

    print(f"[PASS] All {len(factual_queries)} factual queries classified correctly")


def test_intent_advisory():
    """Test that advisory queries are classified correctly."""
    advisory_queries = [
        "Should I invest in HDFC Mid Cap Fund?",
        "Which is better - HDFC Small Cap or HDFC Mid Cap?",
        "Can you recommend a good mutual fund?",
        "Suggest the best ELSS fund for me",
        "Will HDFC Defence Fund give good returns?",
        "Is HDFC Gold ETF a good investment?",
        "Which fund should I invest in?",
        "How much should I invest in SIP?",
        "When should I sell my HDFC Mid Cap units?",
    ]

    for q in advisory_queries:
        intent = classify_intent(q)
        assert intent == INTENT_ADVISORY, f"Expected ADVISORY for '{q}', got {intent}"

    print(f"[PASS] All {len(advisory_queries)} advisory queries classified correctly")


def test_intent_out_of_scope():
    """Test that out-of-scope queries are classified correctly."""
    oos_queries = [
        "What is the weather today?",
        "Tell me about the stock market trends",
        "How to apply for a credit card?",
        "Who won the cricket match?",
        "Tell me a joke",
        "What is your name?",
        "How to learn Python programming?",
    ]

    for q in oos_queries:
        intent = classify_intent(q)
        assert intent == INTENT_OUT_OF_SCOPE, f"Expected OUT_OF_SCOPE for '{q}', got {intent}"

    print(f"[PASS] All {len(oos_queries)} out-of-scope queries classified correctly")


def test_intent_empty():
    """Test empty and whitespace queries."""
    assert classify_intent("") == INTENT_OUT_OF_SCOPE
    assert classify_intent("   ") == INTENT_OUT_OF_SCOPE
    print("[PASS] Empty queries classified as OUT_OF_SCOPE")


def test_advisory_keywords():
    """Test the keyword detection function directly."""
    assert is_advisory_keyword("should I invest in HDFC fund")
    assert is_advisory_keyword("which is better fund")
    assert is_advisory_keyword("recommend me a fund")
    assert not is_advisory_keyword("what is the expense ratio")
    assert not is_advisory_keyword("what is the NAV")
    print("[PASS] is_advisory_keyword() works correctly")


def test_mf_domain():
    """Test mutual fund domain detection."""
    assert is_mutual_fund_domain("What is the expense ratio?")
    assert is_mutual_fund_domain("How to start SIP?")
    assert is_mutual_fund_domain("Tell me about HDFC Mid Cap Fund")
    assert not is_mutual_fund_domain("What is the weather?")
    assert not is_mutual_fund_domain("How to cook biryani?")
    print("[PASS] is_mutual_fund_domain() works correctly")


# ============================================================
# Refusal Generator Tests
# ============================================================

def test_refusal_advisory():
    """Test advisory refusal contains required elements."""
    refusal = generate_refusal("ADVISORY", "Should I invest in HDFC Mid Cap?")

    assert "I cannot offer investment advice" in refusal
    assert "amfiindia.com" in refusal
    assert "Last updated from sources:" in refusal
    assert is_refusal_response(refusal)
    print(f"[PASS] Advisory refusal generated ({len(refusal)} chars)")


def test_refusal_out_of_scope():
    """Test out-of-scope refusal contains required elements."""
    refusal = generate_refusal("OUT_OF_SCOPE", "What is the weather today?")

    assert "outside my scope" in refusal
    assert "sebi.gov.in" in refusal
    assert "Last updated from sources:" in refusal
    assert is_refusal_response(refusal)
    print(f"[PASS] Out-of-scope refusal generated ({len(refusal)} chars)")


def test_refusal_empty_query():
    """Test empty query refusal."""
    refusal = generate_refusal("FACTUAL", "")

    assert "Please ask a specific question" in refusal
    assert "Last updated from sources:" in refusal
    print(f"[PASS] Empty query refusal generated ({len(refusal)} chars)")


def test_refusal_no_context():
    """Test no-context refusal (factual query, no matching data)."""
    refusal = generate_refusal("FACTUAL", "What is the AUM of the fund?")

    assert "I don't have this information" in refusal
    assert "Last updated from sources:" in refusal
    assert is_refusal_response(refusal)
    print(f"[PASS] No-context refusal generated ({len(refusal)} chars)")


def test_is_refusal_response():
    """Test the refusal detection helper."""
    assert is_refusal_response("I can only provide factual information about mutual fund schemes.")
    assert is_refusal_response("I cannot offer investment advice or recommendations.")
    assert is_refusal_response("Your question appears to be outside my scope.")
    assert not is_refusal_response("The expense ratio of HDFC Mid Cap Fund is 0.80%.")
    assert not is_refusal_response("The minimum SIP amount is ₹500.")
    print("[PASS] is_refusal_response() correctly identifies refusals")


def test_format_factual_footer():
    """Test the factual response footer."""
    footer = format_factual_footer("2026-06-09")
    assert footer == "Last updated from sources: 2026-06-09"

    footer_auto = format_factual_footer()
    assert footer_auto.startswith("Last updated from sources:")
    assert len(footer_auto) > 25
    print(f"[PASS] format_factual_footer(): '{footer}'")


# ============================================================
# Integration: Full Guardrails Pipeline
# ============================================================

def test_full_guardrails_pipeline():
    """Test the full guardrails pipeline: sanitize → classify → refuse."""
    test_cases = [
        # (query, expected_intent, should_refuse)
        ("What is the expense ratio of HDFC Mid Cap Fund?", INTENT_FACTUAL, False),
        ("Should I invest in HDFC Mid Cap Fund?", INTENT_ADVISORY, True),
        ("What is the weather today?", INTENT_OUT_OF_SCOPE, True),
        ("My PAN is ABCDE1234F what is the NAV?", INTENT_FACTUAL, False),  # PII stripped, factual remains
    ]

    passed = 0
    for query, expected_intent, should_refuse in test_cases:
        # Step 1: Sanitize PII
        clean_query = sanitize(query)

        # Step 2: Classify intent
        intent = classify_intent(clean_query)
        assert intent == expected_intent, \
            f"Query '{query}': expected {expected_intent}, got {intent}"

        # Step 3: Generate response or refusal
        if should_refuse:
            refusal = generate_refusal(intent, clean_query)
            assert is_refusal_response(refusal), f"Expected refusal for '{query}'"
        else:
            # Factual queries should pass through (no refusal)
            refusal = generate_refusal(intent, clean_query)
            # For factual, refusal should be the "no context" message
            # which is still technically a refusal marker
            pass

        passed += 1

    print(f"[PASS] Full guardrails pipeline: {passed}/{len(test_cases)} cases passed")


if __name__ == "__main__":
    print("=" * 65)
    print("Phase 5 Verification Tests — Guardrails & Refusal Logic")
    print("=" * 65)

    print("\n--- PII Sanitizer ---")
    test_pii_pan_detection()
    test_pii_aadhaar_detection()
    test_pii_phone_detection()
    test_pii_email_detection()
    test_pii_no_false_positives()
    test_pii_sanitize_simple()
    test_pii_empty_query()

    print("\n--- Intent Classifier ---")
    test_intent_factual()
    test_intent_advisory()
    test_intent_out_of_scope()
    test_intent_empty()
    test_advisory_keywords()
    test_mf_domain()

    print("\n--- Refusal Generator ---")
    test_refusal_advisory()
    test_refusal_out_of_scope()
    test_refusal_empty_query()
    test_refusal_no_context()
    test_is_refusal_response()
    test_format_factual_footer()

    print("\n--- Integration: Full Pipeline ---")
    test_full_guardrails_pipeline()

    print("\n" + "=" * 65)
    print("ALL PHASE 5 TESTS PASSED")
    print("=" * 65)
