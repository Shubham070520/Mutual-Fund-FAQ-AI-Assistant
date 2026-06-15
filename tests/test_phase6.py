"""
Phase 6 verification tests — LLM Integration & Generation.

Tests (no API key required for most):
1. Prompt template building and structure
2. Context block formatting
3. Post-processor validation (sentence count, citation, footer, advisory leak)
4. Generator behavior with no context / no API key
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.prompts import (
    build_prompt,
    build_context_block,
    build_no_context_prompt,
    SYSTEM_PROMPT,
)
from src.generation.postprocessor import (
    validate_and_format,
    count_sentences,
    _check_advisory_leak,
    _extract_urls,
    _ensure_footer,
    _ensure_citation,
    FOOTER_PATTERN,
)
from src.generation.generator import generate_response, FALLBACK_RESPONSE
from src.generation.llm import is_available

# --- Test context data ---
SAMPLE_CHUNKS = [
    {
        "text": "HDFC Mid Cap Fund Direct Plan has an expense ratio of 0.80%.",
        "source_url": "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
        "scheme_name": "HDFC Mid Cap Fund",
        "document_type": "factsheet",
        "similarity": 0.89,
    },
    {
        "text": "The minimum SIP amount for HDFC Mid Cap Fund is ₹500.",
        "source_url": "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
        "scheme_name": "HDFC Mid Cap Fund",
        "document_type": "faq",
        "similarity": 0.75,
    },
]


# ============================================================
# Prompt Template Tests
# ============================================================

def test_system_prompt():
    """Test that system prompt has required rules."""
    assert "facts-only" in SYSTEM_PROMPT.lower()
    assert "ONLY the provided context" in SYSTEM_PROMPT
    assert "MAXIMUM" in SYSTEM_PROMPT
    assert "EXACTLY ONE source URL" in SYSTEM_PROMPT
    assert "Last updated from sources" in SYSTEM_PROMPT
    assert "Do NOT provide investment advice" in SYSTEM_PROMPT
    print("[PASS] System prompt contains all required rules")


def test_build_context_block():
    """Test context block formatting."""
    block = build_context_block(SAMPLE_CHUNKS)

    assert "[Source 1]" in block
    assert "[Source 2]" in block
    assert "HDFC Mid Cap Fund" in block
    assert "expense ratio" in block
    assert "groww.in" in block
    print(f"[PASS] Context block: {len(block)} chars, 2 sources")


def test_build_context_block_empty():
    """Test empty context block."""
    block = build_context_block([])
    assert block == "[No context available]"
    print("[PASS] Empty context block returns placeholder")


def test_build_prompt():
    """Test full prompt construction."""
    messages = build_prompt(SAMPLE_CHUNKS, "What is the expense ratio?", date="2026-06-09")

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "expense ratio" in messages[1]["content"]
    assert "2026-06-09" in messages[1]["content"]
    assert "CONTEXT" in messages[1]["content"]
    print("[PASS] Prompt built: system + user messages with context")


def test_build_no_context_prompt():
    """Test no-context prompt."""
    messages = build_no_context_prompt("What is the NAV?", date="2026-06-09")

    assert len(messages) == 2
    assert "don't have this information" in messages[1]["content"]
    print("[PASS] No-context prompt built correctly")


# ============================================================
# Post-Processor Tests
# ============================================================

def test_count_sentences():
    """Test sentence counting."""
    assert count_sentences("Hello. World.") == 2
    assert count_sentences("One sentence.") == 1
    assert count_sentences("A. B. C. Last updated from sources: 2026-06-09") == 3
    assert count_sentences("") == 0
    print("[PASS] Sentence counting works correctly")


def test_postprocessor_good_response():
    """Test validation of a well-formed response."""
    response = (
        "The expense ratio of HDFC Mid Cap Fund is 0.80%. "
        "Source: https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth\n\n"
        "Last updated from sources: 2026-06-09"
    )

    result = validate_and_format(
        response=response,
        source_url="https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
        date="2026-06-09",
    )

    assert result["sentence_count"] <= 3
    assert result["has_citation"] is True
    assert result["has_footer"] is True
    assert len(result["advisory_leak"]) == 0
    assert len(result["warnings"]) == 0
    print("[PASS] Well-formed response passes all validations")


def test_postprocessor_trims_long_response():
    """Test that responses with too many sentences get trimmed."""
    response = (
        "First sentence. Second sentence. Third sentence. "
        "Fourth sentence. Fifth sentence. "
        "Source: https://groww.in/test\n\n"
        "Last updated from sources: 2026-06-09"
    )

    result = validate_and_format(
        response=response,
        source_url="https://groww.in/test",
        date="2026-06-09",
        max_sentences=3,
    )

    assert result["sentence_count"] <= 3, f"Expected ≤3 sentences, got {result['sentence_count']}"
    assert any("Trimmed" in w for w in result["warnings"]), "Should have trim warning"
    print(f"[PASS] Long response trimmed: {result['sentence_count']} sentences, warnings={result['warnings']}")


def test_postprocessor_adds_footer():
    """Test that missing footer gets appended."""
    response = "The NAV is ₹150.50. Source: https://groww.in/test"

    result = validate_and_format(
        response=response,
        source_url="https://groww.in/test",
        date="2026-06-09",
    )

    assert result["has_footer"] is True
    assert "2026-06-09" in result["formatted_response"]
    assert any("Footer" in w for w in result["warnings"])
    print(f"[PASS] Footer appended: '{result['formatted_response'][-50:]}'")


def test_postprocessor_adds_citation():
    """Test that missing citation gets appended."""
    response = "The expense ratio is 0.80%.\n\nLast updated from sources: 2026-06-09"

    result = validate_and_format(
        response=response,
        source_url="https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
        date="2026-06-09",
    )

    assert result["has_citation"] is True
    assert "groww.in" in result["formatted_response"]
    print(f"[PASS] Citation appended to response")


def test_postprocessor_detects_advisory_leak():
    """Test that advisory language in responses is flagged."""
    response = (
        "I recommend investing in HDFC Mid Cap Fund. "
        "It is a good investment with guaranteed returns. "
        "Source: https://groww.in/test\n\n"
        "Last updated from sources: 2026-06-09"
    )

    result = validate_and_format(
        response=response,
        source_url="https://groww.in/test",
        date="2026-06-09",
    )

    assert len(result["advisory_leak"]) > 0, "Should detect advisory language"
    print(f"[PASS] Advisory leak detected: {result['advisory_leak']}")


def test_check_advisory_leak_clean():
    """Test that clean factual responses don't trigger advisory detection."""
    clean_response = (
        "The expense ratio of HDFC Mid Cap Fund is 0.80%. "
        "The minimum SIP amount is ₹500."
    )

    leaks = _check_advisory_leak(clean_response)
    assert len(leaks) == 0, f"False positive advisory: {leaks}"
    print("[PASS] Clean response has no advisory leaks")


def test_extract_urls():
    """Test URL extraction from text."""
    text = "Visit https://groww.in/test and http://example.com for more."
    urls = _extract_urls(text)
    assert len(urls) == 2
    assert "https://groww.in/test" in urls
    print(f"[PASS] Extracted {len(urls)} URLs from text")


# ============================================================
# Generator Tests (no API key needed)
# ============================================================

def test_generator_no_context():
    """Test generator returns 'no info' when context is empty."""
    result = generate_response(
        "What is the expense ratio?",
        context_chunks=[],
        date="2026-06-09",
    )

    assert result["status"] == "no_context"
    assert "don't have this information" in result["response"]
    assert "2026-06-09" in result["response"]
    assert result["context_used"] == 0
    print(f"[PASS] No-context generator: status={result['status']}")


def test_generator_no_api_key():
    """Test generator fallback when API key is not set."""
    if is_available():
        print("[SKIP] API key is set — skipping no-key fallback test")
        return

    result = generate_response(
        "What is the expense ratio?",
        context_chunks=SAMPLE_CHUNKS,
        date="2026-06-09",
    )

    assert result["status"] == "error"
    assert "unable to process" in result["response"].lower() or "try again" in result["response"].lower()
    assert result["context_used"] == 2
    print(f"[PASS] No-API-key fallback: status={result['status']}")


def test_generator_metadata():
    """Test that generator returns correct metadata structure."""
    result = generate_response(
        "What is the expense ratio?",
        context_chunks=SAMPLE_CHUNKS,
        date="2026-06-09",
    )

    assert "response" in result
    assert "source_url" in result
    assert "context_used" in result
    assert "latency_ms" in result
    assert "status" in result
    assert "date" in result
    assert result["source_url"] == SAMPLE_CHUNKS[0]["source_url"]
    assert result["context_used"] == 2
    assert result["date"] == "2026-06-09"
    print(f"[PASS] Generator metadata: url={result['source_url'][:40]}..., chunks={result['context_used']}")


# ============================================================
# Integration: Post-Process Generated Response
# ============================================================

def test_end_to_end_post_processing():
    """Test post-processing a simulated LLM response."""
    # Simulate what an LLM might generate
    raw_response = (
        "The expense ratio of HDFC Mid Cap Fund Direct Plan is 0.80%. "
        "This information is available at https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth\n\n"
        "Last updated from sources: 2026-06-09"
    )

    result = validate_and_format(
        response=raw_response,
        source_url="https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
        date="2026-06-09",
    )

    assert result["sentence_count"] <= 3
    assert result["has_citation"]
    assert result["has_footer"]
    assert not result["advisory_leak"]
    assert not result["warnings"]
    print(f"[PASS] End-to-end post-processing: clean response, no warnings")


if __name__ == "__main__":
    print("=" * 65)
    print("Phase 6 Verification Tests — LLM Integration & Generation")
    print("=" * 65)

    print("\n--- Prompt Templates ---")
    test_system_prompt()
    test_build_context_block()
    test_build_context_block_empty()
    test_build_prompt()
    test_build_no_context_prompt()

    print("\n--- Post-Processor ---")
    test_count_sentences()
    test_postprocessor_good_response()
    test_postprocessor_trims_long_response()
    test_postprocessor_adds_footer()
    test_postprocessor_adds_citation()
    test_postprocessor_detects_advisory_leak()
    test_check_advisory_leak_clean()
    test_extract_urls()

    print("\n--- Generator (No API Key Required) ---")
    test_generator_no_context()
    test_generator_no_api_key()
    test_generator_metadata()

    print("\n--- Integration ---")
    test_end_to_end_post_processing()

    print("\n" + "=" * 65)
    print("ALL PHASE 6 TESTS PASSED")
    print("=" * 65)
