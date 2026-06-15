"""
Phase 9 — Unit Tests.

Tests individual functions across all modules for correctness:
- Scraper: scraping logic, PDF routing
- Chunker: splitting, metadata, classification
- Cleaner: unicode, whitespace, boilerplate, headings
- Post-processor: sentence count, trim, citation, footer, advisory leak
- Prompts: context block, prompt building
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ====================================================================
# Imports
# ====================================================================

from src.processing.chunker import (
    chunk_text,
    chunk_documents,
    create_splitter,
    _classify_chunk_content,
    _get_section_heading,
)
from src.ingestion.cleaner import (
    clean_text,
    normalize_unicode,
    normalize_whitespace,
    remove_boilerplate,
    remove_urls_and_emails,
    clean_section_headings,
)
from src.generation.postprocessor import (
    count_sentences,
    validate_and_format,
    _extract_urls,
    _check_advisory_leak,
    _ensure_footer,
    _ensure_citation,
    _trim_to_sentences,
    _strip_footer,
)
from src.generation.prompts import (
    build_context_block,
    build_prompt,
    build_no_context_prompt,
    SYSTEM_PROMPT,
)
from src.ingestion.scraper import (
    _compute_hash,
    _guess_pdf_type,
    scrape_url,
)
from src.guardrails.sanitizer import sanitize, sanitize_detailed
from src.guardrails.refusal import (
    generate_refusal,
    is_refusal_response,
    format_factual_footer,
)


# ====================================================================
# Scraper Unit Tests
# ====================================================================

def test_compute_hash_deterministic():
    """Same input always produces the same hash."""
    h1 = _compute_hash("hello world")
    h2 = _compute_hash("hello world")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest
    print("✓ test_compute_hash_deterministic passed")


def test_compute_hash_different_inputs():
    """Different inputs produce different hashes."""
    h1 = _compute_hash("hello")
    h2 = _compute_hash("world")
    assert h1 != h2
    print("✓ test_compute_hash_different_inputs passed")


def test_guess_pdf_type_factsheet():
    """PDF type detection for factsheet URLs."""
    assert _guess_pdf_type("https://hdfc.com/docs/factsheet-june.pdf") == "factsheet"
    print("✓ test_guess_pdf_type_factsheet passed")


def test_guess_pdf_type_kim():
    """PDF type detection for KIM URLs."""
    assert _guess_pdf_type("https://hdfc.com/docs/key-information-memorandum.pdf") == "kim"
    print("✓ test_guess_pdf_type_kim passed")


def test_guess_pdf_type_sid():
    """PDF type detection for SID URLs."""
    assert _guess_pdf_type("https://hdfc.com/docs/scheme-information-document.pdf") == "sid"
    print("✓ test_guess_pdf_type_sid passed")


def test_guess_pdf_type_unknown():
    """PDF type detection for unknown URLs."""
    assert _guess_pdf_type("https://hdfc.com/docs/report.pdf") == "pdf_document"
    print("✓ test_guess_pdf_type_unknown passed")


def test_scrape_url_routes_to_html():
    """scrape_url routes HTML format correctly."""
    with patch("src.ingestion.scraper.scrape_html") as mock_html, \
         patch("src.ingestion.scraper.time.sleep"):
        mock_html.return_value = {
            "text": "test content",
            "metadata": {"source_url": "https://example.com"},
        }
        result = scrape_url({
            "url": "https://example.com",
            "format": "html",
            "scheme": "Test Fund",
            "type": "scheme_page",
        })
        mock_html.assert_called_once_with("https://example.com")
        assert result["metadata"]["scheme_name"] == "Test Fund"
    print("✓ test_scrape_url_routes_to_html passed")


def test_scrape_url_routes_to_pdf():
    """scrape_url routes PDF format correctly."""
    with patch("src.ingestion.scraper.extract_pdf") as mock_pdf, \
         patch("src.ingestion.scraper.time.sleep"):
        mock_pdf.return_value = {
            "text": "pdf content",
            "metadata": {"source_url": "https://example.com/doc.pdf"},
        }
        result = scrape_url({
            "url": "https://example.com/doc.pdf",
            "format": "pdf",
            "scheme": "Test Fund",
            "type": "factsheet",
        })
        mock_pdf.assert_called_once()
        assert result["metadata"]["document_type"] == "factsheet"
    print("✓ test_scrape_url_routes_to_pdf passed")


# ====================================================================
# Chunker Unit Tests
# ====================================================================

def test_create_splitter_defaults():
    """Default splitter has expected config."""
    splitter = create_splitter()
    assert splitter is not None
    print("✓ test_create_splitter_defaults passed")


def test_chunk_text_basic():
    """chunk_text produces non-empty chunks with metadata."""
    text = "This is a test paragraph.\n\nThis is another test paragraph."
    meta = {"source_url": "https://test.com", "scheme_name": "Test Fund"}
    chunks = chunk_text(text, meta, chunk_size=50, chunk_overlap=10)

    assert len(chunks) >= 1
    for c in chunks:
        assert "chunk_id" in c
        assert "text" in c
        assert "metadata" in c
        assert c["metadata"]["source_url"] == "https://test.com"
        assert c["metadata"]["scheme_name"] == "Test Fund"
    print("✓ test_chunk_text_basic passed")


def test_chunk_text_empty_input():
    """Empty text produces no chunks."""
    chunks = chunk_text("", {"source_url": "x"})
    assert chunks == []
    print("✓ test_chunk_text_empty_input passed")


def test_chunk_text_metadata_enriched():
    """Each chunk gets chunk_index, chunk_total, content_types."""
    text = "NAV: ₹142.35\n\nExpense Ratio: 0.74%\n\nFund Manager: John Doe"
    meta = {"source_url": "https://test.com"}
    chunks = chunk_text(text, meta, chunk_size=60, chunk_overlap=10)

    for c in chunks:
        m = c["metadata"]
        assert "chunk_index" in m
        assert "chunk_total" in m
        assert "content_types" in m
        assert "has_structured_data" in m
        assert "char_count" in m
    print("✓ test_chunk_text_metadata_enriched passed")


def test_classify_chunk_content_nav():
    """NAV content is tagged correctly."""
    types = _classify_chunk_content("The NAV of HDFC Mid Cap Fund is ₹142.35")
    assert "nav" in types
    print("✓ test_classify_chunk_content_nav passed")


def test_classify_chunk_content_sip():
    """SIP content is tagged correctly."""
    types = _classify_chunk_content("The minimum SIP amount is ₹100 per month")
    assert "sip" in types
    print("✓ test_classify_chunk_content_sip passed")


def test_classify_chunk_content_fund_manager():
    """Fund manager content is tagged correctly."""
    types = _classify_chunk_content("The fund is managed by Chirag Setalvad since 2020")
    assert "fund_manager" in types
    print("✓ test_classify_chunk_content_fund_manager passed")


def test_classify_chunk_content_expense_ratio():
    """Expense ratio content is tagged correctly."""
    types = _classify_chunk_content("The expense ratio of this fund is 0.74%")
    assert "expense_ratio" in types
    print("✓ test_classify_chunk_content_expense_ratio passed")


def test_classify_chunk_content_faq():
    """FAQ content is tagged correctly."""
    types = _classify_chunk_content("Q: What is the minimum SIP? A: The minimum SIP is ₹100")
    assert "faq" in types
    print("✓ test_classify_chunk_content_faq passed")


def test_classify_chunk_content_empty():
    """Generic content has no tags."""
    types = _classify_chunk_content("This is a generic sentence about nothing special.")
    assert types == []
    print("✓ test_classify_chunk_content_empty passed")


def test_get_section_heading():
    """Section heading detection finds the nearest heading."""
    text = "## Key Metrics\n\nSome data here.\n\nMore content follows."
    heading = _get_section_heading(text, 32)
    assert heading == "Key Metrics"
    print("✓ test_get_section_heading passed")


def test_get_section_heading_none():
    """No heading returns None."""
    text = "Just a plain paragraph without any headings."
    heading = _get_section_heading(text, 20)
    assert heading is None
    print("✓ test_get_section_heading_none passed")


def test_chunk_documents_skips_empty():
    """chunk_documents skips documents with empty text."""
    docs = [
        {"text": "", "metadata": {"source_url": "a"}},
        {"text": "Valid content here.", "metadata": {"source_url": "b"}},
    ]
    chunks = chunk_documents(docs)
    assert all(c["metadata"]["source_url"] == "b" for c in chunks)
    print("✓ test_chunk_documents_skips_empty passed")


# ====================================================================
# Cleaner Unit Tests
# ====================================================================

def test_normalize_unicode_special_chars():
    """Unicode special characters are normalized."""
    text = "\u201cHello\u201d \u2014 \u2018world\u2019"
    result = normalize_unicode(text)
    assert '"' in result  # left/right double quotes → plain
    assert "'" in result  # left/right single quotes → plain
    assert "-" in result  # em dash → hyphen
    print("✓ test_normalize_unicode_special_chars passed")


def test_normalize_unicode_bom():
    """BOM and zero-width space are removed."""
    text = "\ufeffHello\u200bWorld"
    result = normalize_unicode(text)
    assert "\ufeff" not in result
    assert "\u200b" not in result
    assert "HelloWorld" in result
    print("✓ test_normalize_unicode_bom passed")


def test_normalize_whitespace():
    """Excessive whitespace is collapsed."""
    text = "hello    world\n\n\n\n\nnew paragraph"
    result = normalize_whitespace(text)
    assert "hello world" in result
    assert "\n\n\n" not in result
    print("✓ test_normalize_whitespace passed")


def test_normalize_whitespace_tabs():
    """Tabs are converted to spaces."""
    text = "col1\tcol2\tcol3"
    result = normalize_whitespace(text)
    assert "\t" not in result
    assert "col1 col2 col3" in result
    print("✓ test_normalize_whitespace_tabs passed")


def test_remove_boilerplate_cookie():
    """Cookie policy text is removed."""
    text = "Main content here.\nCookie Policy: We use cookies to improve experience.\nMore content."
    result = remove_boilerplate(text)
    assert "cookie" not in result.lower() or "Cookie Policy" not in result
    print("✓ test_remove_boilerplate_cookie passed")


def test_remove_boilerplate_legal():
    """Mutual fund legal disclaimer is removed."""
    text = "Fund data here.\nMutual Fund investments are subject to market risks, read all documents carefully.\nMore data."
    result = remove_boilerplate(text)
    assert "market risks" not in result.lower()
    print("✓ test_remove_boilerplate_legal passed")


def test_remove_urls_and_emails():
    """URLs and emails are stripped."""
    text = "Visit https://groww.in or email test@example.com for info."
    result = remove_urls_and_emails(text)
    assert "https" not in result
    assert "@" not in result
    print("✓ test_remove_urls_and_emails passed")


def test_clean_section_headings_preserves_structured():
    """Structured fund data lines are NOT converted to headings."""
    text = "NAV: ₹142.35\nFund Manager: John Doe\nQ: What is SIP?"
    result = clean_section_headings(text)
    assert "## NAV" not in result
    assert "## Fund Manager" not in result
    assert "## Q:" not in result
    print("✓ test_clean_section_headings_preserves_structured passed")


def test_clean_text_full_pipeline():
    """Full cleaning pipeline produces non-empty output."""
    text = "\ufeffHello   World\n\n\n\n## Section One\n\nContent here."
    result = clean_text(text)
    assert result != ""
    assert "\ufeff" not in result
    print("✓ test_clean_text_full_pipeline passed")


def test_clean_text_empty():
    """Empty input returns empty."""
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    print("✓ test_clean_text_empty passed")


# ====================================================================
# Post-processor Unit Tests
# ====================================================================

def test_count_sentences_basic():
    """Basic sentence counting."""
    assert count_sentences("Hello. World.") == 2
    assert count_sentences("One sentence.") == 1
    assert count_sentences("Wow! Really? Yes.") == 3
    print("✓ test_count_sentences_basic passed")


def test_count_sentences_excludes_footer():
    """Footer line is excluded from sentence count."""
    text = "The NAV is ₹142.\n\nLast updated from sources: 2026-06-09"
    assert count_sentences(text) == 1
    print("✓ test_count_sentences_excludes_footer passed")


def test_count_sentences_excludes_citation():
    """Parenthetical citations are excluded from count."""
    text = "The NAV is ₹142. (Source: https://groww.in/test)\n\nLast updated from sources: 2026-06-09"
    assert count_sentences(text) == 1
    print("✓ test_count_sentences_excludes_citation passed")


def test_trim_to_sentences():
    """Trimming to max sentences works correctly."""
    text = "First. Second. Third. Fourth.\n\nLast updated from sources: 2026-06-09"
    result = _trim_to_sentences(text, 3)
    assert "First." in result
    assert "Second." in result
    assert "Third." in result
    assert "Fourth." not in result
    assert "Last updated from sources" in result
    print("✓ test_trim_to_sentences passed")


def test_extract_urls():
    """URL extraction finds http/https URLs."""
    urls = _extract_urls("Visit https://groww.in or http://example.com for more.")
    assert len(urls) == 2
    assert "https://groww.in" in urls[0]
    print("✓ test_extract_urls passed")


def test_check_advisory_leak_clean():
    """No advisory language → empty list."""
    leaks = _check_advisory_leak("The NAV is ₹142.35 as of June 2026.")
    assert leaks == []
    print("✓ test_check_advisory_leak_clean passed")


def test_check_advisory_leak_detected():
    """Advisory language is detected."""
    leaks = _check_advisory_leak("I recommend you should invest in this fund.")
    assert len(leaks) > 0
    print("✓ test_check_advisory_leak_detected passed")


def test_ensure_footer_appends():
    """Missing footer is appended."""
    text = "The NAV is ₹142.35."
    result = _ensure_footer(text, "2026-06-09")
    assert "Last updated from sources: 2026-06-09" in result
    print("✓ test_ensure_footer_appends passed")


def test_ensure_footer_fixes_date():
    """Wrong date is corrected."""
    text = "The NAV is ₹142.35.\n\nLast updated from sources: 2025-01-01"
    result = _ensure_footer(text, "2026-06-09")
    assert "2026-06-09" in result
    assert "2025-01-01" not in result
    print("✓ test_ensure_footer_fixes_date passed")


def test_ensure_citation_appends():
    """Missing citation is appended."""
    text = "The NAV is ₹142.35.\n\nLast updated from sources: 2026-06-09"
    result = _ensure_citation(text, "https://groww.in/test")
    assert "https://groww.in/test" in result
    print("✓ test_ensure_citation_appends passed")


def test_ensure_citation_keeps_existing():
    """Existing citation is not duplicated."""
    text = "The NAV is ₹142.35 (Source: https://groww.in/test).\n\nLast updated from sources: 2026-06-09"
    result = _ensure_citation(text, "https://groww.in/test")
    assert result.count("groww.in/test") == 1
    print("✓ test_ensure_citation_keeps_existing passed")


def test_validate_and_format_full():
    """Full validation pipeline produces correct output."""
    response = "The NAV is ₹142.35. (Source: https://groww.in/test)\n\nLast updated from sources: 2026-06-09"
    result = validate_and_format(response, "https://groww.in/test", "2026-06-09")

    assert result["has_citation"] is True
    assert result["has_footer"] is True
    assert result["advisory_leak"] == []
    assert result["sentence_count"] <= 3
    print("✓ test_validate_and_format_full passed")


def test_validate_and_format_adds_missing():
    """Missing citation and footer are added."""
    response = "The NAV is ₹142.35."
    result = validate_and_format(response, "https://groww.in/test", "2026-06-09")

    assert result["has_citation"] is True
    assert result["has_footer"] is True
    assert "Last updated from sources: 2026-06-09" in result["formatted_response"]
    print("✓ test_validate_and_format_adds_missing passed")


# ====================================================================
# Prompts Unit Tests
# ====================================================================

def test_build_context_block_empty():
    """Empty context produces placeholder."""
    block = build_context_block([])
    assert "[No context available]" in block
    print("✓ test_build_context_block_empty passed")


def test_build_context_block_with_chunks():
    """Context block includes source info."""
    chunks = [{
        "text": "NAV is ₹142.35",
        "source_url": "https://groww.in/test",
        "scheme_name": "HDFC Mid Cap Fund",
        "document_type": "scheme_page",
        "similarity": 0.9,
    }]
    block = build_context_block(chunks)
    assert "HDFC Mid Cap Fund" in block
    assert "https://groww.in/test" in block
    assert "NAV is ₹142.35" in block
    print("✓ test_build_context_block_with_chunks passed")


def test_build_prompt_returns_messages():
    """build_prompt returns system + user messages."""
    chunks = [{
        "text": "NAV is ₹142.35",
        "source_url": "https://groww.in/test",
        "scheme_name": "Test",
        "document_type": "page",
        "similarity": 0.9,
    }]
    messages = build_prompt(chunks, "What is the NAV?", "2026-06-09")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "What is the NAV?" in messages[1]["content"]
    assert "2026-06-09" in messages[1]["content"]
    print("✓ test_build_prompt_returns_messages passed")


def test_build_no_context_prompt():
    """No-context prompt instructs LLM to say no info."""
    messages = build_no_context_prompt("What is NAV?", "2026-06-09")
    assert len(messages) == 2
    assert "No relevant context" in messages[1]["content"]
    assert "2026-06-09" in messages[1]["content"]
    print("✓ test_build_no_context_prompt passed")


def test_system_prompt_contains_rules():
    """System prompt has all required rules."""
    assert "factual" in SYSTEM_PROMPT.lower()
    assert "MAXIMUM" in SYSTEM_PROMPT or "maximum" in SYSTEM_PROMPT.lower()
    assert "source URL" in SYSTEM_PROMPT or "source url" in SYSTEM_PROMPT.lower()
    assert "Last updated from sources" in SYSTEM_PROMPT
    print("✓ test_system_prompt_contains_rules passed")


# ====================================================================
# Guardrails Unit Tests (supplemental)
# ====================================================================

def test_sanitize_no_pii():
    """Clean query passes through unchanged."""
    result = sanitize("What is the NAV of HDFC Mid Cap Fund?")
    assert result == "What is the NAV of HDFC Mid Cap Fund?"
    print("✓ test_sanitize_no_pii passed")


def test_sanitize_detailed_pan():
    """PAN card number is detected and redacted."""
    result = sanitize_detailed("My PAN is ABCDE1234F and I want NAV info")
    assert result.was_modified
    assert "pan" in result.pii_detected
    assert "[REDACTED]" in result.sanitized_query
    assert "ABCDE1234F" not in result.sanitized_query
    print("✓ test_sanitize_detailed_pan passed")


def test_sanitize_detailed_aadhaar():
    """Aadhaar number is detected and redacted."""
    result = sanitize_detailed("My Aadhaar is 1234 5678 9012 please help")
    assert result.was_modified
    assert "aadhaar" in result.pii_detected
    print("✓ test_sanitize_detailed_aadhaar passed")


def test_sanitize_empty():
    """Empty input returns unchanged."""
    result = sanitize_detailed("")
    assert not result.was_modified
    assert result.pii_detected == []
    print("✓ test_sanitize_empty passed")


def test_generate_refusal_advisory():
    """Advisory refusal contains AMFI link."""
    refusal = generate_refusal("ADVISORY", "should I invest?")
    assert "factual" in refusal.lower()
    assert "amfiindia.com" in refusal
    assert "Last updated from sources:" in refusal
    print("✓ test_generate_refusal_advisory passed")


def test_generate_refusal_out_of_scope():
    """Out-of-scope refusal contains SEBI link."""
    refusal = generate_refusal("OUT_OF_SCOPE", "what's the weather?")
    assert "outside my scope" in refusal.lower()
    assert "sebi.gov.in" in refusal
    print("✓ test_generate_refusal_out_of_scope passed")


def test_is_refusal_response_true():
    """Refusal text is detected."""
    assert is_refusal_response("I can only provide factual information about mutual fund schemes.")
    assert is_refusal_response("I don't have this information in my current sources.")
    print("✓ test_is_refusal_response_true passed")


def test_is_refusal_response_false():
    """Normal factual text is not flagged as refusal."""
    assert not is_refusal_response("The NAV of HDFC Mid Cap Fund is ₹142.35.")
    print("✓ test_is_refusal_response_false passed")


def test_format_factual_footer():
    """Footer is formatted correctly."""
    footer = format_factual_footer("2026-06-09")
    assert footer == "Last updated from sources: 2026-06-09"
    print("✓ test_format_factual_footer passed")


# ====================================================================
# Run all tests
# ====================================================================

if __name__ == "__main__":
    tests = [
        # Scraper
        test_compute_hash_deterministic,
        test_compute_hash_different_inputs,
        test_guess_pdf_type_factsheet,
        test_guess_pdf_type_kim,
        test_guess_pdf_type_sid,
        test_guess_pdf_type_unknown,
        test_scrape_url_routes_to_html,
        test_scrape_url_routes_to_pdf,
        # Chunker
        test_create_splitter_defaults,
        test_chunk_text_basic,
        test_chunk_text_empty_input,
        test_chunk_text_metadata_enriched,
        test_classify_chunk_content_nav,
        test_classify_chunk_content_sip,
        test_classify_chunk_content_fund_manager,
        test_classify_chunk_content_expense_ratio,
        test_classify_chunk_content_faq,
        test_classify_chunk_content_empty,
        test_get_section_heading,
        test_get_section_heading_none,
        test_chunk_documents_skips_empty,
        # Cleaner
        test_normalize_unicode_special_chars,
        test_normalize_unicode_bom,
        test_normalize_whitespace,
        test_normalize_whitespace_tabs,
        test_remove_boilerplate_cookie,
        test_remove_boilerplate_legal,
        test_remove_urls_and_emails,
        test_clean_section_headings_preserves_structured,
        test_clean_text_full_pipeline,
        test_clean_text_empty,
        # Post-processor
        test_count_sentences_basic,
        test_count_sentences_excludes_footer,
        test_count_sentences_excludes_citation,
        test_trim_to_sentences,
        test_extract_urls,
        test_check_advisory_leak_clean,
        test_check_advisory_leak_detected,
        test_ensure_footer_appends,
        test_ensure_footer_fixes_date,
        test_ensure_citation_appends,
        test_ensure_citation_keeps_existing,
        test_validate_and_format_full,
        test_validate_and_format_adds_missing,
        # Prompts
        test_build_context_block_empty,
        test_build_context_block_with_chunks,
        test_build_prompt_returns_messages,
        test_build_no_context_prompt,
        test_system_prompt_contains_rules,
        # Guardrails (supplemental)
        test_sanitize_no_pii,
        test_sanitize_detailed_pan,
        test_sanitize_detailed_aadhaar,
        test_sanitize_empty,
        test_generate_refusal_advisory,
        test_generate_refusal_out_of_scope,
        test_is_refusal_response_true,
        test_is_refusal_response_false,
        test_format_factual_footer,
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
