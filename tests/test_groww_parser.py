"""
Tests for Groww-specific scheme page parser and integration with ingestion pipeline.
Tests structured data extraction (NAV, AUM, Exit Load, SIP, Fund Manager, etc.).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.groww_parser import (
    parse_groww_scheme,
    build_scheme_facts_block,
    enrich_chunk_with_facts,
    _normalize_label,
)
from src.ingestion.cleaner import clean_section_headings, clean_text
from src.processing.chunker import chunk_text, _classify_chunk_content


# --- Sample HTML fixtures ---

SAMPLE_GROWW_HTML = """
<html>
<head><title>HDFC Mid Cap Opportunities Fund - Groww</title></head>
<body>
<main>
<div class="scheme-header">
  <h1>HDFC Mid Cap Opportunities Fund</h1>
  <p>HDFC Mutual Fund</p>
</div>

<div class="key-metrics">
  <div class="metric-card">
    <span class="label">NAV as on Jun 09, 2026</span>
    <span class="value">₹142.35</span>
  </div>
  <div class="metric-card">
    <span class="label">Fund Size (AUM)</span>
    <span class="value">₹48,235 Cr</span>
  </div>
  <div class="metric-card">
    <span class="label">Expense Ratio</span>
    <span class="value">0.74%</span>
  </div>
  <div class="metric-card">
    <span class="label">Exit Load</span>
    <span class="value">1% if redeemed within 1 year</span>
  </div>
</div>

<section class="fund-details">
  <h2>Fund Details</h2>
  <table>
    <tr><td>Min SIP Amount</td><td>₹100</td></tr>
    <tr><td>Min Lumpsum Investment</td><td>₹5,000</td></tr>
    <tr><td>Fund Manager</td><td>Chirag Setalvad</td></tr>
    <tr><td>Benchmark Index</td><td>NIFTY Midcap 150 TRI</td></tr>
    <tr><td>Category</td><td>Mid Cap</td></tr>
    <tr><td>AMC</td><td>HDFC Asset Management Company</td></tr>
    <tr><td>Launch Date</td><td>Jun 2007</td></tr>
    <tr><td>Lock-in Period</td><td>None</td></tr>
  </table>
</section>

<section class="returns">
  <h2>Returns</h2>
  <table>
    <tr><th>Period</th><th>1M</th><th>3M</th><th>6M</th><th>1Y</th><th>3Y</th><th>5Y</th></tr>
    <tr><td>Returns</td><td>2.5%</td><td>8.3%</td><td>15.2%</td><td>28.7%</td><td>22.1%</td><td>19.8%</td></tr>
  </table>
</section>

<section class="faq-section">
  <h2>Frequently Asked Questions</h2>
  <div class="faq-item">
    <h3>What is the minimum SIP for HDFC Mid Cap Fund?</h3>
    <p>The minimum SIP amount for HDFC Mid Cap Opportunities Fund is ₹100.</p>
  </div>
  <div class="faq-item">
    <h3>Who manages HDFC Mid Cap Opportunities Fund?</h3>
    <p>HDFC Mid Cap Opportunities Fund is managed by Chirag Setalvad.</p>
  </div>
</section>

</main>
</body>
</html>
"""

SAMPLE_GROWW_HTML_DT_DD = """
<html>
<body>
<dl>
  <dt>NAV</dt>
  <dd>₹98.52</dd>
  <dt>AUM</dt>
  <dd>₹12,450 Cr</dd>
  <dt>Exit Load</dt>
  <dd>Nil after 1 year</dd>
  <dt>Expense Ratio</dt>
  <dd>0.85%</dd>
  <dt>Minimum SIP</dt>
  <dd>₹500</dd>
  <dt>Fund Manager</dt>
  <dd>Rohit Sharma</dd>
</dl>
</body>
</html>
"""

SAMPLE_GROWW_HTML_TH_TD = """
<html>
<body>
<table>
  <tr><th>Net Asset Value</th><td>₹215.67</td></tr>
  <tr><th>Fund Size</th><td>₹8,920 Cr</td></tr>
  <tr><th>Exit Load</th><td>1% within 365 days</td></tr>
  <tr><th>Min SIP</th><td>₹500</td></tr>
  <tr><th>Min Lumpsum</th><td>₹1,000</td></tr>
  <tr><th>Fund Manager</th><td>Priya Nair</td></tr>
  <tr><th>Benchmark</th><td>NIFTY Smallcap 250 TRI</td></tr>
  <tr><th>Inception Date</th><td>Jan 2020</td></tr>
</table>
</body>
</html>
"""

SAMPLE_HTML_NO_STRUCTURED_DATA = """
<html>
<body>
<main>
<h1>About Mutual Funds</h1>
<p>Mutual funds pool money from many investors to invest in stocks, bonds, and other securities.</p>
<p>They are managed by professional fund managers.</p>
</main>
</body>
</html>
"""


# --- Test normalize_label ---

def test_normalize_label_basic():
    """Test label normalization."""
    assert _normalize_label("  NAV  as  on  ") == "nav as on"
    assert _normalize_label("Fund Size (AUM)") == "fund size (aum)"
    assert _normalize_label("EXIT LOAD") == "exit load"
    print("✓ test_normalize_label_basic passed")


# --- Test parse_groww_scheme with metric cards ---

def test_parse_groww_metric_cards():
    """Test extraction from Groww metric card layout."""
    result = parse_groww_scheme(SAMPLE_GROWW_HTML, "https://groww.in/test")

    structured = result["structured_data"]

    # Should extract NAV
    assert "nav" in structured, f"NAV not found. Keys: {list(structured.keys())}"
    assert "142.35" in structured["nav"], f"NAV value wrong: {structured['nav']}"

    # Should extract AUM/Fund Size
    assert "aum" in structured, f"AUM not found. Keys: {list(structured.keys())}"
    assert "48,235" in structured["aum"], f"AUM value wrong: {structured['aum']}"

    # Should extract Exit Load
    assert "exit_load" in structured, f"Exit Load not found. Keys: {list(structured.keys())}"
    assert "1%" in structured["exit_load"], f"Exit Load value wrong: {structured['exit_load']}"

    # Should extract Expense Ratio
    assert "expense_ratio" in structured, f"Expense Ratio not found. Keys: {list(structured.keys())}"
    assert "0.74" in structured["expense_ratio"], f"Expense Ratio wrong: {structured['expense_ratio']}"

    print("✓ test_parse_groww_metric_cards passed")


def test_parse_groww_fund_details():
    """Test extraction of fund details (SIP, lumpsum, fund manager, etc.)."""
    result = parse_groww_scheme(SAMPLE_GROWW_HTML, "https://groww.in/test")

    structured = result["structured_data"]

    # Should extract SIP minimum
    assert "sip_minimum" in structured, f"SIP minimum not found. Keys: {list(structured.keys())}"
    assert "100" in structured["sip_minimum"], f"SIP value wrong: {structured['sip_minimum']}"

    # Should extract lumpsum minimum
    assert "lumpsum_minimum" in structured, f"Lumpsum minimum not found. Keys: {list(structured.keys())}"
    assert "5,000" in structured["lumpsum_minimum"], f"Lumpsum value wrong: {structured['lumpsum_minimum']}"

    # Should extract fund manager
    assert "fund_manager" in structured, f"Fund Manager not found. Keys: {list(structured.keys())}"
    assert "Chirag Setalvad" in structured["fund_manager"], f"Fund Manager wrong: {structured['fund_manager']}"

    # Should extract benchmark
    assert "benchmark" in structured, f"Benchmark not found. Keys: {list(structured.keys())}"
    assert "NIFTY Midcap 150" in structured["benchmark"], f"Benchmark wrong: {structured['benchmark']}"

    # Should extract category
    assert "category" in structured, f"Category not found. Keys: {list(structured.keys())}"
    assert "Mid Cap" in structured["category"], f"Category wrong: {structured['category']}"

    # Should extract AMC
    assert "amc" in structured, f"AMC not found. Keys: {list(structured.keys())}"

    # Should extract launch date
    assert "launch_date" in structured, f"Launch date not found. Keys: {list(structured.keys())}"

    print("✓ test_parse_groww_fund_details passed")


def test_parse_groww_dt_dd_layout():
    """Test extraction from dt/dd definition list layout."""
    result = parse_groww_scheme(SAMPLE_GROWW_HTML_DT_DD, "https://groww.in/test")

    structured = result["structured_data"]

    assert "nav" in structured, f"NAV not found in dt/dd. Keys: {list(structured.keys())}"
    assert "98.52" in structured["nav"], f"NAV value wrong: {structured['nav']}"

    assert "aum" in structured, f"AUM not found in dt/dd. Keys: {list(structured.keys())}"
    assert "12,450" in structured["aum"], f"AUM value wrong: {structured['aum']}"

    assert "exit_load" in structured, f"Exit load not found in dt/dd. Keys: {list(structured.keys())}"
    assert "Nil" in structured["exit_load"], f"Exit load value wrong: {structured['exit_load']}"

    assert "sip_minimum" in structured, f"SIP minimum not found. Keys: {list(structured.keys())}"
    assert "500" in structured["sip_minimum"], f"SIP value wrong: {structured['sip_minimum']}"

    assert "fund_manager" in structured, f"Fund manager not found. Keys: {list(structured.keys())}"
    assert "Rohit Sharma" in structured["fund_manager"], f"Fund manager wrong: {structured['fund_manager']}"

    print("✓ test_parse_groww_dt_dd_layout passed")


def test_parse_groww_th_td_layout():
    """Test extraction from th/td table layout."""
    result = parse_groww_scheme(SAMPLE_GROWW_HTML_TH_TD, "https://groww.in/test")

    structured = result["structured_data"]

    assert "nav" in structured, f"NAV not found in th/td. Keys: {list(structured.keys())}"
    assert "215.67" in structured["nav"], f"NAV value wrong: {structured['nav']}"

    assert "aum" in structured, f"AUM not found in th/td. Keys: {list(structured.keys())}"
    assert "8,920" in structured["aum"], f"AUM value wrong: {structured['aum']}"

    assert "exit_load" in structured, f"Exit load not found in th/td. Keys: {list(structured.keys())}"
    assert "1%" in structured["exit_load"], f"Exit load value wrong: {structured['exit_load']}"

    assert "sip_minimum" in structured, f"SIP minimum not found. Keys: {list(structured.keys())}"
    assert "500" in structured["sip_minimum"], f"SIP value wrong: {structured['sip_minimum']}"

    assert "fund_manager" in structured, f"Fund manager not found. Keys: {list(structured.keys())}"
    assert "Priya Nair" in structured["fund_manager"], f"Fund manager wrong: {structured['fund_manager']}"

    assert "launch_date" in structured, f"Launch date not found. Keys: {list(structured.keys())}"
    assert "Jan 2020" in structured["launch_date"], f"Launch date wrong: {structured['launch_date']}"

    print("✓ test_parse_groww_th_td_layout passed")


def test_parse_groww_no_structured_data():
    """Test that pages without structured data return empty structured_data."""
    result = parse_groww_scheme(SAMPLE_HTML_NO_STRUCTURED_DATA, "https://example.com/test")

    structured = result["structured_data"]

    # Should have no structured metrics
    assert "nav" not in structured
    assert "aum" not in structured
    assert "fund_manager" not in structured

    # Text should be minimal/empty since no FAQs or metrics
    assert result["faqs"] == []

    print("✓ test_parse_groww_no_structured_data passed")


def test_parse_groww_faqs():
    """Test FAQ extraction from Groww page."""
    result = parse_groww_scheme(SAMPLE_GROWW_HTML, "https://groww.in/test")

    faqs = result["faqs"]

    assert len(faqs) >= 1, f"Expected at least 1 FAQ, got {len(faqs)}"

    # Check first FAQ
    q1 = faqs[0]["question"].lower()
    a1 = faqs[0]["answer"].lower()
    assert "sip" in q1 or "minimum" in q1, f"FAQ question unexpected: {faqs[0]['question']}"
    assert len(a1) > 0, "FAQ answer is empty"

    print("✓ test_parse_groww_faqs passed")


def test_parse_groww_returns_text():
    """Test that the text output contains structured sections."""
    result = parse_groww_scheme(SAMPLE_GROWW_HTML, "https://groww.in/test")

    text = result["text"]

    # Should have Key Fund Metrics section
    assert "## Key Fund Metrics" in text, "Missing Key Fund Metrics section"

    # Should have NAV line
    assert "NAV:" in text, "Missing NAV in text output"

    # Should have Fund Size line
    assert "Fund Size (AUM):" in text, "Missing Fund Size in text output"

    # Should have Exit Load line
    assert "Exit Load:" in text, "Missing Exit Load in text output"

    # Should have SIP line
    assert "Minimum SIP" in text or "SIP" in text, "Missing SIP in text output"

    # Should have Fund Manager line
    assert "Fund Manager:" in text, "Missing Fund Manager in text output"

    print("✓ test_parse_groww_returns_text passed")


# --- Test build_scheme_facts_block ---

def test_build_scheme_facts_block():
    """Test building a dense facts block from structured data."""
    structured_data = {
        "nav": "₹142.35",
        "aum": "₹48,235 Cr",
        "exit_load": "1% if redeemed within 1 year",
        "expense_ratio": "0.74%",
        "sip_minimum": "₹100",
        "fund_manager": "Chirag Setalvad",
        "benchmark": "NIFTY Midcap 150 TRI",
    }

    facts = build_scheme_facts_block(structured_data, "HDFC Mid Cap Opportunities Fund")

    assert "Scheme: HDFC Mid Cap Opportunities Fund" in facts
    assert "NAV: ₹142.35" in facts
    assert "Fund Size (AUM): ₹48,235 Cr" in facts
    assert "Exit Load: 1%" in facts
    assert "Min SIP: ₹100" in facts
    assert "Fund Manager: Chirag Setalvad" in facts
    assert "Benchmark: NIFTY Midcap 150 TRI" in facts

    print("✓ test_build_scheme_facts_block passed")


def test_build_scheme_facts_block_empty():
    """Test facts block with minimal data."""
    facts = build_scheme_facts_block({}, "Test Fund")

    assert "Scheme: Test Fund" in facts
    # Should only have scheme name, no other fields
    assert facts.count("|") == 0

    print("✓ test_build_scheme_facts_block_empty passed")


# --- Test enrich_chunk_with_facts ---

def test_enrich_chunk_with_facts():
    """Test that chunks without metrics get facts block prepended."""
    chunk = {
        "text": "HDFC Mid Cap Fund invests primarily in mid-cap companies.",
        "metadata": {"source_url": "https://groww.in/test"},
    }

    facts = "Scheme: HDFC Mid Cap | NAV: ₹142.35 | Fund Size (AUM): ₹48,235 Cr"

    enriched = enrich_chunk_with_facts(chunk, facts)

    assert facts in enriched["text"]
    assert enriched["metadata"].get("has_facts_block") is True
    assert "mid-cap companies" in enriched["text"]

    print("✓ test_enrich_chunk_with_facts passed")


def test_enrich_chunk_already_has_metrics():
    """Test that chunks with metrics are NOT double-enriched."""
    chunk = {
        "text": "The NAV of the fund is ₹142.35 and AUM is ₹48,235 Cr.",
        "metadata": {"source_url": "https://groww.in/test"},
    }

    facts = "Scheme: HDFC Mid Cap | NAV: ₹142.35"

    enriched = enrich_chunk_with_facts(chunk, facts)

    # Should NOT prepend facts since chunk already has nav/aum
    assert not enriched["text"].startswith("Scheme:")
    assert enriched["metadata"].get("has_facts_block") is not True

    print("✓ test_enrich_chunk_already_has_metrics passed")


# --- Test cleaner preserves structured data ---

def test_cleaner_preserves_fund_metrics():
    """Test that cleaner doesn't convert metric lines to headings."""
    text = """## Key Fund Metrics
NAV: ₹142.35
Fund Size (AUM): ₹48,235 Cr
Exit Load: 1% if redeemed within 1 year
Expense Ratio: 0.74%
Minimum SIP Amount: ₹100
Fund Manager: Chirag Setalvad"""

    cleaned = clean_section_headings(text)

    # Metric lines should NOT be converted to headings (no ## prefix)
    assert "## NAV:" not in cleaned, "NAV line was converted to heading"
    assert "## Fund Size" not in cleaned, "Fund Size line was converted to heading"
    assert "## Exit Load" not in cleaned, "Exit Load line was converted to heading"
    assert "## Fund Manager" not in cleaned, "Fund Manager line was converted to heading"

    # They should remain as regular lines
    assert "NAV: ₹142.35" in cleaned
    assert "Fund Size (AUM): ₹48,235 Cr" in cleaned
    assert "Fund Manager: Chirag Setalvad" in cleaned

    print("✓ test_cleaner_preserves_fund_metrics passed")


def test_cleaner_preserves_faq_format():
    """Test that cleaner preserves Q:/A: format."""
    text = """## Frequently Asked Questions
Q: What is the minimum SIP?
A: The minimum SIP amount is ₹100.

Q: Who is the fund manager?
A: The fund is managed by Chirag Setalvad."""

    cleaned = clean_section_headings(text)

    assert "## Q:" not in cleaned, "Q: was converted to heading"
    assert "## A:" not in cleaned, "A: was converted to heading"
    assert "Q: What is the minimum SIP?" in cleaned
    assert "A: The minimum SIP amount is ₹100." in cleaned

    print("✓ test_cleaner_preserves_faq_format passed")


# --- Test chunker content classification ---

def test_classify_chunk_content_nav():
    """Test chunk classification for NAV content."""
    text = "The NAV of HDFC Mid Cap Fund is ₹142.35 as on June 2026."
    types = _classify_chunk_content(text)
    assert "nav" in types, f"Expected 'nav' in types: {types}"
    print("✓ test_classify_chunk_content_nav passed")


def test_classify_chunk_content_aum():
    """Test chunk classification for AUM content."""
    text = "The fund size (AUM) is ₹48,235 Cr as of latest data."
    types = _classify_chunk_content(text)
    assert "aum" in types, f"Expected 'aum' in types: {types}"
    print("✓ test_classify_chunk_content_aum passed")


def test_classify_chunk_content_sip():
    """Test chunk classification for SIP content."""
    text = "You can start a SIP with minimum ₹100 per month."
    types = _classify_chunk_content(text)
    assert "sip" in types, f"Expected 'sip' in types: {types}"
    print("✓ test_classify_chunk_content_sip passed")


def test_classify_chunk_content_fund_manager():
    """Test chunk classification for fund manager content."""
    text = "The fund is managed by Chirag Setalvad since 2020."
    types = _classify_chunk_content(text)
    assert "fund_manager" in types, f"Expected 'fund_manager' in types: {types}"
    print("✓ test_classify_chunk_content_fund_manager passed")


def test_classify_chunk_content_exit_load():
    """Test chunk classification for exit load content."""
    text = "Exit load is 1% if units are redeemed within 1 year of purchase."
    types = _classify_chunk_content(text)
    assert "exit_load" in types, f"Expected 'exit_load' in types: {types}"
    print("✓ test_classify_chunk_content_exit_load passed")


def test_classify_chunk_content_multiple():
    """Test chunk classification with multiple content types."""
    text = """## Key Fund Metrics
NAV: ₹142.35
Fund Size (AUM): ₹48,235 Cr
Exit Load: 1% if redeemed within 1 year
Minimum SIP Amount: ₹100
Fund Manager: Chirag Setalvad"""
    types = _classify_chunk_content(text)
    assert "nav" in types, f"Expected 'nav': {types}"
    assert "aum" in types, f"Expected 'aum': {types}"
    assert "exit_load" in types, f"Expected 'exit_load': {types}"
    assert "sip" in types, f"Expected 'sip': {types}"
    assert "fund_manager" in types, f"Expected 'fund_manager': {types}"
    print("✓ test_classify_chunk_content_multiple passed")


def test_classify_chunk_content_faq():
    """Test chunk classification for FAQ content."""
    text = "Q: What is the minimum SIP?\nA: The minimum SIP is ₹100."
    types = _classify_chunk_content(text)
    assert "faq" in types, f"Expected 'faq': {types}"
    assert "sip" in types, f"Expected 'sip': {types}"
    print("✓ test_classify_chunk_content_faq passed")


def test_classify_chunk_content_generic():
    """Test chunk classification for generic content (no specific types)."""
    text = "Mutual funds are investment vehicles that pool money from multiple investors."
    types = _classify_chunk_content(text)
    assert len(types) == 0, f"Expected empty types for generic content, got: {types}"
    print("✓ test_classify_chunk_content_generic passed")


def test_chunker_tags_metadata():
    """Test that chunker adds content_types and has_structured_data to metadata."""
    text = """## Key Fund Metrics
NAV: ₹142.35
Fund Size (AUM): ₹48,235 Cr
Exit Load: 1% if redeemed within 1 year"""

    metadata = {"source_url": "https://groww.in/test", "scheme_name": "HDFC Mid Cap"}
    chunks = chunk_text(text, metadata, chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 0
    chunk = chunks[0]
    assert "content_types" in chunk["metadata"], "Missing content_types in metadata"
    assert "has_structured_data" in chunk["metadata"], "Missing has_structured_data"
    assert chunk["metadata"]["has_structured_data"] is True
    assert "nav" in chunk["metadata"]["content_types"]
    assert "aum" in chunk["metadata"]["content_types"]

    print("✓ test_chunker_tags_metadata passed")


def test_chunker_no_tags_for_generic():
    """Test that generic content chunks have empty content_types."""
    text = "Mutual funds pool money from investors. They invest in stocks and bonds. Professional managers handle the portfolio."

    metadata = {"source_url": "https://example.com/test"}
    chunks = chunk_text(text, metadata, chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 0
    chunk = chunks[0]
    assert chunk["metadata"]["has_structured_data"] is False
    assert chunk["metadata"]["content_types"] == []

    print("✓ test_chunker_no_tags_for_generic passed")


# --- Integration test: Full pipeline from HTML to chunks ---

def test_full_pipeline_integration():
    """Test the full flow: HTML -> parse -> clean -> chunk -> classify."""
    # Step 1: Parse HTML
    result = parse_groww_scheme(SAMPLE_GROWW_HTML, "https://groww.in/mutual-funds/hdfc-mid-cap")
    structured = result["structured_data"]
    groww_text = result["text"]

    # Step 2: Clean
    cleaned = clean_text(groww_text, preserve_headings=True)

    # Step 3: Build facts block
    facts = build_scheme_facts_block(structured, "HDFC Mid Cap Opportunities Fund")

    # Step 4: Chunk
    metadata = {
        "source_url": "https://groww.in/mutual-funds/hdfc-mid-cap",
        "scheme_name": "HDFC Mid Cap Opportunities Fund",
        "structured_data": structured,
    }
    chunks = chunk_text(cleaned, metadata, chunk_size=500, chunk_overlap=50)

    # Step 5: Enrich
    for chunk in chunks:
        enrich_chunk_with_facts(chunk, facts)

    # Verify
    assert len(chunks) > 0, "No chunks produced"

    # At least one chunk should have structured data tags
    has_structured = any(
        c["metadata"].get("has_structured_data", False) for c in chunks
    )
    assert has_structured, "No chunk has structured data tags"

    # At least one chunk should mention NAV
    has_nav = any("NAV" in c["text"] for c in chunks)
    assert has_nav, "No chunk mentions NAV"

    # At least one chunk should mention AUM/Fund Size
    has_aum = any("AUM" in c["text"] or "Fund Size" in c["text"] for c in chunks)
    assert has_aum, "No chunk mentions AUM/Fund Size"

    # At least one chunk should mention Fund Manager
    has_fm = any("Fund Manager" in c["text"] for c in chunks)
    assert has_fm, "No chunk mentions Fund Manager"

    # At least one chunk should mention SIP
    has_sip = any("SIP" in c["text"] for c in chunks)
    assert has_sip, "No chunk mentions SIP"

    # At least one chunk should mention Exit Load
    has_exit = any("Exit Load" in c["text"] for c in chunks)
    assert has_exit, "No chunk mentions Exit Load"

    print("✓ test_full_pipeline_integration passed")


# --- Run all tests ---

if __name__ == "__main__":
    tests = [
        test_normalize_label_basic,
        test_parse_groww_metric_cards,
        test_parse_groww_fund_details,
        test_parse_groww_dt_dd_layout,
        test_parse_groww_th_td_layout,
        test_parse_groww_no_structured_data,
        test_parse_groww_faqs,
        test_parse_groww_returns_text,
        test_build_scheme_facts_block,
        test_build_scheme_facts_block_empty,
        test_enrich_chunk_with_facts,
        test_enrich_chunk_already_has_metrics,
        test_cleaner_preserves_fund_metrics,
        test_cleaner_preserves_faq_format,
        test_classify_chunk_content_nav,
        test_classify_chunk_content_aum,
        test_classify_chunk_content_sip,
        test_classify_chunk_content_fund_manager,
        test_classify_chunk_content_exit_load,
        test_classify_chunk_content_multiple,
        test_classify_chunk_content_faq,
        test_classify_chunk_content_generic,
        test_chunker_tags_metadata,
        test_chunker_no_tags_for_generic,
        test_full_pipeline_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*60}")
