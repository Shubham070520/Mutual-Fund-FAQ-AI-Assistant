"""Quick verification test for Phase 2 modules."""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.cleaner import clean_text
from src.processing.chunker import chunk_text, chunk_documents


def test_cleaner():
    raw = "  Hello   World  \n\n\n  Test  \n\n  Another paragraph.  "
    result = clean_text(raw)
    assert result, "Cleaner returned empty text"
    assert "Hello World" in result
    print(f"[PASS] Cleaner: {repr(result)}")


def test_chunker():
    text = "This is a test sentence. " * 200
    metadata = {"source_url": "https://test.com", "scheme_name": "Test Fund", "document_type": "faq"}
    chunks = chunk_text(text, metadata)

    assert len(chunks) > 0, "No chunks produced"
    assert "chunk_id" in chunks[0], "Missing chunk_id"
    assert "text" in chunks[0], "Missing text"
    assert "metadata" in chunks[0], "Missing metadata"
    assert chunks[0]["metadata"]["source_url"] == "https://test.com"
    assert "chunk_index" in chunks[0]["metadata"]
    assert "chunk_total" in chunks[0]["metadata"]

    print(f"[PASS] Chunker: produced {len(chunks)} chunks from {len(text)} chars")
    print(f"       First chunk: {len(chunks[0]['text'])} chars")
    print(f"       Metadata keys: {list(chunks[0]['metadata'].keys())}")


def test_chunk_documents():
    docs = [
        {"text": "Document one content. " * 100, "metadata": {"source_url": "url1", "scheme_name": "Fund A"}},
        {"text": "Document two content. " * 100, "metadata": {"source_url": "url2", "scheme_name": "Fund B"}},
        {"text": "", "metadata": {"source_url": "url3", "scheme_name": "Fund C"}},  # empty - should be skipped
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) > 0, "No chunks from chunk_documents"
    print(f"[PASS] chunk_documents: {len(chunks)} chunks from {len(docs)} docs (1 empty skipped)")


if __name__ == "__main__":
    print("Running Phase 2 verification tests...\n")
    test_cleaner()
    test_chunker()
    test_chunk_documents()
    print("\nAll Phase 2 tests passed!")
