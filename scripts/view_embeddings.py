"""
View embeddings stored in ChromaDB.

Usage:
    python scripts/view_embeddings.py              # Show first 5 embeddings
    python scripts/view_embeddings.py --count 10   # Show first 10
    python scripts/view_embeddings.py --scheme "HDFC Mid Cap Fund"
    python scripts/view_embeddings.py --search "expense ratio"
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import VECTORDB_DIR, VECTORDB_COLLECTION_NAME, EMBEDDING_DIMENSIONS

import chromadb


def get_client():
    return chromadb.PersistentClient(path=str(VECTORDB_DIR))


def view_embeddings(count: int = 5, scheme_filter: str = None, search_query: str = None):
    client = get_client()

    # List all collections
    collections = client.list_collections()
    print(f"\n{'='*70}")
    print(f" ChromaDB — Embedding Viewer")
    print(f"{'='*70}")
    print(f" DB Path      : {VECTORDB_DIR}")
    print(f" Collections  : {[c.name for c in collections]}")

    try:
        collection = client.get_collection(VECTORDB_COLLECTION_NAME)
    except Exception:
        print(f"\n Collection '{VECTORDB_COLLECTION_NAME}' not found.")
        print(" Run the ingestion pipeline first to populate the vector store.")
        return

    total = collection.count()
    print(f" Collection   : {VECTORDB_COLLECTION_NAME}")
    print(f" Total vectors: {total}")
    print(f" Embedding dim: {EMBEDDING_DIMENSIONS}")
    print(f"{'='*70}\n")

    if total == 0:
        print(" Collection is empty. Run: python -m src.ingestion.pipeline")
        return

    # --- Search mode: query by text similarity ---
    if search_query:
        print(f" Searching for: \"{search_query}\"\n")
        from src.processing.embeddings import embed_query
        query_embedding = embed_query(search_query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(count, total),
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        embs = results["embeddings"][0]

        for i, (doc_id, doc, meta, dist, emb) in enumerate(zip(ids, docs, metas, dists, embs)):
            similarity = 1.0 - dist  # cosine distance → similarity
            _print_entry(i + 1, doc_id, doc, meta, emb, similarity=similarity)
        return

    # --- Browse mode: fetch first N ---
    where = None
    if scheme_filter:
        where = {"scheme_name": scheme_filter}
        print(f" Filter: scheme_name = \"{scheme_filter}\"\n")

    kwargs = {
        "include": ["documents", "metadatas", "embeddings"],
    }
    if where:
        kwargs["where"] = where

    # ChromaDB .get() returns all matching; we limit manually
    result = collection.get(**kwargs)

    ids = result["ids"][:count]
    docs = result["documents"][:count]
    metas = result["metadatas"][:count]
    embs = result["embeddings"][:count]

    showing = len(ids)
    matched = len(result["ids"])
    print(f" Showing {showing} of {matched} matching vectors\n")

    for i, (doc_id, doc, meta, emb) in enumerate(zip(ids, docs, metas, embs)):
        _print_entry(i + 1, doc_id, doc, meta, emb)


def _print_entry(index, doc_id, doc, meta, emb, similarity=None):
    """Pretty-print a single embedding entry."""
    text_preview = doc[:200].replace("\n", " ") if doc else "(empty)"
    if len(doc) > 200:
        text_preview += "..."

    print(f" [{index}] ID: {doc_id}")

    if similarity is not None:
        print(f"     Similarity : {similarity:.4f}")

    print(f"     Text       : {text_preview}")

    # Metadata — show key fields
    scheme = meta.get("scheme_name", "—")
    doc_type = meta.get("document_type", "—")
    source = meta.get("source_url", "—")
    content_types = meta.get("content_types", "[]")

    print(f"     Scheme     : {scheme}")
    print(f"     Doc Type   : {doc_type}")
    print(f"     Source URL : {source}")
    print(f"     Content    : {content_types}")

    # Embedding vector preview
    emb_preview = str(emb[:5])
    print(f"     Embedding  : {emb_preview} ... (dim={len(emb)})")

    print()


def main():
    parser = argparse.ArgumentParser(description="View ChromaDB embeddings")
    parser.add_argument("--count", "-n", type=int, default=5, help="Number of embeddings to show (default: 5)")
    parser.add_argument("--scheme", "-s", type=str, default=None, help="Filter by scheme name")
    parser.add_argument("--search", "-q", type=str, default=None, help="Search by semantic similarity")
    args = parser.parse_args()

    view_embeddings(count=args.count, scheme_filter=args.scheme, search_query=args.search)


if __name__ == "__main__":
    main()
