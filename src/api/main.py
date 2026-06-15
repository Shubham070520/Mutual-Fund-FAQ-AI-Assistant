"""
FastAPI application — Mutual Fund FAQ Assistant API.

Wires together the full RAG pipeline:
1. PII sanitization
2. Intent classification
3. Refusal generation (for advisory/out-of-scope)
4. Vector retrieval + reranking
5. LLM generation
6. Post-processing validation

Endpoints:
- POST /query     — Answer a user query
- GET  /health    — Health check
- GET  /schemes   — List supported schemes
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException

# Ensure project root is on sys.path for module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    AMC_NAME,
    RERANK_ENABLED,
    RERANK_TOP_N,
    RETRIEVAL_TOP_K,
)
from config.settings import CONFIG_DIR

from src.api.schemas import (
    QueryRequest,
    QueryResponseEnvelope,
    HealthResponse,
    SchemeListResponse,
    SchemeInfo,
)
from src.api.middleware import configure_middleware

# --- Pipeline modules ---
from src.guardrails.sanitizer import sanitize_detailed
from src.guardrails.intent import classify_intent, INTENT_FACTUAL, INTENT_ADVISORY, INTENT_OUT_OF_SCOPE
from src.guardrails.refusal import generate_refusal, is_refusal_response
from src.retrieval.retriever import retrieve, retrieve_with_rerank
from src.generation.generator import generate_response
from src.generation.postprocessor import validate_and_format
from src.generation.llm import is_available as llm_is_available
from src.processing.vectorstore import get_collection_count

logger = logging.getLogger(__name__)

# --- App initialization ---

app = FastAPI(
    title="Mutual Fund FAQ Assistant",
    description=(
        "A facts-only FAQ assistant for HDFC Mutual Fund schemes. "
        "Answers factual questions about NAV, expense ratio, SIP, fund manager, etc. "
        "Does NOT provide investment advice or recommendations."
    ),
    version="1.0.0",
)

# Apply middleware
configure_middleware(app)


def _load_schemes() -> list[dict]:
    """Load scheme metadata from config/schemes.json."""
    schemes_path = CONFIG_DIR / "schemes.json"
    if not schemes_path.exists():
        logger.warning("schemes.json not found at %s", schemes_path)
        return []

    with open(schemes_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("schemes", [])


def _get_current_date() -> str:
    """Return current UTC date in YYYY-MM-DD format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# --- Endpoints ---

@app.post("/query", response_model=QueryResponseEnvelope)
async def answer_query(request: QueryRequest) -> QueryResponseEnvelope:
    """
    Process a user query through the full RAG pipeline.

    Pipeline:
    1. Sanitize input (PII removal)
    2. Classify intent (FACTUAL / ADVISORY / OUT_OF_SCOPE)
    3. If advisory → return refusal with educational link
    4. If factual → retrieve context → generate response → validate
    5. Return unified response envelope
    """
    start_time = time.time()
    date = _get_current_date()

    raw_query = request.query.strip()
    if not raw_query:
        raise HTTPException(status_code=422, detail="Query cannot be empty")

    # --- Step 1: PII Sanitization ---
    san_result = sanitize_detailed(raw_query)
    query = san_result.sanitized_query
    pii_detected = san_result.pii_detected

    if san_result.was_modified:
        logger.warning("PII detected and redacted: %s", pii_detected)

    # --- Step 2: Intent Classification ---
    intent = classify_intent(query)
    logger.info("Query classified as: %s", intent)

    # --- Step 3: Handle non-factual intents ---
    if intent in (INTENT_ADVISORY, INTENT_OUT_OF_SCOPE):
        refusal_text = generate_refusal(intent, query)

        # Extract educational link from refusal
        educational_link = ""
        if "amfiindia.com" in refusal_text:
            educational_link = "https://www.amfiindia.com/investor-education"
        elif "sebi.gov.in" in refusal_text:
            educational_link = "https://www.sebi.gov.in/investor-education"

        elapsed = (time.time() - start_time) * 1000

        return QueryResponseEnvelope(
            answer=refusal_text,
            educational_link=educational_link,
            last_updated=date,
            intent=intent.lower(),
            is_refusal=True,
            latency_ms=round(elapsed, 1),
            pii_detected=pii_detected,
        )

    # --- Step 4: Factual query — Retrieve context ---
    logger.info("Retrieving context for factual query: '%s'", query[:80])

    try:
        if RERANK_ENABLED:
            context_chunks = retrieve_with_rerank(
                query=query,
                top_k=RETRIEVAL_TOP_K,
                scheme_filter=request.scheme_filter,
                document_type_filter=request.document_type_filter,
                rerank_top_n=RERANK_TOP_N,
            )
        else:
            context_chunks = retrieve(
                query=query,
                top_k=RETRIEVAL_TOP_K,
                scheme_filter=request.scheme_filter,
                document_type_filter=request.document_type_filter,
            )
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        context_chunks = []

    # --- Step 5: Generate response ---
    gen_result = generate_response(query, context_chunks, date=date)

    raw_response = gen_result["response"]
    source_url = gen_result.get("source_url", "")
    context_used = gen_result.get("context_used", 0)
    status = gen_result.get("status", "success")

    # --- Step 6: Post-process ---
    warnings = []

    if status == "no_context":
        # No context found — use refusal generator
        answer = generate_refusal("FACTUAL", query)
        is_refusal = True
    else:
        # Validate and format the LLM response
        post_result = validate_and_format(
            response=raw_response,
            source_url=source_url,
            date=date,
        )
        answer = post_result["formatted_response"]
        warnings = post_result.get("warnings", [])
        is_refusal = is_refusal_response(answer)

        # If advisory language leaked, override with refusal
        if post_result.get("advisory_leak"):
            logger.warning("Advisory language leak detected — replacing with refusal")
            answer = generate_refusal("ADVISORY", query)
            is_refusal = True
            warnings.append("Response replaced due to advisory language leak")

    # --- Determine scheme name ---
    scheme = request.scheme_filter
    if not scheme and context_chunks:
        scheme = context_chunks[0].get("metadata", {}).get("scheme_name", "")

    elapsed = (time.time() - start_time) * 1000

    return QueryResponseEnvelope(
        answer=answer,
        source_url=source_url if not is_refusal else None,
        educational_link=None,
        last_updated=date,
        intent=INTENT_FACTUAL.lower(),
        scheme=scheme or None,
        is_refusal=is_refusal,
        context_used=context_used,
        latency_ms=round(elapsed, 1),
        warnings=warnings,
        pii_detected=pii_detected,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns system status including:
    - Vector store chunk count
    - LLM service availability
    """
    try:
        vector_count = get_collection_count()
    except Exception as e:
        logger.warning("Vector store health check failed: %s", e)
        vector_count = -1

    return HealthResponse(
        status="ok",
        version="1.0.0",
        vector_store_count=vector_count,
        llm_available=llm_is_available(),
    )


@app.get("/schemes", response_model=SchemeListResponse)
async def list_schemes() -> SchemeListResponse:
    """
    Return the list of supported mutual fund schemes.
    """
    schemes_raw = _load_schemes()

    schemes = [
        SchemeInfo(
            name=s.get("name", ""),
            category=s.get("category", ""),
            groww_url=s.get("groww_url", ""),
        )
        for s in schemes_raw
    ]

    return SchemeListResponse(
        amc_name=AMC_NAME,
        scheme_count=len(schemes),
        schemes=schemes,
    )


# --- Startup logging ---

@app.on_event("startup")
async def startup_event():
    """Log startup info."""
    logger.info("=" * 60)
    logger.info("Mutual Fund FAQ Assistant API starting...")
    logger.info("  AMC: %s", AMC_NAME)
    logger.info("  LLM available: %s", llm_is_available())
    try:
        logger.info("  Vector store chunks: %d", get_collection_count())
    except Exception:
        logger.info("  Vector store: not available")
    logger.info("=" * 60)


# --- Entrypoint for standalone execution ---

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
