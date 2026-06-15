"""
Response generator — orchestrates prompt building and LLM inference
to produce factual, cited responses for mutual fund queries.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from config.settings import LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P
from src.generation.prompts import build_prompt, build_no_context_prompt
from src.generation.llm import chat_completion, is_available

logger = logging.getLogger(__name__)

# LLM generation parameters
LLM_PARAMS = {
    "temperature": LLM_TEMPERATURE,
    "max_tokens": LLM_MAX_TOKENS,
    "top_p": LLM_TOP_P,
}

# Fallback response when LLM is unavailable or fails
FALLBACK_RESPONSE = (
    "I'm currently unable to process your request. "
    "Please try again later.\n\n"
    "Last updated from sources: {date}"
)


def generate_response(
    query: str,
    context_chunks: list[dict],
    date: Optional[str] = None,
) -> dict:
    """
    Generate a factual response to a user query using retrieved context.

    Pipeline:
    1. Check if context is available
    2. Build prompt with context chunks
    3. Call LLM for generation
    4. Return response with metadata

    Args:
        query: User's question (already sanitized and classified as factual)
        context_chunks: Retrieved chunks from the retriever
        date: Optional date string for footer (defaults to today)

    Returns:
        Dict with keys:
        - response: str (generated text)
        - source_url: str (primary citation URL)
        - context_used: int (number of chunks used)
        - latency_ms: float (generation time in milliseconds)
        - status: str ("success" | "no_context" | "error")
    """
    start_time = time.time()

    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    result = {
        "response": "",
        "source_url": "",
        "context_used": 0,
        "latency_ms": 0.0,
        "status": "success",
        "date": date,
    }

    # No context — return "no information" response
    if not context_chunks:
        logger.info("No context chunks provided — generating no-context response")
        result["response"] = (
            f"I don't have this information in my current sources.\n\n"
            f"Last updated from sources: {date}"
        )
        result["status"] = "no_context"
        result["latency_ms"] = (time.time() - start_time) * 1000
        return result

    # Extract primary source URL from top chunk
    result["source_url"] = context_chunks[0].get("source_url", "")
    result["context_used"] = len(context_chunks)

    # Check LLM availability
    if not is_available():
        logger.error("LLM not available — returning fallback response")
        result["response"] = FALLBACK_RESPONSE.format(date=date)
        result["status"] = "error"
        result["latency_ms"] = (time.time() - start_time) * 1000
        return result

    # Build prompt and generate
    try:
        messages = build_prompt(context_chunks, query, date=date)

        logger.info(
            "Generating response: query='%s', chunks=%d",
            query[:60], len(context_chunks),
        )

        response_text = chat_completion(
            messages=messages,
            **LLM_PARAMS,
        )

        result["response"] = response_text
        logger.info("Response generated: %d chars", len(response_text))

    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        result["response"] = FALLBACK_RESPONSE.format(date=date)
        result["status"] = "error"

    result["latency_ms"] = (time.time() - start_time) * 1000
    return result


def generate_response_simple(
    query: str,
    context_chunks: list[dict],
) -> str:
    """
    Simplified response generation — returns just the response text.

    Convenience wrapper around generate_response() for callers that
    only need the text output.

    Args:
        query: User question
        context_chunks: Retrieved context chunks

    Returns:
        Generated response string
    """
    result = generate_response(query, context_chunks)
    return result["response"]
