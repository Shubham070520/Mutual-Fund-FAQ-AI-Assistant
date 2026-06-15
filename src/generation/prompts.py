"""
Prompt templates for the mutual fund FAQ assistant.
Constructs system and user prompts with context injection,
enforcing strict factual-only response generation.
"""

import logging
from datetime import datetime, timezone

from config.settings import MAX_RESPONSE_SENTENCES

logger = logging.getLogger(__name__)

# --- System Prompt ---
# Enforces strict factual-only behavior with citation requirements

SYSTEM_PROMPT = f"""You are a facts-only mutual fund FAQ assistant for HDFC Mutual Fund schemes.

STRICT RULES:
1. Answer ONLY factual, verifiable questions about mutual fund schemes.
2. Use ONLY the provided context below. Do NOT use external knowledge or make assumptions.
3. Response must be MAXIMUM {MAX_RESPONSE_SENTENCES} sentences (not counting the footer).
4. Include EXACTLY ONE source URL from the context in your answer.
5. End every response with this exact footer on a new line: "Last updated from sources: <date>"
6. Do NOT provide investment advice, recommendations, or opinions.
7. Do NOT compare fund performance or predict future returns.
8. If the context does not contain the answer, say exactly: "I don't have this information in my current sources."
9. Do NOT use phrases like "I recommend", "you should", "it might be good", or any advisory language.
10. Keep responses concise, objective, and professional."""

# --- User Prompt Template ---
# Injects retrieved context chunks and the user's query

USER_PROMPT_TEMPLATE = """Based on the following context from official sources, answer the user's question.

--- CONTEXT ---
{context}
--- END CONTEXT ---

User's question: {query}

Instructions:
- Answer using ONLY information from the context above.
- Include the source URL in your answer.
- Keep response to maximum {max_sentences} sentences.
- End with: "Last updated from sources: {date}"
- If the answer is not in the context, say: "I don't have this information in my current sources."

Answer:"""


def build_context_block(context_chunks: list[dict]) -> str:
    """
    Build a formatted context block from retrieved chunks.

    Each chunk is formatted with its text and metadata for the LLM
    to reference when generating the response.

    Args:
        context_chunks: List of result dicts from the retriever, each containing:
            - text: str (chunk content)
            - source_url: str
            - scheme_name: str
            - document_type: str
            - similarity: float

    Returns:
        Formatted string with all context chunks
    """
    if not context_chunks:
        return "[No context available]"

    blocks = []
    for i, chunk in enumerate(context_chunks, 1):
        text = chunk.get("text", "").strip()
        source_url = chunk.get("source_url", "Unknown")
        scheme_name = chunk.get("scheme_name", "Unknown")
        doc_type = chunk.get("document_type", "Unknown")

        block = (
            f"[Source {i}] Scheme: {scheme_name} | Type: {doc_type} | "
            f"URL: {source_url}\n{text}"
        )
        blocks.append(block)

    result = "\n\n".join(blocks)
    logger.debug("Built context block: %d sources, %d chars", len(blocks), len(result))
    return result


def build_prompt(
    context_chunks: list[dict],
    query: str,
    date: str | None = None,
) -> list[dict]:
    """
    Construct the full prompt messages for the LLM.

    Returns a list of message dicts in OpenAI chat format:
    - system message with behavioral rules
    - user message with context and query

    Args:
        context_chunks: Retrieved chunks from the retriever
        query: User's question
        date: Optional date string for the footer (defaults to today)

    Returns:
        List of message dicts for the chat completion API
    """
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    context_block = build_context_block(context_chunks)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        context=context_block,
        query=query,
        max_sentences=MAX_RESPONSE_SENTENCES,
        date=date,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    logger.info(
        "Built prompt: %d context chunks, query='%s', date=%s",
        len(context_chunks), query[:60], date,
    )

    return messages


def build_no_context_prompt(query: str, date: str | None = None) -> list[dict]:
    """
    Build a prompt for when no relevant context is found.

    Instructs the LLM to respond with the "no information" message.

    Args:
        query: User's question
        date: Optional date string

    Returns:
        List of message dicts
    """
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"No relevant context was found for the following question.\n"
                f"Respond with: 'I don't have this information in my current sources.'\n"
                f"End with: 'Last updated from sources: {date}'\n\n"
                f"Question: {query}"
            ),
        },
    ]

    return messages
