"""
LLM client wrapper for Groq API.
Provides a singleton Groq client with OpenAI-compatible interface
for fast inference using Llama models.
"""

import logging
from typing import Optional

from config.settings import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

# Module-level singleton
_client = None


def get_llm_client():
    """
    Get or create the Groq LLM client (singleton pattern).

    Groq provides an OpenAI-compatible API with ultra-fast inference
    via their custom LPU (Language Processing Unit) hardware.

    Returns:
        Groq client instance

    Raises:
        ValueError: If GROQ_API_KEY is not set
    """
    global _client

    if _client is not None:
        return _client

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Please set it in your .env file or environment variables."
        )

    from groq import Groq

    logger.info("Initializing Groq LLM client (model=%s)", LLM_MODEL)
    _client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client initialized successfully")
    return _client


def chat_completion(
    messages: list[dict],
    model: str = LLM_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 200,
    top_p: float = 0.9,
    stop: Optional[list[str]] = None,
) -> str:
    """
    Generate a chat completion using the Groq API.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
                  Roles: 'system', 'user', 'assistant'
        model: Model identifier (default: llama-3.3-70b-versatile)
        temperature: Sampling temperature (0.0-2.0, lower = more deterministic)
        max_tokens: Maximum tokens in the response
        top_p: Nucleus sampling probability threshold
        stop: Optional list of stop sequences

    Returns:
        Generated text string

    Raises:
        Exception: If the API call fails
    """
    client = get_llm_client()

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }

    if stop:
        kwargs["stop"] = stop

    logger.debug(
        "LLM call: model=%s, temp=%.2f, max_tokens=%d, messages=%d",
        model, temperature, max_tokens, len(messages),
    )

    response = client.chat.completions.create(**kwargs)

    content = response.choices[0].message.content.strip()
    usage = response.usage

    logger.info(
        "LLM response: %d tokens (prompt=%d, completion=%d)",
        usage.total_tokens if usage else 0,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )

    return content


def is_available() -> bool:
    """
    Check if the Groq LLM client is available and configured.

    Returns:
        True if the API key is set and client can be created
    """
    try:
        get_llm_client()
        return True
    except (ValueError, Exception):
        return False
