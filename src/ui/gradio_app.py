"""
Gradio UI — Mutual Fund FAQ Assistant (Alternative Interface).

Provides a chatbot interface using Gradio's ChatInterface component.
Connects to the FastAPI backend at API_BASE_URL.

Run with:
    python -m src.ui.gradio_app
"""

import sys
from pathlib import Path

import gradio as gr
import requests

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import API_BASE_URL, AMC_NAME

# --- Constants ---

APP_TITLE = "Mutual Fund FAQ Assistant"
APP_DESCRIPTION = (
    "📊 **Facts-only. No investment advice.**\n\n"
    "Ask factual questions about HDFC Mutual Fund schemes — "
    "NAV, expense ratio, SIP, fund manager, AUM, exit load, benchmark, etc."
)

EXAMPLE_QUESTIONS = [
    ["What is the expense ratio of HDFC Mid Cap Fund?"],
    ["What is the minimum SIP amount for HDFC Small Cap Fund?"],
    ["Who is the fund manager of HDFC Defence Fund?"],
    ["What is the fund size (AUM) of HDFC Mid Cap Fund?"],
    ["What is the exit load for HDFC Small Cap Fund?"],
    ["What is the benchmark index for HDFC Gold ETF Fund of Fund?"],
]


# --- API Client ---

def call_query_api(query: str) -> dict:
    """
    Send a query to the /query endpoint.

    Returns:
        Response envelope dict, or error dict on failure.
    """
    url = f"{API_BASE_URL}/query"

    try:
        response = requests.post(
            url,
            json={"query": query},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to the API server. Please ensure it's running."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The server might be busy."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# --- Chat Function ---

def chat_fn(message: str, history: list) -> str:
    """
    Gradio chat function.

    Args:
        message: User's input message
        history: List of [user, assistant] pairs

    Returns:
        Assistant response string
    """
    if not message.strip():
        return "Please ask a specific question about HDFC Mutual Fund schemes."

    result = call_query_api(message)

    if "error" in result:
        return f"⚠️ {result['error']}"

    # Build formatted response
    parts = [result.get("answer", "No response received.")]

    # Add metadata
    meta = []
    if result.get("source_url"):
        meta.append(f"🔗 Source: {result['source_url']}")
    if result.get("scheme"):
        meta.append(f"📋 Scheme: {result['scheme']}")
    if result.get("last_updated"):
        meta.append(f"📅 {result['last_updated']}")
    if result.get("latency_ms"):
        meta.append(f"⚡ {result['latency_ms']:.0f}ms")

    if meta:
        parts.append("\n---\n" + " | ".join(meta))

    if result.get("is_refusal") and result.get("intent") == "advisory":
        parts.append("\n\n🚫 *Investment advice not provided — this is a facts-only assistant.*")
    elif result.get("is_refusal") and result.get("intent") == "out_of_scope":
        parts.append("\n\n🚫 *Out of scope — only mutual fund questions are supported.*")

    if result.get("pii_detected"):
        pii = ", ".join(result["pii_detected"])
        parts.append(f"\n\n⚠️ *Personal data detected and redacted: {pii}*")

    if result.get("educational_link"):
        parts.append(f"\n\n📚 [Investor Education]({result['educational_link']})")

    return "\n\n".join(parts)


# --- Build Gradio Interface ---

def create_interface() -> gr.ChatInterface:
    """Create and configure the Gradio ChatInterface."""
    demo = gr.ChatInterface(
        fn=chat_fn,
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        examples=EXAMPLE_QUESTIONS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="green",
        ),
        retry_btn="🔄 Retry",
        undo_btn="↩️ Undo",
        clear_btn="🗑️ Clear",
    )
    return demo


# --- Main ---

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
