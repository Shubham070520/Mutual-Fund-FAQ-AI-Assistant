"""
Streamlit UI — Mutual Fund FAQ Assistant.

Provides a chat interface for asking factual questions about HDFC Mutual Fund schemes.
Connects to the FastAPI backend at API_BASE_URL.

Features:
- Disclaimer banner (facts-only, no investment advice)
- 3 clickable example questions
- Chat history with streaming-style display
- Response display with source URL, footer, and metadata
- Sidebar with supported schemes and system health
- Copy-to-clipboard for responses
"""

import sys
from pathlib import Path

import streamlit as st
import requests

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import API_BASE_URL, AMC_NAME

# --- Constants ---

APP_TITLE = "Mutual Fund FAQ Assistant"
APP_ICON = "📊"
DISCLAIMER = (
    "**Facts-only. No investment advice.** "
    "This assistant provides factual information about HDFC Mutual Fund schemes only. "
    "It does not offer investment recommendations, comparisons, or opinions."
)

EXAMPLE_QUESTIONS = [
    "What is the expense ratio of HDFC Mid Cap Fund?",
    "What is the minimum SIP amount for HDFC Small Cap Fund?",
    "Who is the fund manager of HDFC Defence Fund?",
]

# Category display names
CATEGORY_LABELS = {
    "mid-cap": "📈 Mid-Cap",
    "small-cap": "📊 Small-Cap",
    "gold-etf-fof": "🥇 Gold ETF FoF",
    "sectoral-defence": "🛡️ Sectoral (Defence)",
    "silver-etf-fof": "🥈 Silver ETF FoF",
}


# --- API Client ---

def call_api(endpoint: str, method: str = "GET", json_body: dict | None = None) -> dict | None:
    """
    Make an API call to the FastAPI backend.

    Args:
        endpoint: API path (e.g., "/query")
        method: HTTP method
        json_body: Request body for POST requests

    Returns:
        Response JSON dict, or None on failure
    """
    url = f"{API_BASE_URL}{endpoint}"

    try:
        if method == "POST":
            response = requests.post(url, json=json_body, timeout=60)
        else:
            response = requests.get(url, timeout=10)

        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to the API server. Please ensure it's running."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The server might be busy."}
    except requests.exceptions.HTTPError as e:
        return {"error": f"API error: {e.response.status_code} — {e.response.text[:200]}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def send_query(query: str, scheme_filter: str | None = None) -> dict:
    """Send a query to the /query endpoint and return the response envelope."""
    body = {"query": query}
    if scheme_filter:
        body["scheme_filter"] = scheme_filter

    result = call_api("/query", method="POST", json_body=body)

    if result and "error" in result:
        return {
            "answer": f"⚠️ {result['error']}",
            "is_refusal": True,
            "intent": "error",
            "source_url": None,
            "educational_link": None,
            "last_updated": "",
            "scheme": None,
            "context_used": 0,
            "latency_ms": 0,
            "warnings": [],
            "pii_detected": [],
        }

    return result


# --- Page Configuration ---

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Session State Initialization ---

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "schemes" not in st.session_state:
        result = call_api("/schemes")
        if result and "schemes" in result:
            st.session_state.schemes = result["schemes"]
            st.session_state.amc_name = result.get("amc_name", AMC_NAME)
        else:
            st.session_state.schemes = []
            st.session_state.amc_name = AMC_NAME

    if "health" not in st.session_state:
        st.session_state.health = call_api("/health") or {}

    if "selected_scheme" not in st.session_state:
        st.session_state.selected_scheme = None


init_session_state()


# --- Sidebar ---

def render_sidebar():
    """Render the sidebar with schemes list and system health."""
    with st.sidebar:
        st.header(f"📋 {st.session_state.amc_name}")
        st.caption("Supported Schemes")

        # Scheme list
        if st.session_state.schemes:
            # "All Schemes" option
            scheme_names = ["All Schemes"] + [s["name"] for s in st.session_state.schemes]
            selected = st.selectbox(
                "Filter by scheme:",
                scheme_names,
                key="scheme_selector",
            )
            st.session_state.selected_scheme = selected if selected != "All Schemes" else None

            st.divider()

            for scheme in st.session_state.schemes:
                cat = scheme.get("category", "")
                label = CATEGORY_LABELS.get(cat, cat)
                st.markdown(f"**{scheme['name']}**  \n{label}")
                st.caption(f"[View on Groww]({scheme.get('groww_url', '#')})")
        else:
            st.warning("Could not load scheme list. Check API connection.")

        # System health
        st.divider()
        st.caption("**System Status**")
        health = st.session_state.health
        if health and "error" not in health:
            col1, col2 = st.columns(2)
            with col1:
                status = "🟢 Online" if health.get("status") == "ok" else "🔴 Offline"
                st.markdown(f"**API:** {status}")
            with col2:
                llm = "🟢 Ready" if health.get("llm_available") else "🟡 Unavailable"
                st.markdown(f"**LLM:** {llm}")
            chunks = health.get("vector_store_count", 0)
            st.caption(f"Vector store: {chunks} chunks indexed")
        else:
            st.markdown("**API:** 🔴 Offline")
            st.caption("Start the API server: `python -m src.api.main`")

        # Clear chat button
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


render_sidebar()


# --- Main Content ---

# Title and disclaimer
st.title(f"{APP_ICON} {APP_TITLE}")
st.info(DISCLAIMER)

# --- Welcome section (shown when no messages) ---

def render_welcome():
    """Render welcome section with example questions."""
    if st.session_state.messages:
        return

    st.markdown("### 👋 Welcome!")
    st.markdown("Ask factual questions about HDFC Mutual Fund schemes. Try one of these examples:")

    cols = st.columns(3)
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        with cols[i]:
            if st.button(question, key=f"example_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.pending_query = question
                st.rerun()

    st.divider()
    st.markdown(
        "**What you can ask about:** NAV, expense ratio, fund size (AUM), exit load, "
        "minimum SIP, fund manager, benchmark, category, and more."
    )
    st.markdown(
        "**What you can't ask:** Investment advice, fund comparisons, "
        "recommendations, or questions outside mutual funds."
    )


render_welcome()


# --- Chat History Display ---

def render_chat_history():
    """Render all messages in the chat history."""
    for msg in st.session_state.messages:
        role = msg["role"]

        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])

        elif role == "assistant":
            with st.chat_message("assistant", avatar="📊"):
                render_response(msg)


def render_response(msg: dict):
    """Render an assistant response with metadata."""
    # Main answer
    st.markdown(msg["content"])

    # Metadata row
    meta_parts = []

    if msg.get("source_url"):
        meta_parts.append(f"🔗 [Source]({msg['source_url']})")

    if msg.get("scheme"):
        meta_parts.append(f"📋 {msg['scheme']}")

    if msg.get("last_updated"):
        meta_parts.append(f"📅 {msg['last_updated']}")

    if msg.get("latency_ms"):
        meta_parts.append(f"⚡ {msg['latency_ms']:.0f}ms")

    if msg.get("context_used"):
        meta_parts.append(f"📄 {msg['context_used']} sources")

    if meta_parts:
        st.caption(" | ".join(meta_parts))

    # Refusal indicators
    if msg.get("is_refusal"):
        intent = msg.get("intent", "")
        if intent == "advisory":
            st.caption("🚫 Investment advice not provided — this is a facts-only assistant")
        elif intent == "out_of_scope":
            st.caption("🚫 Out of scope — only mutual fund questions are supported")

    # PII warning
    if msg.get("pii_detected"):
        pii_types = ", ".join(msg["pii_detected"])
        st.warning(f"⚠️ Personal data detected and redacted: {pii_types}")

    # Educational link
    if msg.get("educational_link"):
        st.caption(f"📚 [Investor Education]({msg['educational_link']})")

    # Copy button
    st.button(
        "📋 Copy",
        key=f"copy_{id(msg)}",
        on_click=lambda text=msg["content"]: st.session_state.update({"clipboard": text}),
        help="Copy response to clipboard",
    )


render_chat_history()


# --- Pending Query Processing ---

def process_pending_query():
    """Process a query that was set by an example button click."""
    if "pending_query" not in st.session_state:
        return

    query = st.session_state.pending_query
    del st.session_state.pending_query

    scheme_filter = st.session_state.get("selected_scheme")

    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="📊"):
        with st.spinner("Searching and generating response..."):
            response = send_query(query, scheme_filter=scheme_filter)

        if response:
            msg = {
                "role": "assistant",
                "content": response.get("answer", "No response received."),
                "source_url": response.get("source_url"),
                "scheme": response.get("scheme"),
                "last_updated": response.get("last_updated", ""),
                "is_refusal": response.get("is_refusal", False),
                "intent": response.get("intent", ""),
                "latency_ms": response.get("latency_ms", 0),
                "context_used": response.get("context_used", 0),
                "educational_link": response.get("educational_link"),
                "pii_detected": response.get("pii_detected", []),
            }
            st.session_state.messages.append(msg)
            render_response(msg)
        else:
            error_msg = {
                "role": "assistant",
                "content": "⚠️ Failed to get a response from the API.",
                "is_refusal": True,
                "intent": "error",
            }
            st.session_state.messages.append(error_msg)
            st.error("Could not connect to the API.")


process_pending_query()


# --- Chat Input ---

if prompt := st.chat_input("Ask a factual question about HDFC Mutual Fund schemes..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Get response
    scheme_filter = st.session_state.get("selected_scheme")

    with st.chat_message("assistant", avatar="📊"):
        with st.spinner("Searching and generating response..."):
            response = send_query(prompt, scheme_filter=scheme_filter)

        if response:
            msg = {
                "role": "assistant",
                "content": response.get("answer", "No response received."),
                "source_url": response.get("source_url"),
                "scheme": response.get("scheme"),
                "last_updated": response.get("last_updated", ""),
                "is_refusal": response.get("is_refusal", False),
                "intent": response.get("intent", ""),
                "latency_ms": response.get("latency_ms", 0),
                "context_used": response.get("context_used", 0),
                "educational_link": response.get("educational_link"),
                "pii_detected": response.get("pii_detected", []),
            }
            st.session_state.messages.append(msg)
            render_response(msg)
        else:
            error_msg = {
                "role": "assistant",
                "content": "⚠️ Failed to get a response from the API.",
                "is_refusal": True,
                "intent": "error",
            }
            st.session_state.messages.append(error_msg)
            st.error("Could not connect to the API. Make sure it's running on " + API_BASE_URL)


# --- Footer ---

st.divider()
st.caption(
    f"Mutual Fund FAQ Assistant — {AMC_NAME} | "
    "Powered by RAG (Retrieval-Augmented Generation) | "
    "Facts-only. No investment advice."
)
