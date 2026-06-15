# Mutual Fund FAQ AI Assistant

A **facts-only** RAG (Retrieval-Augmented Generation) chatbot that answers factual questions about HDFC Mutual Fund schemes. It does **not** provide investment advice, recommendations, or comparisons.

## Architecture

```
User Query
    │
    ▼
┌──────────────────┐
│  Streamlit       │      
└───────┬──────────┘
        │ POST /query
        ▼
┌──────────────────────────────────────────────────┐
│                  FastAPI Backend                  │
│                                                   │
│  1. PII Sanitizer    ── detect & redact PAN,     │
│                        Aadhaar, phone, email      │
│  2. Intent Classifier── keyword + LLM fallback   │
│       ├─ ADVISORY   → refusal + edu link          │
│       ├─ OUT_OF_SCOPE → refusal                   │
│       └─ FACTUAL    → proceed to retrieval        │
│  3. Retriever      ── ChromaDB cosine search + MMR│
│  4. LLM Generator  ── Groq (Llama 3.3 70B)       │
│  5. Post-Processor ── trim, cite URL, footer      │
└──────────────────────────────────────────────────┘
        │
        ▼
   Response + metadata
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Groq — Llama 3.3 70B Versatile |
| **Embeddings** | BAAI/bge-small-en-v1.5 (384-dim) |
| **Vector DB** | ChromaDB (local persistent) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit (+ optional Gradio) |
| **Scraping** | BeautifulSoup4, PyMuPDF, Playwright |
| **Text Processing** | LangChain Text Splitters |
| **Scheduler** | GitHub Actions (daily at 10:30 AM IST) |
| **Testing** | Pytest (279 tests across 10 files) |

## Supported Schemes

| Category | Scheme |
|----------|--------|
| Mid-Cap | HDFC Mid Cap Fund |
| Small-Cap | HDFC Small Cap Fund |
| Gold ETF FoF | HDFC Gold ETF Fund of Fund |
| Sectoral (Defence) | HDFC Defence Fund |
| Silver ETF FoF | HDFC Silver ETF FoF |

## What You Can Ask

**Factual queries** (will be answered):
- "What is the expense ratio of HDFC Mid Cap Fund?"
- "What is the minimum SIP amount for HDFC Small Cap Fund?"
- "Who is the fund manager of HDFC Defence Fund?"
- "What is the benchmark index for HDFC Gold ETF Fund of Fund?"
- "What is the fund size (AUM) of HDFC Silver ETF FoF?"

**Advisory queries** (politely refused):
- "Should I invest in HDFC Mid Cap Fund?"
- "Which fund is better?"

**Out-of-scope queries** (refused):
- Weather, sports, politics, cooking, stock tips, etc.

## Project Structure

```
RAG/
├── src/
│   ├── ingestion/          # Scraping & extraction
│   │   ├── scraper.py      # HTML/PDF scraper with Groww parser
│   │   ├── groww_parser.py # Structured data extraction from Groww
│   │   ├── cleaner.py      # Text cleaning pipeline
│   │   └── pipeline.py     # Full ingestion orchestrator
│   ├── processing/         # Chunking & embedding
│   │   ├── chunker.py      # Text chunking with metadata
│   │   ├── embeddings.py   # BGE embedding model
│   │   ├── vectorstore.py  # ChromaDB wrapper
│   │   └── indexer.py      # Embed + store pipeline
│   ├── retrieval/          # Vector search & reranking
│   │   ├── retriever.py    # ChromaDB retrieval with MMR
│   │   └── reranker.py     # Cross-encoder reranking (optional)
│   ├── generation/         # LLM prompts & response
│   │   ├── llm.py          # Groq client setup
│   │   ├── prompts.py      # System/user prompt templates
│   │   ├── generator.py    # Response generation
│   │   └── postprocessor.py# Validation, citation, footer
│   ├── guardrails/         # Safety & compliance
│   │   ├── sanitizer.py    # PII detection & redaction
│   │   ├── intent.py       # Intent classification (keyword + LLM)
│   │   └── refusal.py      # Refusal response generation
│   ├── api/                # FastAPI endpoints
│   │   ├── main.py         # /query, /health, /schemes
│   │   ├── schemas.py      # Pydantic models
│   │   └── middleware.py    # CORS, logging, error handlers
│   └── ui/                 # Frontend
│       ├── app.py          # Streamlit chat UI
│       └── gradio_app.py   # Gradio chat UI (alternative)
├── config/
│   ├── urls.json           # Curated URL corpus (7 sources)
│   ├── schemes.json        # Scheme metadata
│   └── settings.py         # All configuration constants
├── tests/                  # 279 tests (unit, integration, e2e, edge cases)
├── scripts/
│   └── view_embeddings.py  # CLI tool to inspect ChromaDB embeddings
├── docs/
│   ├── architecture.md     # System architecture document
│   └── implementation_plan.md
├── .github/workflows/
│   └── ingestion.yml       # Daily ingestion scheduler (10:30 AM IST)
├── data/
│   ├── raw/                # Scraped documents
│   ├── processed/          # Chunked text with metadata
│   └── vectordb/           # ChromaDB persistent storage
├── requirements.txt
├── .env.example
└── .gitignore
```

## Quick Start

### Prerequisites

- **Python 3.12+**
- **Groq API Key** — [Get one free at groq.com](https://console.groq.com/)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Shubham070520/Mutual-Fund-FAQ-AI-Assistant.git
cd Mutual-Fund-FAQ-AI-Assistant

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Run the Application

**Terminal 1 — Start the API backend:**
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Start the Streamlit frontend:**
```bash
python -m streamlit run src/ui/app.py --server.port 8501
```

- **API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Frontend:** http://localhost:8501

### Run Ingestion Pipeline (one-time data setup)

```bash
# Scrape → clean → chunk → enrich
python -m src.ingestion.pipeline

# Embed chunks → store in ChromaDB
python -m src.processing.indexer
```

### Run Tests

```bash
pytest tests/ -v
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Submit a question and get an answer |
| `GET` | `/health` | Health check with vector store count |
| `GET` | `/schemes` | List supported mutual fund schemes |

### Example Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the expense ratio of HDFC Mid Cap Fund?"}'
```

### Example Response

```json
{
  "answer": "The expense ratio of HDFC Mid Cap Fund is 0.76%.",
  "source_url": "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
  "last_updated": "2026-06-15",
  "intent": "factual",
  "scheme": "HDFC Mid Cap Fund",
  "is_refusal": false,
  "context_used": 5,
  "latency_ms": 1200
}
```

## Guardrails

| Guardrail | What It Does |
|-----------|-------------|
| **PII Detection** | Detects and redacts PAN, Aadhaar, phone numbers, and email from queries |
| **Intent Classification** | Two-layer (keyword + LLM) classifier: FACTUAL / ADVISORY / OUT_OF_SCOPE |
| **Advisory Refusal** | Politely declines investment advice with educational links (AMFI, SEBI) |
| **Citation Injection** | Every factual response includes exactly one source URL |
| **Response Validation** | Max 3 sentences, advisory leak detection, footer with date |

## Daily Ingestion Scheduler

A GitHub Actions workflow runs **daily at 10:30 AM IST** to keep the knowledge base fresh:

1. Scrapes all URLs from `config/urls.json`
2. Cleans and chunks the content
3. Generates BGE embeddings
4. Upserts vectors to local ChromaDB
5. Commits updated data back to the repo

Trigger manually via **Actions → Daily ingest → Run workflow**.

## Known Limitations

- Single AMC scope (HDFC Mutual Fund only)
- Data sourced from Groww scheme pages — subject to their page structure
- No real-time NAV (daily scrape only)
- English-only responses
- PDF parsing limited to text-based PDFs (no scanned documents)

## Disclaimer

> **Facts-only. No investment advice.** This assistant provides factual information about mutual fund schemes only. It does not offer investment recommendations, comparisons, or opinions. For investment guidance, visit [AMFI Investor Education](https://www.amfiindia.com/investor-education) or [SEBI Investor Education](https://www.sebi.gov.in/investor-education).

## License

This project is built for educational purposes.
