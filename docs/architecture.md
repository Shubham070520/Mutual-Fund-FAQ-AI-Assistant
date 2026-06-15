# Architecture: Mutual Fund FAQ Assistant (RAG-Based)

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Welcome Message | Example Questions | Disclaimer Display     │  │
│  │  Chat Input | Response Display with Citations                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Query        │  │ Guardrails   │  │ Response Formatter       │  │
│  │ Preprocessor │  │ (Refusal     │  │ (Citations, Footer,      │  │
│  │              │  │  Detection)  │  │  Sentence Limit)         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Embed    │  │ Vector   │  │ Retrieve │  │ Rerank           │   │
│  │ Query    │→ │ Search   │→ │ Top-K    │→ │ (Optional)       │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       LLM LAYER                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Prompt Template (System + Context + User Query)              │  │
│  │  → Generate Facts-Only Response                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA LAYER (Offline)                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Scrape   │  │ Chunk    │  │ Embed    │  │ Store in Vector  │   │
│  │ Official │  │ Documents│  │ Chunks   │  │ Database         │   │
│  │ URLs     │  │          │  │          │  │                  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Data Ingestion Pipeline (Offline)

#### 1.1 URL Corpus

**Selected AMC:** HDFC Mutual Fund

**Selected Schemes (5, category-diverse):**
| Category | Scheme Name | Source URL |
|----------|-------------|------------|
| Mid-Cap | HDFC Mid Cap Fund | https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth |
| Small-Cap | HDFC Small Cap Fund | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| Gold ETF FoF | HDFC Gold ETF Fund of Fund | https://groww.in/mutual-funds/hdfc-gold-etf-fund-of-fund-direct-plan-growth |
| Sectoral (Defence) | HDFC Defence Fund | https://groww.in/mutual-funds/hdfc-defence-fund-direct-growth |
| Silver ETF FoF | HDFC Silver ETF FoF | https://groww.in/mutual-funds/hdfc-silver-etf-fof-direct-growth |

**Source Types (15–25 URLs):**
- Groww scheme pages (primary source — HTML)
- Scheme factsheets (PDF/HTML from HDFC AMC)
- Key Information Memorandum (KIM)
- Scheme Information Document (SID)
- HDFC AMC FAQ/help pages
- AMFI investor guidance pages
- SEBI investor education pages
- Statement download guides

#### 1.2 Scraping & Extraction

```python
# Tools
- BeautifulSoup / Playwright (for HTML pages)
- PyMuPDF / pdfplumber (for PDFs)
- Custom extractors per source type

# Output
- Raw text with metadata (source URL, document type, scheme name, date)
```

**Metadata Schema:**
```json
{
  "source_url": "https://...",
  "document_type": "factsheet | kim | sid | faq | guidance",
  "scheme_name": "AMC Large Cap Fund",
  "scheme_category": "large-cap",
  "scraped_date": "2026-06-09",
  "content_hash": "sha256..."
}
```

#### 1.3 Chunking Strategy

```
Approach: Recursive Character Splitting with Overlap

Parameters:
- Chunk Size: 500–800 tokens
- Overlap: 50–100 tokens
- Split boundaries: Paragraph → Sentence → Word

Metadata preserved per chunk:
- parent_document_id
- source_url
- scheme_name
- section_heading (if available)
```

#### 1.4 Embedding & Storage

```
Embedding Model: BAAI/bge-small-en-v1.5 or BAAI/bge-base-en-v1.5 (from HuggingFace)
Vector Dimensions: 384 (bge-small) or 768 (bge-base)

Vector Database: ChromaDB (lightweight, local) or FAISS
```

**Storage Schema:**
```json
{
  "chunk_id": "uuid",
  "embedding": [0.12, -0.34, ...],
  "text": "The expense ratio of XYZ fund is 0.80%...",
  "metadata": {
    "source_url": "https://...",
    "scheme_name": "...",
    "document_type": "factsheet"
  }
}
```

---

### 2. Query Processing Pipeline (Online)

#### 2.1 Query Preprocessing

```python
Input: User's natural language question
Steps:
1. Sanitize input (strip PII patterns: PAN, Aadhaar, phone, email)
2. Classify query intent:
   - FACTUAL → proceed to retrieval
   - ADVISORY → trigger refusal
   - OUT_OF_SCOPE → trigger refusal
3. Normalize query (lowercase, strip noise)
```

#### 2.2 Guardrails: Refusal Detection

**Detection Methods:**
```
1. Keyword Matching:
   - "should I invest", "which is better", "recommend", "suggest",
     "best fund", "good investment", "will it grow", "returns"

2. LLM-Based Intent Classification:
   - System prompt classifies query as FACTUAL vs ADVISORY
   - Confidence threshold: 0.8

3. Regex Patterns:
   - "What (is|are|was)" → likely factual
   - "Should|Would|Could I" → likely advisory
   - "Which is (best|better)" → advisory
```

**Refusal Response Template:**
```
I can only provide factual information about mutual fund schemes. 
I cannot offer investment advice or recommendations.

For guidance on making investment decisions, please visit:
[AMFI Investor Education](https://www.amfiindia.com/investor-education)

Last updated from sources: <date>
```

#### 2.3 Retrieval

```python
Steps:
1. Embed user query using same embedding model
2. Vector similarity search (cosine similarity)
3. Retrieve top-K chunks (K = 5)
4. Optional: Cross-encoder reranking for relevance
5. Filter by scheme/document type if query is scheme-specific
```

**Retrieval Parameters:**
```
- Top-K: 5
- Similarity Threshold: 0.7
- MMR (Maximal Marginal Relevance): enabled to reduce redundancy
```

#### 2.4 Prompt Construction

```python
SYSTEM_PROMPT = """
You are a facts-only mutual fund FAQ assistant.
Rules:
1. Answer ONLY factual, verifiable questions about mutual fund schemes.
2. Use ONLY the provided context to answer.
3. Response must be MAX 3 sentences.
4. Include EXACTLY ONE source URL from the context.
5. End with: "Last updated from sources: <date>"
6. If the question is advisory/opinion-based, politely refuse and provide 
   an educational link (AMFI or SEBI).
7. Do NOT provide investment advice, recommendations, or performance comparisons.
"""

USER_PROMPT = """
Context (retrieved documents):
{context_chunks}

User Question: {query}

Answer (facts-only, max 3 sentences, include one source link):
"""
```

#### 2.5 Response Generation

```python
LLM Provider: Groq (fast inference via GroqCloud API)
Models: llama-3.3-70b-versatile or llama-3.1-8b-instant

Parameters:
- temperature: 0.1 (deterministic, factual)
- max_tokens: 200
- top_p: 0.9
```

#### 2.6 Response Post-Processing

```python
Validation:
1. Check sentence count ≤ 3
2. Verify exactly one URL present
3. Verify footer "Last updated from sources:" present
4. Check for PII leakage (reject if found)
5. Check for advisory language (reject if detected)

Formatting:
- Trim to 3 sentences if exceeded
- Append footer if missing
- Standardize URL format
```

---

### 3. User Interface

#### 3.1 Layout

```
┌─────────────────────────────────────────────────────┐
│  🏦 Mutual Fund FAQ Assistant                        │
│  ─────────────────────────────────────               │
│  ⚠️ Facts-only. No investment advice.                │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Welcome! I can answer factual questions about      │
│  mutual fund schemes. Try asking:                    │
│                                                      │
│  • "What is the expense ratio of [Scheme Name]?"    │
│  • "What is the minimum SIP amount for [Scheme]?"   │
│  • "What is the lock-in period for ELSS funds?"     │
│                                                      │
├─────────────────────────────────────────────────────┤
│  [Type your question here...]              [Send]   │
└─────────────────────────────────────────────────────┘
```

#### 3.2 Tech Stack (UI)

```
Option A: Streamlit (rapid prototyping)
Option B: Gradio (chat interface)
Option C: React + FastAPI (production-ready)
```

---

### 4. Tech Stack (Backend)

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Web Scraping | BeautifulSoup4, Playwright, PyMuPDF |
| Chunking | LangChain TextSplitter |
| Embeddings | BAAI/bge-small-en-v1.5 or BAAI/bge-base-en-v1.5 |
| Vector DB | ChromaDB or FAISS |
| LLM | Groq (llama-3.3-70b-versatile / llama-3.1-8b-instant) |
| Orchestration | LangChain / LlamaIndex |
| API | FastAPI |
| UI | Streamlit / Gradio / React |
| Guardrails | NeMo Guardrails / custom logic |

---

## Data Flow Diagram

```
[Offline: Data Preparation]

Official URLs
    │
    ▼
[Web Scraper] ──→ Raw HTML/PDF
    │
    ▼
[Text Extractor] ──→ Clean Text + Metadata
    │
    ▼
[Chunker] ──→ Text Chunks (500-800 tokens)
    │
    ▼
[Embedding Model] ──→ Vector Embeddings
    │
    ▼
[Vector Database] ──→ Stored Index

─────────────────────────────────────────────

[Online: Query Processing]

User Query
    │
    ▼
[Query Preprocessor] ──→ Sanitized Query
    │
    ▼
[Guardrails: Intent Classifier]
    │
    ├─── ADVISORY ──→ [Refusal Response] ──→ User
    │
    └─── FACTUAL ──→ [Embed Query]
                          │
                          ▼
                    [Vector Search] ──→ Top-5 Chunks
                          │
                          ▼
                    [Prompt Builder] ──→ System + Context + Query
                          │
                          ▼
                    [LLM] ──→ Draft Response
                          │
                          ▼
                    [Post-Processor] ──→ Validated Response
                          │
                          ▼
                    [Response to User]
                          │
                          ▼
                    "The expense ratio of XYZ Fund is 0.80% (Direct Plan).
                     Source: https://www.amc.com/factsheet/xyz
                     Last updated from sources: 2026-06-09"
```

---

## Scheduled Ingestion (GitHub Actions)

The data ingestion pipeline runs **daily at 10:30 AM IST (05:00 UTC)** via GitHub Actions to keep the corpus fresh.

```yaml
# .github/workflows/ingestion.yml
name: Daily Data Ingestion

on:
  schedule:
    - cron: '0 5 * * *'   # 10:30 AM IST = 05:00 UTC
  workflow_dispatch:        # Allow manual triggers

jobs:
  ingest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - name: Run Ingestion Pipeline
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: python -m src.ingestion.pipeline
      - name: Commit updated corpus
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/
          git diff --staged --quiet || git commit -m "chore: daily ingestion $(date -u +%Y-%m-%d)"
          git push
```

**Key Behaviors:**
- Runs automatically every day at 10:30 AM IST
- Can be triggered manually via `workflow_dispatch`
- Scrapes all URLs in `config/urls.json`
- Updates ChromaDB index with new/changed content
- Commits updated data back to the repository
- Logs success/failure to GitHub Actions dashboard

---

## Security & Compliance

| Concern | Mitigation |
|---------|-----------|
| PII Collection | Regex-based input sanitizer strips PAN, Aadhaar, phone, email patterns before processing |
| Advisory Leakage | Dual-layer guardrail: keyword filter + LLM intent classifier |
| Source Accuracy | Only whitelisted official URLs ingested; no third-party content |
| Hallucination | Strict "use only provided context" instruction + low temperature (0.1) |
| Citation Validity | Post-processor verifies URL exists in retrieved context |
| Data Freshness | Metadata tracks `scraped_date`; footer shows last-updated date |

---

## Known Limitations

1. **Static Corpus** — Data is scraped periodically, not real-time. Facts may be stale between updates.
2. **PDF Parsing** — Complex layouts in factsheets/SIDs may lead to extraction errors.
3. **Ambiguous Queries** — Queries that blend factual + advisory intent may be incorrectly refused.
4. **Single AMC Scope** — Limited to one AMC's schemes; cross-AMC queries not supported.
5. **Language** — English-only; no vernacular support.
6. **Performance Queries** — Cannot compute or compare returns; can only link to factsheets.

---

## Deployment Considerations

```
Development:
- Local ChromaDB + Groq API (free tier, fast inference)
- Streamlit UI for rapid iteration

Production:
- Managed vector DB (Pinecone / Weaviate)
- Groq API (production plan)
- React frontend + FastAPI backend
- Docker containerization
- Scheduled re-scraping (weekly/monthly)
```
