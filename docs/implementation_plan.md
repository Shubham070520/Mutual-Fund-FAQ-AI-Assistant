# Implementation Plan: Mutual Fund FAQ Assistant

## Phase 1: Project Setup & AMC Selection (Week 1)

### Task 1.1: Initialize Project Structure
```
RAG/
├── src/
│   ├── ingestion/        # Scraping & extraction
│   ├── processing/       # Chunking & embedding
│   ├── retrieval/        # Vector search & reranking
│   ├── generation/       # LLM prompts & response
│   ├── guardrails/       # Refusal detection & validation
│   ├── api/              # FastAPI endpoints
│   └── ui/               # Streamlit/Gradio app
├── data/
│   ├── raw/              # Scraped documents
│   ├── processed/        # Chunked text
│   └── vectordb/         # ChromaDB storage
├── config/
│   ├── urls.json         # Curated URL corpus
│   ├── schemes.json      # Scheme metadata
│   └── settings.py       # App configuration
├── tests/
├── notebooks/            # Experimentation
├── requirements.txt
└── .env
```

**Files to create:**
- `requirements.txt` with core dependencies
- `.env.example` with API key placeholders
- `config/settings.py` with configuration constants

### Task 1.2: Select AMC & Schemes

**AMC:** HDFC Mutual Fund

**Finalized Schemes (5, category-diverse):**

| Category | Scheme Name | Groww URL |
|----------|-------------|------------|
| Mid-Cap | HDFC Mid Cap Fund | https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth |
| Small-Cap | HDFC Small Cap Fund | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| Gold ETF FoF | HDFC Gold ETF Fund of Fund | https://groww.in/mutual-funds/hdfc-gold-etf-fund-of-fund-direct-plan-growth |
| Sectoral (Defence) | HDFC Defence Fund | https://groww.in/mutual-funds/hdfc-defence-fund-direct-growth |
| Silver ETF FoF | HDFC Silver ETF FoF | https://groww.in/mutual-funds/hdfc-silver-etf-fof-direct-growth |

**Output:** `config/schemes.json`
```json
{
  "amc_name": "HDFC Mutual Fund",
  "schemes": [
    {
      "name": "HDFC Mid Cap Fund",
      "category": "mid-cap",
      "groww_url": "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
      "factsheet_url": "...",
      "kim_url": "...",
      "sid_url": "..."
    },
    {
      "name": "HDFC Small Cap Fund",
      "category": "small-cap",
      "groww_url": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"
    },
    {
      "name": "HDFC Gold ETF Fund of Fund",
      "category": "gold-etf-fof",
      "groww_url": "https://groww.in/mutual-funds/hdfc-gold-etf-fund-of-fund-direct-plan-growth"
    },
    {
      "name": "HDFC Defence Fund",
      "category": "sectoral-defence",
      "groww_url": "https://groww.in/mutual-funds/hdfc-defence-fund-direct-growth"
    },
    {
      "name": "HDFC Silver ETF FoF",
      "category": "silver-etf-fof",
      "groww_url": "https://groww.in/mutual-funds/hdfc-silver-etf-fof-direct-growth"
    }
  ]
}
```

### Task 1.3: Curate URL Corpus
- Collect 15–25 official public URLs
- **Primary source:** Groww scheme pages (5 URLs above)
- Supplementary sources: HDFC AMC official site, AMFI, SEBI
- Validate each URL is accessible
- Categorize by document type

**Output:** `config/urls.json`
```json
[
  {
    "url": "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
    "type": "scheme_page",
    "scheme": "HDFC Mid Cap Fund",
    "format": "html"
  },
  {
    "url": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    "type": "scheme_page",
    "scheme": "HDFC Small Cap Fund",
    "format": "html"
  },
  {
    "url": "https://groww.in/mutual-funds/hdfc-gold-etf-fund-of-fund-direct-plan-growth",
    "type": "scheme_page",
    "scheme": "HDFC Gold ETF Fund of Fund",
    "format": "html"
  },
  {
    "url": "https://groww.in/mutual-funds/hdfc-defence-fund-direct-growth",
    "type": "scheme_page",
    "scheme": "HDFC Defence Fund",
    "format": "html"
  },
  {
    "url": "https://groww.in/mutual-funds/hdfc-silver-etf-fof-direct-growth",
    "type": "scheme_page",
    "scheme": "HDFC Silver ETF FoF",
    "format": "html"
  }
]
```

**Acceptance Criteria:**
- [x] Project structure created
- [x] AMC selected with 3–5 schemes
- [x] 15–25 URLs collected and validated
- [x] Configuration files populated

---

## Phase 2: Data Ingestion Pipeline (Week 2)

### Task 2.1: Build Web Scraper
**File:** `src/ingestion/scraper.py`

```python
# Implement:
- HTML scraper using BeautifulSoup4
- PDF extractor using PyMuPDF
- Handle different page structures per source type
- Rate limiting and error handling
- Output: raw text + metadata
```

**Functions:**
```python
def scrape_html(url: str) -> dict:
    """Scrape HTML page and return text + metadata"""

def extract_pdf(url: str) -> dict:
    """Download and extract text from PDF"""

def scrape_url(url_entry: dict) -> dict:
    """Route to appropriate extractor based on format"""
```

### Task 2.2: Build Text Cleaner
**File:** `src/ingestion/cleaner.py`

```python
# Implement:
- Remove headers/footers/nav elements
- Strip excessive whitespace
- Normalize unicode characters
- Remove boilerplate legal text (optional)
- Preserve section headings for metadata
```

### Task 2.3: Build Chunker
**File:** `src/processing/chunker.py`

```python
# Use LangChain's RecursiveCharacterTextSplitter
# Parameters:
- chunk_size: 600 tokens
- chunk_overlap: 80 tokens
- separators: ["\n\n", "\n", ". ", " "]

# Preserve metadata per chunk:
- source_url
- scheme_name
- document_type
- section_heading
```

### Task 2.4: Run Ingestion Pipeline
**File:** `src/ingestion/pipeline.py`

```python
# Orchestrate:
1. Load urls.json
2. Scrape each URL
3. Clean extracted text
4. Chunk documents
5. Save chunks with metadata to data/processed/
```

**Acceptance Criteria:**
- [ ] All 15–25 URLs scraped successfully
- [ ] Text extracted cleanly from HTML and PDFs
- [ ] Chunks generated with correct metadata
- [ ] Chunks saved to disk as JSON/Parquet

---

## Phase 3: Vector Store & Embeddings (Week 3)

### Task 3.1: Set Up Embedding Model
**File:** `src/processing/embeddings.py`

```python
# Using BGE (BAAI General Embedding) from HuggingFace
from sentence_transformers import SentenceTransformer

# Option A: bge-small (384 dimensions, faster, lighter)
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# Option B: bge-base (768 dimensions, better accuracy)
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# Note: BGE models are optimized for retrieval tasks and
# perform well on financial/domain-specific text
```

### Task 3.2: Initialize ChromaDB
**File:** `src/processing/vectorstore.py`

```python
import chromadb

# Create persistent client
client = chromadb.PersistentClient(path="./data/vectordb")

# Create collection
collection = client.get_or_create_collection(
    name="mf_faq_corpus",
    metadata={"hnsw:space": "cosine"}
)
```

### Task 3.3: Embed & Store Chunks
**File:** `src/processing/indexer.py`

```python
# Pipeline:
1. Load processed chunks from disk
2. Generate embeddings for each chunk
3. Store in ChromaDB with metadata
4. Verify index count matches chunk count
```

**Storage format per entry:**
```python
collection.add(
    ids=[chunk_id],
    embeddings=[embedding_vector],
    documents=[chunk_text],
    metadatas=[{
        "source_url": "...",
        "scheme_name": "...",
        "document_type": "factsheet",
        "scraped_date": "2026-06-09"
    }]
)
```

**Acceptance Criteria:**
- [ ] Embedding model loaded successfully
- [ ] ChromaDB collection created
- [ ] All chunks embedded and stored
- [ ] Index queryable with test vectors

---

## Phase 4: Retrieval Pipeline (Week 3–4)

### Task 4.1: Build Retriever
**File:** `src/retrieval/retriever.py`

```python
def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    1. Embed user query
    2. Search ChromaDB for similar chunks
    3. Filter by similarity threshold (0.7)
    4. Return top-K chunks with metadata
    """
```

**Features:**
- Cosine similarity search
- Optional metadata filtering (by scheme, document type)
- MMR (Maximal Marginal Relevance) to reduce redundancy

### Task 4.2: Add Reranking (Optional)
**File:** `src/retrieval/reranker.py`

```python
# Use cross-encoder for reranking
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query: str, chunks: list, top_n: int = 3) -> list:
    """Rerank retrieved chunks for better relevance"""
```

### Task 4.3: Test Retrieval Quality
**File:** `tests/test_retrieval.py`

```python
# Test queries:
test_queries = [
    ("What is the expense ratio of HDFC Mid Cap Fund?", "factsheet"),
    ("What is the minimum SIP amount for HDFC Small Cap Fund?", "faq"),
    ("What is the benchmark index for HDFC Defence Fund?", "sid"),
    ("How to download capital gains statement?", "guidance"),
]

# Validate:
- Relevant chunks retrieved
- Correct source URLs returned
- Metadata filtering works
```

**Acceptance Criteria:**
- [ ] Retriever returns relevant chunks for test queries
- [ ] Similarity threshold filters noise
- [ ] Metadata filtering works correctly
- [ ] Retrieval latency < 500ms

---

## Phase 5: Guardrails & Refusal Logic (Week 4)

### Task 5.1: Build PII Sanitizer
**File:** `src/guardrails/sanitizer.py`

```python
import re

PII_PATTERNS = {
    "pan": r"[A-Z]{5}[0-9]{4}[A-Z]",
    "aadhaar": r"\d{4}\s?\d{4}\s?\d{4}",
    "phone": r"(\+91)?[-\s]?\d{10}",
    "email": r"[\w.-]+@[\w.-]+\.\w+",
}

def sanitize(query: str) -> str:
    """Strip PII patterns from user input"""
    for pattern in PII_PATTERNS.values():
        query = re.sub(pattern, "[REDACTED]", query)
    return query
```

### Task 5.2: Build Intent Classifier
**File:** `src/guardrails/intent.py`

```python
# Layer 1: Keyword-based detection
ADVISORY_KEYWORDS = [
    "should I invest", "which is better", "recommend",
    "suggest", "best fund", "good investment", "will it grow",
    "which fund", "compare", "better returns"
]

def is_advisory_keyword(query: str) -> bool:
    """Fast keyword-based advisory detection"""

# Layer 2: LLM-based classification (for ambiguous cases)
def classify_intent(query: str) -> str:
    """
    Returns: FACTUAL | ADVISORY | OUT_OF_SCOPE
    Uses LLM with confidence threshold 0.8
    """
```

### Task 5.3: Build Refusal Response Generator
**File:** `src/guardrails/refusal.py`

```python
REFUSAL_TEMPLATE = """
I can only provide factual information about mutual fund schemes. 
I cannot offer investment advice or recommendations.

For guidance on making investment decisions, please visit:
{educational_link}

Last updated from sources: {date}
"""

EDUCATIONAL_LINKS = {
    "default": "https://www.amfiindia.com/investor-education",
    "sebi": "https://www.sebi.gov.in/investor-education",
}

def generate_refusal(intent: str) -> str:
    """Generate polite refusal with educational link"""
```

**Acceptance Criteria:**
- [ ] PII patterns correctly detected and stripped
- [ ] Advisory queries refused with >90% accuracy
- [ ] Factual queries pass through without false refusals
- [ ] Refusal responses include educational links

---

## Phase 6: LLM Integration & Generation (Week 5)

### Task 6.1: Set Up LLM Client
**File:** `src/generation/llm.py`

```python
# Primary: Groq (fast inference via GroqCloud API)
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Available models:
# - llama-3.3-70b-versatile  (70B, best quality)
# - llama-3.1-8b-instant     (8B, fastest)
# - mixtral-8x7b-32768        (Mixtral, good balance)

# Groq provides OpenAI-compatible API, making it easy to
# integrate with LangChain and other frameworks
```

### Task 6.2: Design Prompt Templates
**File:** `src/generation/prompts.py`

```python
SYSTEM_PROMPT = """
You are a facts-only mutual fund FAQ assistant.

STRICT RULES:
1. Answer ONLY factual, verifiable questions about mutual fund schemes.
2. Use ONLY the provided context below. Do NOT use external knowledge.
3. Response must be MAXIMUM 3 sentences.
4. Include EXACTLY ONE source URL from the context in your answer.
5. End every response with: "Last updated from sources: <date>"
6. Do NOT provide investment advice, recommendations, or opinions.
7. Do NOT compare fund performance or calculate returns.
8. If the context doesn't contain the answer, say: 
   "I don't have this information in my current sources."
"""

def build_prompt(context_chunks: list[str], query: str, date: str) -> str:
    """Construct user prompt with context and query"""
```

### Task 6.3: Build Response Generator
**File:** `src/generation/generator.py`

```python
def generate_response(
    query: str,
    context_chunks: list[dict],
    llm_client
) -> str:
    """
    1. Build prompt with context
    2. Call LLM with temperature=0.1
    3. Return generated response
    """

LLM_PARAMS = {
    "temperature": 0.1,
    "max_tokens": 200,
    "top_p": 0.9,
}
```

### Task 6.4: Build Response Post-Processor
**File:** `src/generation/postprocessor.py`

```python
def validate_and_format(response: str, source_url: str, date: str) -> str:
    """
    1. Check sentence count ≤ 3 (trim if exceeded)
    2. Verify exactly one URL present
    3. Verify footer present (append if missing)
    4. Check for advisory language (flag if detected)
    5. Return formatted response
    """
```

**Acceptance Criteria:**
- [ ] LLM generates responses with correct format
- [ ] Temperature 0.1 produces consistent outputs
- [ ] Responses limited to 3 sentences max
- [ ] Citation URL included in every response
- [ ] Footer with date appended correctly

---

## Phase 7: API Layer (Week 6)

### Task 7.1: Build FastAPI Backend
**File:** `src/api/main.py`

```python
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Mutual Fund FAQ Assistant")

@app.post("/query")
async def answer_query(request: QueryRequest) -> QueryResponse:
    """
    1. Sanitize input
    2. Classify intent
    3. If advisory → return refusal
    4. If factual → retrieve + generate
    5. Validate & format response
    6. Return response with metadata
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""

@app.get("/schemes")
async def list_schemes():
    """Return list of supported schemes"""
```

### Task 7.2: Define API Schemas
**File:** `src/api/schemas.py`

```python
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    scheme_filter: str | None = None

class QueryResponse(BaseModel):
    answer: str
    source_url: str
    last_updated: str
    intent: str  # "factual" | "advisory" | "out_of_scope"
    scheme: str | None

class RefusalResponse(BaseModel):
    answer: str
    educational_link: str
    last_updated: str
```

### Task 7.3: Add Middleware
**File:** `src/api/middleware.py`

```python
# CORS configuration
# Request logging
# Error handling
# Rate limiting (optional)
```

**Acceptance Criteria:**
- [ ] `/query` endpoint returns correct responses
- [ ] Advisory queries return refusal responses
- [ ] `/schemes` returns scheme list
- [ ] API documented via FastAPI auto-docs
- [ ] Error handling covers edge cases

---

## Phase 8: User Interface (Week 6–7)

### Task 8.1: Build Streamlit UI
**File:** `src/ui/app.py`

```python
import streamlit as st

# Layout:
- Title: "Mutual Fund FAQ Assistant"
- Disclaimer banner: "Facts-only. No investment advice."
- Welcome message with 3 example questions (clickable)
- Chat input box
- Response display with:
  - Answer text
  - Source URL (hyperlink)
  - Footer with last-updated date
- Sidebar with supported schemes list
```

### Task 8.2: Add Chat Functionality
```python
# Chat history management
# Clickable example questions
# Loading spinner during generation
# Error display for API failures
# Copy-to-clipboard for responses
```

### Task 8.3: Alternative UI (Optional)
**File:** `src/ui/gradio_app.py`

```python
# Gradio chatbot interface as alternative
import gradio as gr

demo = gr.ChatInterface(
    fn=chat_fn,
    title="Mutual Fund FAQ Assistant",
    description="Facts-only. No investment advice.",
    examples=[...],
)
```

**Acceptance Criteria:**
- [ ] UI displays welcome message and disclaimer
- [ ] 3 example questions shown and clickable
- [ ] Chat input works and displays responses
- [ ] Source URL and footer visible in responses
- [ ] UI is responsive and user-friendly

---

## Phase 9: Integration & Testing (Week 7–8)

### Task 9.1: End-to-End Integration
```python
# Connect all components:
UI → API → Guardrails → Retriever → Generator → Post-processor → UI

# Test full flow:
1. User types query in UI
2. API receives request
3. Input sanitized
4. Intent classified
5. Context retrieved
6. Response generated
7. Response validated
8. Answer displayed with citation
```

### Task 9.2: Write Unit Tests
**File:** `tests/`

```python
# test_scraper.py — URL scraping correctness
# test_chunker.py — Chunk size and metadata
# test_retriever.py — Retrieval relevance
# test_guardrails.py — Refusal accuracy
# test_generator.py — Response format compliance
# test_api.py — Endpoint behavior
```

### Task 9.3: Write Integration Tests
```python
# Test scenarios:
test_cases = [
    # Factual queries (should answer)
    ("What is the expense ratio of HDFC Mid Cap Fund?", "factual"),
    ("What is the minimum SIP amount for HDFC Small Cap Fund?", "factual"),
    ("What is the benchmark index for HDFC Defence Fund?", "factual"),
    
    # Advisory queries (should refuse)
    ("Should I invest in HDFC Mid Cap Fund?", "advisory"),
    ("Which fund is better - HDFC Small Cap or HDFC Mid Cap?", "advisory"),
    ("Will HDFC Defence Fund give good returns?", "advisory"),
    
    # Out of scope (should refuse)
    ("What is the weather today?", "out_of_scope"),
    ("How to invest in stocks?", "out_of_scope"),
]
```

### Task 9.4: Edge Case Testing
```python
# Test:
- Empty queries
- Queries with PII (PAN, phone numbers)
- Very long queries
- Queries about unsupported schemes
- Queries with no matching context
- Multiple questions in one query
```

**Acceptance Criteria:**
- [x] End-to-end flow works without errors
- [x] Unit tests pass (>80% coverage)
- [x] Integration tests pass (>90% accuracy)
- [x] Edge cases handled gracefully
- [x] No PII leakage in responses
- [x] No advisory content in factual responses

---

## Phase 10: Documentation & Deployment (Week 8)

### Task 10.1: Write README
**File:** `README.md`

```markdown
# Mutual Fund FAQ Assistant

## Setup Instructions
- Prerequisites (Python 3.10+, API keys)
- Installation steps
- Environment configuration
- Running the application

## Selected AMC & Schemes
- AMC name
- List of schemes with categories

## Architecture Overview
- High-level diagram
- RAG pipeline explanation
- Guardrails description

## Known Limitations
- Static corpus
- PDF parsing limitations
- Single AMC scope
- English-only

## Disclaimer
Facts-only. No investment advice.
```

### Task 10.2: Create Docker Configuration
**Files:** `Dockerfile`, `docker-compose.yml`

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Task 10.3: Set Up Daily Ingestion Scheduler (GitHub Actions)
**File:** `.github/workflows/ingestion.yml`

```yaml
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
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Cache pip dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run Ingestion Pipeline
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: python -m src.ingestion.pipeline

      - name: Commit and push updated corpus
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/
          git diff --staged --quiet || git commit -m "chore: daily ingestion $(date -u +%Y-%m-%d)"
          git push
```

**Scheduler Behaviors:**
- Runs automatically every day at **10:30 AM IST (05:00 UTC)**
- Can be triggered manually via `workflow_dispatch` for on-demand updates
- Scrapes all URLs in `config/urls.json`
- Rebuilds ChromaDB index with fresh content
- Commits updated data back to the repository
- Logs success/failure to GitHub Actions dashboard

**Setup Steps:**
1. Create `.github/workflows/` directory
2. Add `ingestion.yml` with the workflow above
3. Add `GROQ_API_KEY` to repository secrets (Settings > Secrets > Actions)
4. Ensure workflow has write permissions (Settings > Actions > General > Workflow permissions)
5. Test manually via "Run workflow" button

### Task 10.4: Set Up CI/CD (Optional)
**File:** `.github/workflows/ci.yml`

```yaml
# Run tests on push
# Lint code
# Build Docker image
# Deploy (optional)
```

### Task 10.5: Final Review
- [ ] All acceptance criteria met
- [ ] Documentation complete
- [ ] Demo recorded
- [ ] Code reviewed
- [ ] Deployed (local or cloud)

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **1. Setup** | Week 1 | Project structure, AMC/schemes selected, URL corpus |
| **2. Ingestion** | Week 2 | Scraper, cleaner, chunker, processed data |
| **3. Vector Store** | Week 3 | Embeddings, ChromaDB, indexed corpus |
| **4. Retrieval** | Week 3–4 | Retriever, reranker, retrieval tests |
| **5. Guardrails** | Week 4 | PII sanitizer, intent classifier, refusal logic |
| **6. Generation** | Week 5 | LLM integration, prompts, post-processor |
| **7. API** | Week 6 | FastAPI endpoints, schemas, middleware |
| **8. UI** | Week 6–7 | Streamlit/Gradio app, chat interface |
| **9. Testing** | Week 7–8 | Unit tests, integration tests, edge cases |
| **10. Deployment** | Week 8 | README, Docker, scheduler, CI/CD, final review |

**Total Duration:** 8 weeks

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Poor PDF extraction | Fallback to manual text entry for critical documents |
| Low retrieval accuracy | Tune chunk size, overlap, and similarity threshold |
| LLM hallucination | Strict context-only prompting + low temperature |
| Advisory leakage | Dual-layer guardrails + post-processing validation |
| API latency | Cache frequent queries, optimize retrieval |
| Scope creep | Stick to facts-only; defer features to v2 |
