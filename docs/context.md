# Project Context: Mutual Fund FAQ Assistant

## Project Overview

A **facts-only FAQ assistant** for mutual fund schemes, using **Groww** as the reference product context. Built as a lightweight **RAG (Retrieval-Augmented Generation)** system that answers objective, verifiable queries by retrieving information exclusively from official public sources (AMC websites, AMFI, SEBI).

**Core Principle:** Accuracy over intelligence. No advice, no opinions, no recommendations.

---

## Target Users

- Retail investors comparing mutual fund schemes
- Customer support and content teams handling repetitive mutual fund queries

---

## Scope

### AMC & Schemes

- Select **one** Asset Management Company (AMC)
- Choose **3–5 mutual fund schemes** with category diversity (e.g., large-cap, flexi-cap, ELSS)

### Corpus

- Collect **15–25 official public URLs** covering:
  - Scheme factsheets
  - KIM (Key Information Memorandum)
  - SID (Scheme Information Document)
  - AMC FAQ/help pages
  - AMFI/SEBI guidance pages
  - Statement and tax document download guides

---

## Assistant Requirements

### Must Answer (Facts-Only)

- Expense ratio of a scheme
- Exit load details
- Minimum SIP amount
- ELSS lock-in period
- Riskometer classification
- Benchmark index
- Process to download statements or capital gains reports

### Response Format

- **Max 3 sentences** per response
- **Exactly one citation link** per response
- Footer: `"Last updated from sources: <date>"`

### Refusal Handling

- Refuse non-factual / advisory queries (e.g., "Should I invest?", "Which fund is better?")
- Refusals must be polite, reinforce facts-only limitation, and include a relevant educational link (AMFI/SEBI)

---

## Constraints

| Category | Rule |
|---|---|
| **Data Sources** | Only official public sources (AMC, AMFI, SEBI). No third-party blogs or aggregators. |
| **Privacy** | No PAN, Aadhaar, account numbers, OTPs, email, or phone numbers collected/stored. |
| **Content** | No investment advice, no recommendations, no performance comparisons, no return calculations. For performance queries → link to official factsheet only. |
| **Transparency** | Short, factual, verifiable responses. Every answer must include source link + last updated date. |

---

## UI Requirements (Minimal)

- Welcome message
- Three example questions
- Visible disclaimer: `"Facts-only. No investment advice."`

---

## Architecture Approach

**RAG Pipeline:**

1. **Ingestion** — Crawl/scrape official documents from curated URLs
2. **Chunking & Embedding** — Split documents into chunks, generate vector embeddings
3. **Storage** — Store embeddings in a vector database
4. **Retrieval** — On user query, retrieve top-k relevant chunks
5. **Generation** — LLM generates a concise, facts-only answer grounded in retrieved context
6. **Guardrails** — Enforce refusal logic, citation format, and response constraints

---

## Deliverables

- Application code (RAG pipeline + UI)
- README with setup instructions, selected AMC/schemes, architecture overview, known limitations
- Disclaimer snippet: `"Facts-only. No investment advice."`

---

## Success Criteria

- Accurate retrieval of factual mutual fund information
- Strict adherence to facts-only responses
- Consistent inclusion of valid source citations
- Proper refusal of advisory queries
- Clean, minimal, and user-friendly interface
