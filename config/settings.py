"""
Application configuration constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTORDB_DIR = DATA_DIR / "vectordb"
CONFIG_DIR = BASE_DIR / "config"

# === API Keys ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# === API Server Configuration ===
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{API_PORT}")

# === LLM Configuration ===
LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"  # or "llama-3.1-8b-instant" for faster inference
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 200
LLM_TOP_P = 0.9

# === Embedding Configuration ===
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # or "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSIONS = 384  # 384 for bge-small, 768 for bge-base

# === Vector Database ===
VECTORDB_COLLECTION_NAME = "mf_faq_corpus"
VECTORDB_DISTANCE_METRIC = "cosine"

# === Retrieval Configuration ===
RETRIEVAL_TOP_K = 5
RETRIEVAL_SIMILARITY_THRESHOLD = 0.7
RETRIEVAL_USE_MMR = True
RERANK_ENABLED = False
RERANK_TOP_N = 3

# === Chunking Configuration ===
CHUNK_SIZE = 600  # tokens
CHUNK_OVERLAP = 80  # tokens
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " "]

# === Response Configuration ===
MAX_RESPONSE_SENTENCES = 3
CITATION_REQUIRED = True
FOOTER_TEXT = "Last updated from sources: {date}"

# === Guardrails ===
PII_DETECTION_ENABLED = True
INTENT_CLASSIFICATION_ENABLED = True
ADVISORY_CONFIDENCE_THRESHOLD = 0.8

# === Educational Links (for refusals) ===
EDUCATIONAL_LINKS = {
    "default": "https://www.amfiindia.com/investor-education",
    "sebi": "https://www.sebi.gov.in/investor-education",
}

# === AMC Configuration ===
AMC_NAME = "HDFC Mutual Fund"
