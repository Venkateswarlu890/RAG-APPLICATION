"""
config.py — Central configuration for the RAG pipeline.

All tuneable parameters live here so every other module can import
from a single source of truth. Changing a value here propagates
everywhere automatically.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env (must be done before reading os.getenv)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "❌  GEMINI_API_KEY is not set.\n"
        "    1. Open the .env file in the project root.\n"
        "    2. Paste your key: GEMINI_API_KEY=AIza...\n"
        "    3. Get a free key at https://aistudio.google.com/app/apikey"
    )

# ---------------------------------------------------------------------------
# Directory Paths  (resolved relative to the project root, not this file)
# ---------------------------------------------------------------------------
# Project root is two levels up from src/config.py  →  document-qa-bot/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
DB_DIR: Path = PROJECT_ROOT / "db"

# Ensure data dir exists so users aren't confused by missing folder
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ChromaDB / Vector Store Settings
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_NAME: str = "document_knowledge_base"

# ---------------------------------------------------------------------------
# Google AI Model Names
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "models/gemini-embedding-001"
GENERATION_MODEL: str = "gemini-2.5-flash"   # Latest stable Flash

# ---------------------------------------------------------------------------
# Chunking Parameters
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 1000        # Maximum characters per chunk
CHUNK_OVERLAP: int = 200      # Overlap between consecutive chunks

# ---------------------------------------------------------------------------
# Retrieval Parameters
# ---------------------------------------------------------------------------
TOP_K_RESULTS: int = 5        # Number of top chunks returned per query

# ---------------------------------------------------------------------------
# Supported file extensions for document ingestion
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".txt")
