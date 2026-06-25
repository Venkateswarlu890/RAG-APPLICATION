"""
ingest.py — Document Ingestion Pipeline

Responsibilities:
  1. Scan the `data/` directory for supported documents (.pdf, .docx, .txt)
  2. Extract raw text page-by-page (preserving source + page metadata)
  3. Split extracted text into overlapping chunks
  4. Embed each chunk via Google `text-embedding-004`
  5. Persist everything into a local ChromaDB vector database

Run this script ONCE (or every time you add new documents):
    python src/ingest.py
"""

import os
import sys
import hashlib

# Force UTF-8 output so emoji print correctly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -- Make sure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from tqdm import tqdm

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai

from src.config import (
    GEMINI_API_KEY,
    DATA_DIR,
    DB_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SUPPORTED_EXTENSIONS,
)

# ---------------------------------------------------------------------------
# CUSTOM EMBEDDING FUNCTION  (uses new google-genai SDK — no deprecated code)
# ---------------------------------------------------------------------------

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function backed by the new google-genai SDK.
    Replaces the built-in GoogleGenerativeAiEmbeddingFunction which still
    depends on the deprecated google-generativeai package.
    """

    def __init__(self, api_key: str, model_name: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model  = model_name

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        response = self._client.models.embed_content(
            model=self._model,
            contents=list(input),
        )
        # response.embeddings is a list of ContentEmbedding objects
        return [emb.values for emb in response.embeddings]

# ---------------------------------------------------------------------------
# 1.  DOCUMENT EXTRACTION HELPERS
# ---------------------------------------------------------------------------

def extract_pdf_pages(file_path: str) -> list[dict]:
    """
    Extract text page-by-page from a PDF file.

    Returns a list of dicts:
        [{"text": "...", "metadata": {"source": "file.pdf", "page": 1}}, ...]
    """
    from pypdf import PdfReader

    extracted: list[dict] = []
    file_name = Path(file_path).name

    try:
        reader = PdfReader(file_path)
        for idx, page in enumerate(reader.pages):
            raw = page.extract_text()
            if raw and raw.strip():
                clean = " ".join(raw.split())   # collapse whitespace
                extracted.append({
                    "text": clean,
                    "metadata": {
                        "source": file_name,
                        "page": idx + 1,   # 1-indexed for human readability
                    },
                })
    except Exception as exc:
        print(f"  ⚠️  Error reading PDF '{file_name}': {exc}")

    return extracted


def extract_docx_pages(file_path: str) -> list[dict]:
    """
    Extract text paragraph-by-paragraph from a .docx file.

    Because Word documents don't have discrete page numbers, we group every
    30 paragraphs into a logical "page" for metadata purposes.
    """
    from docx import Document

    extracted: list[dict] = []
    file_name = Path(file_path).name
    GROUP_SIZE = 30   # paragraphs per logical page

    try:
        doc = Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        for page_num, start in enumerate(range(0, len(paragraphs), GROUP_SIZE), start=1):
            group = paragraphs[start : start + GROUP_SIZE]
            combined = " ".join(group)
            if combined:
                extracted.append({
                    "text": combined,
                    "metadata": {
                        "source": file_name,
                        "page": page_num,
                    },
                })
    except Exception as exc:
        print(f"  ⚠️  Error reading DOCX '{file_name}': {exc}")

    return extracted


def extract_txt_pages(file_path: str) -> list[dict]:
    """
    Extract text from a plain .txt file.

    Splits on blank lines to create logical page groups.
    """
    extracted: list[dict] = []
    file_name = Path(file_path).name
    GROUP_LINES = 50   # logical lines per page

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = [ln.strip() for ln in fh.readlines() if ln.strip()]

        for page_num, start in enumerate(range(0, len(lines), GROUP_LINES), start=1):
            group = lines[start : start + GROUP_LINES]
            combined = " ".join(group)
            if combined:
                extracted.append({
                    "text": combined,
                    "metadata": {
                        "source": file_name,
                        "page": page_num,
                    },
                })
    except Exception as exc:
        print(f"  ⚠️  Error reading TXT '{file_name}': {exc}")

    return extracted


def extract_document(file_path: str) -> list[dict]:
    """
    Dispatch extraction to the correct handler based on file extension.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_pdf_pages(file_path)
    elif ext == ".docx":
        return extract_docx_pages(file_path)
    elif ext == ".txt":
        return extract_txt_pages(file_path)
    else:
        print(f"  ⚠️  Unsupported file type: {ext}")
        return []


# ---------------------------------------------------------------------------
# 2.  TEXT CHUNKING
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split page-level documents into smaller, overlapping chunks.

    Strategy — Recursive character splitting:
      • Try to split on paragraph breaks (\\n\\n), then line breaks (\\n),
        then spaces, and finally character boundaries.
      • A sliding window with `chunk_overlap` ensures boundary context is
        preserved between consecutive chunks.

    Returns a list of dicts:
        [{"text": "...", "metadata": {"source": ..., "page": ..., "chunk_range": ...}}, ...]
    """
    chunks: list[dict] = []

    for page in pages:
        text: str = page["text"]
        metadata: dict = page["metadata"]
        text_length = len(text)
        start = 0

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]

            # Try to find a natural break point near the end of the window
            # so we don't cut mid-sentence. Priority: \n\n > \n > space
            if end < text_length:
                for sep in ("\n\n", "\n", " "):
                    last_sep = chunk_text.rfind(sep)
                    if last_sep != -1 and last_sep > chunk_size // 2:
                        end = start + last_sep + len(sep)
                        chunk_text = text[start:end]
                        break

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text.strip(),
                    "metadata": {
                        "source": metadata["source"],
                        "page": metadata["page"],
                        "chunk_range": f"{start}-{end}",
                    },
                })

            # Slide window forward (ensuring we always make progress)
            stride = max(chunk_size - chunk_overlap, 1)
            start += stride

    return chunks


# ---------------------------------------------------------------------------
# 3.  VECTOR DATABASE — SAVE
# ---------------------------------------------------------------------------

def _stable_chunk_id(source: str, page: int, chunk_range: str) -> str:
    """
    Generate a deterministic, collision-resistant ID for a chunk.
    Using a hash means re-running ingest won't create duplicate entries
    (ChromaDB upsert behaviour).
    """
    raw = f"{source}|{page}|{chunk_range}"
    return "id_" + hashlib.md5(raw.encode()).hexdigest()


def save_to_vector_db(chunks: list[dict], db_path: str = str(DB_DIR)) -> None:
    """
    Embed text chunks and persist them into a local ChromaDB database.

    • Uses Google `text-embedding-004` via ChromaDB's built-in embedding fn.
    • Uploads in batches of 100 to stay within API rate limits.
    • Upserts so the function is safely re-runnable (idempotent).
    """
    if not chunks:
        print("  ⚠️  No chunks to index. Is your data/ folder empty?")
        return

    client = chromadb.PersistentClient(path=db_path)

    embedding_fn = GeminiEmbeddingFunction(
        api_key=GEMINI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},   # cosine distance for semantic search
    )

    # Prepare batch data
    ids        = [_stable_chunk_id(c["metadata"]["source"],
                                   c["metadata"]["page"],
                                   c["metadata"]["chunk_range"]) for c in chunks]
    documents  = [c["text"]     for c in chunks]
    metadatas  = [c["metadata"] for c in chunks]

    # Upload in batches of 100 (avoids hitting API payload limits)
    BATCH_SIZE = 100
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\n📤  Uploading {len(chunks)} chunks in {total_batches} batch(es)…")
    for batch_idx in tqdm(range(total_batches), desc="  Indexing", unit="batch"):
        lo = batch_idx * BATCH_SIZE
        hi = lo + BATCH_SIZE
        collection.upsert(
            ids=ids[lo:hi],
            documents=documents[lo:hi],
            metadatas=metadatas[lo:hi],
        )

    print(f"\n✅  Successfully indexed {len(chunks)} chunks into '{CHROMA_COLLECTION_NAME}'.")
    print(f"    Database stored at: {db_path}\n")


# ---------------------------------------------------------------------------
# 4.  MAIN INGESTION PIPELINE
# ---------------------------------------------------------------------------

def run_ingestion() -> None:
    """
    Full ingestion pipeline:
      Scan data/ → Extract → Chunk → Embed → Persist to ChromaDB
    """
    print("=" * 60)
    print("  📂  Document Q&A Bot — Ingestion Pipeline")
    print("=" * 60)

    # ── Discover documents ──────────────────────────────────────────────
    data_path = Path(DATA_DIR)
    files = [
        str(f)
        for f in sorted(data_path.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        print(f"\n⚠️   No supported documents found in: {data_path}")
        print(f"    Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        print("    Please add your files and run ingest.py again.\n")
        return

    print(f"\n🔍  Found {len(files)} document(s) to ingest:")
    for f in files:
        print(f"    • {Path(f).name}")

    # ── Extract text ────────────────────────────────────────────────────
    all_pages: list[dict] = []
    print("\n📄  Extracting text…")
    for file_path in tqdm(files, desc="  Extracting", unit="file"):
        pages = extract_document(file_path)
        print(f"    ✔ {Path(file_path).name}  →  {len(pages)} page(s)")
        all_pages.extend(pages)

    if not all_pages:
        print("\n⚠️   No text was extracted. Check that your documents are text-based (not scanned images).")
        return

    print(f"\n   Total pages extracted: {len(all_pages)}")

    # ── Chunk ───────────────────────────────────────────────────────────
    print("\n✂️   Chunking text…")
    all_chunks = chunk_pages(all_pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"   Total chunks created: {len(all_chunks)}")

    # ── Save to vector DB ───────────────────────────────────────────────
    save_to_vector_db(all_chunks, db_path=str(DB_DIR))


if __name__ == "__main__":
    run_ingestion()
