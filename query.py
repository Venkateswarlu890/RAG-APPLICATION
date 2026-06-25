"""
query.py — RAG Query Pipeline

Responsibilities:
  1. Load the persisted ChromaDB vector database from disk
  2. Embed the user's natural-language question with the same model used at
     ingestion time (`text-embedding-004`)
  3. Retrieve the top-k most semantically similar document chunks
  4. Build a strictly-grounded system prompt with inline source citations
  5. Call Gemini to generate the final answer
  6. Return a structured result dict with answer, citations, and raw context

Can also be run directly as a CLI interactive loop:
    python src/query.py
"""

import os
import sys

# Force UTF-8 output so emoji print correctly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from google import genai                                              # new SDK
# pyrefly: ignore [missing-import]
from google.genai import types as genai_types
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

from src.config import (
    GEMINI_API_KEY,
    DB_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    TOP_K_RESULTS,
)

load_dotenv()
# Initialise the new google-genai client once at module level
_genai_client = genai.Client(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# CUSTOM EMBEDDING FUNCTION  (uses new google-genai SDK — no deprecated code)
# ---------------------------------------------------------------------------

class GeminiEmbeddingFunction(EmbeddingFunction):
    """ChromaDB-compatible embedding fn using the new google-genai SDK."""

    def __init__(self, api_key: str, model_name: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model  = model_name

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        response = self._client.models.embed_content(
            model=self._model,
            contents=list(input),
        )
        return [emb.values for emb in response.embeddings]


# ---------------------------------------------------------------------------
# SYSTEM PROMPT — Hallucination prevention instructions
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a professional and precise document Q&A assistant. "
    "Your sole source of truth is the CONTEXT INFORMATION provided below — "
    "never use your own parametric memory or external knowledge.\n\n"
    "Rules you MUST follow:\n"
    "1. Answer ONLY using facts present in the provided context.\n"
    "2. After every factual claim, cite the source inline in the format: "
    "   (source_filename.pdf, Page X).\n"
    "3. If the provided context does not contain enough information to answer "
    "   the question, respond with exactly: "
    "   'I am sorry, but the provided documents do not contain the answer to your question.'\n"
    "4. Do NOT fabricate, extrapolate, or guess any information.\n"
    "5. Be concise, structured, and professional."
)


# ---------------------------------------------------------------------------
# CORE QUERY FUNCTION
# ---------------------------------------------------------------------------

def query_rag_pipeline(
    user_query: str,
    db_path: str = str(DB_DIR),
    k: int = TOP_K_RESULTS,
) -> dict:
    """
    Run the full RAG query pipeline for a single user question.

    Args:
        user_query:  The natural-language question from the user.
        db_path:     Path to the persisted ChromaDB directory.
        k:           Number of top-k chunks to retrieve.

    Returns:
        A dict with keys:
          • "answer"      — The LLM-generated grounded answer (str)
          • "citations"   — List of citation strings (list[str])
          • "raw_context" — The raw retrieved chunk texts (list[str])
          • "metadatas"   — The metadata for each chunk (list[dict])
          • "error"       — Error message string if something went wrong, else None
    """
    # ── 1. Connect to persisted ChromaDB ────────────────────────────────
    try:
        client = chromadb.PersistentClient(path=db_path)
        embedding_fn = GeminiEmbeddingFunction(
            api_key=GEMINI_API_KEY,
            model_name=EMBEDDING_MODEL,
        )
        collection = client.get_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except Exception as exc:
        msg = (
            f"Could not load the vector database: {exc}\n"
            "Have you run `python src/ingest.py` yet?"
        )
        return {"answer": msg, "citations": [], "raw_context": [], "metadatas": [], "error": str(exc)}

    # ── 2. Query for top-k relevant chunks ──────────────────────────────
    try:
        results = collection.query(
            query_texts=[user_query],
            n_results=min(k, collection.count()),   # guard against empty DB
        )
    except Exception as exc:
        msg = f"Vector search failed: {exc}"
        return {"answer": msg, "citations": [], "raw_context": [], "metadatas": [], "error": str(exc)}

    retrieved_docs      = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]
    retrieved_distances = results.get("distances", [[]])[0]

    # ── 3. Build context blocks with inline citations ────────────────────
    context_blocks: list[str] = []
    citations:      list[str] = []

    for doc, meta, dist in zip(retrieved_docs, retrieved_metadatas, retrieved_distances):
        source    = meta.get("source", "unknown")
        page      = meta.get("page", "?")
        relevance = round((1 - dist) * 100, 1) if dist is not None else "N/A"
        citation  = f"{source}, Page {page}"

        context_blocks.append(
            f"[Source: {citation}  |  Relevance: {relevance}%]\n{doc}"
        )
        citations.append(citation)

    context_payload = "\n\n---\n\n".join(context_blocks)

    # ── 4. Formulate grounded prompt ─────────────────────────────────────
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT INFORMATION:\n"
        f"{context_payload}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"GROUNDED ANSWER:"
    )

    # ── 5. Generate answer with Gemini (new google-genai SDK) ───────────────
    try:
        response = _genai_client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
        )
        answer = response.text
    except Exception as exc:
        answer = f"LLM generation failed: {exc}"
        return {
            "answer": answer,
            "citations": citations,
            "raw_context": retrieved_docs,
            "metadatas": retrieved_metadatas,
            "error": str(exc),
        }

    return {
        "answer":      answer,
        "citations":   citations,
        "raw_context": retrieved_docs,
        "metadatas":   retrieved_metadatas,
        "error":       None,
    }


# ---------------------------------------------------------------------------
# INTERACTIVE CLI LOOP  (when run directly: python src/query.py)
# ---------------------------------------------------------------------------

def run_cli() -> None:
    """Start an interactive command-line Q&A session."""
    print("=" * 60)
    print("  🤖  Document Q&A Bot — CLI Mode")
    print("  Type your question and press Enter.")
    print("  Type  'exit'  or  'quit'  to stop.")
    print("=" * 60)

    while True:
        print()
        try:
            user_input = input("❓  Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print("\n👋  Goodbye!")
            break

        print("\n⏳  Searching documents and generating answer…\n")
        result = query_rag_pipeline(user_input)

        print("─" * 60)
        print("💬  ANSWER:\n")
        print(result["answer"])
        print()

        if result["citations"]:
            print("📚  SOURCES RETRIEVED:")
            for i, cite in enumerate(result["citations"], start=1):
                print(f"    {i}. {cite}")

        print("─" * 60)


if __name__ == "__main__":
    run_cli()
