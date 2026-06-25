"""
main.py — Streamlit Web UI for the Document Q&A Bot

Run with:
    streamlit run src/main.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from src.config import (
    DB_DIR,
    DATA_DIR,
    CHROMA_COLLECTION_NAME,
    GENERATION_MODEL,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    SUPPORTED_EXTENSIONS,
)
from src.query import query_rag_pipeline


# ---------------------------------------------------------------------------
# Page configuration  (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Document Q&A Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS — Premium dark-mode design
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        color: #e6edf3;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(22, 27, 34, 0.95);
        border-right: 1px solid rgba(48, 54, 61, 0.6);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #58a6ff;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label {
        color: #8b949e;
    }

    /* ── Header banner ── */
    .hero-banner {
        background: linear-gradient(90deg, #1f3a5f 0%, #1a2f4e 50%, #0d2137 100%);
        border: 1px solid rgba(88, 166, 255, 0.25);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    .hero-banner h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        background: linear-gradient(90deg, #58a6ff, #a5d6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-banner p {
        color: #8b949e;
        font-size: 0.95rem;
        margin: 0;
    }

    /* ── Chat messages ── */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 16px;
        margin-bottom: 24px;
    }

    .chat-user {
        background: linear-gradient(135deg, #1f3a5f, #1a2f4e);
        border: 1px solid rgba(88, 166, 255, 0.3);
        border-radius: 16px 16px 4px 16px;
        padding: 14px 18px;
        margin-left: 15%;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }

    .chat-assistant {
        background: rgba(22, 27, 34, 0.9);
        border: 1px solid rgba(48, 54, 61, 0.6);
        border-radius: 16px 16px 16px 4px;
        padding: 14px 18px;
        margin-right: 8%;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }

    .chat-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .label-user      { color: #58a6ff; }
    .label-assistant { color: #3fb950; }

    /* ── Citations card ── */
    .citations-card {
        background: rgba(31, 111, 31, 0.08);
        border: 1px solid rgba(63, 185, 80, 0.25);
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 10px;
        font-size: 0.85rem;
        color: #8b949e;
    }
    .citations-card strong {
        color: #3fb950;
    }

    /* ── Input area ── */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background: rgba(22, 27, 34, 0.8) !important;
        border: 1px solid rgba(48, 54, 61, 0.8) !important;
        border-radius: 10px !important;
        color: #e6edf3 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        padding: 10px 24px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(31, 111, 235, 0.35);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(31, 111, 235, 0.5);
    }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(48, 54, 61, 0.6);
        border-radius: 12px;
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label {
        color: #8b949e !important;
        font-size: 0.8rem !important;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #58a6ff !important;
        font-weight: 700 !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(22, 27, 34, 0.6) !important;
        border: 1px solid rgba(48, 54, 61, 0.5) !important;
        border-radius: 8px !important;
        color: #8b949e !important;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(48, 54, 61, 0.5) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role": "user"|"assistant", "content": ..., "citations": [...]}

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

if "db_ready" not in st.session_state:
    # Check if the database folder exists and is non-empty
    st.session_state.db_ready = DB_DIR.exists() and any(DB_DIR.iterdir())


# ---------------------------------------------------------------------------
# Helper — check document count
# ---------------------------------------------------------------------------
def _count_documents() -> int:
    """Count supported documents in the data/ folder."""
    if not DATA_DIR.exists():
        return 0
    return sum(
        1 for f in DATA_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _count_db_chunks() -> int:
    """Return number of vectors in the ChromaDB collection, or 0 on error."""
    try:
        import chromadb
        from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
        from src.config import GEMINI_API_KEY, EMBEDDING_MODEL
        client = chromadb.PersistentClient(path=str(DB_DIR))
        emb_fn = GoogleGenerativeAiEmbeddingFunction(
            api_key=GEMINI_API_KEY,
            model_name=EMBEDDING_MODEL,
        )
        col = client.get_collection(name=CHROMA_COLLECTION_NAME, embedding_function=emb_fn)
        return col.count()
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    top_k = st.slider(
        "Top-K Chunks Retrieved",
        min_value=1, max_value=10, value=TOP_K_RESULTS,
        help="How many document chunks to retrieve per query. Higher = more context, slower response."
    )

    st.divider()
    st.markdown("## 📊 Status")

    doc_count = _count_documents()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", doc_count)
    with col2:
        st.metric("Queries", st.session_state.total_queries)

    if st.session_state.db_ready:
        st.success("✅ Database ready")
    else:
        st.warning("⚠️ Database not indexed yet")
        st.caption("Run `python src/ingest.py` first.")

    st.divider()
    st.markdown("## 🔬 Models")
    st.caption(f"**Generation:** `{GENERATION_MODEL}`")
    st.caption(f"**Embeddings:** `{EMBEDDING_MODEL}`")

    st.divider()
    st.markdown("## 🗂️ Pipeline")
    st.caption(f"Chunk size: `{CHUNK_SIZE}` chars")
    st.caption(f"Chunk overlap: `{CHUNK_OVERLAP}` chars")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main Area — Hero banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>🤖 Document Q&amp;A Bot</h1>
        <p>Ask questions about your documents. Answers are strictly grounded in your uploaded files — no hallucinations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Database not ready — show prominent warning
# ---------------------------------------------------------------------------
if not st.session_state.db_ready:
    st.error(
        "**⚠️ Vector database not found.**\n\n"
        "Before you can ask questions, you need to ingest your documents:\n\n"
        "```bash\n# 1. Add documents to the data/ folder\n# 2. Run:\npython src/ingest.py\n```",
        icon="🔴",
    )
    st.stop()


# ---------------------------------------------------------------------------
# Chat History Display
# ---------------------------------------------------------------------------
chat_placeholder = st.container()
with chat_placeholder:
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            st.markdown(
                f"""<div class="chat-user">
                    <div class="chat-label label-user">You</div>
                    {entry["content"]}
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            citations_html = ""
            if entry.get("citations"):
                cites = "".join(f"<li>{c}</li>" for c in entry["citations"])
                citations_html = (
                    f'<div class="citations-card"><strong>📚 Sources:</strong>'
                    f"<ol style='margin:6px 0 0 0;padding-left:18px'>{cites}</ol></div>"
                )
            st.markdown(
                f"""<div class="chat-assistant">
                    <div class="chat-label label-assistant">Assistant</div>
                    {entry["content"]}
                    {citations_html}
                </div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Query Input Form
# ---------------------------------------------------------------------------
st.divider()

with st.form(key="query_form", clear_on_submit=True):
    user_question = st.text_area(
        "Ask a question about your documents:",
        placeholder="e.g. What are the key findings of the annual report?",
        height=80,
        label_visibility="visible",
    )
    submitted = st.form_submit_button("🔍 Ask", use_container_width=False)


if submitted and user_question.strip():
    question = user_question.strip()

    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
        "citations": [],
    })
    st.session_state.total_queries += 1

    # Run pipeline with spinner
    with st.spinner("🔎 Searching documents and generating grounded answer…"):
        start_time = time.time()
        result = query_rag_pipeline(user_query=question, k=top_k)
        elapsed = round(time.time() - start_time, 2)

    # Add assistant response to history
    answer = result["answer"]
    citations = result["citations"]

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
    })

    # Show timing in sidebar (store in session)
    st.session_state.last_query_time = elapsed

    # Show raw context in an expander (optional debug view)
    if result.get("raw_context"):
        with st.expander(f"🔬 View Retrieved Context Chunks ({len(result['raw_context'])})  |  ⏱ {elapsed}s"):
            for i, (chunk, meta) in enumerate(
                zip(result["raw_context"], result["metadatas"]), start=1
            ):
                st.markdown(
                    f"**Chunk {i}** — `{meta.get('source', '?')}` | "
                    f"Page {meta.get('page', '?')} | "
                    f"Range {meta.get('chunk_range', '?')}"
                )
                st.text(chunk)
                st.divider()

    # Re-render updated history
    st.rerun()


elif submitted and not user_question.strip():
    st.warning("Please enter a question before submitting.", icon="⚠️")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; margin-top:40px; color:#484f58; font-size:0.8rem;">
        Powered by <strong style="color:#58a6ff">Google Gemini</strong> &amp;
        <strong style="color:#58a6ff">ChromaDB</strong> &nbsp;|&nbsp;
        RAG Pipeline &nbsp;|&nbsp; Answers grounded in your documents only.
    </div>
    """,
    unsafe_allow_html=True,
)
