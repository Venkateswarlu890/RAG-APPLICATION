# 📄 Document Q&A Bot — RAG Pipeline

A production-quality **Retrieval-Augmented Generation (RAG)** application that answers questions grounded strictly in your own documents (PDFs, DOCX, TXT). Powered by **Google Gemini 2.5 Flash** and **ChromaDB**.

---

## 🏗️ Architecture

```
User Query ──► Embed Query ──► ChromaDB Vector Search ──► Top-k Chunks
                                                                │
                                                                ▼
                                                    Grounded Prompt Builder
                                                                │
                                                                ▼
                                                    Gemini 2.5 Flash LLM
                                                                │
                                                                ▼
                                                    Answer + Citations
```

---

## 📁 Project Structure

```
document-qa-bot/
├── .env                  # API keys (never commit this!)
├── .gitignore
├── README.md
├── requirements.txt
├── data/                 # Drop your documents here
│   ├── business_doc.pdf
│   ├── science_paper.pdf
│   └── factsheet.docx
├── db/                   # Auto-created — persistent ChromaDB storage
└── src/
    ├── __init__.py
    ├── config.py         # Central configuration constants
    ├── ingest.py         # Ingestion pipeline (run once)
    ├── query.py          # Query pipeline (core RAG logic)
    └── main.py           # Streamlit UI
```

---

## ⚙️ Setup & Installation

### 1. Clone / Navigate to Project

```bash
cd document-qa-bot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate It

- **Windows:** `venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Your API Key

Edit `.env` and paste your key:

```
GEMINI_API_KEY=AIza...your_real_key_here
```

Get a free key at: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

## 🚀 Usage

### Step 1 — Add Documents

Place your `.pdf`, `.docx`, or `.txt` files inside the `data/` folder.

### Step 2 — Ingest Documents (Run Once)

```bash
python src/ingest.py
```

This will:
- Extract text from all documents
- Chunk text into overlapping segments
- Embed each chunk with `text-embedding-004`
- Persist the vector database to `./db/`

> **Note:** You only need to run this once (or whenever you add new documents).

### Step 3a — Run the Streamlit UI (Recommended)

```bash
streamlit run src/main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Step 3b — Run the Command-Line Interface

```bash
python src/query.py
```

Type your questions in the interactive loop. Type `exit` or `quit` to stop.

---

## 🔧 Configuration

Edit `src/config.py` to tune the pipeline:

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `TOP_K_RESULTS` | 5 | Number of chunks retrieved per query |
| `EMBEDDING_MODEL` | `text-embedding-004` | Google embedding model |
| `GENERATION_MODEL` | `gemini-2.5-flash-preview-09-2025` | Gemini chat model |

---

## 🛡️ Hallucination Prevention

The system prompt strictly instructs the LLM to:

1. **Only use provided context** — never its own parametric memory.
2. **Cite sources** inline with every factual claim.
3. **Admit uncertainty** if the context doesn't contain the answer.

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Google `text-embedding-004` |
| Vector DB | ChromaDB (local, disk-persistent) |
| PDF Parsing | pypdf |
| DOCX Parsing | python-docx |
| UI | Streamlit |
| Env Management | python-dotenv |

---

## 🐛 Troubleshooting

- **`ModuleNotFoundError`** → Make sure your virtual environment is activated and `pip install -r requirements.txt` was run.
- **`API key not found`** → Check your `.env` file. The key must be `GEMINI_API_KEY=...`.
- **`Collection not found`** → Run `python src/ingest.py` before querying.
- **Empty results** → Make sure PDF/DOCX files are text-based (not scanned images).
