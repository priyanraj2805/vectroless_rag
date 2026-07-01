# Vectorless RAG System

A production-ready RAG (Retrieval-Augmented Generation) system that uses **SQLite FTS5** and **knowledge graphs** instead of vector embeddings. Upload 50+ PDFs and query them using natural language.

---

## How Vectorless Works

Traditional RAG stores float vector embeddings for every chunk and retrieves them via cosine similarity. This project replaces that entirely with two things: a **knowledge graph** (entities + relationships) and **SQLite full-text search (FTS5)**.

### End-to-End Flow

#### Ingestion (`app/ingestion/`)

When you upload a PDF, the following happens in sequence:

1. **`pdf_parser.py`** — extracts raw text per page using PyMuPDF
2. **`chunker.py`** — splits text into sections/paragraphs. No embedding is computed — just plain text chunks stored as-is
3. **`extractor.py`** — sends each chunk to the LLM and asks it to extract **entities** (people, orgs, concepts) and **relationships** as JSON. This is the LLM doing the "understanding" work that vector embeddings normally handle
4. **`graph_builder.py`** — saves those entities as `nodes` and relationships as `edges` in SQLite
5. **`pipeline.py`** — orchestrates the above, running extraction for all chunks in parallel (5 concurrent API calls)

#### Storage (`app/storage/database.py`)

This is where "vectorless" is most explicit. The entire retrieval index is two SQLite FTS5 virtual tables — no vector columns, no embedding model, no external vector database:

```sql
-- Full-text index over chunk content (replaces vector similarity search)
CREATE VIRTUAL TABLE chunks_fts USING fts5(content, section_title, ...)

-- Full-text index over entity names/attributes
CREATE VIRTUAL TABLE nodes_fts  USING fts5(name, attributes, ...)
```

SQLite's FTS5 uses **BM25** (keyword ranking) instead of cosine similarity. There is no pgvector, no ChromaDB, no embedding column anywhere in the schema.

#### Query (`app/query/`)

When you ask a question, three steps happen:

1. **`planner.py`** — LLM converts your question into a structured search plan:
   ```json
   { "search_terms": ["revenue", "Acme"], "entity_types": ["metric"], "traverse_edges": ["relates_to"] }
   ```

2. **`executor.py`** — the heart of vectorless retrieval:
   - **FTS keyword search** on `nodes_fts` — finds matching entities by name
   - **Graph traversal** — walks the `edges` table to find connected entities
   - **FTS keyword search** on `chunks_fts` — pulls relevant text passages
   - No embedding lookup. No cosine distance. Just SQL `MATCH` + `JOIN`

3. **`synthesizer.py`** — LLM generates a final answer from the retrieved graph context + chunks

### Why It Works Without Vectors

Vectors capture semantic similarity ("car" ≈ "vehicle"). This project gets the same effect differently:
- The **LLM extracts explicit relationships** during ingestion (so "car" and "vehicle" end up connected as nodes)
- **Graph traversal** finds semantically related content by following those relationships
- **BM25** handles keyword matching for direct term hits

The trade-off: slightly slower ingestion (one LLM call per chunk) but zero dependency on an embedding model and no vector database infrastructure.

### Vectorless File Map

| File | Role |
|---|---|
| `app/storage/database.py` | FTS5 indexes — the vectorless retrieval index |
| `app/query/executor.py` | Core retrieval — keyword search + graph traversal |
| `app/ingestion/extractor.py` | Builds the graph from text using LLM |
| `app/ingestion/graph_builder.py` | Writes nodes/edges to SQLite |
| `app/query/planner.py` | Converts question into a search plan |
| `app/query/synthesizer.py` | Generates answer from retrieved context |

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Groq API Key — free at [console.groq.com](https://console.groq.com/keys) (primary)
- OpenRouter API Key — optional at [openrouter.ai](https://openrouter.ai) (fallback)

### 2. Installation

```bash
git clone <your-repo-url>
cd vectorless_rag
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=gsk_your_groq_key_here
OPENROUTER_API_KEY=sk-or-v1-your_openrouter_key_here  # optional fallback
DATABASE_PATH=./data/graph.db
UPLOAD_DIR=./uploads
```

### 4. Run the Server

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000** in your browser.

### 5. Upload & Query

1. Drag & drop PDFs into the sidebar
2. Wait for **Ready** status
3. Ask questions in natural language
4. Get answers with source citations

---

## Features

- **No Vector Embeddings** — Uses SQLite FTS5 full-text search + knowledge graph
- **Automatic Provider Fallback** — Groq is primary; switches to OpenRouter automatically on rate limit or credit errors
- **Parallel Extraction** — 5 concurrent LLM calls per PDF for faster ingestion
- **Real-time Status** — Live processing timer per document
- **Conversational Mode** — Handles greetings naturally
- **Source Citations** — Every answer includes page numbers and sections
- **Knowledge Graph** — Extracts entities and relationships from every chunk

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                      │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
      PDF Upload → PyMuPDF Text Extraction (page-by-page)
                             │
                             ▼
            Section-aware Chunking (headings, paragraphs)
                             │
                             ▼
         Entity Extraction (LLM) → {entities, relationships}
         [Groq primary → OpenRouter fallback]
                             │
                             ▼
                   Knowledge Graph Builder
                             │
                   ┌─────────┴─────────┐
                   ▼                   ▼
              SQLite Tables      FTS5 Indexes
         (docs, nodes, edges)   (chunks, entities)

┌─────────────────────────────────────────────────────────────┐
│                       QUERY PIPELINE                         │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
          User Question → Conversational Detection
                             │
                   ┌─────────┴─────────┐
                   ▼                   ▼
           Conversational          Query Planner (LLM)
           Response Only           {search_terms, entity_types}
                                         │
                                         ▼
                                  Query Executor
                                  (FTS5 + Graph Traversal)
                                         │
                                         ▼
                                Answer Synthesizer (LLM)
                                {answer, sources}
```

---

## LLM Provider Fallback

The system uses a `FallbackLLMClient` (`app/llm_client.py`) that chains providers:

| Priority | Provider | Model | Cost |
|---|---|---|---|
| 1st | Groq | `llama-3.1-8b-instant` | Free |
| 2nd | OpenRouter | `meta-llama/llama-3.1-8b-instruct` | Paid |

If Groq hits a rate limit (429) or OpenRouter hits a credit error (402), the client automatically retries with the next provider. You'll see a log line like:
```
[groq] limit hit, switching to next provider...
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | FastAPI | Async web framework + background tasks |
| **Database** | SQLite (WAL mode) | Document storage + knowledge graph |
| **Search** | FTS5 (Full-Text Search) | Fast text retrieval — no vectors needed |
| **LLM** | Groq + OpenRouter (fallback) | Entity extraction, query planning, synthesis |
| **PDF** | PyMuPDF (`fitz`) | Text extraction with page metadata |
| **Frontend** | Vanilla JS + CSS | Single-page app with real-time updates |
| **Parallelization** | ThreadPoolExecutor | Concurrent chunk extraction per PDF |

---

## Project Structure

```
vectorless_rag/
├── app/
│   ├── api/
│   │   ├── chat.py          # Chat endpoint (question → answer)
│   │   └── upload.py        # PDF upload + background processing
│   ├── ingestion/
│   │   ├── pdf_parser.py    # PyMuPDF text extraction
│   │   ├── chunker.py       # Section-aware text chunking
│   │   ├── extractor.py     # Entity extraction (LLM)
│   │   ├── graph_builder.py # Knowledge graph construction
│   │   └── pipeline.py      # Orchestrates ingestion with parallel extraction
│   ├── query/
│   │   ├── planner.py       # Question → search plan (LLM)
│   │   ├── executor.py      # FTS5 search + graph traversal
│   │   └── synthesizer.py   # Context → answer (LLM)
│   ├── storage/
│   │   └── database.py      # SQLite schema + FTS5 indexes + CRUD
│   ├── llm_client.py        # FallbackLLMClient (Groq → OpenRouter)
│   ├── config.py            # Environment variables (.env)
│   └── main.py              # FastAPI app + routes
├── static/
│   ├── index.html           # UI layout
│   ├── style.css            # Styling
│   └── app.js               # Frontend logic (upload, chat, polling)
├── tests/                   # Unit tests
├── data/                    # SQLite database (auto-created)
├── uploads/                 # Uploaded PDFs (auto-created)
├── .env                     # API keys (you create this)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Database Schema

```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    page_count INTEGER,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT DEFAULT 'processing'  -- 'processing', 'ready', 'error'
);

CREATE TABLE nodes (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,  -- 'person', 'organization', 'concept', etc.
    attributes TEXT,     -- JSON metadata
    document_id INTEGER
);

CREATE TABLE edges (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    type TEXT NOT NULL,  -- 'mentions', 'relates_to', 'part_of', etc.
    confidence REAL DEFAULT 1.0
);

CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    page_number INTEGER,
    section_title TEXT
);

-- Vectorless retrieval indexes (replaces vector store)
CREATE VIRTUAL TABLE chunks_fts USING fts5(content, section_title);
CREATE VIRTUAL TABLE nodes_fts  USING fts5(name, attributes);
```

---

## API Reference

### Upload PDFs
```http
POST /api/upload
Content-Type: multipart/form-data

files: [file1.pdf, file2.pdf, ...]
```

### List Documents
```http
GET /api/documents
```

### Ask a Question
```http
POST /api/chat
Content-Type: application/json

{ "question": "What are the key findings?" }
```

**Response:**
```json
{
  "answer": "The key findings are...",
  "sources": [{"page": 5, "section": "Executive Summary"}],
  "entities_found": 8,
  "chunks_used": 5
}
```

### Delete Document
```http
DELETE /api/documents/{id}
```

### Graph Statistics
```http
GET /api/graph/stats
```

---

## Configuration

### Environment Variables

```bash
GROQ_API_KEY=gsk_...           # Primary LLM provider (free)
OPENROUTER_API_KEY=sk-or-v1-.. # Fallback LLM provider (optional)
DATABASE_PATH=./data/graph.db  # Database location
UPLOAD_DIR=./uploads           # PDF storage directory
```

---

## Troubleshooting

### "Invalid API Key" (401)
- `.env` has the placeholder value instead of a real key
- Fix: get a free key at [console.groq.com](https://console.groq.com) and paste it in `.env`

### "Insufficient credits" (402)
- OpenRouter account has no credits
- Fix: Groq is free — make sure `GROQ_API_KEY` is set correctly and it will be used instead

### "Rate limit exceeded" (429)
- Groq free tier limit hit during heavy ingestion
- The system automatically falls back to OpenRouter if configured
- Or reduce parallel workers in `app/ingestion/pipeline.py`: `max_workers=2`

### Database locked / corrupted
```bash
del data\graph.db data\graph.db-wal data\graph.db-shm
```
Restart the server — database rebuilds automatically.

### Processing stuck at "Processing..."
- Background task crashed or LLM API error
- Check server logs, then restart the server

---

## Testing

```bash
pytest tests/ -v
```

---

## Deployment

```bash
# Production run with Gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## License

MIT License

---

**Built for document intelligence without the vector complexity.**
