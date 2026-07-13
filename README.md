# Vectorless RAG System

A production-ready RAG (Retrieval-Augmented Generation) system that uses **SQLite FTS5** and **knowledge graphs** instead of vector embeddings. Upload 50+ documents (PDF, DOCX, XLSX, PPTX, images, text) and query them using natural language.

---

## How Vectorless Works

Traditional RAG stores float vector embeddings for every chunk and retrieves them via cosine similarity. This project replaces that entirely with two things: a **knowledge graph** (entities + relationships) and **SQLite full-text search (FTS5)**.

### End-to-End Flow

#### Ingestion (`app/ingestion/`)

When you upload a document (PDF, Word, Excel, PowerPoint, image, or text), the following happens in sequence:

1. **`tika_detector.py`** — Apache Tika detects file type via magic bytes and extracts metadata (author, title, language, page count)
2. **`document_router.py`** — Routes the MIME type to the appropriate parser
3. **`docling_parser.py`** — IBM Docling parses the document into **structured elements**: headings, paragraphs, tables (as markdown), figures (with captions) — works for all supported formats
4. **`chunker.py`** — Converts structured elements into context-aware chunks; tables and figures become dedicated chunks with `chunk_type` preserved
5. **`extractor.py`** — sends text chunks to the LLM to extract **entities** (people, orgs, concepts) and **relationships** as JSON
6. **`graph_builder.py`** — saves those entities as `nodes` and relationships as `edges` in SQLite
7. **`pipeline.py`** — orchestrates the above, running extraction in parallel (2 concurrent workers, 10 chunks per batch)

*Fallback: If Docling is unavailable, PyMuPDF handles PDF-only text extraction.*

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

1. **`query_analysis.py`** — LLM converts your question into a structured search plan:
   ```json
   { "keywords": ["revenue", "Acme"], "query_type": "factual", "search_terms": [...] }
   ```

2. **`hierarchical_retriever.py`** — the heart of vectorless retrieval:
   - **Query expansion** via knowledge graph (1-hop traversal from matched entities)
   - **Single global FTS5 BM25 search** across all chunks (not per-document)
   - **Document-level scoring** (BM25 + heading match + entity density + metadata)
   - **Dynamic document selection** (threshold-based, min/max docs)
   - **Cross-encoder reranking** (ms-marco-MiniLM-L-6-v2)
   - **Neighbor window expansion** (±1 chunk) for context continuity

3. **`synthesizer.py`** — LLM generates a final answer from retrieved context with per-document citations

### Why It Works Without Vectors

Vectors capture semantic similarity ("car" ≈ "vehicle"). This project gets the same effect differently:
- The **LLM extracts explicit relationships** during ingestion (so "car" and "vehicle" end up connected as nodes)
- **Graph traversal** finds semantically related content by following those relationships
- **BM25** handles keyword matching for direct term hits

The trade-off: slightly slower ingestion (one LLM call per chunk batch) but zero dependency on an embedding model and no vector database infrastructure.

### Vectorless File Map

| File | Role |
|---|---|
| `app/storage/database.py` | FTS5 indexes — the vectorless retrieval index |
| `app/query/hierarchical_retriever.py` | Core retrieval — keyword search + graph traversal + reranking |
| `app/ingestion/extractor.py` | Builds the graph from text using LLM |
| `app/ingestion/graph_builder.py` | Writes nodes/edges to SQLite |
| `app/query/query_analysis.py` | Converts question into a search plan |
| `app/query/synthesizer.py` | Generates answer from retrieved context |
| `app/ingestion/docling_parser.py` | Structured parsing via IBM Docling (tables, figures, headings) |
| `app/ingestion/chunker.py` | Structure-aware chunking (tables/figures as dedicated chunks) |
| `app/ingestion/tika_detector.py` | File type detection + metadata via Apache Tika |
| `app/ingestion/document_router.py` | Routes MIME types to appropriate parser |

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- **Java 17+** (required for Apache Tika)
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

1. Drag & drop documents (PDF, DOCX, XLSX, PPTX, PNG, JPG, TXT, etc.) into the sidebar
2. Wait for **Ready** status
3. Ask questions in natural language
4. Get answers with source citations (page numbers, sections, table/figure references)

---

## Features

- **No Vector Embeddings** — Uses SQLite FTS5 full-text search + knowledge graph
- **Multi-Format Support** — PDF, DOCX, XLSX, PPTX, images (PNG/JPG/TIFF), text/HTML/RTF
- **Structured Document Parsing** — IBM Docling extracts headings, tables (markdown), figures (captions)
- **Table & Figure Awareness** — Tables and figures become dedicated searchable chunks
- **Automatic Provider Fallback** — Groq primary → OpenRouter fallback on rate limit/credit errors
- **Parallel Extraction** — 2 concurrent workers, 10-chunk batches for faster ingestion
- **Real-time Status** — Live processing timer per document (queued → processing → ready)
- **Conversational Mode** — Handles greetings and meta-questions naturally
- **Source Citations** — Every answer includes page numbers, sections, and chunk types
- **Knowledge Graph** — Extracts entities and relationships from every text chunk
- **Cross-Encoder Reranking** — ms-marco-MiniLM-L-6-v2 for precision
- **Rich Metadata** — Author, title, language, MIME type stored per document

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
       Document Upload → Tika Detection (MIME + metadata)
                              │
                              ▼
              DocumentRouter → Docling (preferred) / PyMuPDF (fallback)
                              │
                              ▼
       Structured Elements: headings, paragraphs, tables, figures
                              │
                              ▼
         TextChunker → chunks (text | table | figure) + section + page
                              │
                              ▼
              BM25 Index (chunks_fts) — searchable immediately
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
```

```
┌─────────────────────────────────────────────────────────────┐
│                       QUERY PIPELINE                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
           User Question → Conversational Detection
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            Conversational          Query Analysis (LLM)
            Response Only           {keywords, query_type, search_terms}
                                          │
                                          ▼
                                   Hierarchical Retriever
                                   (FTS5 + Graph + Rerank)
                                          │
                                          ▼
                                 Answer Synthesizer (LLM)
                                 {answer, sources, citations}
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
| **LLM** | Groq + OpenRouter (fallback) | Entity extraction, query analysis, synthesis |
| **Document Parsing** | IBM Docling | Structured parsing (headings, tables, figures) |
| **File Detection** | Apache Tika | Magic-byte MIME detection + metadata |
| **PDF Fallback** | PyMuPDF (`fitz`) | Text extraction when Docling unavailable |
| **Reranking** | sentence-transformers | Cross-encoder (ms-marco-MiniLM-L-6-v2) |
| **Frontend** | Vanilla JS + CSS | Single-page app with real-time updates |
| **Parallelization** | ThreadPoolExecutor | Concurrent chunk extraction per document |

---

## Project Structure

```
vectorless_rag/
├── app/
│   ├── api/
│   │   ├── chat.py          # Chat endpoint (question → answer)
│   │   ├── upload.py        # Multi-format upload + background processing
│   │   └── eval.py          # Evaluation endpoints
│   ├── ingestion/
│   │   ├── pdf_parser.py    # PyMuPDF text extraction (fallback)
│   │   ├── chunker.py       # Structure-aware chunking (text/table/figure)
│   │   ├── extractor.py     # Entity extraction (LLM)
│   │   ├── graph_builder.py # Knowledge graph construction
│   │   ├── pipeline.py      # Orchestrates ingestion with parallel extraction
│   │   ├── docling_parser.py       # IBM Docling structured parsing
│   │   ├── document_router.py      # MIME type → parser routing
│   │   └── tika_detector.py        # Apache Tika file detection + metadata
│   ├── query/
│   │   ├── query_analysis.py       # Question → search plan (LLM)
│   │   ├── hierarchical_retriever.py # FTS5 + Graph + Rerank
│   │   ├── reranker.py             # Cross-encoder reranking
│   │   └── synthesizer.py          # Context → answer (LLM)
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
├── uploads/                 # Uploaded documents (auto-created)
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
    status TEXT DEFAULT 'processing',  -- 'queued', 'processing', 'ready', 'error'
    mime_type TEXT,                    -- e.g., 'application/pdf'
    content_type TEXT,                 -- 'pdf', 'docx', 'xlsx', 'pptx', 'image', 'text'
    author TEXT,
    document_title TEXT,
    language TEXT
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
    section_title TEXT,
    chunk_index INTEGER,
    chunk_type TEXT DEFAULT 'text',  -- 'text', 'table', 'figure'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vectorless retrieval indexes (replaces vector store)
CREATE VIRTUAL TABLE chunks_fts USING fts5(content, section_title);
CREATE VIRTUAL TABLE nodes_fts  USING fts5(name, attributes);
```

---

## API Reference

### Upload Documents
```http
POST /api/upload
Content-Type: multipart/form-data

files: [file1.pdf, file2.docx, file3.xlsx, ...]
```

**Supported extensions:** `.pdf`, `.docx`, `.doc`, `.xlsx`, `.xls`, `.pptx`, `.ppt`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp`, `.txt`, `.html`, `.rtf`

**Response:**
```json
{
  "documents": [
    {"filename": "report.pdf", "status": "queued", "doc_id": 1},
    {"filename": "data.xlsx", "status": "queued", "doc_id": 2}
  ]
}
```

### List Documents
```http
GET /api/documents
```

**Response:**
```json
{
  "documents": [
    {"id": 1, "filename": "report.pdf", "page_count": 12, "upload_date": "...", "status": "ready", "completed_at": "..."}
  ]
}
```

### Ask a Question
```http
POST /api/chat
Content-Type: application/json

{ "question": "What are the key findings?", "document_ids": [1, 2] }
```

**Response:**
```json
{
  "answer": "The key findings are...",
  "sources": [{"document": "Document 1 — report.pdf", "content_preview": "...", "page": 5, "section": "Executive Summary"}],
  "entities_found": 8,
  "chunks_used": 5,
  "trace_id": "abc123",
  "context_texts": ["..."]
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

### Evaluation
```http
POST /api/eval/run
Content-Type: application/json

{ "questions": ["What is X?", "Compare Y and Z"] }
```

```http
POST /api/eval/score
Content-Type: application/json

{ "question": "...", "answer": "...", "context_texts": ["..."], "trace_id": "..." }
```

---

## Configuration

### Environment Variables

```bash
GROQ_API_KEY=gsk_...           # Primary LLM provider (free)
OPENROUTER_API_KEY=sk-or-v1-.. # Fallback LLM provider (optional)
DATABASE_PATH=./data/graph.db  # Database location
UPLOAD_DIR=./uploads           # Document storage directory
REDIS_URL=redis://localhost:6379  # Optional: for answer caching
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

### "Docling not available"
- Install Docling: `pip install docling`
- Requires Python 3.9+ and system dependencies (see Docling docs)

### "Tika not available"
- Install Java 17+: `apt-get install openjdk-17-jre` (Linux) or download from Oracle/Adoptium
- Install Tika Python client: `pip install tika`
- Fallback: extension-based detection works without Tika (no metadata extraction)

### Database locked / corrupted
```bash
del data\graph.db data\graph.db-wal data\graph.db-shm
```
Restart the server — database rebuilds automatically.

### Processing stuck at "Processing..." or "Queued"
- Background task crashed or LLM API error
- Check server logs, then restart the server
- Documents stuck in "queued" → processing starts when background worker picks them up

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

Ensure Java 17+ is installed on the deployment target for Tika.

---

## License

MIT License

---

**Built for document intelligence without the vector complexity.**