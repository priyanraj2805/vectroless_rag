# Vectorless RAG — Knowledge Graph Design

## Overview

A retrieval-augmented generation system that uses knowledge graphs instead of vector embeddings. PDFs are ingested, entities/relationships are extracted via LLM, stored in a SQLite graph, and queried via graph traversal to answer both factual and analytical questions.

## Goals

- Upload 50-500 academic/business PDFs
- Answer factual questions (e.g., "What is X's revenue?")
- Answer relational questions (e.g., "How does X relate to Y across documents?")
- No vector database dependency — graph-based retrieval only
- Simple deployment — SQLite, no external services besides Groq API

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Web UI     │────▶│   FastAPI     │────▶│  Groq LLM    │
│ (HTML/JS)    │◀────│   Backend     │◀────│  (Extract +  │
└─────────────┘     └──────┬───────┘     │   Synthesize) │
                           │             └──────────────┘
                           ▼
                    ┌──────────────┐
                    │   SQLite     │
                    │ (Graph DB)   │
                    └──────────────┘
```

## Components

### 1. PDF Ingestion Pipeline

**Flow:** PDF → Text Extraction → Chunking → Entity Extraction → Graph Construction

- **Text extraction:** PyMuPDF for fast extraction, pdfplumber for table-heavy PDFs
- **Chunking:** Split by section headings (detect via font size, bold, numbering patterns). Fallback: split at paragraph boundaries. Preserve page number metadata.
- **Entity extraction:** Groq LLM call per chunk with structured prompt:
  - Input: text chunk + entity type definitions
  - Output: JSON list of entities with types, attributes, and relationships
- **Graph construction:** Insert extracted entities/edges into SQLite

**Entity Types:**
| Type | Description | Example Attributes |
|------|-------------|-------------------|
| Person | Named individuals | name, role, organization |
| Organization | Companies, institutions | name, type, industry |
| Concept | Technical terms, ideas | name, definition, domain |
| Date | Temporal references | value, precision |
| Metric | Numbers with context | value, unit, period |
| Section | Document sections | title, document_id, page |

**Edge Types:**
| Type | Description |
|------|-------------|
| mentions | Section/Chunk mentions Entity |
| relates_to | Two entities are semantically related |
| part_of | Hierarchical containment (section → document) |
| authored_by | Entity authored/authored a document |
| contains | One entity contains another (org contains person) |

### 2. SQLite Graph Schema

```sql
-- Core entities
CREATE TABLE nodes (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,  -- person, organization, concept, date, metric, section
    attributes TEXT,     -- JSON blob
    document_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Relationships
CREATE TABLE edges (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    type TEXT NOT NULL,  -- mentions, relates_to, part_of, authored_by, contains
    attributes TEXT,     -- JSON blob
    confidence REAL DEFAULT 1.0,
    FOREIGN KEY (source_id) REFERENCES nodes(id),
    FOREIGN KEY (target_id) REFERENCES nodes(id)
);

-- Original text chunks (for context retrieval)
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    page_number INTEGER,
    section_title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document metadata
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    page_count INTEGER,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'processing'  -- processing, ready, error
);

-- Full-text search index
CREATE VIRTUAL TABLE chunks_fts USING fts5(content, section_title, content=chunks, content_rowid=id);
CREATE VIRTUAL TABLE nodes_fts USING fts5(name, attributes, content=nodes, content_rowid=id);
```

### 3. Query Engine

**Query flow:**
1. User asks a question
2. Groq generates a structured graph query plan:
   - Which entity types to find
   - What relationships to traverse
   - What filters to apply
3. Execute query on SQLite graph (join nodes + edges)
4. Retrieve relevant chunks (via FTS on matched nodes/sections)
5. Groq synthesizes answer from chunks + graph context

**Query plan example:**
```json
{
  "find_entities": [{"type": "organization", "name_contains": "revenue"}],
  "traverse": [{"edge": "relates_to", "depth": 2}],
  "filter": {"document_ids": [1, 3, 7]},
  "retrieve_chunks": true
}
```

### 4. Web UI

Minimal chat interface:
- **Left panel:** PDF upload area (drag-and-drop), document list with status
- **Right panel:** Chat interface with message history
- **Features:**
  - Upload PDFs (batch support)
  - View ingestion progress per document
  - Ask questions, see answers with source citations
  - View entity graph visualization (optional, basic D3.js)

## File Structure

```
vectorless_rag/
├── app/
│   ├── main.py              # FastAPI app entry
│   ├── config.py            # Settings (Groq API key, paths)
│   ├── models.py            # Pydantic models
│   ├── ingestion/
│   │   ├── pdf_parser.py    # PDF text extraction
│   │   ├── chunker.py       # Section-aware chunking
│   │   ├── extractor.py     # LLM entity extraction
│   │   └── graph_builder.py # SQLite graph construction
│   ├── query/
│   │   ├── planner.py       # LLM query planning
│   │   ├── executor.py      # Graph traversal execution
│   │   └── synthesizer.py   # LLM answer generation
│   ├── storage/
│   │   ├── database.py      # SQLite connection + schema
│   │   └── queries.py       # SQL query helpers
│   └── api/
│       ├── upload.py        # PDF upload endpoints
│       └── chat.py          # Query endpoints
├── static/
│   ├── index.html           # Main page
│   ├── style.css
│   └── app.js               # Chat + upload UI logic
├── uploads/                  # Uploaded PDF storage
├── requirements.txt
├── .env.example             # GROQ_API_KEY=...
└── README.md
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/upload | Upload PDF(s) |
| GET | /api/documents | List documents + status |
| DELETE | /api/documents/{id} | Remove document + graph data |
| POST | /api/chat | Send query, get answer |
| GET | /api/graph/stats | Graph statistics (node/edge counts) |

## Dependencies

```
fastapi
uvicorn
pymupdf
pdfplumber
groq
python-dotenv
python-multipart
```

## Error Handling

- **PDF parsing failure:** Mark document as `error` status, return partial results
- **LLM extraction failure:** Retry once, then skip chunk (log warning)
- **Graph query failure:** Fall back to FTS-only search
- **Groq API failure:** Return error message to user with retry suggestion

## Limitations

- Graph quality depends on LLM extraction accuracy
- Large PDFs (1000+ pages) may be slow to ingest
- SQLite has practical limit ~100GB, sufficient for thousands of PDFs
- No incremental graph updates — re-extraction needed for modified PDFs
