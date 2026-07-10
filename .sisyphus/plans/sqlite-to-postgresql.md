# Plan: SQLite → PostgreSQL Migration

## Goal
Replace SQLite with PostgreSQL + pgvector for million-PDF scale. Handle FTS, vector search, and concurrent access.

## Scope
- Rewrite `database.py` (core change)
- Update `config.py` and `.env` (connection string)
- Update `executor.py` (FTS query syntax)
- Update `chat.py` (placeholder syntax)
- Update `upload.py` (placeholder syntax)
- Add `psycopg2-binary` and `pgvector` to requirements

## Changes

### 1. `.env` / `.env.example`
Replace `DATABASE_PATH=./data/graph.db` with:
```
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/vectorless_rag
```

### 2. `app/config.py`
Replace `database_path: str = "./data/graph.db"` with:
```
database_url: str = "postgresql://postgres:postgres@localhost:5432/vectorless_rag"
```

### 3. `app/storage/database.py` — Full Rewrite
Key changes:
- `sqlite3` → `psycopg2`
- `INTEGER PRIMARY KEY AUTOINCREMENT` → `SERIAL PRIMARY KEY`
- `BLOB` → `vector(384)` (pgvector)
- FTS5 virtual tables → `tsvector` column + GIN index
- FTS triggers → `tsvector` auto-update trigger
- `?` placeholders → `%s`
- `cursor.lastrowid` → `RETURNING id` + `cursor.fetchone()[0]`
- `INSERT OR REPLACE` → `INSERT ... ON CONFLICT ... DO UPDATE`
- Remove PRAGMAs
- Add connection pooling via `psycopg2.pool`

New schema:
```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    page_count INTEGER,
    upload_date TIMESTAMPTZ DEFAULT NOW(),
    status TEXT DEFAULT 'processing',
    completed_at TIMESTAMPTZ
);

CREATE TABLE nodes (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    attributes TEXT,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    name_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(name,'') || ' ' || coalesce(attributes,''))) STORED
);
CREATE INDEX idx_nodes_tsv ON nodes USING GIN(name_tsv);

CREATE TABLE edges (
    id SERIAL PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    attributes TEXT,
    confidence DOUBLE PRECISION DEFAULT 1.0
);

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    page_number INTEGER,
    section_title TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content,'') || ' ' || coalesce(section_title,''))) STORED
);
CREATE INDEX idx_chunks_tsv ON chunks USING GIN(content_tsv);

CREATE TABLE embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    vector vector(384) NOT NULL
);
CREATE INDEX idx_embeddings_ivfflat ON embeddings USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);
```

FTS queries change from:
```sql
WHERE chunks_fts MATCH ? ORDER BY rank
```
To:
```sql
WHERE content_tsv @@ plainto_tsquery('english', ?) ORDER BY ts_rank(content_tsv, plainto_tsquery('english', ?)) DESC
```

### 4. `app/query/executor.py`
- Replace all `?` with `%s`
- Rewrite FTS MATCH queries to use `@@ plainto_tsquery`
- Replace `rank` with `ts_rank()`
- Vector search: load embeddings into Python (same as now) — pgvector `<=>` operator can be added later as optimization

### 5. `app/api/chat.py` and `app/api/upload.py`
- Replace `?` with `%s` in raw SQL
- `Database(settings.database_path)` → `Database(settings.database_url)`

### 6. `requirements.txt`
Add:
```
psycopg2-binary
pgvector
```

### 7. Docker Compose for PostgreSQL
Add `docker-compose.yml` for easy local PostgreSQL + pgvector setup.

## Files to modify
- `app/storage/database.py` (full rewrite)
- `app/config.py`
- `app/query/executor.py`
- `app/api/chat.py`
- `app/api/upload.py`
- `.env`, `.env.example`
- `requirements.txt`
- `docker-compose.yml` (new)

## Verification
1. `docker-compose up -d` to start PostgreSQL
2. `python -c "from app.config import settings; from app.storage.database import Database; db = Database(settings.database_url); print(db.get_graph_stats())"`
3. Upload a PDF and verify it works
4. Ask a question and verify query works
