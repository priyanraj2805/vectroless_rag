# Vectorless RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a knowledge-graph-based RAG system that ingests PDFs, extracts entities/relationships via Groq LLM, stores them in SQLite, and answers questions via graph traversal.

**Architecture:** FastAPI backend with SQLite graph storage. PDF ingestion pipeline extracts text → chunks by sections → LLM extracts entities/edges → builds graph. Query engine generates graph traversal plans → retrieves relevant chunks → synthesizes answers via Groq.

**Tech Stack:** Python, FastAPI, PyMuPDF, pdfplumber, Groq API, SQLite, HTML/JS frontend

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `app/config.py`
- Create: `static/index.html`

- [ ] **Step 1: Create requirements.txt**

```
fastapi==0.115.0
uvicorn==0.30.0
pymupdf==1.24.0
pdfplumber==0.11.0
groq==0.11.0
python-dotenv==1.0.1
python-multipart==0.0.9
pytest==8.3.0
httpx==0.27.0
```

- [ ] **Step 2: Create .env.example**

```
GROQ_API_KEY=gsk_your_key_here
DATABASE_PATH=./data/graph.db
UPLOAD_DIR=./uploads
```

- [ ] **Step 3: Create app/config.py**

```python
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    groq_api_key: str = ""
    database_path: str = "./data/graph.db"
    upload_dir: str = "./uploads"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

Path(settings.database_path).parent.mkdir(parents=True, exist_ok=True)
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Create app/__init__.py**

```python
```

- [ ] **Step 5: Create app/main.py**

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Vectorless RAG", version="0.1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return {"message": "Vectorless RAG API"}


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 6: Create minimal static/index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vectorless RAG</title>
</head>
<body>
    <h1>Vectorless RAG</h1>
    <p>Coming soon...</p>
</body>
</html>
```

- [ ] **Step 7: Install dependencies and verify**

Run: `pip install -r requirements.txt && python -c "from app.main import app; print('OK')"`

- [ ] **Step 8: Commit**

```bash
git init
git add requirements.txt .env.example app/ static/
git commit -m "feat: project scaffolding with FastAPI skeleton"
```

---

### Task 2: SQLite Database Layer

**Files:**
- Create: `app/storage/__init__.py`
- Create: `app/storage/database.py`
- Create: `tests/__init__.py`
- Create: `tests/test_database.py`

- [ ] **Step 1: Write failing test for database schema creation**

```python
# tests/test_database.py
import os
import tempfile
from app.storage.database import Database


def test_schema_creation():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = Database(db_path)
        tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {t[0] for t in tables}
        assert "nodes" in table_names
        assert "edges" in table_names
        assert "chunks" in table_names
        assert "documents" in table_names
        db.close()
    finally:
        os.unlink(db_path)


def test_insert_document():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = Database(db_path)
        doc_id = db.insert_document("test.pdf", 10)
        assert doc_id == 1
        docs = db.execute("SELECT * FROM documents").fetchall()
        assert len(docs) == 1
        assert docs[0][1] == "test.pdf"
        db.close()
    finally:
        os.unlink(db_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_database.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.storage.database'`

- [ ] **Step 3: Create app/storage/__init__.py**

```python
```

- [ ] **Step 4: Create app/storage/database.py with schema**

```python
import sqlite3
import json
from pathlib import Path
from datetime import datetime


class Database:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        page_count INTEGER,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'processing'
    );

    CREATE TABLE IF NOT EXISTS nodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        attributes TEXT,
        document_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER NOT NULL,
        target_id INTEGER NOT NULL,
        type TEXT NOT NULL,
        attributes TEXT,
        confidence REAL DEFAULT 1.0,
        FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
        FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL,
        content TEXT NOT NULL,
        page_number INTEGER,
        section_title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        content, section_title, content=chunks, content_rowid=id
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
        name, attributes, content=nodes, content_rowid=id
    );

    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, content, section_title)
        VALUES (new.id, new.content, new.section_title);
    END;

    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, content, section_title)
        VALUES ('delete', old.id, old.content, old.section_title);
    END;

    CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
        INSERT INTO nodes_fts(rowid, name, attributes)
        VALUES (new.id, new.name, new.attributes);
    END;

    CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
        INSERT INTO nodes_fts(nodes_fts, rowid, name, attributes)
        VALUES ('delete', old.id, old.name, old.attributes);
    END;
    """

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(query, params)

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

    def insert_document(self, filename: str, page_count: int = 0) -> int:
        cursor = self.execute(
            "INSERT INTO documents (filename, page_count) VALUES (?, ?)",
            (filename, page_count),
        )
        self.commit()
        return cursor.lastrowid

    def insert_node(self, name: str, node_type: str, attributes: dict = None, document_id: int = None) -> int:
        cursor = self.execute(
            "INSERT INTO nodes (name, type, attributes, document_id) VALUES (?, ?, ?, ?)",
            (name, node_type, json.dumps(attributes or {}), document_id),
        )
        self.commit()
        return cursor.lastrowid

    def insert_edge(self, source_id: int, target_id: int, edge_type: str, attributes: dict = None, confidence: float = 1.0) -> int:
        cursor = self.execute(
            "INSERT INTO edges (source_id, target_id, type, attributes, confidence) VALUES (?, ?, ?, ?, ?)",
            (source_id, target_id, edge_type, json.dumps(attributes or {}), confidence),
        )
        self.commit()
        return cursor.lastrowid

    def insert_chunk(self, document_id: int, content: str, page_number: int = None, section_title: str = None) -> int:
        cursor = self.execute(
            "INSERT INTO chunks (document_id, content, page_number, section_title) VALUES (?, ?, ?, ?)",
            (document_id, content, page_number, section_title),
        )
        self.commit()
        return cursor.lastrowid

    def update_document_status(self, doc_id: int, status: str):
        self.execute("UPDATE documents SET status = ? WHERE id = ?", (status, doc_id))
        self.commit()

    def get_document(self, doc_id: int):
        return self.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()

    def list_documents(self):
        return self.execute("SELECT * FROM documents ORDER BY upload_date DESC").fetchall()

    def delete_document(self, doc_id: int):
        self.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        self.execute("DELETE FROM edges WHERE source_id IN (SELECT id FROM nodes WHERE document_id = ?) OR target_id IN (SELECT id FROM nodes WHERE document_id = ?)", (doc_id, doc_id))
        self.execute("DELETE FROM nodes WHERE document_id = ?", (doc_id,))
        self.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.commit()

    def search_chunks(self, query: str, limit: int = 10):
        return self.execute(
            "SELECT c.id, c.content, c.page_number, c.section_title, rank FROM chunks_fts fts JOIN chunks c ON c.id = fts.rowid WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, limit),
        ).fetchall()

    def search_nodes(self, query: str, limit: int = 20):
        return self.execute(
            "SELECT n.id, n.name, n.type, n.attributes, rank FROM nodes_fts fts JOIN nodes n ON n.id = fts.rowid WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, limit),
        ).fetchall()

    def get_graph_stats(self):
        nodes = self.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edges = self.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        chunks = self.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        docs = self.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        return {"nodes": nodes, "edges": edges, "chunks": chunks, "documents": docs}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_database.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/storage/ tests/test_database.py
git commit -m "feat: SQLite database layer with graph schema"
```

---

### Task 3: PDF Text Extraction

**Files:**
- Create: `app/ingestion/__init__.py`
- Create: `app/ingestion/pdf_parser.py`
- Create: `tests/test_pdf_parser.py`

- [ ] **Step 1: Write failing test for PDF parsing**

```python
# tests/test_pdf_parser.py
import tempfile
from pathlib import Path
from app.ingestion.pdf_parser import PDFParser


def test_extract_text_from_simple_pdf():
    parser = PDFParser()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>>>endobj\n4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000266 00000 n \ntrailer<</Size 5/Root 1 0 R>>\nstartxref\n340\n%%EOF")
        f.flush()
        result = parser.extract_text(f.name)
        assert "pages" in result
        assert result["page_count"] >= 1
        assert len(result["text"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pdf_parser.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create app/ingestion/__init__.py**

```python
```

- [ ] **Step 4: Create app/ingestion/pdf_parser.py**

```python
import pymupdf
from pathlib import Path


class PDFParser:
    def extract_text(self, pdf_path: str) -> dict:
        doc = pymupdf.open(pdf_path)
        pages = []
        full_text = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            pages.append({"page_num": page_num + 1, "text": text})
            full_text.append(text)

        doc.close()

        return {
            "text": "\n\n".join(full_text),
            "pages": pages,
            "page_count": len(pages),
        }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_pdf_parser.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/ingestion/ tests/test_pdf_parser.py
git commit -m "feat: PDF text extraction with PyMuPDF"
```

---

### Task 4: Section-Aware Chunking

**Files:**
- Create: `app/ingestion/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write failing tests for chunking**

```python
# tests/test_chunker.py
from app.ingestion.chunker import TextChunker


def test_chunk_by_headings():
    chunker = TextChunker()
    text = """# Introduction
This is the intro section.

# Methods
This is the methods section.

# Results
This is the results section."""

    chunks = chunker.chunk(text, document_id=1)
    assert len(chunks) == 3
    assert chunks[0]["section_title"] == "Introduction"
    assert chunks[1]["section_title"] == "Methods"
    assert chunks[2]["section_title"] == "Results"


def test_chunk_preserves_page_number():
    chunker = TextChunker()
    pages = [
        {"page_num": 1, "text": "# Chapter 1\nFirst chapter content."},
        {"page_num": 2, "text": "# Chapter 2\nSecond chapter content."},
    ]

    chunks = chunker.chunk_from_pages(pages, document_id=1)
    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 1
    assert chunks[1]["page_number"] == 2


def test_fallback_to_paragraphs():
    chunker = TextChunker()
    text = """This is paragraph one with some content.

This is paragraph two with more content.

This is paragraph three with even more content."""

    chunks = chunker.chunk(text, document_id=1)
    assert len(chunks) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create app/ingestion/chunker.py**

```python
import re
from typing import List, Dict


class TextChunker:
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    NUMBERED_PATTERN = re.compile(r"^(\d+\.?\d*\.?\d*)\s+(.+)$", re.MULTILINE)
    BOLD_PATTERN = re.compile(r"^\*\*(.+?)\*\*$", re.MULTILINE)

    def chunk(self, text: str, document_id: int) -> List[Dict]:
        sections = self._split_by_headings(text)
        if len(sections) <= 1:
            sections = self._split_by_paragraphs(text)

        chunks = []
        for section in sections:
            if section["content"].strip():
                chunks.append({
                    "document_id": document_id,
                    "content": section["content"].strip(),
                    "section_title": section.get("title", ""),
                    "page_number": None,
                })
        return chunks

    def chunk_from_pages(self, pages: List[Dict], document_id: int) -> List[Dict]:
        chunks = []
        for page in pages:
            page_chunks = self.chunk(page["text"], document_id)
            for chunk in page_chunks:
                chunk["page_number"] = page["page_num"]
            chunks.extend(page_chunks)
        return chunks

    def _split_by_headings(self, text: str) -> List[Dict]:
        headings = []
        for match in self.HEADING_PATTERN.finditer(text):
            headings.append({
                "title": match.group(2).strip(),
                "start": match.start(),
                "end": match.end(),
            })

        if not headings:
            for match in self.NUMBERED_PATTERN.finditer(text):
                headings.append({
                    "title": match.group(2).strip(),
                    "start": match.start(),
                    "end": match.end(),
                })

        if not headings:
            return [{"content": text, "title": ""}]

        sections = []
        if headings[0]["start"] > 0:
            pre_content = text[:headings[0]["start"]].strip()
            if pre_content:
                sections.append({"content": pre_content, "title": ""})

        for i, heading in enumerate(headings):
            content_start = heading["end"]
            content_end = headings[i + 1]["start"] if i + 1 < len(headings) else len(text)
            content = text[content_start:content_end].strip()
            sections.append({"content": content, "title": heading["title"]})

        return sections

    def _split_by_paragraphs(self, text: str) -> List[Dict]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            return [{"content": text, "title": ""}]

        merged = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > 2000:
                if current:
                    merged.append({"content": current, "title": ""})
                current = para
            else:
                current = current + "\n\n" + para if current else para
        if current:
            merged.append({"content": current, "title": ""})

        return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chunker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/ingestion/chunker.py tests/test_chunker.py
git commit -m "feat: section-aware text chunking"
```

---

### Task 5: Groq LLM Entity Extractor

**Files:**
- Create: `app/ingestion/extractor.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 1: Write failing test for entity extraction**

```python
# tests/test_extractor.py
import json
from unittest.mock import patch, MagicMock
from app.ingestion.extractor import EntityExtractor


def test_extract_entities_returns_structured_data():
    extractor = EntityExtractor(api_key="test_key")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps({
        "entities": [
            {"name": "Acme Corp", "type": "organization", "attributes": {"industry": "tech"}},
            {"name": "John Doe", "type": "person", "attributes": {"role": "CEO"}},
        ],
        "relationships": [
            {"source": "John Doe", "target": "Acme Corp", "type": "authored_by"}
        ]
    }))]

    with patch("app.ingestion.extractor.groq.Client") as MockClient:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockClient.return_value = mock_client

        result = extractor.extract("John Doe is the CEO of Acme Corp.", document_id=1)

        assert "entities" in result
        assert "relationships" in result
        assert len(result["entities"]) == 2
        assert len(result["relationships"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create app/ingestion/extractor.py**

```python
import json
import groq
from typing import Dict, List


EXTRACTION_PROMPT = """Extract entities and relationships from the following text.

Return JSON with this exact structure:
{
  "entities": [
    {"name": "entity name", "type": "person|organization|concept|date|metric", "attributes": {"key": "value"}}
  ],
  "relationships": [
    {"source": "source entity name", "target": "target entity name", "type": "mentions|relates_to|part_of|authored_by|contains"}
  ]
}

Rules:
- Extract ALL named entities (people, organizations, concepts)
- Only create relationships between entities you extracted
- Use exact names from the text
- Return ONLY valid JSON, no other text

Text:
"""


class EntityExtractor:
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.client = groq.Client(api_key=api_key)
        self.model = model

    def extract(self, text: str, document_id: int) -> Dict[str, List]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an entity extraction assistant. Return only valid JSON."},
                    {"role": "user", "content": EXTRACTION_PROMPT + text[:6000]},
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]

            result = json.loads(content)
            result.setdefault("entities", [])
            result.setdefault("relationships", [])

            for entity in result["entities"]:
                entity["document_id"] = document_id

            return result

        except json.JSONDecodeError:
            return {"entities": [], "relationships": []}
        except Exception as e:
            print(f"Extraction error: {e}")
            return {"entities": [], "relationships": []}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_extractor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/ingestion/extractor.py tests/test_extractor.py
git commit -m "feat: Groq LLM entity and relationship extraction"
```

---

### Task 6: Graph Builder (Ingestion Pipeline)

**Files:**
- Create: `app/ingestion/graph_builder.py`
- Create: `tests/test_graph_builder.py`

- [ ] **Step 1: Write failing tests for graph building**

```python
# tests/test_graph_builder.py
import tempfile
import os
from app.storage.database import Database
from app.ingestion.graph_builder import GraphBuilder


def test_build_graph_from_extraction():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = Database(db_path)
        builder = GraphBuilder(db)

        extraction = {
            "entities": [
                {"name": "Acme Corp", "type": "organization", "attributes": {"industry": "tech"}, "document_id": 1},
                {"name": "John Doe", "type": "person", "attributes": {"role": "CEO"}, "document_id": 1},
            ],
            "relationships": [
                {"source": "John Doe", "target": "Acme Corp", "type": "authored_by"}
            ]
        }

        node_ids = builder.build_graph(extraction, document_id=1)
        assert len(node_ids) == 2

        stats = db.get_graph_stats()
        assert stats["nodes"] == 2
        assert stats["edges"] == 1
        db.close()
    finally:
        os.unlink(db_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph_builder.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create app/ingestion/graph_builder.py**

```python
from typing import Dict, List
from app.storage.database import Database


class GraphBuilder:
    def __init__(self, db: Database):
        self.db = db

    def build_graph(self, extraction: Dict[str, List], document_id: int) -> List[int]:
        name_to_id = {}

        for entity in extraction.get("entities", []):
            node_id = self.db.insert_node(
                name=entity["name"],
                node_type=entity["type"],
                attributes=entity.get("attributes", {}),
                document_id=document_id,
            )
            name_to_id[entity["name"]] = node_id

        for rel in extraction.get("relationships", []):
            source_id = name_to_id.get(rel["source"])
            target_id = name_to_id.get(rel["target"])
            if source_id and target_id:
                self.db.insert_edge(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=rel["type"],
                    attributes=rel.get("attributes", {}),
                )

        return list(name_to_id.values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph_builder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/ingestion/graph_builder.py tests/test_graph_builder.py
git commit -m "feat: graph builder for entity/relationship insertion"
```

---

### Task 7: Full Ingestion Pipeline

**Files:**
- Create: `app/ingestion/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test for full pipeline**

```python
# tests/test_pipeline.py
import tempfile
import os
from unittest.mock import patch, MagicMock
from app.storage.database import Database
from app.ingestion.pipeline import IngestionPipeline


def test_full_ingestion_pipeline():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = Database(db_path)

        mock_text_result = {
            "text": "Acme Corp is a tech company. John Doe is the CEO.",
            "pages": [{"page_num": 1, "text": "Acme Corp is a tech company. John Doe is the CEO."}],
            "page_count": 1,
        }

        mock_extraction = {
            "entities": [
                {"name": "Acme Corp", "type": "organization", "attributes": {}},
                {"name": "John Doe", "type": "person", "attributes": {}},
            ],
            "relationships": [
                {"source": "John Doe", "target": "Acme Corp", "type": "authored_by"}
            ]
        }

        with patch("app.ingestion.pipeline.PDFParser") as MockParser, \
             patch("app.ingestion.pipeline.EntityExtractor") as MockExtractor:
            MockParser.return_value.extract_text.return_value = mock_text_result
            MockExtractor.return_value.extract.return_value = mock_extraction

            pipeline = IngestionPipeline(db, groq_key="test_key")
            doc_id = pipeline.ingest("fake.pdf")

            assert doc_id is not None
            doc = db.get_document(doc_id)
            assert doc[4] == "ready"

            stats = db.get_graph_stats()
            assert stats["nodes"] == 2
            assert stats["edges"] == 1
            assert stats["chunks"] >= 1

        db.close()
    finally:
        os.unlink(db_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create app/ingestion/pipeline.py**

```python
from pathlib import Path
from app.storage.database import Database
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import TextChunker
from app.ingestion.extractor import EntityExtractor
from app.ingestion.graph_builder import GraphBuilder


class IngestionPipeline:
    def __init__(self, db: Database, groq_key: str, model: str = "llama3-8b-8192"):
        self.db = db
        self.parser = PDFParser()
        self.chunker = TextChunker()
        self.extractor = EntityExtractor(api_key=groq_key, model=model)
        self.graph_builder = GraphBuilder(db)

    def ingest(self, pdf_path: str) -> int:
        filename = Path(pdf_path).name
        doc_id = self.db.insert_document(filename)

        try:
            parsed = self.parser.extract_text(pdf_path)
            self.db.execute(
                "UPDATE documents SET page_count = ? WHERE id = ?",
                (parsed["page_count"], doc_id),
            )
            self.db.commit()

            chunks = self.chunker.chunk_from_pages(parsed["pages"], document_id=doc_id)

            for chunk in chunks:
                chunk_id = self.db.insert_chunk(
                    document_id=doc_id,
                    content=chunk["content"],
                    page_number=chunk.get("page_number"),
                    section_title=chunk.get("section_title", ""),
                )

                extraction = self.extractor.extract(chunk["content"], document_id=doc_id)
                node_ids = self.graph_builder.build_graph(extraction, document_id=doc_id)

                for node_id in node_ids:
                    self.db.insert_edge(
                        source_id=chunk_id,
                        target_id=node_id,
                        edge_type="mentions",
                    )

            self.db.update_document_status(doc_id, "ready")
            return doc_id

        except Exception as e:
            self.db.update_document_status(doc_id, "error")
            raise e
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/ingestion/pipeline.py tests/test_pipeline.py
git commit -m "feat: complete PDF ingestion pipeline"
```

---

### Task 8: Query Planner

**Files:**
- Create: `app/query/__init__.py`
- Create: `app/query/planner.py`
- Create: `tests/test_planner.py`

- [ ] **Step 1: Write failing test for query planning**

```python
# tests/test_planner.py
import json
from unittest.mock import patch, MagicMock
from app.query.planner import QueryPlanner


def test_plan_query_returns_structured_plan():
    planner = QueryPlanner(api_key="test_key")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps({
        "search_terms": ["Acme Corp", "revenue"],
        "entity_types": ["organization", "metric"],
        "traverse_edges": ["relates_to"],
        "max_results": 10,
    }))]

    with patch("app.query.planner.groq.Client") as MockClient:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockClient.return_value = mock_client

        plan = planner.plan("What is Acme Corp's revenue?")

        assert "search_terms" in plan
        assert "entity_types" in plan
        assert len(plan["search_terms"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_planner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create app/query/__init__.py**

```python
```

- [ ] **Step 4: Create app/query/planner.py**

```python
import json
import groq
from typing import Dict


PLANNER_PROMPT = """Given a user question, create a search plan to find relevant information in a knowledge graph.

Return JSON with this exact structure:
{
  "search_terms": ["term1", "term2"],
  "entity_types": ["person", "organization", "concept", "date", "metric"],
  "traverse_edges": ["relates_to", "mentions"],
  "max_results": 10,
  "query_type": "factual|relational|analytical"
}

Rules:
- Extract key terms from the question for full-text search
- Identify which entity types are relevant
- Determine which relationship types to traverse
- Return ONLY valid JSON, no other text

Question:
"""


class QueryPlanner:
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.client = groq.Client(api_key=api_key)
        self.model = model

    def plan(self, question: str) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a search planning assistant. Return only valid JSON."},
                    {"role": "user", "content": PLANNER_PROMPT + question},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]

            return json.loads(content)

        except (json.JSONDecodeError, Exception):
            return {
                "search_terms": question.split()[:5],
                "entity_types": ["person", "organization", "concept"],
                "traverse_edges": ["relates_to"],
                "max_results": 10,
                "query_type": "factual",
            }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_planner.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/query/ tests/test_planner.py
git commit -m "feat: LLM query planner for graph traversal"
```

---

### Task 9: Query Executor

**Files:**
- Create: `app/query/executor.py`
- Create: `tests/test_executor.py`

- [ ] **Step 1: Write failing tests for query execution**

```python
# tests/test_executor.py
import tempfile
import os
from app.storage.database import Database
from app.query.executor import QueryExecutor


def test_execute_search_plan():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = Database(db_path)

        org_id = db.insert_node("Acme Corp", "organization", {"industry": "tech"})
        person_id = db.insert_node("John Doe", "person", {"role": "CEO"})
        db.insert_edge(person_id, org_id, "authored_by")

        chunk_id = db.insert_chunk(1, "Acme Corp reported revenue of $10M.", 1, "Financial Results")

        executor = QueryExecutor(db)
        plan = {
            "search_terms": ["Acme Corp"],
            "entity_types": ["organization"],
            "traverse_edges": ["authored_by"],
            "max_results": 10,
        }

        results = executor.execute(plan)
        assert len(results["nodes"]) >= 1
        assert len(results["chunks"]) >= 1
        db.close()
    finally:
        os.unlink(db_path)


def test_execute_fts_only_fallback():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = Database(db_path)

        chunk_id = db.insert_chunk(1, "Machine learning is a subset of AI.", 1, "Introduction")

        executor = QueryExecutor(db)
        plan = {
            "search_terms": ["machine learning"],
            "entity_types": [],
            "traverse_edges": [],
            "max_results": 10,
        }

        results = executor.execute(plan)
        assert len(results["chunks"]) >= 1
        db.close()
    finally:
        os.unlink(db_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_executor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create app/query/executor.py**

```python
from typing import Dict, List
from app.storage.database import Database


class QueryExecutor:
    def __init__(self, db: Database):
        self.db = db

    def execute(self, plan: Dict) -> Dict[str, List]:
        nodes = self._search_nodes(plan)
        node_ids = [n[0] for n in nodes]

        related_nodes = []
        for node_id in node_ids:
            for edge_type in plan.get("traverse_edges", []):
                related = self.db.execute(
                    """SELECT n.id, n.name, n.type, n.attributes
                       FROM edges e
                       JOIN nodes n ON (n.id = e.target_id OR n.id = e.source_id)
                       WHERE (e.source_id = ? OR e.target_id = ?) AND e.type = ? AND n.id != ?""",
                    (node_id, node_id, edge_type, node_id),
                ).fetchall()
                related_nodes.extend(related)

        all_node_ids = list(set(node_ids + [r[0] for r in related_nodes]))

        chunks = self._search_chunks(plan, all_node_ids)

        return {
            "nodes": nodes + related_nodes,
            "chunks": chunks,
            "node_ids": all_node_ids,
        }

    def _search_nodes(self, plan: Dict) -> List:
        results = []
        entity_types = plan.get("entity_types", [])
        search_terms = plan.get("search_terms", [])

        for term in search_terms:
            try:
                found = self.db.search_nodes(f'"{term}"', limit=plan.get("max_results", 10))
                results.extend(found)
            except Exception:
                pass

        if entity_types and not results:
            placeholders = ",".join("?" * len(entity_types))
            results = self.db.execute(
                f"SELECT id, name, type, attributes FROM nodes WHERE type IN ({placeholders}) LIMIT ?",
                (*entity_types, plan.get("max_results", 10)),
            ).fetchall()

        return results

    def _search_chunks(self, plan: Dict, node_ids: List[int]) -> List:
        chunks = []
        search_terms = plan.get("search_terms", [])

        for term in search_terms:
            try:
                found = self.db.search_chunks(f'"{term}"', limit=plan.get("max_results", 10))
                chunks.extend(found)
            except Exception:
                pass

        if not chunks and node_ids:
            placeholders = ",".join("?" * len(node_ids))
            chunks = self.db.execute(
                f"""SELECT c.id, c.content, c.page_number, c.section_title, 0
                    FROM chunks c
                    JOIN edges e ON e.source_id = c.id OR e.target_id = c.id
                    WHERE e.source_id IN ({placeholders}) OR e.target_id IN ({placeholders})
                    LIMIT ?""",
                (*node_ids, *node_ids, plan.get("max_results", 10)),
            ).fetchall()

        return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_executor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/query/executor.py tests/test_executor.py
git commit -m "feat: graph query executor with FTS fallback"
```

---

### Task 10: Answer Synthesizer

**Files:**
- Create: `app/query/synthesizer.py`
- Create: `tests/test_synthesizer.py`

- [ ] **Step 1: Write failing test for synthesis**

```python
# tests/test_synthesizer.py
from unittest.mock import patch, MagicMock
from app.query.synthesizer import AnswerSynthesizer


def test_synthesize_answer():
    synthesizer = AnswerSynthesizer(api_key="test_key")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Acme Corp is a tech company with $10M revenue."))]

    with patch("app.query.synthesizer.groq.Client") as MockClient:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockClient.return_value = mock_client

        context = {
            "nodes": [("Acme Corp", "organization", "{}")],
            "chunks": [("Acme Corp reported revenue of $10M.", 1, "Financial Results")],
        }

        answer = synthesizer.synthesize("What is Acme Corp's revenue?", context)

        assert isinstance(answer, dict)
        assert "answer" in answer
        assert "sources" in answer
        assert len(answer["answer"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthesizer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create app/query/synthesizer.py**

```python
import groq
from typing import Dict, List


SYNTHESIS_PROMPT = """Answer the user's question based on the following context from a knowledge graph.

Context includes:
- Relevant entities found in the graph
- Text chunks from source documents

Rules:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so
- Cite source document names when possible
- Be concise and factual

Context:
{context}

Question:
"""


class AnswerSynthesizer:
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.client = groq.Client(api_key=api_key)
        self.model = model

    def synthesize(self, question: str, context: Dict[str, List]) -> Dict:
        context_str = self._format_context(context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on document knowledge graphs."},
                    {"role": "user", "content": SYNTHESIS_PROMPT.format(context=context_str, question=question)},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            answer_text = response.choices[0].message.content.strip()
            sources = self._extract_sources(context)

            return {
                "answer": answer_text,
                "sources": sources,
                "entity_count": len(context.get("nodes", [])),
                "chunk_count": len(context.get("chunks", [])),
            }

        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "entity_count": 0,
                "chunk_count": 0,
            }

    def _format_context(self, context: Dict[str, List]) -> str:
        parts = []

        nodes = context.get("nodes", [])
        if nodes:
            parts.append("Entities found:")
            for node in nodes:
                name = node[1] if len(node) > 1 else str(node)
                node_type = node[2] if len(node) > 2 else "unknown"
                parts.append(f"- {name} ({node_type})")

        chunks = context.get("chunks", [])
        if chunks:
            parts.append("\nRelevant text:")
            for chunk in chunks:
                content = chunk[1] if len(chunk) > 1 else str(chunk)
                page = chunk[2] if len(chunk) > 2 else "?"
                section = chunk[3] if len(chunk) > 3 else ""
                parts.append(f"- [Page {page}, {section}] {content[:500]}")

        return "\n".join(parts)

    def _extract_sources(self, context: Dict[str, List]) -> List[Dict]:
        sources = []
        seen = set()
        for chunk in context.get("chunks", []):
            source = {
                "content_preview": (chunk[1][:200] if len(chunk) > 1 else "") if isinstance(chunk[1], str) else "",
                "page": chunk[2] if len(chunk) > 2 else None,
                "section": chunk[3] if len(chunk) > 3 else "",
            }
            key = (source["page"], source["section"])
            if key not in seen:
                seen.add(key)
                sources.append(source)
        return sources
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthesizer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/query/synthesizer.py tests/test_synthesizer.py
git commit -m "feat: LLM answer synthesizer with source citations"
```

---

### Task 11: API Routes — Upload & Documents

**Files:**
- Create: `app/api/__init__.py`
- Create: `app/api/upload.py`
- Create: `tests/test_upload_api.py`

- [ ] **Step 1: Write failing tests for upload endpoint**

```python
# tests/test_upload_api.py
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
from app.storage.database import Database


client = TestClient(app)


def test_upload_pdf():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 dummy")
        f.flush()

        with patch("app.api.upload.settings") as mock_settings, \
             patch("app.api.upload.IngestionPipeline") as MockPipeline:
            mock_settings.upload_dir = tempfile.gettempdir()
            mock_settings.groq_api_key = "test_key"
            mock_settings.database_path = ":memory:"

            MockPipeline.return_value.ingest.return_value = 1

            with open(f.name, "rb") as pdf_file:
                response = client.post(
                    "/api/upload",
                    files=[("files", ("test.pdf", pdf_file, "application/pdf"))],
                )

            assert response.status_code == 200
            data = response.json()
            assert "documents" in data
    os.unlink(f.name)


def test_list_documents():
    with patch("app.api.upload.get_db") as mock_get_db:
        mock_db = MagicMock()
        mock_db.list_documents.return_value = [(1, "test.pdf", 10, "2024-01-01", "ready")]
        mock_get_db.return_value = mock_db

        response = client.get("/api/documents")
        assert response.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_upload_api.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create app/api/__init__.py**

```python
```

- [ ] **Step 4: Create app/api/upload.py**

```python
import os
import shutil
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from typing import List
from app.config import settings
from app.storage.database import Database
from app.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/api")


def get_db():
    return Database(settings.database_path)


def process_pdf_background(pdf_path: str, groq_key: str):
    db = Database(settings.database_path)
    try:
        pipeline = IngestionPipeline(db, groq_key)
        pipeline.ingest(pdf_path)
    finally:
        db.close()


@router.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    results = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            results.append({"filename": file.filename, "status": "error", "message": "Not a PDF"})
            continue

        save_path = os.path.join(settings.upload_dir, file.filename)
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        if background_tasks:
            background_tasks.add_task(process_pdf_background, save_path, settings.groq_api_key)
            results.append({"filename": file.filename, "status": "processing"})
        else:
            try:
                db = get_db()
                pipeline = IngestionPipeline(db, settings.groq_api_key)
                doc_id = pipeline.ingest(save_path)
                results.append({"filename": file.filename, "status": "ready", "document_id": doc_id})
                db.close()
            except Exception as e:
                results.append({"filename": file.filename, "status": "error", "message": str(e)})

    return {"documents": results}


@router.get("/documents")
async def list_documents():
    db = get_db()
    try:
        docs = db.list_documents()
        return {"documents": [
            {"id": d[0], "filename": d[1], "page_count": d[2], "upload_date": d[3], "status": d[4]}
            for d in docs
        ]}
    finally:
        db.close()


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    db = get_db()
    try:
        db.delete_document(doc_id)
        return {"status": "deleted"}
    finally:
        db.close()
```

- [ ] **Step 5: Update app/main.py to include router**

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.upload import router as upload_router

app = FastAPI(title="Vectorless RAG", version="0.1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(upload_router)


@app.get("/")
async def root():
    return {"message": "Vectorless RAG API"}


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_upload_api.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add app/api/ app/main.py tests/test_upload_api.py
git commit -m "feat: PDF upload and document management API"
```

---

### Task 12: API Routes — Chat

**Files:**
- Create: `app/api/chat.py`
- Create: `tests/test_chat_api.py`

- [ ] **Step 1: Write failing test for chat endpoint**

```python
# tests/test_chat_api.py
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

client = TestClient(app)


def test_chat_returns_answer():
    with patch("app.api.chat.settings") as mock_settings, \
         patch("app.api.chat.Database") as MockDB, \
         patch("app.api.chat.QueryPlanner") as MockPlanner, \
         patch("app.api.chat.QueryExecutor") as MockExecutor, \
         patch("app.api.chat.AnswerSynthesizer") as MockSynth:

        mock_settings.groq_api_key = "test_key"
        mock_settings.database_path = ":memory:"

        MockPlanner.return_value.plan.return_value = {
            "search_terms": ["revenue"],
            "entity_types": ["metric"],
            "traverse_edges": ["relates_to"],
            "max_results": 10,
        }

        MockExecutor.return_value.execute.return_value = {
            "nodes": [(1, "Revenue", "metric", "{}")],
            "chunks": [(1, "Revenue was $10M", 1, "Financials")],
            "node_ids": [1],
        }

        MockSynth.return_value.synthesize.return_value = {
            "answer": "Revenue was $10M.",
            "sources": [{"page": 1, "section": "Financials"}],
        }

        response = client.post("/api/chat", json={"question": "What is the revenue?"})

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chat_api.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create app/api/chat.py**

```python
from fastapi import APIRouter
from pydantic import BaseModel
from app.config import settings
from app.storage.database import Database
from app.query.planner import QueryPlanner
from app.query.executor import QueryExecutor
from app.query.synthesizer import AnswerSynthesizer

router = APIRouter(prefix="/api")


class ChatRequest(BaseModel):
    question: str


@router.post("/chat")
async def chat(request: ChatRequest):
    db = Database(settings.database_path)
    try:
        planner = QueryPlanner(api_key=settings.groq_api_key)
        executor = QueryExecutor(db)
        synthesizer = AnswerSynthesizer(api_key=settings.groq_api_key)

        plan = planner.plan(request.question)
        context = executor.execute(plan)
        result = synthesizer.synthesize(request.question, context)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "entities_found": result["entity_count"],
            "chunks_used": result["chunk_count"],
            "plan": plan,
        }
    finally:
        db.close()


@router.get("/graph/stats")
async def graph_stats():
    db = Database(settings.database_path)
    try:
        stats = db.get_graph_stats()
        return stats
    finally:
        db.close()
```

- [ ] **Step 4: Update app/main.py to include chat router**

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router

app = FastAPI(title="Vectorless RAG", version="0.1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(upload_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    return {"message": "Vectorless RAG API"}


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_chat_api.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/api/chat.py app/main.py tests/test_chat_api.py
git commit -m "feat: chat API with query planning, execution, synthesis"
```

---

### Task 13: Web Frontend

**Files:**
- Modify: `static/index.html`
- Create: `static/style.css`
- Create: `static/app.js`

- [ ] **Step 1: Create static/style.css**

```css
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; height: 100vh; }
.container { display: flex; height: 100vh; }
.sidebar { width: 320px; background: #fff; border-right: 1px solid #e0e0e0; display: flex; flex-direction: column; }
.sidebar h2 { padding: 16px; border-bottom: 1px solid #e0e0e0; font-size: 16px; }
.upload-area { padding: 16px; border-bottom: 1px solid #e0e0e0; }
.dropzone { border: 2px dashed #ccc; border-radius: 8px; padding: 32px; text-align: center; cursor: pointer; transition: border-color 0.2s; }
.dropzone:hover, .dropzone.active { border-color: #4a90d9; }
.dropzone p { color: #666; font-size: 14px; }
.file-list { flex: 1; overflow-y: auto; padding: 8px; }
.file-item { padding: 8px 12px; border-radius: 6px; margin-bottom: 4px; display: flex; justify-content: space-between; align-items: center; font-size: 13px; }
.file-item.ready { background: #e8f5e9; }
.file-item.processing { background: #fff3e0; }
.file-item.error { background: #ffebee; }
.file-item .name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-item .status { font-size: 11px; color: #666; margin-left: 8px; }
.file-item .delete { cursor: pointer; color: #999; margin-left: 8px; }
.file-item .delete:hover { color: #d32f2f; }
.main { flex: 1; display: flex; flex-direction: column; }
.chat-header { padding: 16px; background: #fff; border-bottom: 1px solid #e0e0e0; }
.chat-header h1 { font-size: 18px; }
.messages { flex: 1; overflow-y: auto; padding: 16px; }
.message { margin-bottom: 16px; max-width: 80%; }
.message.user { margin-left: auto; }
.message .bubble { padding: 12px 16px; border-radius: 12px; font-size: 14px; line-height: 1.5; }
.message.user .bubble { background: #4a90d9; color: #fff; border-bottom-right-radius: 4px; }
.message.assistant .bubble { background: #fff; border: 1px solid #e0e0e0; border-bottom-left-radius: 4px; }
.message .sources { font-size: 11px; color: #888; margin-top: 4px; }
.input-area { padding: 16px; background: #fff; border-top: 1px solid #e0e0e0; }
.input-row { display: flex; gap: 8px; }
.input-row input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 8px; font-size: 14px; }
.input-row button { padding: 12px 24px; background: #4a90d9; color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; }
.input-row button:hover { background: #357abd; }
.input-row button:disabled { background: #ccc; cursor: not-allowed; }
.stats { padding: 8px 16px; font-size: 11px; color: #888; border-top: 1px solid #e0e0e0; }
```

- [ ] **Step 2: Create static/app.js**

```javascript
let documents = [];

document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const messages = document.getElementById('messages');

    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('active'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('active'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('active');
        handleFiles(e.dataTransfer.files);
    });
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });

    loadDocuments();
    loadStats();
});

async function handleFiles(files) {
    const formData = new FormData();
    for (const file of files) {
        if (file.name.endsWith('.pdf')) {
            formData.append('files', file);
        }
    }

    try {
        const response = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await response.json();
        loadDocuments();
        addMessage('assistant', `Uploaded ${data.documents.length} file(s). Processing...`);
    } catch (error) {
        addMessage('assistant', 'Error uploading files: ' + error.message);
    }
}

async function loadDocuments() {
    try {
        const response = await fetch('/api/documents');
        const data = await response.json();
        documents = data.documents;
        renderDocuments();
    } catch (error) {
        console.error('Failed to load documents:', error);
    }
}

function renderDocuments() {
    const list = document.getElementById('fileList');
    list.innerHTML = documents.map(doc => `
        <div class="file-item ${doc.status}">
            <span class="name" title="${doc.filename}">${doc.filename}</span>
            <span class="status">${doc.status}</span>
            <span class="delete" onclick="deleteDocument(${doc.id})">&times;</span>
        </div>
    `).join('');
}

async function deleteDocument(id) {
    try {
        await fetch(`/api/documents/${id}`, { method: 'DELETE' });
        loadDocuments();
    } catch (error) {
        console.error('Failed to delete:', error);
    }
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const question = input.value.trim();
    if (!question) return;

    addMessage('user', question);
    input.value = '';
    document.getElementById('sendBtn').disabled = true;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });
        const data = await response.json();

        let answerHtml = data.answer;
        if (data.sources && data.sources.length > 0) {
            answerHtml += `<div class="sources">Sources: ${data.sources.map(s => `p.${s.page || '?'}`).join(', ')}</div>`;
        }
        addMessage('assistant', answerHtml);
    } catch (error) {
        addMessage('assistant', 'Error: ' + error.message);
    } finally {
        document.getElementById('sendBtn').disabled = false;
    }
}

function addMessage(role, content) {
    const messages = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.innerHTML = `<div class="bubble">${content}</div>`;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

async function loadStats() {
    try {
        const response = await fetch('/api/graph/stats');
        const data = await response.json();
        document.getElementById('stats').textContent =
            `Graph: ${data.nodes} nodes, ${data.edges} edges, ${data.chunks} chunks`;
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}
```

- [ ] **Step 3: Create static/index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vectorless RAG</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>Documents</h2>
            <div class="upload-area">
                <div class="dropzone" id="dropzone">
                    <p>Drop PDFs here or click to upload</p>
                </div>
                <input type="file" id="fileInput" multiple accept=".pdf" hidden>
            </div>
            <div class="file-list" id="fileList"></div>
            <div class="stats" id="stats">Loading...</div>
        </div>
        <div class="main">
            <div class="chat-header">
                <h1>Vectorless RAG</h1>
            </div>
            <div class="messages" id="messages">
                <div class="message assistant">
                    <div class="bubble">Upload PDFs on the left, then ask me anything about them.</div>
                </div>
            </div>
            <div class="input-area">
                <div class="input-row">
                    <input type="text" id="chatInput" placeholder="Ask a question...">
                    <button id="sendBtn">Send</button>
                </div>
            </div>
        </div>
    </div>
    <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Commit**

```bash
git add static/
git commit -m "feat: web frontend with chat and PDF upload UI"
```

---

### Task 14: Environment Setup & Documentation

**Files:**
- Create: `.env`
- Create: `README.md`
- Create: `.gitignore`

- [ ] **Step 1: Create .env**

```
GROQ_API_KEY=gsk_your_key_here
DATABASE_PATH=./data/graph.db
UPLOAD_DIR=./uploads
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
.env
data/
uploads/
*.db
.venv/
venv/
```

- [ ] **Step 3: Create README.md**

```markdown
# Vectorless RAG

Knowledge-graph-based Retrieval-Augmented Generation without vector embeddings.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your Groq API key
   ```

3. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

4. Open http://localhost:8000

## Usage

1. Upload PDFs via the sidebar
2. Wait for processing (entity extraction)
3. Ask questions in the chat

## Architecture

- **Ingestion:** PDF → Text → Chunks → LLM extracts entities/edges → SQLite graph
- **Query:** Question → LLM plans graph traversal → Execute → Retrieve chunks → LLM synthesizes answer
```

- [ ] **Step 4: Commit**

```bash
git add .env .gitignore README.md
git commit -m "feat: environment setup and documentation"
```

---

### Task 15: Run Full Test Suite & Verify

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Start server and test manually**

Run: `uvicorn app.main:app --reload`

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final verification and cleanup"
```
