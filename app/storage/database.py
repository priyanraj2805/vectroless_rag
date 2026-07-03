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

    CREATE TABLE IF NOT EXISTS embeddings (
        chunk_id   INTEGER PRIMARY KEY,
        vector     BLOB NOT NULL,
        FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
    );
    """

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(self.SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self):
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(documents)").fetchall()}
        if "completed_at" not in cols:
            self.conn.execute("ALTER TABLE documents ADD COLUMN completed_at TIMESTAMP")
        # Rebuild FTS indexes to fix any sync issues from previous failed writes
        try:
            self.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            self.conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')")
        except Exception:
            pass

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        try:
            return self.conn.execute(query, params)
        except Exception:
            self.conn.rollback()
            raise

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

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
        if status in ("ready", "error"):
            self.execute(
                "UPDATE documents SET status = ?, completed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, doc_id),
            )
        else:
            self.execute("UPDATE documents SET status = ? WHERE id = ?", (status, doc_id))
        self.commit()

    def get_document(self, doc_id: int):
        return self.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()

    def list_documents(self):
        return self.execute(
            "SELECT id, filename, page_count, upload_date, status, completed_at FROM documents ORDER BY upload_date DESC"
        ).fetchall()

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

    def insert_embedding(self, chunk_id: int, vector: bytes):
        self.execute(
            "INSERT OR REPLACE INTO embeddings (chunk_id, vector) VALUES (?, ?)",
            (chunk_id, vector),
        )
        self.commit()

    def get_all_embeddings(self, document_ids: list = None):
        """Return [(chunk_id, vector_blob, content, page_number, section_title)]"""
        if document_ids:
            placeholders = ",".join("?" * len(document_ids))
            return self.execute(
                f"""SELECT e.chunk_id, e.vector, c.content, c.page_number, c.section_title
                    FROM embeddings e
                    JOIN chunks c ON c.id = e.chunk_id
                    WHERE c.document_id IN ({placeholders})""",
                tuple(document_ids),
            ).fetchall()
        return self.execute(
            """SELECT e.chunk_id, e.vector, c.content, c.page_number, c.section_title
               FROM embeddings e
               JOIN chunks c ON c.id = e.chunk_id"""
        ).fetchall()

    def get_graph_stats(self):
        nodes = self.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edges = self.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        chunks = self.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        docs = self.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        return {"nodes": nodes, "edges": edges, "chunks": chunks, "documents": docs}
