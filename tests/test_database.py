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


def test_insert_node():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = Database(db_path)
        doc_id = db.insert_document("test.pdf")
        node_id = db.insert_node("Acme Corp", "organization", {"industry": "tech"}, doc_id)
        assert node_id == 1
        nodes = db.execute("SELECT * FROM nodes").fetchall()
        assert len(nodes) == 1
        assert nodes[0][1] == "Acme Corp"
        assert nodes[0][2] == "organization"
        db.close()
    finally:
        os.unlink(db_path)


def test_insert_edge():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = Database(db_path)
        n1 = db.insert_node("A", "concept")
        n2 = db.insert_node("B", "concept")
        edge_id = db.insert_edge(n1, n2, "relates_to")
        assert edge_id == 1
        edges = db.execute("SELECT * FROM edges").fetchall()
        assert len(edges) == 1
        db.close()
    finally:
        os.unlink(db_path)


def test_search_chunks():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = Database(db_path)
        doc_id = db.insert_document("test.pdf")
        db.insert_chunk(doc_id, "Machine learning is a subset of AI", 1, "Introduction")
        results = db.search_chunks("machine learning")
        assert len(results) >= 1
        db.close()
    finally:
        os.unlink(db_path)


def test_get_graph_stats():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = Database(db_path)
        stats = db.get_graph_stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0
        assert stats["chunks"] == 0
        assert stats["documents"] == 0
        db.close()
    finally:
        os.unlink(db_path)
