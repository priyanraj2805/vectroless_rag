import tempfile
import os
from app.storage.database import Database
from app.query.executor import QueryExecutor


def test_execute_search_plan():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = None
    try:
        db = Database(db_path)

        doc_id = db.insert_document("test.pdf", 1)
        org_id = db.insert_node("Acme Corp", "organization", {"industry": "tech"}, doc_id)
        person_id = db.insert_node("John Doe", "person", {"role": "CEO"}, doc_id)
        db.insert_edge(person_id, org_id, "authored_by")
        db.insert_chunk(doc_id, "Acme Corp reported revenue of $10M.", 1, "Financial Results")

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
    finally:
        if db:
            db.close()
        os.unlink(db_path)


def test_execute_fts_only_fallback():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = None
    try:
        db = Database(db_path)
        doc_id = db.insert_document("test.pdf", 1)
        db.insert_chunk(doc_id, "Machine learning is a subset of AI.", 1, "Introduction")

        executor = QueryExecutor(db)
        plan = {
            "search_terms": ["machine learning"],
            "entity_types": [],
            "traverse_edges": [],
            "max_results": 10,
        }

        results = executor.execute(plan)
        assert len(results["chunks"]) >= 1
    finally:
        if db:
            db.close()
        os.unlink(db_path)
