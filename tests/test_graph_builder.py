import tempfile
import os
from app.storage.database import Database
from app.ingestion.graph_builder import GraphBuilder


def test_build_graph_from_extraction():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = None
    try:
        db = Database(db_path)
        builder = GraphBuilder(db)
        doc_id = db.insert_document("test.pdf", 1)

        extraction = {
            "entities": [
                {"name": "Acme Corp", "type": "organization", "attributes": {"industry": "tech"}, "document_id": doc_id},
                {"name": "John Doe", "type": "person", "attributes": {"role": "CEO"}, "document_id": doc_id},
            ],
            "relationships": [
                {"source": "John Doe", "target": "Acme Corp", "type": "authored_by"}
            ]
        }

        node_ids = builder.build_graph(extraction, document_id=doc_id)
        assert len(node_ids) == 2

        stats = db.get_graph_stats()
        assert stats["nodes"] == 2
        assert stats["edges"] == 1
    finally:
        if db:
            db.close()
        os.unlink(db_path)
