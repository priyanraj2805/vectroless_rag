import tempfile
import os
from unittest.mock import patch, MagicMock
from app.storage.database import Database
from app.ingestion.pipeline import IngestionPipeline


def test_full_ingestion_pipeline():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = None
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
            assert stats["edges"] >= 1
            assert stats["chunks"] >= 1

    finally:
        if db:
            db.close()
        os.unlink(db_path)
