import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

client = TestClient(app)


def test_list_documents():
    with patch("app.api.upload.get_db") as mock_get_db:
        mock_db = MagicMock()
        mock_db.list_documents.return_value = [(1, "test.pdf", 10, "2024-01-01", "ready")]
        mock_get_db.return_value = mock_db

        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data


def test_delete_document():
    with patch("app.api.upload.get_db") as mock_get_db:
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        response = client.delete("/api/documents/1")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"
