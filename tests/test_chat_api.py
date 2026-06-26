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
            "chunks": [(1, "Revenue was $10M", 1, "Financials", 0)],
            "node_ids": [1],
        }

        MockSynth.return_value.synthesize.return_value = {
            "answer": "Revenue was $10M.",
            "sources": [{"page": 1, "section": "Financials", "content_preview": "Revenue was $10M"}],
            "entity_count": 1,
            "chunk_count": 1,
        }

        response = client.post("/api/chat", json={"question": "What is the revenue?"})

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0


def test_graph_stats():
    with patch("app.api.chat.Database") as MockDB:
        mock_db = MagicMock()
        mock_db.get_graph_stats.return_value = {"nodes": 5, "edges": 3, "chunks": 10, "documents": 2}
        MockDB.return_value = mock_db

        response = client.get("/api/graph/stats")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
