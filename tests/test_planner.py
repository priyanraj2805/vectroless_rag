import json
from unittest.mock import patch, MagicMock
from app.query.planner import QueryPlanner


def test_plan_query_returns_structured_plan():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps({
        "search_terms": ["Acme Corp", "revenue"],
        "entity_types": ["organization", "metric"],
        "traverse_edges": ["relates_to"],
        "max_results": 10,
    })))]

    with patch("app.query.planner.groq.Client") as MockClient:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockClient.return_value = mock_client

        planner = QueryPlanner(api_key="test_key")
        plan = planner.plan("What is Acme Corp's revenue?")

        assert "search_terms" in plan
        assert "entity_types" in plan
        assert len(plan["search_terms"]) == 2
