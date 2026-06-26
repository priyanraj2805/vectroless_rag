import json
from unittest.mock import patch, MagicMock
from app.ingestion.extractor import EntityExtractor


def test_extract_entities_returns_structured_data():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps({
        "entities": [
            {"name": "Acme Corp", "type": "organization", "attributes": {"industry": "tech"}},
            {"name": "John Doe", "type": "person", "attributes": {"role": "CEO"}},
        ],
        "relationships": [
            {"source": "John Doe", "target": "Acme Corp", "type": "authored_by"}
        ]
    })))]

    with patch("app.ingestion.extractor.groq.Client") as MockClient:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockClient.return_value = mock_client

        extractor = EntityExtractor(api_key="test_key")
        result = extractor.extract("John Doe is the CEO of Acme Corp.", document_id=1)

        assert "entities" in result
        assert "relationships" in result
        assert len(result["entities"]) == 2
        assert len(result["relationships"]) == 1
