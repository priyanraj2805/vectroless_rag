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
            "nodes": [(1, "Acme Corp", "organization", "{}")],
            "chunks": [(1, "Acme Corp reported revenue of $10M.", 1, "Financial Results", 0)],
        }

        answer = synthesizer.synthesize("What is Acme Corp's revenue?", context)

        assert isinstance(answer, dict)
        assert "answer" in answer
        assert "sources" in answer
        assert len(answer["answer"]) > 0
