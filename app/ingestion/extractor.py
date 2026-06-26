import json
import groq
from typing import Dict, List


EXTRACTION_PROMPT = """Extract entities and relationships from the following text.

Return JSON with this exact structure:
{
  "entities": [
    {"name": "entity name", "type": "person|organization|concept|date|metric", "attributes": {"key": "value"}}
  ],
  "relationships": [
    {"source": "source entity name", "target": "target entity name", "type": "mentions|relates_to|part_of|authored_by|contains"}
  ]
}

Rules:
- Extract ALL named entities (people, organizations, concepts)
- Only create relationships between entities you extracted
- Use exact names from the text
- Return ONLY valid JSON, no other text

Text:
"""


class EntityExtractor:
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.client = groq.Client(api_key=api_key)
        self.model = model

    def extract(self, text: str, document_id: int) -> Dict[str, List]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an entity extraction assistant. Return only valid JSON."},
                    {"role": "user", "content": EXTRACTION_PROMPT + text[:6000]},
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]

            result = json.loads(content)
            result.setdefault("entities", [])
            result.setdefault("relationships", [])

            for entity in result["entities"]:
                entity["document_id"] = document_id

            return result

        except json.JSONDecodeError:
            return {"entities": [], "relationships": []}
        except Exception as e:
            print(f"Extraction error: {e}")
            return {"entities": [], "relationships": []}
