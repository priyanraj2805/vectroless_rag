import json
import time
from app.llm_client import FallbackLLMClient
from app.cache import get_redis, make_key, cache_get, cache_set
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
    def __init__(self, openrouter_key: str = "", opencode_key: str = "", redis_url: str = ""):
        self.client = FallbackLLMClient(openrouter_api_key=openrouter_key, opencode_api_key=opencode_key)
        self.r = get_redis(redis_url)

    def extract(self, text: str, document_id: int) -> Dict[str, List]:
        # Check cache — same text chunk never re-extracted
        cache_key = make_key("extract", text[:500])
        cached = cache_get(self.r, cache_key)
        if cached:
            # Re-attach document_id since it varies per document
            for entity in cached.get("entities", []):
                entity["document_id"] = document_id
            return cached

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an entity extraction assistant. Return only valid JSON."},
                        {"role": "user", "content": EXTRACTION_PROMPT + text[:6000]},
                    ],
                    temperature=0.1,
                    max_tokens=800,
                )

                raw = response.choices[0].message.content
                if not raw or not raw.strip():
                    return {"entities": [], "relationships": []}
                content = raw.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    if content.endswith("```"):
                        content = content[:-3]

                result = json.loads(content)
                if not isinstance(result, dict):
                    return {"entities": [], "relationships": []}
                result.setdefault("entities", [])
                result.setdefault("relationships", [])

                for entity in result["entities"]:
                    entity["document_id"] = document_id

                # Cache for 24 hours (chunks rarely change)
                cache_set(self.r, cache_key, result, ttl=86400)

                return result

            except json.JSONDecodeError:
                return {"entities": [], "relationships": []}
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < 2:
                    wait = 20 if attempt == 0 else 40
                    print(f"Rate limit hit, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"Extraction error: {e}")
                    return {"entities": [], "relationships": []}
