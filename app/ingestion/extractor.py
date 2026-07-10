import json
from app.llm_client import FallbackLLMClient
from app.cache import get_redis, make_key, cache_get, cache_set
from typing import Dict, List


EXTRACTION_PROMPT = """Extract entities and relationships from the following text passages.

Return JSON with this exact structure:
{
  "entities": [
    {"name": "entity name", "type": "person|organization|concept|date|metric", "attributes": {}}
  ],
  "relationships": [
    {"source": "source entity name", "target": "target entity name", "type": "mentions|relates_to|part_of|authored_by|contains"}
  ]
}

Rules:
- Extract ALL named entities (people, organizations, concepts) across all passages
- Only create relationships between entities you extracted
- Use exact names from the text
- Return ONLY valid JSON, no other text

Text passages:
"""


def _parse_extraction_response(raw: str) -> dict | None:
    if not raw or not raw.strip():
        return None
    content = raw.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict):
        return None
    result.setdefault("entities", [])
    result.setdefault("relationships", [])
    return result


class EntityExtractor:
    def __init__(self, groq_api_key: str = "", groq_base_url: str = "", groq_model: str = "",
                 ollama_base_url: str = "", ollama_model: str = "", ollama_api_key: str = "", redis_url: str = ""):
        self.client = FallbackLLMClient(
            groq_api_key=groq_api_key, groq_base_url=groq_base_url, groq_model=groq_model,
            ollama_base_url=ollama_base_url, ollama_model=ollama_model, ollama_api_key=ollama_api_key,
        )
        self.r = get_redis(redis_url)

    def extract_batch(self, texts: List[str], document_id: int) -> Dict[str, List]:
        """Extract entities from multiple text chunks in a single LLM call."""
        # 800 chars per chunk keeps the total prompt under ~4 000 chars — safe for free-tier models
        combined = "\n\n---\n\n".join(t[:800] for t in texts)
        cache_key = make_key("extract_batch", combined[:500])
        cached = cache_get(self.r, cache_key)
        if cached:
            for entity in cached.get("entities", []):
                entity["document_id"] = document_id
            return cached

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an entity extraction assistant. Return only valid JSON."},
                    {"role": "user", "content": EXTRACTION_PROMPT + combined[:4000]},
                ],
                temperature=0.1,
                max_tokens=800,
            )
            result = _parse_extraction_response(response.choices[0].message.content)
            if result is None:
                return {"entities": [], "relationships": []}

            for entity in result["entities"]:
                entity["document_id"] = document_id

            cache_set(self.r, cache_key, result, ttl=86400)
            return result

        except Exception as e:
            print(f"[extractor] batch failed (doc {document_id}), skipping graph for this batch: {e!s:.200}")
            return {"entities": [], "relationships": []}

    def extract(self, text: str, document_id: int) -> Dict[str, List]:
        """Single-chunk extraction (kept for backward compatibility)."""
        return self.extract_batch([text], document_id)
