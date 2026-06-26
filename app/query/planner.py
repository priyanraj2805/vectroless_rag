import json
import groq
from typing import Dict


PLANNER_PROMPT = """Given a user question, create a search plan to find relevant information in a knowledge graph.

Return JSON with this exact structure:
{
  "search_terms": ["term1", "term2"],
  "entity_types": ["person", "organization", "concept", "date", "metric"],
  "traverse_edges": ["relates_to", "mentions"],
  "max_results": 10,
  "query_type": "factual|relational|analytical"
}

Rules:
- Extract key terms from the question for full-text search
- Identify which entity types are relevant
- Determine which relationship types to traverse
- Return ONLY valid JSON, no other text

Question:
"""


class QueryPlanner:
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.client = groq.Client(api_key=api_key)
        self.model = model

    def plan(self, question: str) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a search planning assistant. Return only valid JSON."},
                    {"role": "user", "content": PLANNER_PROMPT + question},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]

            return json.loads(content)

        except (json.JSONDecodeError, Exception):
            return {
                "search_terms": question.split()[:5],
                "entity_types": ["person", "organization", "concept"],
                "traverse_edges": ["relates_to"],
                "max_results": 10,
                "query_type": "factual",
            }
