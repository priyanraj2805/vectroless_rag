import json
from app.llm_client import FallbackLLMClient
from typing import Dict


PLANNER_PROMPT = """Given a user question, create a search plan to find relevant information in a knowledge graph.

Return JSON with this exact structure:
{{
  "search_terms": ["term1", "term2"],
  "entity_types": ["person", "organization", "concept", "date", "metric"],
  "traverse_edges": ["relates_to", "mentions"],
  "max_results": {max_results},
  "query_type": "factual|relational|analytical|summary"
}}

Rules:
- Extract key terms from the question for full-text search
- Identify which entity types are relevant
- Determine which relationship types to traverse
- Set max_results to at least {max_results} since {num_docs} document(s) are selected
- If the question asks for summaries or overview of multiple documents, set query_type to "summary"
- Return ONLY valid JSON, no other text

Question:
"""


class QueryPlanner:
    def __init__(self, openrouter_key: str = "", opencode_key: str = ""):
        self.client = FallbackLLMClient(openrouter_api_key=openrouter_key, opencode_api_key=opencode_key)

    def plan(self, question: str, num_docs: int = 1) -> Dict:
        # Scale max_results: at least 5 chunks per document, minimum 10
        max_results = max(10, num_docs * 5)

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a search planning assistant. Return only valid JSON."},
                    {"role": "user", "content": PLANNER_PROMPT.format(
                        max_results=max_results,
                        num_docs=num_docs,
                    ) + question},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]

            plan = json.loads(content)
            # Always enforce the scaled max_results — don't let LLM set it too low
            plan["max_results"] = max(plan.get("max_results", max_results), max_results)
            plan["num_docs"] = num_docs
            return plan

        except (json.JSONDecodeError, Exception):
            return {
                "search_terms": question.split()[:5],
                "entity_types": ["person", "organization", "concept"],
                "traverse_edges": ["relates_to"],
                "max_results": max_results,
                "num_docs": num_docs,
                "query_type": "factual",
            }
