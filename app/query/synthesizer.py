import groq
from typing import Dict, List


SYNTHESIS_PROMPT = """Answer the user's question based on the following context from a knowledge graph.

Context includes:
- Relevant entities found in the graph
- Text chunks from source documents

Rules:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so
- Cite source document names when possible
- Be concise and factual

Context:
{context}

Question:
"""


class AnswerSynthesizer:
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.client = groq.Client(api_key=api_key)
        self.model = model

    def synthesize(self, question: str, context: Dict[str, List]) -> Dict:
        context_str = self._format_context(context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on document knowledge graphs."},
                    {"role": "user", "content": SYNTHESIS_PROMPT.format(context=context_str) + question},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            answer_text = response.choices[0].message.content.strip()
            sources = self._extract_sources(context)

            return {
                "answer": answer_text,
                "sources": sources,
                "entity_count": len(context.get("nodes", [])),
                "chunk_count": len(context.get("chunks", [])),
            }

        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "entity_count": 0,
                "chunk_count": 0,
            }

    def _format_context(self, context: Dict[str, List]) -> str:
        parts = []

        nodes = context.get("nodes", [])
        if nodes:
            parts.append("Entities found:")
            for node in nodes:
                name = node[1] if len(node) > 1 else str(node)
                node_type = node[2] if len(node) > 2 else "unknown"
                parts.append(f"- {name} ({node_type})")

        chunks = context.get("chunks", [])
        if chunks:
            parts.append("\nRelevant text:")
            for chunk in chunks:
                content = chunk[1] if len(chunk) > 1 else str(chunk)
                page = chunk[2] if len(chunk) > 2 else "?"
                section = chunk[3] if len(chunk) > 3 else ""
                parts.append(f"- [Page {page}, {section}] {content[:500]}")

        return "\n".join(parts)

    def _extract_sources(self, context: Dict[str, List]) -> List[Dict]:
        sources = []
        seen = set()
        for chunk in context.get("chunks", []):
            source = {
                "content_preview": (chunk[1][:200] if len(chunk) > 1 else "") if isinstance(chunk[1], str) else "",
                "page": chunk[2] if len(chunk) > 2 else None,
                "section": chunk[3] if len(chunk) > 3 else "",
            }
            key = (source["page"], source["section"])
            if key not in seen:
                seen.add(key)
                sources.append(source)
        return sources
