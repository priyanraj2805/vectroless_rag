from app.llm_client import FallbackLLMClient
from typing import Dict, List


SYSTEM_PROMPT = """You are a helpful AI assistant that can both chat naturally and answer questions about uploaded documents.

Behavior rules:
- If the user says hi, hello, or asks a general conversational question — respond warmly and naturally like a friendly assistant.
- If the user asks about the uploaded documents — use ONLY the provided document context to answer, citing sources.
- If the user asks for a summary of each document — provide a clear, separate summary paragraph for EACH document shown in the context. Label each one.
- If the user asks about documents but the context is empty or irrelevant — say you couldn't find relevant information in the uploaded documents, and suggest they try rephrasing or uploading more PDFs.
- Never make up information. Never say you have no context if the user is just greeting you."""

SYNTHESIS_PROMPT = """Document context from the knowledge graph:
{context}

User question: {question}

Answer:"""

CONVERSATIONAL_PROMPT = """User message: {question}

Respond naturally. If they seem to be asking about documents, let them know they can upload PDFs and ask questions about them."""


class AnswerSynthesizer:
    def __init__(self, groq_key: str = "", openrouter_key: str = ""):
        self.client = FallbackLLMClient(groq_api_key=groq_key, openrouter_api_key=openrouter_key)

    def synthesize(self, question: str, context: Dict[str, List], has_documents: bool = True) -> Dict:
        context_str = self._format_context(context)
        is_conversational = not has_documents or self._is_conversational(question)

        if is_conversational:
            prompt = CONVERSATIONAL_PROMPT.format(question=question)
        else:
            prompt = SYNTHESIS_PROMPT.format(context=context_str, question=question)

        # Scale max_tokens based on how many docs are in context
        num_docs = len(context.get("doc_groups", {})) or 1
        max_tokens = min(2000, max(1000, num_docs * 600))

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=max_tokens,
            )

            answer_text = response.choices[0].message.content.strip()
            sources = [] if is_conversational else self._extract_sources(context)

            return {
                "answer": answer_text,
                "sources": sources,
                "entity_count": len(context.get("nodes", [])),
                "chunk_count": len(context.get("chunks", [])),
                "conversational": is_conversational,
            }

        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "entity_count": 0,
                "chunk_count": 0,
                "conversational": False,
            }

    def _is_conversational(self, question: str) -> bool:
        GREETINGS = {
            'hi', 'hii', 'hiii', 'hello', 'hey', 'howdy', 'hiya', 'yo', 'sup',
            'good morning', 'good afternoon', 'good evening', 'good night',
            'how are you', 'how r u', 'whats up', "what's up",
            'thanks', 'thank you', 'ty', 'thx', 'bye', 'goodbye', 'ok', 'okay',
            'cool', 'nice', 'great', 'awesome', 'got it', 'sure', 'alright',
        }
        q = question.lower().strip().rstrip('!?.,')
        return q in GREETINGS or len(q.split()) <= 2

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
        doc_groups = context.get("doc_groups", {})

        if chunks:
            if doc_groups and len(doc_groups) > 1:
                # Multi-document: group chunks by document so the LLM sees clear per-doc sections
                parts.append("\nRelevant text per document:")
                for doc_label, doc_chunks in doc_groups.items():
                    parts.append(f"\n--- {doc_label} ---")
                    for chunk in doc_chunks:
                        content = chunk[1] if len(chunk) > 1 else str(chunk)
                        page = chunk[2] if len(chunk) > 2 else "?"
                        section = chunk[3] if len(chunk) > 3 else ""
                        # Increased from 600 → 1500 chars per chunk for richer summaries
                        parts.append(f"[Page {page}{', ' + section if section else ''}]\n{content[:1500]}")
            else:
                parts.append("\nRelevant text from documents:")
                for chunk in chunks:
                    content = chunk[1] if len(chunk) > 1 else str(chunk)
                    page = chunk[2] if len(chunk) > 2 else "?"
                    section = chunk[3] if len(chunk) > 3 else ""
                    parts.append(f"[Page {page}{', ' + section if section else ''}]\n{content[:1500]}")

        return "\n".join(parts) if parts else "No document context available."

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
