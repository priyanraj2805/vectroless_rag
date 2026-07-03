from app.llm_client import FallbackLLMClient
from typing import Dict, List


SYSTEM_PROMPT = """You are a document Q&A assistant. Answer questions strictly based on the provided document context.

Rules:
- Use ONLY the information from the document context below to answer.
- If the context does not contain enough information to answer, say: "I couldn't find relevant information in the uploaded documents."
- If the user asks for a summary of each document, provide a clear separate summary paragraph for EACH document shown in the context, labeled by document name.
- Never make up information that is not in the context.
- Do not engage in small talk or general conversation."""

SYNTHESIS_PROMPT = """Document context from the knowledge graph:
{context}

User question: {question}

Answer:"""

NO_DOCUMENTS_PROMPT = """The user asked: {question}

No documents have been uploaded yet. Respond by letting them know they need to upload PDF documents first before you can answer questions."""


class AnswerSynthesizer:
    def __init__(self, groq_key: str = "", openrouter_key: str = "", opencode_key: str = ""):
        self.client = FallbackLLMClient(groq_api_key=groq_key, openrouter_api_key=openrouter_key, opencode_api_key=opencode_key)

    def synthesize(self, question: str, context: Dict[str, List], has_documents: bool = True) -> Dict:
        context_str = self._format_context(context)

        if not has_documents:
            prompt = NO_DOCUMENTS_PROMPT.format(question=question)
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
            sources = [] if not has_documents else self._extract_sources(context)

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
