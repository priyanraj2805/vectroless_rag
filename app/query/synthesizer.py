from app.llm_client import AnswerLLMClient
from app.opik_tracer import track, get_current_trace_id
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
    def __init__(self, opencode_api_key: str = "", opencode_base_url: str = "", opencode_model: str = "",
                 groq_api_key: str = "", groq_base_url: str = "", groq_model: str = "",
                 ollama_base_url: str = "", ollama_model: str = "", ollama_api_key: str = ""):
        self.client = AnswerLLMClient(
            opencode_api_key=opencode_api_key, opencode_base_url=opencode_base_url, opencode_model=opencode_model,
            groq_api_key=groq_api_key, groq_base_url=groq_base_url, groq_model=groq_model,
            ollama_base_url=ollama_base_url, ollama_model=ollama_model, ollama_api_key=ollama_api_key,
        )

    @track
    def synthesize(self, question: str, context: Dict[str, List], has_documents: bool = True) -> Dict:
        trace_id = get_current_trace_id()
        context_str = self._format_context(context)

        # If the user pasted a large block of text (equations, page refs, etc.) along with
        # their question, truncate to the first 600 chars so the LLM sees only the question.
        # BM25 retrieval already used the full text for keyword matching upstream.
        llm_question = question[:600].strip() if len(question) > 600 else question

        if not has_documents:
            prompt = NO_DOCUMENTS_PROMPT.format(question=llm_question)
        else:
            prompt = SYNTHESIS_PROMPT.format(context=context_str, question=llm_question)

        # Scale max_tokens by doc count and question length (multi-question queries need more room)
        num_docs = len(context.get("doc_groups", {})) or 1
        question_lines = len([l for l in question.strip().splitlines() if l.strip()])
        multi_q_boost = min(question_lines, 6) * 200  # 200 extra tokens per question line, cap at 6
        max_tokens = min(4000, max(1500, num_docs * 600 + multi_q_boost))

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
            sources = [] if not has_documents else self._extract_sources(context, context.get("doc_groups"))

            return {
                "answer": answer_text,
                "sources": sources,
                "entity_count": len(context.get("nodes", [])),
                "chunk_count": len(context.get("chunks", [])),
                "trace_id": trace_id,
            }

        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "entity_count": 0,
                "chunk_count": 0,
                "trace_id": trace_id,
            }

    def _is_conversational(self, question: str) -> bool:
        GREETINGS = {
            'hi', 'hii', 'hiii', 'hello', 'hey', 'howdy', 'hiya', 'yo', 'sup',
            'good morning', 'good afternoon', 'good evening', 'good night',
            'how are you', 'how r u', 'whats up', "what's up",
            'thanks', 'thank you', 'ty', 'thx', 'bye', 'goodbye', 'ok', 'okay',
            'cool', 'nice', 'great', 'awesome', 'got it', 'sure', 'alright',
        }
        ABOUT_YOU = (
            'what do you do', 'what can you do', 'what is your work',
            'what are you', 'who are you',
            'tell me about you', 'tell me about yourself',
            'about you', 'about yourself',
            'what are your capabilities', 'what can you help with',
            'what is your purpose', 'how do you work', 'what do you help with',
            'what is your job', 'what are you for', 'what do you help me with',
            'describe yourself', 'introduce yourself',
        )
        q = question.lower().strip().rstrip('!?.,')
        if q in GREETINGS:
            return True
        return any(q == s or q.startswith(s) for s in ABOUT_YOU)

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

    def _extract_sources(self, context: Dict[str, List], doc_groups: Dict[str, List] = None) -> List[Dict]:
        """Extract sources with document names from context, filtered to high-relevance docs only."""
        sources = []
        seen = set()

        # Build chunk_id -> document label mapping from doc_groups
        chunk_to_doc = {}
        if doc_groups:
            for doc_label, chunks in doc_groups.items():
                for chunk in chunks:
                    chunk_id = chunk[0] if len(chunk) > 0 else None
                    if chunk_id:
                        chunk_to_doc[chunk_id] = doc_label

        # Build chunk_id -> doc_id mapping for score filtering
        chunk_to_doc_id = {}
        for chunk in context.get("chunks", []):
            chunk_id = chunk[0] if len(chunk) > 0 else None
            doc_id = chunk[5] if len(chunk) > 5 else None
            if chunk_id and doc_id:
                chunk_to_doc_id[chunk_id] = doc_id

        # Only cite documents that scored >= 65% of the top-scoring document
        doc_scores = context.get("doc_scores", {})
        if doc_scores:
            best = max(doc_scores.values())
            citation_threshold = best * 0.65
            allowed_doc_ids = {doc_id for doc_id, score in doc_scores.items() if score >= citation_threshold}
        else:
            allowed_doc_ids = None

        # Only cite directly-matched chunks (rank != 0).
        # Neighbor-expanded chunks have rank=0 and are included for LLM context
        # but should not appear as citations — they weren't directly relevant.
        direct_chunks = [c for c in context.get("chunks", []) if len(c) > 4 and c[4] != 0]
        # Fall back to all chunks if nothing was directly matched (e.g. all neighbors)
        cite_chunks = direct_chunks if direct_chunks else context.get("chunks", [])

        # Sort by BM25 rank — most negative = most relevant first
        cite_chunks = sorted(cite_chunks, key=lambda c: c[4] if len(c) > 4 and c[4] is not None else 0)

        pages_per_doc: Dict[str, int] = {}
        MAX_PAGES_PER_DOC = 5

        for chunk in cite_chunks:
            chunk_id = chunk[0] if len(chunk) > 0 else None
            doc_id = chunk_to_doc_id.get(chunk_id)

            # Skip chunks from low-relevance documents
            if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                continue

            doc_label = chunk_to_doc.get(chunk_id, "Unknown Document")
            source = {
                "document": doc_label,
                "content_preview": (chunk[1][:200] if len(chunk) > 1 else "") if isinstance(chunk[1], str) else "",
                "page": chunk[2] if len(chunk) > 2 else None,
                "section": chunk[3] if len(chunk) > 3 else "",
            }
            key = (doc_label, source["page"], source["section"])
            if key not in seen:
                if pages_per_doc.get(doc_label, 0) >= MAX_PAGES_PER_DOC:
                    continue
                seen.add(key)
                pages_per_doc[doc_label] = pages_per_doc.get(doc_label, 0) + 1
                sources.append(source)
        return sources
