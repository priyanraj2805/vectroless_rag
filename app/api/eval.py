import json
import re
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from app.config import settings
from app.storage.database import Database
from app.query.hierarchical_retriever import HierarchicalRetriever
from app.query.synthesizer import AnswerSynthesizer
from app.llm_client import FallbackLLMClient
from app.opik_tracer import log_feedback_scores

router = APIRouter(prefix="/api")


class EvalRequest(BaseModel):
    questions: List[str]


class ScoreRequest(BaseModel):
    question: str
    answer: str
    context_texts: List[str] = []
    trace_id: Optional[str] = None


class LLMScorer:
    def __init__(self):
        # Use 70b model as judge — larger model avoids self-rating bias vs the answer model
        self.client = FallbackLLMClient(
            groq_api_key=settings.groq_api_key,
            groq_base_url=settings.groq_base_url,
            groq_model="llama-3.3-70b-versatile",
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
            ollama_api_key=settings.ollama_api_key,
        )

    def _ask(self, system: str, user: str, max_tokens: int = 200) -> str:
        try:
            resp = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[LLMScorer] Error: {e}")
            return ""

    def _parse_score(self, text: str):
        try:
            data = json.loads(text.strip())
            val = float(data.get("score", data.get("value", data.get("rating", -1))))
            if 0 <= val <= 100:
                return round(val / 100.0, 4) if val > 1.0 else round(val, 4)
        except Exception:
            pass
        m = re.search(r'(?:score|rating|value)["\s:]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if not m:
            m = re.search(r'(\d+(?:\.\d+)?)\s*/\s*100', text)
        if m:
            val = float(m.group(1))
            if val <= 100:
                return round(val / 100.0, 4) if val > 1.0 else round(val, 4)
        nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        if nums:
            val = float(nums[-1])
            if val <= 1.0:
                return round(val, 4)
            if val <= 100:
                return round(val / 100.0, 4)
        return None

    def _build_judge_context(self, context_texts: list, max_chars: int = 12000) -> str:
        parts = []
        total = 0
        for text in context_texts:
            if total >= max_chars:
                break
            parts.append(text)
            total += len(text)
        return "\n\n".join(parts)

    def score(self, question: str, answer: str, context_texts: list) -> dict:
        ctx = self._build_judge_context(context_texts, max_chars=3000)
        try:
            resp = self._ask(
                'You are an evaluation judge. Reply ONLY with a valid JSON object, no explanation or extra text.',
                f'Evaluate this RAG answer on three metrics. All scores are integers 0-100.\n\n'
                f'- hallucination: 0 = fully supported by context, 100 = completely made up\n'
                f'- answer_relevance: 0 = irrelevant to question, 100 = perfectly answers it\n'
                f'- context_precision: 0 = context irrelevant to question, 100 = perfectly on-topic\n\n'
                f'Context:\n{ctx}\n\nQuestion: {question}\nAnswer: {answer}\n\n'
                f'Respond with exactly: {{"hallucination": <int>, "answer_relevance": <int>, "context_precision": <int>}}',
                max_tokens=80,
            )
            data = json.loads(resp.strip())
            def norm(v):
                if v is None: return None
                v = float(v)
                return round(v / 100.0 if v > 1.0 else v, 4)
            return {
                "hallucination":     norm(data.get("hallucination")),
                "answer_relevance":  norm(data.get("answer_relevance")),
                "context_precision": norm(data.get("context_precision")),
            }
        except Exception as e:
            print(f"[LLMScorer] score error: {e}")
            return {"hallucination": None, "answer_relevance": None, "context_precision": None}


def _build_doc_groups(db: Database, chunks: list, document_ids: Optional[List[int]]) -> dict:
    if not chunks or not document_ids:
        return {}

    doc_names = {}
    for doc_id in document_ids:
        row = db.execute("SELECT id, filename FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row:
            doc_names[doc_id] = row[1]

    chunk_ids = [c[0] for c in chunks]
    if not chunk_ids:
        return {}

    placeholders = ",".join("?" * len(chunk_ids))
    rows = db.execute(
        f"SELECT id, document_id FROM chunks WHERE id IN ({placeholders})",
        tuple(chunk_ids),
    ).fetchall()
    chunk_to_doc = {row[0]: row[1] for row in rows}

    groups: dict = {}
    for chunk in chunks:
        chunk_id = chunk[0]
        doc_id = chunk_to_doc.get(chunk_id)
        if doc_id is None:
            continue
        filename = doc_names.get(doc_id, f"Document {doc_id}")
        label = f"Document {doc_id} — {filename}"
        groups.setdefault(label, []).append(chunk)

    return groups


def _query_rag(question: str, document_ids: list = None) -> dict:
    db = Database(settings.database_path)
    try:
        stats = db.get_graph_stats()
        if stats["documents"] == 0 or stats["chunks"] == 0:
            return {"answer": "No documents uploaded.", "sources": [], "chunks_used": 0, "context_chunks": []}

        retriever = HierarchicalRetriever(db, settings=settings)
        context = retriever.retrieve(question, document_ids=document_ids)
        context["doc_groups"] = _build_doc_groups(db, context["chunks"], document_ids)

        synthesizer = AnswerSynthesizer(
            opencode_api_key=settings.opencode_api_key,
            opencode_base_url=settings.opencode_base_url,
            opencode_model=settings.opencode_model,
        )
        result = synthesizer.synthesize(question, context, has_documents=True)

        result["context_chunks"] = [
            {"content": c[1] if len(c) > 1 else "", "page": c[2] if len(c) > 2 else None, "section": c[3] if len(c) > 3 else None}
            for c in context.get("chunks", [])
        ]
        return result
    finally:
        db.close()


@router.post("/eval/score")
def score_single(request: ScoreRequest):
    """Score a single already-generated answer — no RAG re-run."""
    if not request.context_texts or not request.answer:
        return {"scores": {"hallucination": None, "answer_relevance": None, "context_precision": None}, "skipped": True}
    scorer = LLMScorer()
    scores = scorer.score(request.question, request.answer, request.context_texts)
    log_feedback_scores(request.trace_id, scores)
    return {"scores": scores}


@router.post("/eval/run")
def run_evaluation(request: EvalRequest):
    if not request.questions:
        return {"error": "No questions provided.", "results": [], "averages": {}}

    scorer = LLMScorer()
    results = []
    totals: dict = {"hallucination": [], "answer_relevance": [], "context_precision": []}

    for i, question in enumerate(request.questions):
        print(f"[eval] {i+1}/{len(request.questions)}: {question}")
        response = _query_rag(question)
        answer = response.get("answer", "")
        context_texts = [c["content"] for c in response.get("context_chunks", []) if c.get("content")]
        scores = scorer.score(question, answer, context_texts) if context_texts else {}

        try:
            log_feedback_scores(response.get("trace_id"), scores)
        except Exception:
            pass

        for metric in totals:
            if scores.get(metric) is not None:
                totals[metric].append(scores[metric])

        results.append({
            "question": question,
            "answer": answer,
            "scores": scores,
            "sources": response.get("sources", []),
        })

    averages = {
        metric: round(sum(vals) / len(vals), 4) if vals else None
        for metric, vals in totals.items()
    }
    return {"results": results, "averages": averages, "total": len(results)}
