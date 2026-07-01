from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel
from app.config import settings
from app.storage.database import Database
from app.query.planner import QueryPlanner
from app.query.executor import QueryExecutor
from app.query.synthesizer import AnswerSynthesizer

router = APIRouter(prefix="/api")


class ChatRequest(BaseModel):
    question: str
    document_ids: Optional[List[int]] = None


@router.post("/chat")
async def chat(request: ChatRequest):
    db = Database(settings.database_path)
    try:
        if request.document_ids:
            valid_ids = []
            for doc_id in request.document_ids:
                doc = db.get_document(doc_id)
                if doc and doc[4] == "ready":
                    valid_ids.append(doc_id)
            if not valid_ids:
                return {
                    "answer": "Selected documents are not available or still processing.",
                    "sources": [],
                    "entities_found": 0,
                    "chunks_used": 0,
                }
            effective_ids = valid_ids
        else:
            effective_ids = None

        stats = db.get_graph_stats()
        has_documents = stats["documents"] > 0 and stats["chunks"] > 0

        synthesizer = AnswerSynthesizer(groq_key=settings.groq_api_key, openrouter_key=settings.openrouter_api_key)

        if not has_documents or synthesizer._is_conversational(request.question):
            context = {"nodes": [], "chunks": []}
            result = synthesizer.synthesize(request.question, context, has_documents=False)
        else:
            planner = QueryPlanner(groq_key=settings.groq_api_key, openrouter_key=settings.openrouter_api_key)
            executor = QueryExecutor(db)

            plan = planner.plan(request.question)
            context = executor.execute(plan, document_ids=effective_ids)
            result = synthesizer.synthesize(request.question, context, has_documents=True)

        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "entities_found": result.get("entity_count", 0),
            "chunks_used": result.get("chunk_count", 0),
        }
    finally:
        db.close()


@router.get("/graph/stats")
async def graph_stats():
    db = Database(settings.database_path)
    try:
        stats = db.get_graph_stats()
        return stats
    finally:
        db.close()
