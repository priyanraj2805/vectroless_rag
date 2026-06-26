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


@router.post("/chat")
async def chat(request: ChatRequest):
    db = Database(settings.database_path)
    try:
        planner = QueryPlanner(api_key=settings.groq_api_key)
        executor = QueryExecutor(db)
        synthesizer = AnswerSynthesizer(api_key=settings.groq_api_key)

        plan = planner.plan(request.question)
        context = executor.execute(plan)
        result = synthesizer.synthesize(request.question, context)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "entities_found": result["entity_count"],
            "chunks_used": result["chunk_count"],
            "plan": plan,
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
