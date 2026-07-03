from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel
from app.config import settings
from app.storage.database import Database
from app.query.planner import QueryPlanner
from app.query.executor import QueryExecutor
from app.query.synthesizer import AnswerSynthesizer
from app.cache import get_redis, get_version, make_versioned_key, make_key, cache_get, cache_set

router = APIRouter(prefix="/api")


class ChatRequest(BaseModel):
    question: str
    document_ids: Optional[List[int]] = None


@router.post("/chat")
async def chat(request: ChatRequest):
    db = Database(settings.database_path)
    r = get_redis(settings.redis_url)

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

        # Get current version — changes whenever docs are added/deleted
        version = get_version(r)
        doc_key = str(sorted(effective_ids)) if effective_ids else "all"

        # Versioned answer key — stale if version changed
        answer_key = make_versioned_key("answer", version, request.question, doc_key)
        cached = cache_get(r, answer_key)
        if cached:
            print(f"[cache] HIT v{version} — {request.question[:50]}")
            return cached

        stats = db.get_graph_stats()
        has_documents = stats["documents"] > 0 and stats["chunks"] > 0

        synthesizer = AnswerSynthesizer(groq_key=settings.groq_api_key, openrouter_key=settings.openrouter_api_key)

        if not has_documents or synthesizer._is_conversational(request.question):
            context = {"nodes": [], "chunks": []}
            result = synthesizer.synthesize(request.question, context, has_documents=False)
        else:
            planner = QueryPlanner(groq_key=settings.groq_api_key, openrouter_key=settings.openrouter_api_key)
            executor = QueryExecutor(db)

            # Versioned plan key — stale if version changed
            plan_key = make_versioned_key("plan", version, request.question)
            plan = cache_get(r, plan_key)
            if plan:
                print(f"[cache] Plan HIT v{version} — {request.question[:50]}")
            else:
                plan = planner.plan(request.question)
                cache_set(r, plan_key, plan, ttl=3600)

            context = executor.execute(plan, document_ids=effective_ids)
            result = synthesizer.synthesize(request.question, context, has_documents=True)

        response = {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "entities_found": result.get("entity_count", 0),
            "chunks_used": result.get("chunk_count", 0),
        }

        cache_set(r, answer_key, response, ttl=3600)
        return response
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
