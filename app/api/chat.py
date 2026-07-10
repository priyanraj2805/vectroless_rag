from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel
from app.config import settings
from app.storage.database import Database
from app.query.hierarchical_retriever import HierarchicalRetriever
from app.query.synthesizer import AnswerSynthesizer
from app.cache import get_redis, get_version, make_versioned_key, cache_get, cache_set

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

        # Versioned answer key — includes doc selection so different doc combos get different caches
        answer_key = make_versioned_key("answer", version, request.question, doc_key)
        cached = cache_get(r, answer_key)
        if cached:
            print(f"[cache] HIT v{version} — {request.question[:50]}")
            return cached

        stats = db.get_graph_stats()
        has_documents = stats["documents"] > 0 and stats["chunks"] > 0

        synthesizer = AnswerSynthesizer(
            opencode_api_key=settings.opencode_api_key,
            opencode_base_url=settings.opencode_base_url,
            opencode_model=settings.opencode_model,
            groq_api_key=settings.groq_api_key,
            groq_base_url=settings.groq_base_url,
            groq_model=settings.groq_model,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
            ollama_api_key=settings.ollama_api_key,
        )
        is_conversational = synthesizer._is_conversational(request.question)

        if has_documents and not is_conversational and not effective_ids:
            return {
                "answer": "Please select at least one PDF from the sidebar before asking a question.",
                "sources": [],
                "entities_found": 0,
                "chunks_used": 0,
                "needs_selection": True,
            }

        if not has_documents or is_conversational:
            context = {"nodes": [], "chunks": [], "doc_groups": {}}
            result = synthesizer.synthesize(request.question, context, has_documents=has_documents and not is_conversational)
        else:
            retriever = HierarchicalRetriever(db, settings=settings)
            context = retriever.retrieve(request.question, document_ids=effective_ids)

            # If BM25 returned nothing (e.g. summary/overview queries whose keywords
            # don't appear literally in the text), fall back to the first N chunks so
            # the LLM still has document content to work with.
            if not context.get("chunks") and effective_ids:
                placeholders = ",".join("?" * len(effective_ids))
                fallback = db.execute(
                    f"SELECT id, content, page_number, section_title, -1.0, document_id, chunk_index "
                    f"FROM chunks WHERE document_id IN ({placeholders}) "
                    f"ORDER BY document_id, chunk_index LIMIT 12",
                    tuple(effective_ids),
                ).fetchall()
                context["chunks"] = fallback

            # Build per-document groups for the synthesizer to present clear per-doc summaries
            context["doc_groups"] = _build_doc_groups(db, context["chunks"], effective_ids)

            result = synthesizer.synthesize(request.question, context, has_documents=True)

        # Sort by BM25 rank (index 4) so judge sees most relevant chunks first.
        # FTS5 BM25 scores are negative — more negative = more relevant.
        # Neighbor-expanded chunks have rank=0 and naturally fall to the end.
        _raw_chunks = [c for c in context.get("chunks", []) if len(c) > 1 and isinstance(c[1], str) and c[1].strip()]
        _raw_chunks.sort(key=lambda c: c[4] if len(c) > 4 and c[4] is not None else 0)
        context_texts = [c[1] for c in _raw_chunks]

        response = {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "entities_found": result.get("entity_count", 0),
            "chunks_used": result.get("chunk_count", 0),
            "trace_id": result.get("trace_id"),
            "context_texts": context_texts,
        }

        cache_set(r, answer_key, response, ttl=3600)
        return response
    finally:
        db.close()


def _build_doc_groups(db: Database, chunks: list, document_ids: Optional[List[int]]) -> dict:
    """
    Groups chunks by their source document. Returns:
    { "Document 1 — filename.pdf": [chunk, chunk, ...], ... }
    Used by the synthesizer to format multi-doc context with clear per-doc labels.
    """
    if not chunks or not document_ids:
        return {}

    # Fetch filename for each document_id
    doc_names = {}
    for doc_id in document_ids:
        row = db.get_document(doc_id)
        if row:
            doc_names[doc_id] = row[1]  # filename column

    # Look up document_id for each chunk_id
    chunk_ids = [c[0] for c in chunks]
    if not chunk_ids:
        return {}

    placeholders = ",".join("?" * len(chunk_ids))
    rows = db.execute(
        f"SELECT id, document_id FROM chunks WHERE id IN ({placeholders})",
        tuple(chunk_ids),
    ).fetchall()
    chunk_to_doc = {row[0]: row[1] for row in rows}

    # Group chunks by document
    groups: dict = {}
    for chunk in chunks:
        chunk_id = chunk[0]
        doc_id = chunk_to_doc.get(chunk_id)
        if doc_id is None:
            continue
        filename = doc_names.get(doc_id, f"Document {doc_id}")
        label = f"Document {doc_id} — {filename}"
        if label not in groups:
            groups[label] = []
        groups[label].append(chunk)

    return groups


@router.get("/graph/stats")
async def graph_stats():
    db = Database(settings.database_path)
    try:
        stats = db.get_graph_stats()
        return stats
    finally:
        db.close()
