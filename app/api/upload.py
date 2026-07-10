import os
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from typing import List
from concurrent.futures import ThreadPoolExecutor
from app.config import settings
from app.storage.database import Database
from app.ingestion.pipeline import IngestionPipeline
from app.cache import get_redis, bump_version

router = APIRouter(prefix="/api")
executor = ThreadPoolExecutor(max_workers=2)


def get_db():
    return Database(settings.database_path)


def process_pdf_background(pdf_path: str, groq_api_key: str, groq_base_url: str, groq_model: str,
                           ollama_base_url: str, ollama_model: str, ollama_api_key: str, redis_url: str):
    db = Database(settings.database_path)
    try:
        pipeline = IngestionPipeline(
            db,
            groq_api_key=groq_api_key, groq_base_url=groq_base_url, groq_model=groq_model,
            ollama_base_url=ollama_base_url, ollama_model=ollama_model, ollama_api_key=ollama_api_key,
            redis_url=redis_url,
        )
        pipeline.ingest(pdf_path)
        # Bump version after ingestion completes — invalidates all answer/plan caches
        from app.cache import get_redis, bump_version
        r = get_redis(redis_url)
        bump_version(r)
    finally:
        db.close()


@router.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    results = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            results.append({"filename": file.filename, "status": "error", "message": "Not a PDF"})
            continue

        save_path = os.path.join(settings.upload_dir, file.filename)
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        executor.submit(
            process_pdf_background, save_path,
            settings.groq_api_key, settings.groq_base_url, settings.groq_model,
            settings.ollama_base_url, settings.ollama_model, settings.ollama_api_key,
            settings.redis_url,
        )
        results.append({"filename": file.filename, "status": "processing"})

    return {"documents": results}


@router.get("/documents")
async def list_documents():
    db = get_db()
    try:
        docs = db.list_documents()
        return {"documents": [
            {"id": d[0], "filename": d[1], "page_count": d[2], "upload_date": d[3], "status": d[4], "completed_at": d[5]}
            for d in docs
        ]}
    finally:
        db.close()


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    db = get_db()
    r = get_redis(settings.redis_url)
    try:
        db.delete_document(doc_id)
        # Bump version on delete — cached answers about this doc are now stale
        bump_version(r)
        return {"status": "deleted"}
    finally:
        db.close()
