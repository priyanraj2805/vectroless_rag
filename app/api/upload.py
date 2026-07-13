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


SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls",
    ".pptx", ".ppt", ".png", ".jpg", ".jpeg",
    ".tiff", ".bmp", ".webp", ".txt", ".html", ".rtf",
}


def process_document_background(file_path: str, doc_id: int, groq_api_key: str, groq_base_url: str,
                                 groq_model: str, ollama_base_url: str, ollama_model: str,
                                 ollama_api_key: str, redis_url: str):
    db = Database(settings.database_path)
    try:
        pipeline = IngestionPipeline(
            db,
            groq_api_key=groq_api_key, groq_base_url=groq_base_url, groq_model=groq_model,
            ollama_base_url=ollama_base_url, ollama_model=ollama_model, ollama_api_key=ollama_api_key,
            redis_url=redis_url,
        )
        pipeline.ingest(file_path, doc_id=doc_id)
        from app.cache import get_redis, bump_version
        r = get_redis(redis_url)
        bump_version(r)
    except Exception as e:
        print(f"[upload] processing failed for {file_path}: {e}")
        # pipeline.ingest already marks status='error'; this is a safety net
        try:
            db.update_document_status(doc_id, "error")
        except Exception:
            pass
    finally:
        db.close()


@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    results = []
    db = get_db()

    try:
        for file in files:
            ext = os.path.splitext(file.filename or "")[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": f"Unsupported file type '{ext}'. Supported: PDF, DOCX, XLSX, PPTX, PNG, JPG, TXT and more.",
                })
                continue

            save_path = os.path.join(settings.upload_dir, file.filename)
            with open(save_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # Pre-insert as 'queued' — user sees the document in the sidebar immediately
            doc_id = db.insert_document(filename=file.filename, status='queued')

            executor.submit(
                process_document_background, save_path, doc_id,
                settings.groq_api_key, settings.groq_base_url, settings.groq_model,
                settings.ollama_base_url, settings.ollama_model, settings.ollama_api_key,
                settings.redis_url,
            )
            results.append({"filename": file.filename, "status": "queued", "doc_id": doc_id})
    finally:
        db.close()

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
