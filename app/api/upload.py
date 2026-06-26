import os
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from typing import List
from app.config import settings
from app.storage.database import Database
from app.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/api")


def get_db():
    return Database(settings.database_path)


def process_pdf_background(pdf_path: str, groq_key: str):
    db = Database(settings.database_path)
    try:
        pipeline = IngestionPipeline(db, groq_key)
        pipeline.ingest(pdf_path)
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

        if background_tasks:
            background_tasks.add_task(process_pdf_background, save_path, settings.groq_api_key)
            results.append({"filename": file.filename, "status": "processing"})
        else:
            try:
                db = get_db()
                pipeline = IngestionPipeline(db, settings.groq_api_key)
                doc_id = pipeline.ingest(save_path)
                results.append({"filename": file.filename, "status": "ready", "document_id": doc_id})
                db.close()
            except Exception as e:
                results.append({"filename": file.filename, "status": "error", "message": str(e)})

    return {"documents": results}


@router.get("/documents")
async def list_documents():
    db = get_db()
    try:
        docs = db.list_documents()
        return {"documents": [
            {"id": d[0], "filename": d[1], "page_count": d[2], "upload_date": d[3], "status": d[4]}
            for d in docs
        ]}
    finally:
        db.close()


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    db = get_db()
    try:
        db.delete_document(doc_id)
        return {"status": "deleted"}
    finally:
        db.close()
