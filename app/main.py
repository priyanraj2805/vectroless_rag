from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router

app = FastAPI(title="Vectorless RAG", version="0.1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(upload_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}
