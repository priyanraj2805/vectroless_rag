from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Vectorless RAG", version="0.1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {"message": "Vectorless RAG API"}

@app.get("/health")
async def health():
    return {"status": "ok"}