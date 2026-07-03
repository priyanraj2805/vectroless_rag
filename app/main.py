from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router


class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


app = FastAPI(title="Vectorless RAG", version="0.1.0")
app.add_middleware(NoCacheMiddleware)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"answer": f"Server error: {str(exc)}", "sources": [], "entities_found": 0, "chunks_used": 0},
    )

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(upload_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/readme")
async def readme():
    return FileResponse("README.md")


@app.get("/health")
async def health():
    return {"status": "ok"}
