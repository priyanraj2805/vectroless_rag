from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Answer LLMs: Groq (primary) → Ollama (fallback)
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.1-8b-instant"
    ollama_base_url: str = ""
    ollama_model: str = "gemma4:31b"
    ollama_api_key: str = ""
    # Judge LLM: OpenCode (eval scoring only)
    opencode_api_key: str = ""
    opencode_base_url: str = "https://opencode.ai/zen/v1"
    opencode_model: str = "nemotron-3-ultra-free"
    redis_url: str = ""
    database_path: str = "./data/graph.db"
    upload_dir: str = "./uploads"

    # Retrieval settings for HierarchicalRetriever (all vectorless — FTS5 + graph only)
    retrieval_top_k_candidates: int = 80
    retrieval_min_documents: int = 1
    retrieval_max_documents: int = 5
    retrieval_doc_score_threshold: float = 0.4
    retrieval_rerank_enabled: bool = False   # cross-encoder reranker; off = fully vectorless
    retrieval_rerank_top_n: int = 10
    retrieval_neighbor_window: int = 1
    retrieval_weight_bm25: float = 0.50
    retrieval_weight_heading: float = 0.20
    retrieval_weight_entity: float = 0.15
    retrieval_weight_metadata: float = 0.05
    retrieval_weight_density: float = 0.10

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

settings = Settings()
Path(settings.database_path).parent.mkdir(parents=True, exist_ok=True)
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)