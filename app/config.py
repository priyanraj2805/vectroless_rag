from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    opencode_api_key: str = ""
    redis_url: str = ""
    database_path: str = "./data/graph.db"
    upload_dir: str = "./uploads"
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()
Path(settings.database_path).parent.mkdir(parents=True, exist_ok=True)
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)