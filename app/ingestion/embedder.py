import numpy as np
from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("[embedder] Loading all-MiniLM-L6-v2 model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[embedder] Model loaded")
    return _model


class Embedder:
    def __init__(self):
        self.model = get_model()

    def embed(self, text: str) -> bytes:
        """Convert text to a 384-float vector, returned as raw bytes for SQLite storage."""
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.astype(np.float32).tobytes()

    def embed_batch(self, texts: list) -> list:
        """Encode all texts in a single forward pass — much faster than one-by-one."""
        vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return [v.astype(np.float32).tobytes() for v in vectors]

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query string and return as numpy array for cosine similarity."""
        return self.model.encode(text, normalize_embeddings=True).astype(np.float32)

    @staticmethod
    def from_bytes(blob: bytes) -> np.ndarray:
        """Deserialize a stored vector back to numpy array."""
        return np.frombuffer(blob, dtype=np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two normalized vectors (dot product suffices)."""
        return float(np.dot(a, b))
