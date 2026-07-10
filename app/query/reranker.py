"""Cross-encoder reranking wrapper.

Lazily loads cross-encoder/ms-marco-MiniLM-L-6-v2 via sentence-transformers.
If the library or model isn't available, rerank() degrades gracefully to
returning the input order (callers pass items pre-sorted by BM25/doc score),
mirroring the guarded-import pattern already used in app/opik_tracer.py.
"""
import logging
from typing import Callable, List, TypeVar

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

T = TypeVar("T")

_model = None
_load_attempted = False


def _get_model():
    global _model, _load_attempted
    if _load_attempted:
        return _model
    _load_attempted = True
    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(MODEL_NAME)
    except Exception as e:
        logger.warning(f"[reranker] Cross-encoder unavailable ({e}); falling back to BM25 ordering.")
        _model = None
    return _model


def is_available() -> bool:
    return _get_model() is not None


def rerank(query: str, items: List[T], content_fn: Callable[[T], str], top_n: int) -> List[T]:
    """Score items against the query and return the top_n, best-first.

    Falls back to items[:top_n] (preserving caller-supplied order) if the
    cross-encoder model can't be loaded.
    """
    if not items:
        return []

    model = _get_model()
    if model is None:
        return items[:top_n]

    pairs = [(query, content_fn(item)) for item in items]
    try:
        scores = model.predict(pairs)
    except Exception as e:
        logger.warning(f"[reranker] Scoring failed ({e}); falling back to BM25 ordering.")
        return items[:top_n]

    scored = sorted(zip(items, scores), key=lambda pair: pair[1], reverse=True)
    return [item for item, _ in scored[:top_n]]
