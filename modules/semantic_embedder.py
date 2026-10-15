"""Shared sentence-transformer embedding wrapper for transcript-window scoring."""
import logging

import numpy as np

import config

logger = logging.getLogger(__name__)
_model = None
_load_attempted = False


def _load() -> bool:
    global _model, _load_attempted
    if _load_attempted:
        return _model is not None
    _load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model %s loaded", config.EMBEDDING_MODEL)
        return True
    except Exception as exc:
        logger.warning("Embedding model unavailable; using lexical fallback: %s", exc)
        return False


def embed(texts: list[str]):
    if not texts or not _load():
        return None
    try:
        return _model.encode(texts, normalize_embeddings=True)
    except Exception as exc:
        logger.warning("Embedding failed: %s", exc)
        return None


def similarity_scores(query: str, texts: list[str]):
    """Cosine similarity in [0, 1] between the query and each text."""
    query_emb = embed([query])
    if query_emb is None:
        return None
    text_emb = embed(texts)
    if text_emb is None:
        return None
    q = np.asarray(query_emb[0], dtype=np.float64)
    matrix = np.asarray(text_emb, dtype=np.float64)
    dots = matrix @ q
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(norms > 0, dots / np.where(norms > 0, norms, 1.0), 0.0)
    return np.maximum(0.0, cos).tolist()
