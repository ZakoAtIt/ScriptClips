"""Local CrossEncoder reranking and final score fusion."""
import logging
import math

from modules.scoring import combine_score
from modules.text_match import score as lexical_score

logger = logging.getLogger(__name__)
_reranker = None
_attempted = False


def _load() -> bool:
    global _reranker, _attempted
    if _attempted:
        return _reranker is not None
    _attempted = True
    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("CrossEncoder loaded")
        return True
    except Exception as exc:
        logger.info("Cross-encoder not available: %s", exc)
        return False


def _sigmoid(score: float) -> float:
    value = max(-15.0, min(15.0, float(score)))
    return 1.0 / (1.0 + math.exp(-value))


def _reranker_predict(pairs) -> list[float]:
    """Raw sigmoid relevance of (query, text) pairs, or [] if unavailable."""
    if not pairs:
        return []
    if _load():
        try:
            return [_sigmoid(value) for value in _reranker.predict(list(pairs))]
        except Exception as exc:
            logger.warning("CrossEncoder scoring failed: %s", exc)
    return [lexical_score(q, t) for q, t in pairs]


def rerank_with_model(candidates: list[dict], transcript_lookup, query: str) -> list[dict]:
    """Attach rerank_score and combined_score, then sort candidates descending."""
    if not candidates:
        return []

    inputs = []
    for c in candidates:
        snippet = c.get("snippet") or ""
        if not snippet and callable(transcript_lookup):
            try:
                snippet = transcript_lookup(c.get("timestamp") or c.get("start"))
            except Exception:
                snippet = ""
        title = c.get("title", "") or ""
        description = c.get("description", "") or ""
        metadata_context = " ".join(part for part in (title, description) if part).strip()
        text = snippet
        if metadata_context:
            text = f"Title: {title}\nDescription: {description}\nTranscript: {snippet}"
        inputs.append((query, text))

    scores: list[float] = []
    if _load():
        try:
            scores = [_sigmoid(value) for value in _reranker.predict(inputs)]
        except Exception as exc:
            logger.warning("CrossEncoder scoring failed: %s", exc)
    if not scores:
        scores = [lexical_score(query, text) for _, text in inputs]

    for c, rerank_score in zip(candidates, scores):
        c["rerank_score"] = float(rerank_score)
        c["combined_score"] = combine_score(c)

    candidates.sort(key=lambda item: item.get("combined_score", 0.0), reverse=True)
    return candidates
