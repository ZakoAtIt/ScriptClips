"""Shared scoring helpers for the pipeline."""
import math
import re

import config
from modules.text_match import STOPWORDS, stem, tokens

# NOISE = generic filler only.  Intent/entity words (confront, guy, man, woman,
# person, people, harass*) are KEPT so they contribute to topical matching.
NOISE = STOPWORDS | {"video", "clip", "footage",
                     "what", "when", "how", "why", "did", "does", "got"}


def contains_keyword(text: str, keyword: str) -> bool:
    """Word-boundary substring test so 'live' does not match 'lives'."""
    return re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE) is not None


def metadata_rawness_score(title: str | None, description: str | None, channel: str | None) -> float:
    """Soft 0..1 score of how likely the video is raw source footage (not a recap/edit)."""
    text = " ".join(filter(None, [title or "", description or "", channel or ""])).lower()
    if not text.strip():
        return 0.5
    penalty = 0.0
    for keyword in config.EDITED_VIDEO_KEYWORDS:
        if contains_keyword(text, keyword):
            penalty += 0.30
    boost = 0.0
    for keyword in config.RAW_VIDEO_KEYWORDS:
        if contains_keyword(text, keyword):
            boost += 0.16
    score = 0.5 - min(0.45, penalty) + min(0.35, boost)
    score -= 0.5 * recap_score(title, description)
    return max(0.0, min(1.0, score))


def recap_score(title: str | None, description: str | None) -> float:
    """0..1 score of how strongly the video looks like a recap/watch-party of someone
    else's footage rather than the event itself."""
    text = " ".join(filter(None, [title or "", description or ""])).lower()
    if not text.strip():
        return 0.0
    hits = sum(1 for marker in config.TWITCH_EDITORIAL_MARKERS if contains_keyword(text, marker))
    if any(contains_keyword(text, marker) for marker in ("reacting", "watch party", "watching")):
        hits += 2
    return min(1.0, hits * 0.35)


def title_match_score(query: str, title: str | None, description: str | None) -> float:
    text = " ".join(filter(None, [title or "", description or ""])).lower()
    title_lower = (title or "").lower()
    query_lower = query.lower()
    score = 0.0
    if query_lower in title_lower:
        score += 0.6
    elif query_lower in text:
        score += 0.25

    wanted = {stem(w) for w in tokens(query)}
    observed = {stem(w) for w in tokens(text)}
    if wanted:
        score += 0.4 * len(wanted & observed) / len(wanted)

    if any(contains_keyword(text, marker) for marker in ("stream", "live", "raw", "unedited",
                                                         "full stream", "recording", "original", "vod")):
        score += 0.15
    if any(contains_keyword(title_lower, marker) for marker in (
            "highlight", "highlights", "recap", "reaction", "edited", "compilation", "montage",
            "best of", "story time", "storytime", "review", "summary", "analysis", "supercut")):
        score -= 0.35
    return max(0.0, min(1.0, score))


def evidence_editorial_score(text: str) -> float:
    """Penalize evidence that reads like a recap/narration rather than raw event audio."""
    lowered = (text or "").lower()
    penalty = 0.0
    for keyword in config.EDITORIAL_EVIDENCE_KEYWORDS:
        if contains_keyword(lowered, keyword):
            penalty += 0.16
    return max(0.0, 1.0 - min(0.6, penalty))


def visual_prompt(query: str) -> str:
    return f"{query}, event taking place, people, live"


def expand_synonyms(term: str) -> set[str]:
    extra = config.SYNONYMS.get(term, set())
    return {term, *extra}


def combine_score(candidate: dict, weights: dict | None = None) -> float:
    """Weighted sum of the feature scores already attached to a candidate."""
    weights = weights or config.SCORE_WEIGHTS
    total = 0.0
    for key, weight in weights.items():
        field = _WEIGHT_TO_FIELD.get(key, key)
        total += weight * float(candidate.get(field) or 0.0)
    return total


def passes_relevance_gate(candidate: dict, plan: dict | None = None) -> tuple[bool, str]:
    """A candidate must clear every relevance gate; returns (ok, first failure reason).

    For event sentences, entity presence alone is never sufficient — the
    candidate also needs action evidence, object evidence, or visual proof.
    For narration/context sentences, entity-only is sufficient.

    A candidate with action_hits=0, object_hits=0, and no visual verification
    (event_visual_verified=False, vlm_happening=False) CANNOT pass for event
    sentences.  High combined_score from vlm/vision weights does NOT compensate.
    """
    entity_hits = int(candidate.get("entity_hits", 0))
    action_hits = int(candidate.get("action_hits", 0))
    object_hits = int(candidate.get("object_hits", 0))
    if (bool(candidate.get("llm_found"))
            and float(candidate.get("llm_moment_score", 0.0)) >= config.LLM_MOMENT_MIN_CONFIDENCE):
        if entity_hits < 1 and action_hits < 1:
            return False, "llm-grounded moment has no audio signal"
    combined = float(candidate.get("combined_score", combine_score(candidate)))
    if combined < config.RELEVANCE_GATE_COMBINED:
        return False, f"combined {combined:.2f} < {config.RELEVANCE_GATE_COMBINED}"
    semantic = float(candidate.get("semantic_score", 0.0))
    if semantic < config.RELEVANCE_GATE_SEMANTIC:
        return False, f"semantic {semantic:.2f} < {config.RELEVANCE_GATE_SEMANTIC}"
    rerank = float(candidate.get("rerank_score", 0.0))
    if rerank < config.RELEVANCE_GATE_RERANK:
        return False, f"rerank {rerank:.3f} < {config.RELEVANCE_GATE_RERANK}"
    if entity_hits < config.RELEVANCE_GATE_ENTITY:
        return False, f"entity_hits {entity_hits} < {config.RELEVANCE_GATE_ENTITY}"
    # --- action/object gate for event sentences ---
    sentence_type = (plan or {}).get("sentence_type", "narration")
    if sentence_type == "event":
        has_action_or_object = (
            action_hits >= config.RELEVANCE_GATE_ACTION_OR_OBJECT
            or object_hits >= config.RELEVANCE_GATE_ACTION_OR_OBJECT
            or bool(candidate.get("event_visual_verified"))
            or bool(candidate.get("vlm_happening"))
        )
        if not has_action_or_object:
            return False, (f"event sentence: no action/object/visual evidence "
                           f"(action={action_hits}, object={object_hits})")
    return True, ""


_WEIGHT_TO_FIELD = {
    "semantic": "semantic_score",
    "lexical": "lexical_score",
    "rerank": "rerank_score",
    "llm": "llm_score",
    "llm_moment": "llm_moment_score",
    "vision": "vision_score",
    "rawness": "rawness_score",
    "visual_raw": "visual_raw",
    "editorial": "editorial_score",
    "title_match": "title_match_score",
    "metadata": "metadata_score",
    "anchor": "anchor_sim",
    "event_visual": "event_visual_score",
    "vlm": "vlm_score",
}
