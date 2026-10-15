"""Evidence-gated transcript windows and final clip selection."""
import logging
import re
from collections import defaultdict

import config
from modules.scoring import NOISE, expand_synonyms
from modules.semantic_embedder import similarity_scores
from modules.text_match import STOPWORDS, phrase_hit, stem, tokens

logger = logging.getLogger(__name__)

_TERM_RE = re.compile(r"[a-z0-9']+")


def _query_profile(query: str) -> tuple[set[str], set[str]]:
    """Return (entity stems, expanded stems incl. synonyms) derived from the query."""
    raw_terms = {word for word in _TERM_RE.findall(query.lower()) if len(word) > 1 and word not in STOPWORDS}
    content_terms = raw_terms - NOISE
    entities = {stem(term) for term in content_terms}
    expanded: set[str] = set()
    for term in content_terms:
        expanded |= {stem(word) for word in expand_synonyms(term)}
    return entities, expanded


def _build_windows(segments: list[dict], duration: float, clip_duration: int) -> dict[int, list[str]]:
    """Map every possible clip start (stride) to the transcript text inside it."""
    stride = config.WINDOW_STRIDE_SECONDS
    windows: dict[int, list[str]] = defaultdict(list)
    for segment in segments:
        seg_start = float(segment["start"])
        seg_end = float(segment.get("end", seg_start + 2.0))
        first = max(0, int((seg_end - clip_duration) // stride) * stride)
        last = min(int(duration), int(seg_start))
        for start in range(first, last + 1, stride):
            windows[start].append(segment["text"])
    return windows


def _lexical_features(evidence: str, entities: set[str], expanded: set[str], query: str) -> dict:
    observed = {stem(word) for word in tokens(evidence)}
    entity_hits = len(entities & observed)
    expanded_hits = len(expanded & observed)
    coverage = expanded_hits / len(expanded) if expanded else 0.0
    entity_coverage = entity_hits / len(entities) if entities else 0.0
    exact = phrase_hit(query, evidence)
    lexical_score = 0.6 * coverage + 0.25 * entity_coverage + 0.15 * exact
    return {
        "entity_hits": entity_hits,
        "entity_coverage": entity_coverage,
        "expanded_coverage": coverage,
        "exact_phrase": exact,
        "lexical_score": lexical_score,
    }


def _passes_lexical(features: dict, relaxed: bool) -> bool:
    if features["exact_phrase"]:
        return True
    if relaxed:
        return features["entity_hits"] >= 1 or features["expanded_coverage"] >= 0.12
    return features["expanded_coverage"] >= 0.2 and features["entity_hits"] >= 1


def transcript_candidates(transcript: list[dict], query: str, duration: float,
                          clip_duration: int, relaxed: bool = False) -> list[dict]:
    """Slide over the full transcript and propose clip windows ranked by context match."""
    if not transcript:
        return []
    entities, expanded = _query_profile(query)
    if not entities:
        logger.info("Query produced no content terms; cannot propose moments")
        return []

    windows = _build_windows(transcript, duration, clip_duration)
    lexical_candidates = []
    for start, pieces in windows.items():
        evidence = " ".join(dict.fromkeys(pieces))[:1200]
        if len(tokens(evidence)) < 4:
            continue
        features = _lexical_features(evidence, entities, expanded, query)
        if not _passes_lexical(features, relaxed):
            continue
        lexical_candidates.append((start, evidence, features))

    if not lexical_candidates:
        if not relaxed:
            return transcript_candidates(transcript, query, duration, clip_duration, relaxed=True)
        return []

    # Semantic scoring on the survivors only (batch, cheap).
    scores = similarity_scores(query, [evidence for _, evidence, _ in lexical_candidates])
    candidates = []
    threshold = config.SEMANTIC_MIN_SCORE * (0.6 if relaxed else 1.0)
    for (start, evidence, features), semantic in zip(lexical_candidates, scores or [0.0] * len(lexical_candidates)):
        if relaxed and semantic < threshold * 0.5:
            continue
        proposal_score = 0.6 * float(semantic) + 0.4 * features["lexical_score"]
        candidates.append({
            "start": float(start),
            "duration": float(clip_duration),
            "evidence": evidence,
            "semantic_score": float(semantic),
            "lexical_score": features["lexical_score"],
            "entity_hits": features["entity_hits"],
            "exact_phrase": features["exact_phrase"],
            "proposal_score": proposal_score,
            "match_tier": "relaxed" if relaxed else "strict",
        })

    candidates.sort(key=lambda item: item["proposal_score"], reverse=True)
    return _dedupe_candidates(candidates)


def _dedupe_candidates(candidates: list[dict]) -> list[dict]:
    """Keep the best, well-separated windows (candidates arrive score-descending)."""
    kept: list[dict] = []
    for candidate in candidates:
        if all(abs(candidate["start"] - item["start"]) >= config.MIN_CLIP_GAP_SECONDS for item in kept):
            kept.append(candidate)
        if len(kept) >= config.MAX_CANDIDATES_PER_VIDEO:
            break
    return kept


def choose_diverse(candidates: list[dict], count: int, min_gap: int,
                   max_per_video: int, max_per_source: int | None = None) -> list[dict]:
    """Pick top clips: round-robin across source lanes, capped per source video.

    Candidates arrive ranked per lane; the round-robin guarantees every lane
    that produced verified candidates gets pool slots instead of letting one
    lane (e.g. a flood of YouTube results) monopolize the export backfill pool.
    """
    by_source: dict[str, list[dict]] = {}
    for candidate in candidates:
        by_source.setdefault(candidate.get("source", "other"), []).append(candidate)
    order = list(by_source)
    chosen: list[dict] = []
    for _ in range(count):
        added = False
        for source in order:
            lane = by_source[source]
            if not lane:
                continue
            if max_per_source is not None and sum(
                    1 for item in chosen if item.get("source") == source) >= max_per_source:
                continue
            candidate = lane.pop(0)
            video_id = candidate.get("video_id")
            same_source = [item for item in chosen if item.get("video_id") == video_id]
            if len(same_source) >= max_per_video:
                continue
            if same_source and min(abs(candidate["start"] - item["start"])
                                   for item in same_source) < min_gap:
                continue
            chosen.append(candidate)
            added = True
            if len(chosen) >= count:
                return chosen
        if not added:
            break
    return chosen
