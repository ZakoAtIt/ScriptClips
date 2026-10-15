"""Visual anchors: Google Images of the event -> gate for clip candidates.

The image lane proves a sentence's pixels exist (Chrome Google Images + CLIP).
Clip lanes, by contrast, only ever checked audio + metadata, so a wrong event
whose transcript merely mentions the same words could pass.  An AnchorSet gives
every lane the actual visuals of the event; a candidate survives only when a
sampled frame CLIP-matches the anchors.  Image->image cosine is far more
discriminative than the text->image scoring used for `vision_score`.  Anchors
are cached per sentence so the Chrome search runs once and later runs reuse it.
"""
import hashlib
import json
import logging
import urllib.request
from pathlib import Path

import numpy as np

import config
from modules import image_lane, vision_ranker

logger = logging.getLogger(__name__)


class AnchorSet:
    """A sentence's top unique, CLIP-relevant Google Images + their embeddings."""

    def __init__(self, entries: list[dict], embeddings: list | None, prompt: str,
                 entity_mask: list[bool] | None = None,
                 object_mask: list[bool] | None = None):
        self.entries = entries
        self.embeddings = embeddings or []
        self.prompt = prompt
        self.entity_mask = np.asarray(entity_mask) if entity_mask is not None else None
        self.object_mask = np.asarray(object_mask) if object_mask is not None else None

    @property
    def strong(self) -> bool:
        return bool(self.embeddings)

    @property
    def coherence(self) -> float | None:
        """Mean pairwise cosine among anchors; None when unmeasurable (<2).

        High means the images are photos of the same event; low means a generic
        grab-bag of a person/topic, which cannot distinguish the real event.
        """
        if len(self.embeddings) < 2:
            return None
        mat = np.asarray(self.embeddings)
        off = mat @ mat.T
        return float(np.mean(off[np.triu_indices(len(self.embeddings), 1)]))

    @property
    def enforceable(self) -> bool:
        """True only when anchors are specific enough to hard-reject on a floor."""
        coherence = self.coherence
        return (coherence is not None
                and coherence >= config.ANCHOR_GATE_MIN_COHERENCE)

    def frame_sims(self, paths: list[str]) -> list[float]:
        """Max cosine of each frame path against the anchors (0..1), or []."""
        if not self.strong or not paths:
            return []
        vectors = vision_ranker.image_features(paths)
        if not vectors:
            return []
        anchors = np.asarray(self.embeddings)
        frames = np.asarray(vectors)
        return [float(np.max(row)) for row in (frames @ anchors.T)]

    def frame_sims_strict(self, paths: list[str]) -> list[float]:
        """Per-frame score requiring multiple anchor groups to match.

        Splits anchors into entity-group and object-group.  Each frame must
        match BOTH groups above ANCHOR_TERM_FLOOR or it is penalised.  When
        only one group exists (no objects in query), falls back to plain max.
        """
        min_groups = getattr(config, "ANCHOR_STRICT_MIN_GROUPS", 2)
        if not self.strong or not paths or min_groups <= 1:
            return self.frame_sims(paths)
        vectors = vision_ranker.image_features(paths)
        if not vectors:
            return []
        frames = np.asarray(vectors)
        anchor_mat = np.asarray(self.embeddings)
        has_ent = bool(self.entity_mask is not None and np.any(self.entity_mask))
        has_obj = bool(self.object_mask is not None and np.any(self.object_mask))
        if not (has_ent and has_obj):
            return self.frame_sims(paths)
        ent_sims = frames @ anchor_mat[self.entity_mask].T
        obj_sims = frames @ anchor_mat[self.object_mask].T
        ent_max = np.max(ent_sims, axis=1)
        obj_max = np.max(obj_sims, axis=1)
        min_sim = getattr(config, "ANCHOR_TERM_FLOOR", 0.28)
        ent_ok = ent_max >= min_sim
        obj_ok = obj_max >= min_sim
        groups_ok = ent_ok.astype(float) + obj_ok.astype(float)
        strict_score = np.minimum(ent_max, obj_max)
        fallback = np.maximum(ent_max, obj_max)
        both = groups_ok >= min_groups
        result = np.where(both, strict_score, fallback * 0.3)
        return [float(v) for v in result]

    def anchor_sim(self, paths: list[str]) -> float:
        """Best single frame-to-anchor cosine across the given paths."""
        sims = self.frame_sims(paths)
        return max(sims, default=0.0)


def _cache_path(sentence: str) -> Path:
    digest = hashlib.sha1(sentence.encode("utf-8")).hexdigest()[:16]
    return config.ANCHOR_CACHE / f"{digest}.json"


def build(sentence: str, plan: dict) -> AnchorSet | None:
    """Fetch (or load from cache) the event anchors for a sentence.

    Returns None when anchors are disabled, CLIP is unavailable, or no relevant
    image was found (lanes then fall back to audio/metadata verification only).
    """
    if not config.ANCHOR_ENABLED:
        return None
    clip_prompt = (plan.get("visual_prompt") or sentence) if isinstance(plan, dict) else sentence

    cache = _cache_path(sentence)
    if cache.exists():
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            entries = payload.get("entries") or []
            if entries and all(Path(e["path"]).exists() for e in entries):
                embeddings = vision_ranker.image_features(
                    [e["path"] for e in entries])
                if embeddings:
                    entity_mask = [bool(e.get("_has_entity", e.get("_entity_hits", 0) > 0)) for e in entries]
                    object_mask = [bool(e.get("_has_object", e.get("_object_hits", 0) > 0)) for e in entries]
                    logger.info("Visual anchors loaded from cache for '%s' (%d)",
                                sentence, len(entries))
                    return AnchorSet(entries, embeddings, clip_prompt,
                                     entity_mask, object_mask)
        except Exception as exc:
            logger.info("Anchor cache load failed: %s", exc)

    entries, embeddings = _fetch(sentence, plan)
    if not entries:
        logger.info("Visual anchors: no relevant images for '%s'", sentence)
        return None
    if not embeddings:
        logger.info("Visual anchors unavailable: no CLIP model for '%s'", sentence)
        return None
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"prompt": clip_prompt, "entries": entries},
                                    ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.info("Anchor cache write failed: %s", exc)
    logger.info("Visual anchors built for '%s': %d kept", sentence, len(entries))
    entity_mask = [bool(e.get("_has_entity")) for e in entries]
    object_mask = [bool(e.get("_has_object")) for e in entries]
    return AnchorSet(entries, embeddings, clip_prompt, entity_mask, object_mask)


def enrich(anchors: AnchorSet | None, image_urls: list[str],
           prompt: str, max_add: int = 3) -> AnchorSet | None:
    """Fold discovered-post thumbnails into the anchor set.

    Discovery often finds the canonical viral post whose thumbnail IS the
    event's pixels — far more specific than Google Images results.  Passing
    thumbnails here lets the CLIP anchor gate use them even for brand-new
    events where Google Images returned nothing (anchors=None -> a fresh set
    is built from the thumbnails alone).
    """
    if not config.ANCHOR_ENABLED or not image_urls:
        return anchors
    entries = list(anchors.entries) if anchors is not None else []
    vectors = list(anchors.embeddings) if anchors is not None else []
    added = 0
    for url in image_urls[: 2 * max_add]:
        if added >= max_add:
            break
        stem = "discovery_" + hashlib.sha1(url.encode()).hexdigest()[:12]
        dest = config.ANCHOR_CACHE / f"{stem}.img"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=config.IMAGE_NET_TIMEOUT) as resp:
                data = resp.read(config.IMAGE_MAX_BYTES)
            if len(data) < config.IMAGE_MIN_BYTES:
                continue
            dest.write_bytes(data)
        except Exception as exc:
            logger.info("Anchor enrichment download failed %s: %s", url, exc)
            continue
        scores = vision_ranker.image_query_scores([str(dest)], prompt)
        score = float(scores[0]) if scores else 0.0
        if score < config.IMAGE_SEMANTIC_MIN:
            continue
        vecs = vision_ranker.image_features([str(dest)])
        vec = vecs[0] if vecs else None
        if vec is None:
            continue
        if any(float(np.dot(vec, k)) >= config.ANCHOR_NEAR_DUP_COSINE
               for k in vectors):
            continue
        entries.append({"path": str(dest), "semantic_score": score,
                        "_has_entity": True, "_has_object": False,
                        "source_url": url})
        vectors.append(vec)
        added += 1
    if not added:
        return anchors
    logger.info("Anchor enrichment: +%d discovered thumbnail(s)", added)
    if anchors is not None:
        anchors.entries = entries
        anchors.embeddings = vectors
        return anchors
    return AnchorSet(entries, vectors, prompt,
                     [True] * len(entries), [False] * len(entries))


def _fetch(sentence: str, plan: dict) -> tuple[list[dict], list | None]:
    """Download candidate event images; keep the top unique, relevant ones."""
    config.ANCHOR_CACHE.mkdir(parents=True, exist_ok=True)
    clip_prompt = (plan.get("visual_prompt") or sentence) if isinstance(plan, dict) else sentence
    queries = image_lane._queries(sentence, plan)
    # Visual events (police car / helicopter / gel blaster) need anchors that
    # actually depict the object, so push object-only queries into the search —
    # the full sentence can drown the prop out.
    objects = [str(o) for o in (plan.get("object_terms") or [])]
    if objects:
        queries = queries + [" ".join(objects),
                             " ".join(objects) + " " + " ".join(sentence.split())[:70]]
        queries = queries[: config.IMAGE_QUERY_MAX]
    seen: dict[str, dict] = {}
    try:
        for q_index, query in enumerate(queries):
            for entry in image_lane._search(query, config.IMAGE_MAX_RESULTS):
                key = entry["image"]
                if key not in seen:
                    entry["_q"] = q_index
                    seen[key] = entry
    finally:
        image_lane._close_session()
    if not seen:
        return [], None

    downloaded: list[dict] = []
    for key, entry in seen.items():
        stem = (f"q{entry['_q']}_{len(downloaded):02d}_"
                f"{hashlib.sha1(key.encode()).hexdigest()[:10]}")
        dest = config.ANCHOR_CACHE / f"{stem}.img"
        if not image_lane._download(entry, dest):
            continue
        entry["path"] = str(dest)
        downloaded.append(entry)
    if not downloaded:
        return [], None

    scores = vision_ranker.image_query_scores(
        [e["path"] for e in downloaded], clip_prompt)
    for entry, score in zip(downloaded, scores or []):
        entry["semantic_score"] = float(score) if score is not None else 0.0

    scored = [e for e in downloaded
              if e["semantic_score"] >= config.IMAGE_SEMANTIC_MIN]
    scored.sort(key=lambda e: e["semantic_score"], reverse=True)
    if not scored:
        return [], None

    # --- Subject/object-aware tightness scoring ------------------------------
    # Score each image against individual entity and object terms so anchors
    # actually depict the specific people and objects from the query, not just
    # loosely-related imagery.  Images that show more of the query's subjects
    # and objects rank higher.
    entities = [str(t) for t in (plan.get("core_entities") or []) if t]
    streamer = plan.get("streamer") if isinstance(plan, dict) else None
    if streamer:
        sname = (streamer.get("name") or "").strip()
        if sname and sname not in entities:
            entities.append(sname)
    objects = [str(o) for o in (plan.get("object_terms") or []) if o]
    terms = entities + objects
    if terms:
        paths = [e["path"] for e in scored]
        term_scores = vision_ranker.text_image_scores(terms, paths)
        if term_scores is not None:
            term_mat = np.asarray(term_scores)
            for idx, entry in enumerate(scored):
                col = term_mat[:, idx]
                entity_hits = int(np.sum(col[:len(entities)] >= config.ANCHOR_TERM_FLOOR))
                object_hits = int(np.sum(col[len(entities):] >= config.ANCHOR_TERM_FLOOR))
                total_hits = entity_hits + object_hits
                term_coverage = total_hits / len(terms) if terms else 0.0
                entity_frac = entity_hits / len(entities) if entities else 1.0
                entry["_entity_hits"] = entity_hits
                entry["_object_hits"] = object_hits
                entry["_term_coverage"] = term_coverage
                entry["_tightness"] = (0.40 * entry["semantic_score"]
                                       + 0.35 * term_coverage
                                       + 0.25 * entity_frac)
            # Filter: require minimum subject/object coverage
            min_cov = getattr(config, "ANCHOR_MIN_COVERAGE", 0.40)
            before = len(scored)
            scored = [e for e in scored
                      if e.get("_term_coverage", 1.0) >= min_cov]
            if scored:
                logger.info("Anchor tightness: %d/%d images kept "
                            "(min_coverage=%.2f)", len(scored), before, min_cov)
            else:
                # Relax: keep the best by semantic_score if nothing passes
                scored = sorted(
                    [e for e in downloaded
                     if e["semantic_score"] >= config.IMAGE_SEMANTIC_MIN],
                    key=lambda e: e["semantic_score"], reverse=True,
                )[: config.ANCHOR_TOP_K]
                logger.info("Anchor tightness: no images hit min_coverage, "
                            "relaxed to top %d by semantic_score", len(scored))
            scored.sort(key=lambda e: e.get("_tightness", e["semantic_score"]),
                        reverse=True)
    else:
        for entry in scored:
            entry["_tightness"] = entry["semantic_score"]

    embeddings = vision_ranker.image_features([e["path"] for e in scored])
    if not embeddings:
        return scored[: config.ANCHOR_TOP_K], None

    for entry in scored:
        entry["_has_entity"] = entry.get("_entity_hits", 0) > 0
        entry["_has_object"] = entry.get("_object_hits", 0) > 0

    kept: list[dict] = []
    kept_vecs: list = []
    for entry, vec in zip(scored, embeddings):
        if vec is None:
            continue
        if any(float(np.dot(vec, k)) >= config.ANCHOR_NEAR_DUP_COSINE
               for k in kept_vecs):
            continue
        kept.append(entry)
        kept_vecs.append(vec)
        if len(kept) >= config.ANCHOR_TOP_K:
            break
    return kept, kept_vecs
