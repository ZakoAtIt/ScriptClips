"""Semantic refinement: CLIP re-ranks the per-scene shortlist.

Pure heuristic scoring cannot tell which window actually matches the line being
spoken.  For each scene the top SHORT_SEMANTIC_SHORTLIST candidates get a center
frame CLIP-scored against the scene's label, and the final rank is a weighted
blend of heuristic score and CLIP similarity.  If CLIP is unavailable the
original heuristic order is kept (re-ranker never kills candidates).
"""
import hashlib
import logging
import math
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def _center_frame_path(candidate: dict, out_dir: Path) -> Path | None:
    mid = (candidate["start"] + candidate["end"]) / 2.0
    safe = hashlib.sha1(candidate["clip"].encode("utf-8", "replace")).hexdigest()[:12]
    out = out_dir / f"f_{safe}_{mid:.1f}.jpg"
    if out.exists():
        return out
    commands = (
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-ss", f"{mid:.2f}", "-i", candidate["clip"], "-frames:v", "1",
         "-vf", f"scale={config.VISUAL_FRAME_WIDTH}:-2", str(out)],
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", candidate["clip"], "-ss", f"{mid:.2f}", "-frames:v", "1",
         "-vf", f"scale={config.VISUAL_FRAME_WIDTH}:-2", str(out)],
    )
    for attempt, cmd in enumerate(commands):
        try:
            subprocess.run(cmd, check=True, timeout=45,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            logger.info("Center frame extract attempt %d failed for %s: %s",
                        attempt + 1, candidate["name"], exc)
            continue
        if out.exists():
            return out
        out.unlink(missing_ok=True)
    logger.info("Center frame could not be extracted for %s", candidate["name"])
    return None


def rerank(scene_label: str, candidates: list[dict]) -> list[dict]:
    """Return candidates re-ranked by heuristic + CLIP similarity."""
    if not candidates:
        return candidates
    shortlist = sorted(candidates, key=lambda c: -c["score"])[
        :config.SHORT_SEMANTIC_SHORTLIST]
    if len(shortlist) < 2:
        return candidates

    from modules import vision_ranker
    with tempfile.TemporaryDirectory(prefix="scsem_") as tmpdir:
        out_dir = Path(tmpdir)
        paths: list[Path | None] = [_center_frame_path(c, out_dir) for c in shortlist]
        usable = [p for p in paths if p is not None]
        sims = vision_ranker.image_query_scores([str(p) for p in usable],
                                                scene_label) if usable else []
        by_path = {str(path): sim for path, sim in zip(usable, sims)} if sims else {}

    scores = [c["score"] for c in shortlist]
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    enriched: dict[str, float] = {}
    weight = config.SHORT_SEMANTIC_WEIGHT
    for candidate, path in zip(shortlist, paths):
        sim = by_path.get(str(path), 0.0) if path else 0.0
        normalized = (candidate["score"] - lo) / span
        candidate["clip_sim"] = round(sim, 4)
        candidate["combined"] = round(
            (1 - weight) * normalized + weight * sim
            + config.SHORT_STIMULUS_BONUS * float(candidate.get("motion", 0.0)), 4)
        enriched[id(candidate)] = sim
    if not enriched:
        return candidates

    ranked = sorted(shortlist, key=lambda c: -c["combined"])
    kept_ids = {id(c) for c in ranked}
    rest = [c for c in candidates if id(c) not in kept_ids]
    return ranked + rest


def _safe_stem(name: str) -> str:
    from modules.short_builder.footage_index import _safe_stem as _sf
    return _sf(Path(name))


def _token_set(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t and len(t) > 1}


_CURATED_PERSONS: dict[str, tuple[str, ...]] = {
    "tobey": ("tobey", "maguire"),
    "andrew": ("andrew", "garfield"),
    "tom": ("tom", "holland"),
}

_NOISE_TOKENS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "vs", "from", "by", "as", "is", "was", "are", "were", "s", "t",
    "full", "4k", "hd", "1080p", "clip", "clips", "scene", "scenes",
    "compilation", "best", "official", "officially", "react", "reacts",
    "behind", "footage", "movie", "part", "parts", "video", "videos",
    "stream", "streams", "streaming", "livestream", "live",
}


def _folder_tokens(folder: str) -> list[str]:
    """Clean tokens of a topic-folder name (case/hyphen folded, noise removed)."""
    stem = _safe_stem(folder).lower().replace("-", " ")
    return [t for t in re.split(r"[^a-z0-9]+", stem) if t and t not in _NOISE_TOKENS]


def detect_persons(folders: list[str]) -> dict[str, tuple[str, ...]]:
    """Map person id -> alias words, derived from the folder names.

    A word that leads 2+ topic folders marks a person (Tobey/Andrew/Tom in a
    Spider-Man library, Sykkuno/Zoil/Nmplol/Clavicular in a streamer library),
    so person pinning needs no per-project hardcoding.  Curated names are
    prepended as overrides.
    """
    first: dict[str, int] = {}
    for folder in folders:
        toks = _folder_tokens(folder)
        if toks:
            first[toks[0]] = first.get(toks[0], 0) + 1
    persons: dict[str, tuple[str, ...]] = dict(_CURATED_PERSONS)
    for token, count in first.items():
        if count >= 2:
            persons.setdefault(token, (token,))
    return persons


def detect_person(text: str, persons: dict[str, tuple[str, ...]] | None = None) -> str | None:
    """Return the person id named in a beat, or None."""
    if not text:
        return None
    persons = persons if persons is not None else dict(_CURATED_PERSONS)
    low = text.lower()
    for person, names in persons.items():
        if any(re.search(rf"\b{re.escape(n)}\b", low) for n in names):
            return person
    aliases = getattr(config, "SHORT_PERSON_ALIASES", {}) or {}
    for alias, person in aliases.items():
        if person in persons and re.search(rf"\b{re.escape(alias)}\b", low):
            return person
    return None


def folder_person(folder: str,
                  persons: dict[str, tuple[str, ...]] | None = None) -> str | None:
    """Person id a topic folder belongs to (name prefix), or None."""
    persons = persons if persons is not None else dict(_CURATED_PERSONS)
    stem = _safe_stem(folder).lower()
    for person in persons:
        if stem.startswith(person):
            return person
    return None


def folders_for_person(folders: list[str], person: str,
                       persons: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    """All topic folders belonging to one person (e.g. 'tom')."""
    return [f for f in folders if folder_person(f, persons) == person]


# Topic folders are matched by keyword hits against the beat's spoken text.
# Weighted keywords: person names and distinctive phrases are strong; generic
# company/show words are weak so they don't win ties against real topics.
_FOLDER_KEYWORDS: dict[str, dict[str, float]] = {
    "Andrew_Garfield_denies_No_Way_Home": {
        "garfield": 1.2, "andrew": 1.2, "denies": 1.0, "deny": 1.0,
        "denying": 1.2, "denied": 1.0, "return": 1.0, "filming": 1.0},
    "Andrew_Garfield_fake_production_title": {
        "fake": 1.2, "project name": 1.2, "project names": 1.2,
        "production": 1.0, "paperwork": 1.2, "involvement": 1.0},
    "Jean_Grey_Hulk_out_scene_Spider-Man_Brand_New_Day": {
        "hulk": 1.0, "mcu": 1.0, "out of the mcu": 1.2,
        "brand new day": 1.2, "out of the": 0.8},
    "Sony_Marvel_Spider-Man_deal_restored": {
        "deal": 1.2, "new deal": 1.2, "fell apart": 1.2, "apart": 1.0,
        "restored": 1.0, "marvel": 0.5, "sony": 0.5},
    "Spider-Man_MCU_swinging_action_montage_4k": {
        "swing": 1.0, "swinging": 1.0, "montage": 1.2, "action": 1.0,
        "save": 1.2, "protect": 1.0},
    "Spider-Man_Sony_Marvel_split_2019": {
        "split": 1.2, "2019": 1.2, "splitting": 1.0},
    "Spider-Man_The_Other_comic_panels_cocoon": {
        "comic": 1.0, "cocoon": 1.0, "the other": 1.2, "panels": 1.0,
        "secrets": 1.2, "secret": 1.0, "hidden": 1.2, "biggest": 1.0,
        "discover": 1.2, "four": 0.8},
    "Spider-Man_biological_webs_vs_mechanical_web_shooters": {
        "web": 1.0, "webs": 1.0, "web shooters": 1.2, "organic": 1.2,
        "mechanical": 1.2},
    "Spider-Man_panic_attack_MIT_graduation_party_Brand_New_Day": {
        "panic": 1.2, "mit": 1.2, "graduation": 1.2},
    "Spiderman_No_Way_Home_behind_the_scenes": {
        "behind the scenes": 1.2, "no way home": 1.2, "way home": 1.2,
        "last minute": 1.2},
    "Tobey_Maguire_No_Way_Home_interview": {
        "tobey": 1.2, "maguire": 1.2, "interview": 0.8},
    "Tobey_Maguire_No_Way_Home_negotiations": {
        "tobey": 1.2, "maguire": 1.2, "negotiat": 1.0, "negotiation": 1.0,
        "negotiate": 1.0, "signed": 1.2, "signing": 1.0,
        "officially signed": 1.2, "called": 0.5, "executives": 0.5,
        "disney": 0.5},
    "Tobey_Maguire_Spider-Man_organic_webs_wrist_close_up": {
        "wrist": 1.2, "close up": 1.2, "organic web": 1.2},
    "Tom_Holland_didn_t_get_Marvel_full_script": {
        "tom": 1.2, "holland": 1.2, "script": 1.2, "full script": 1.2,
        "pages": 1.2, "received": 1.0, "film": 0.8, "filmed": 0.8},
    "Tom_Holland_Infinity_War_script_interview": {
        "tom": 1.2, "holland": 1.2, "infinity war": 1.2, "infinity": 1.0,
        "war": 1.0, "interview": 0.8, "called": 1.0, "executives": 1.0,
        "disney": 0.5, "personally": 1.0, "convince": 1.0},
    "Tom_Holland_spoilers_compilation": {
        "tom": 1.2, "holland": 1.2, "spoiler": 1.2, "spoilers": 1.2,
        "trust": 1.0, "trusted": 1.2},
}


def _normalize_text(text: str) -> str:
    t = text.lower().replace("2e19", "2019").replace("2e18", "2018")
    t = t.replace("-", " ")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t)


def _folder_ngrams(folder: str) -> list[tuple[str, ...]]:
    toks = _folder_tokens(folder)
    grams: list[tuple[str, ...]] = []
    for n in (1, 2, 3):
        for i in range(len(toks) - n + 1):
            grams.append(tuple(toks[i:i + n]))
    return grams


def _auto_keywords(folders: list[str], folder: str) -> dict[str, float]:
    """N-gram keywords for a folder, weighted by rarity across the library."""
    grams = _folder_ngrams(folder)
    total = max(len(folders), 1)
    out: dict[str, float] = {}
    for gram in grams:
        freq = sum(1 for other in folders if gram in _folder_ngrams(other))
        idf = math.log(total / (1.0 + freq)) + 1.0
        weight = len(gram) * idf
        key = " ".join(gram)
        out[key] = max(out.get(key, 0.0), weight)
    return out


@lru_cache(maxsize=128)
def _keywords_for(folders_key: tuple[str, ...], folder: str) -> dict[str, float]:
    """Curated keyword table when present, else auto-derived n-grams."""
    if folder in _FOLDER_KEYWORDS:
        return dict(_FOLDER_KEYWORDS[folder])
    return _auto_keywords(list(folders_key), folder)


def _folder_match_score(folders: list[str], text: str) -> tuple[str | None, float]:
    """Best folder + keyword score for a beat's spoken text."""
    if not folders or not text:
        return None, 0.0
    text_n = _normalize_text(text)
    best: str | None = None
    best_score = 0.0
    key_folders = tuple(folders)
    for folder in folders:
        keywords = _keywords_for(key_folders, folder)
        score = sum(weight for key, weight in keywords.items() if key in text_n)
        if score > best_score:
            best, best_score = folder, score
    return best if best_score > 0 else None, best_score


def _fallback_folder(folders: list[str], text: str) -> str | None:
    """Best folder pick by weighted keyword hits against the beat text."""
    best, _ = _folder_match_score(folders, text)
    return best


def assign_folders(folders: list[str], beats: list[dict]) -> dict[int, str | None]:
    """Map each beat (by index) to its best-matching topic folder.

    The folders are named after the topics themselves and the script's wording
    lines up with them, so a keyword matcher against the raw spoken text is the
    reliable signal (an LLM pass proved slower and less accurate).  Beats that
    name a person are pinned to that person's folders (identity is the hard
    signal); beats with no folder match map to None -> global pool.
    """
    persons = detect_persons(folders)
    by_index: dict[int, str | None] = {}
    for i, beat in enumerate(beats):
        text = beat.get("text", "")
        person = detect_person(text, persons)
        if person:
            family = folders_for_person(folders, person, persons)
            if family:
                best, _ = _folder_match_score(family, text)
                by_index[i] = best or family[0]
                continue
        by_index[i] = _fallback_folder(folders, text)
    return by_index


def beat_contexts(folders: list[str], beats: list[dict]) -> dict[int, dict]:
    """Per-beat {folder, score, person} so the assigner can enforce person pins."""
    persons = detect_persons(folders)
    contexts: dict[int, dict] = {}
    for i, beat in enumerate(beats):
        text = beat.get("text", "")
        person = detect_person(text, persons)
        if person:
            family = folders_for_person(folders, person, persons)
            if family:
                folder, score = _folder_match_score(family, text)
                contexts[i] = {"folder": folder or family[0],
                               "score": score, "person": person}
                continue
        folder, score = _folder_match_score(folders, text)
        contexts[i] = {"folder": folder, "score": score, "person": person}
    return contexts


def best_image(images: list[dict], scene_label: str, used: set[str],
               folder: str | None = None) -> dict | None:
    """Pick the best unused fallback image, CLIP-matched to the scene label.

    When the beat is matched to a topic folder, images from that folder are
    preferred before falling back to the full pool.
    """
    unused = [image for image in images if image["path"] not in used]
    if not unused:
        return None
    if folder:
        preferred = [image for image in unused
                     if _safe_stem(image.get("folder", "")) == _safe_stem(folder)]
        if preferred:
            unused = preferred
    if config.SHORT_IMAGE_FALLBACK_SEMANTIC:
        try:
            from modules import vision_ranker
            sims = vision_ranker.image_query_scores([i["path"] for i in unused],
                                                    scene_label)
            if sims:
                index = int(max(range(len(unused)), key=lambda i: sims[i]))
                chosen = dict(unused[index])
                chosen["clip_sim"] = round(float(sims[index]), 4)
                return chosen
        except Exception as exc:
            logger.info("Image CLIP fallback failed: %s", exc)
    return dict(unused[0])
