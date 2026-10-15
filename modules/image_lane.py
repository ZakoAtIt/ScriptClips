"""Image fallback lane (Google Images via a real headed Chrome, Bing fallback).

Used only when neither Twitch clips nor YouTube raw footage can supply
verified clips for a sentence.  Searches the sentence (+ streamer name) in a
real headed Chrome on Google Images, then on Bing if Google returns nothing.
The top tiles are downloaded (non-overlapping filenames) and ranked by CLIP
relevance to the sentence so the best-matching images win.
"""
import hashlib
import html
import json
import logging
import re
import time
from pathlib import Path

import requests

import config
from modules import vision_ranker

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}
_M_ATTR = re.compile(r'm="([^"]+)"')
_TERM_RE = re.compile(r"[A-Za-z0-9]+")
_GENERIC = {"others", "said", "says", "would", "probably", "think", "tried",
            "felt", "called", "might", "one", "could", "ever"}

_session = None
_session_tried = False


def _chrome_session():
    """Lazily start (once per run) a headed Chrome instance for Google Images."""
    global _session, _session_tried
    if _session_tried:
        return _session
    _session_tried = True
    try:
        from modules.chrome_session import ChromeSession
        _session = ChromeSession().start()
    except Exception as exc:
        logger.warning("Chrome session failed to start: %s", exc)
        _session = None
    return _session


def _close_session() -> None:
    global _session, _session_tried
    if _session is not None:
        try:
            _session.close()
        except Exception as exc:
            logger.info("Chrome session close failed: %s", exc)
    _session = None
    _session_tried = False


def _google_search(query: str, max_results: int) -> list[dict]:
    """Google Images rendered in a real headed Chrome; grab the top tiles."""
    session = _chrome_session()
    if session is None:
        return []
    from urllib.parse import urlencode
    page = session.context.new_page()
    try:
        page.goto("https://www.google.com/search?" + urlencode({"tbm": "isch", "q": query}),
                  timeout=45000)
        page.wait_for_timeout(3000)
        for _ in range(3):
            page.mouse.wheel(0, 2500)
            page.wait_for_timeout(500)
        entries: list[dict] = []
        for img in page.query_selector_all('img[src*="encrypted-tbn"]'):
            src = img.get_attribute("src") or ""
            if not src.lower().startswith("http"):
                continue
            entries.append({
                "image": src,
                "page": "",
                "title": "",
                "source": "",
                "width": 0,
                "height": 0,
            })
            if len(entries) >= max_results:
                break
        return entries
    except Exception as exc:
        logger.info("Google image render failed: %s", exc)
        return []
    finally:
        try:
            page.close()
        except Exception:
            pass


def _bing_search(query: str, max_results: int) -> list[dict]:
    entries: list[dict] = []
    for attempt in range(2):
        try:
            resp = requests.get(
                "https://www.bing.com/images/search",
                params={"q": query, "form": "HDRSC2"},
                headers=_HEADERS, timeout=config.IMAGE_NET_TIMEOUT)
            if resp.status_code != 200:
                return []
        except Exception as exc:
            logger.info("Bing image search failed: %s", exc)
            return []
        for m_attr in _M_ATTR.findall(resp.text):
            try:
                data = json.loads(html.unescape(m_attr))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            url = data.get("murl") or ""
            if not url.lower().startswith("http"):
                continue
            try:
                width = int(data.get("w") or data.get("width") or 0)
                height = int(data.get("h") or data.get("height") or 0)
            except (TypeError, ValueError):
                width = height = 0
            if width and width < config.IMAGE_MIN_SIDE:
                continue
            if height and height < config.IMAGE_MIN_SIDE:
                continue
            entries.append({
                "image": url,
                "page": data.get("purl") or "",
                "title": data.get("t") or "",
                "source": data.get("mu") or "",
                "width": width,
                "height": height,
            })
            if len(entries) >= max_results:
                return entries
        if entries:
            return entries
        time.sleep(2)
    return entries


def _search(query: str, max_results: int) -> list[dict]:
    """Try configured image sources in order (Google via Chrome, then Bing)."""
    handlers = {"google": _google_search, "bing": _bing_search}
    for src in config.IMAGE_LANE_SOURCES:
        handler = handlers.get(src)
        if handler is None:
            continue
        for attempt in range(2):
            try:
                entries = handler(query, max_results)
            except Exception as exc:
                logger.info("%s image search error: %s", src, exc)
                entries = []
            if entries:
                logger.info("Image lane: %s returned %d results for %r",
                            src, len(entries), query)
                return entries
            time.sleep(2)
    return []


def _download(entry: dict, dest: Path) -> bool:
    try:
        with requests.get(entry["image"], headers=_HEADERS,
                          timeout=config.IMAGE_NET_TIMEOUT, stream=True) as resp:
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if not content_type.startswith("image/"):
                return False
            data = b""
            for chunk in resp.iter_content(64 * 1024):
                data += chunk
                if len(data) > config.IMAGE_MAX_BYTES:
                    return False
        if len(data) < config.IMAGE_MIN_BYTES:
            return False
        dest.write_bytes(data)
        return True
    except Exception as exc:
        logger.info("Image download failed for %s: %s", entry.get("image"), exc)
        return False


def _queries(sentence: str, plan: dict) -> list[str]:
    """Use the planner's 2 entity-aware queries (both subjects + action)."""
    queries = [q for q in (plan.get("search_queries") or []) if q]
    if queries:
        return queries[: config.IMAGE_QUERY_MAX]
    base = " ".join(sentence.split())
    streamer = plan.get("streamer") if isinstance(plan, dict) else None
    name = ""
    if streamer:
        name = streamer.get("name") or streamer.get("twitch_handle") or ""
    out: list[str] = []
    if name:
        out.append(f"{name} {base[:80]}")
    out.append(base[:90])
    return out[: config.IMAGE_QUERY_MAX]


def run(sentence: str, plan: dict, need: int,
        anchor_entries: list[dict] | None = None) -> list[dict]:
    """Return ranked, face-verified image candidates for the sentence.

    ``anchor_entries`` are the visual anchors already fetched by the pipeline
    (top CLIP-relevant Google Images, scored); when supplied the search + CLIP
    pass is skipped so the Chrome session is not opened twice per query.
    """
    if not config.IMAGE_LANE_ENABLED or need <= 0:
        return []
    out_dir = config.IMAGES_DIR / "search"
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_prompt = (plan.get("visual_prompt") or sentence) if isinstance(plan, dict) else sentence
    try:
        if anchor_entries:
            downloaded = list(anchor_entries)
        else:
            seen: dict[str, dict] = {}
            for q_index, query in enumerate(_queries(sentence, plan)):
                for entry in _search(query, config.IMAGE_MAX_RESULTS):
                    key = entry["image"]
                    if key not in seen:
                        entry["_q"] = q_index
                        seen[key] = entry
            if not seen:
                logger.info("Image lane: no search results for '%s'", sentence)
                return []

            downloaded = []
            from PIL import Image as PILImage
            for key, entry in seen.items():
                stem = (f"q{entry['_q']}_{len(downloaded):02d}_"
                        f"{hashlib.sha1(key.encode()).hexdigest()[:10]}")
                dest = out_dir / f"{stem}.img"
                if not _download(entry, dest):
                    continue
                try:
                    with PILImage.open(dest) as im:
                        width, height = im.size
                except Exception:
                    width = height = 0
                if width and height and min(width, height) < config.IMAGE_MIN_SIDE:
                    dest.unlink(missing_ok=True)
                    continue
                entry["path"] = str(dest)
                downloaded.append(entry)
            if not downloaded:
                return []

            scores = vision_ranker.image_query_scores(
                [e["path"] for e in downloaded], clip_prompt)
            for entry, score in zip(downloaded, scores or []):
                entry["semantic_score"] = float(score) if score is not None else 0.0

        refs: dict = {}
        if config.FACE_VERIFY_ENABLED:
            subjects = [s for s in (plan.get("core_entities") or [])]
            if subjects:
                try:
                    from modules import face_verify
                    refs, _missing = face_verify.refs_for(subjects)
                except Exception as exc:
                    logger.info("Image lane: face refs unavailable: %s", exc)
                    refs = {}
        if refs:
            for entry in downloaded:
                try:
                    scores_map = face_verify.verify_image(entry["path"], refs)
                except Exception:
                    scores_map = {}
                seen = face_verify.subjects_on_screen(scores_map)
                entry["face_scores"] = {k: round(v, 3) for k, v in scores_map.items()}
                if config.FACE_REQUIRE_ALL:
                    entry["face_verified"] = all(name in seen for name in refs)
                else:
                    entry["face_verified"] = any(name in seen for name in refs)

        ranked = sorted(downloaded,
                        key=lambda e: (bool(e.get("face_verified")), e["semantic_score"]),
                        reverse=True)
        verified: list[dict] = []
        fallbacks: list[dict] = []
        for index, entry in enumerate(ranked):
            if entry["semantic_score"] < config.IMAGE_SEMANTIC_MIN:
                continue
            candidate = {
                "source": "image",
                "video_id": f"img_{index}",
                "image_path": entry["path"],
                "title": entry.get("title") or sentence,
                "webpage_url": entry.get("page") or entry.get("image"),
                "channel": entry.get("source"),
                "duration": 0.0,
                "start": 0.0,
                "query": clip_prompt,
                "semantic_score": entry["semantic_score"],
                "vision_score": entry["semantic_score"],
                "snippet": entry.get("title", ""),
                "face_verified": bool(entry.get("face_verified")),
                "face_scores": entry.get("face_scores", {}),
                "verified": True,
                "verification_reason": (
                    f"face-verified + CLIP {entry['semantic_score']:.2f}"
                    if entry.get("face_verified")
                    else f"CLIP relevance {entry['semantic_score']:.2f}"),
            }
            (verified if candidate["face_verified"] else fallbacks).append(candidate)
            if len(verified) >= need and not refs:
                break
        candidates = verified
        if config.FACE_REF_FALLBACK:
            candidates = (verified + fallbacks)[:need]
        else:
            candidates = verified[:need]
        logger.info("Image lane for '%s' returned %d verified candidate(s) "
                    "(%d face-verified)", sentence, len(candidates), len(verified))
        return candidates
    finally:
        _close_session()
