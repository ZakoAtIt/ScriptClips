"""Twitch VOD discovery, chunked transcription, and clip-moment analysis.

Search uses Twitch's GraphQL `searchFor` operation (public client id) instead of
yt-dlp, which ships no Twitch search extractor.  Long VODs are transcribed in
resumable 30-minute chunks so a crash never redoes the whole stream, and frame
sampling uses the lowest-bitrate HLS rendition for cheap seeks.
"""
import json
import logging
import math
import re
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import config
from modules import audio_transcriber, moment_proposer, vision_ranker
from modules.overlay_detector import visual_rawness
from modules.scoring import (evidence_editorial_score, metadata_rawness_score,
                             recap_score, title_match_score)
from modules.text_match import STOPWORDS, score as lexical_score
from modules.youtube_search import get_best_stream, sample_frames_at, stream_headers

try:
    import yt_dlp
except Exception:
    yt_dlp = None

logger = logging.getLogger(__name__)

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

_SEARCH_QUERY = """query {
  searchFor(userQuery: %(query)s, platform: "web", target: { index: VOD }) {
    videos {
      items {
        __typename
        ... on Video {
          id
          title
          lengthSeconds
          publishedAt
          viewCount
          owner { id login displayName }
          game { id displayName }
        }
      }
    }
  }
}"""


# --------------------------------------------------------------------------
# Search & ranking
# --------------------------------------------------------------------------
def _gql(payload: dict) -> dict:
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "text/plain;charset=UTF-8",
               "Client-ID": config.TWITCH_CLIENT_ID, "User-Agent": _UA}
    request = Request(config.TWITCH_GQL_ENDPOINT, data=data, headers=headers)
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


# Words Twitch search does not meaningfully index; dropped when building the
# "core" search ladder (e.g. 'kai confronts guy tyla' -> 'kai tyla').
_ACTION_WORDS = {
    "confront", "confronts", "confronted", "confrontation",
    "harass", "harasses", "harassed", "harassing",
    "react", "reacts", "reacted", "reacting",
    "flirt", "flirts", "flirting", "friendzone", "friendzones", "friendzoned",
    "announce", "announces", "announced", "announcement",
    "embarrassed", "embarrassing", "shocked", "shock", "surprised",
    "meme", "memes", "viral", "reunion", "awkward", "ship", "shipping",
    "best", "moments", "moment", "momentary", "clip", "clips", "video",
    "watch", "watching", "guy", "guys", "thing", "things", "people", "day",
}


def _search_variants(query: str) -> list[str]:
    """Progressive keyword ladders: Twitch search fails on long natural-language
    queries, so fall back from the full query down to its most specific terms."""
    tokens = [w for w in re.findall(r"[a-z0-9']+", query.lower())
              if len(w) > 1 and w not in STOPWORDS]
    if not tokens:
        return [query]
    variants: list[str] = []
    for size in (7, 6, 5, 4, 3, 2):
        if len(tokens) >= size:
            variants.append(" ".join(tokens[:size]))
    core = [t for t in tokens if t not in _ACTION_WORDS]
    if core and core != tokens:
        for size in (6, 5, 4, 3, 2):
            if len(core) >= size:
                variants.append(" ".join(core[:size]))
    for token in tokens[:3]:
        variants.append(token)
    return list(dict.fromkeys(v for v in variants if v.strip()))[:10]


def search_vods(query: str, limit: int | None = None) -> list[dict]:
    """Relevance-ranked VODs for a query (no auth needed)."""
    limit = limit or config.TWITCH_SEARCH_RESULTS
    vods_by_id: dict[str, dict] = {}
    for variant in _search_variants(query):
        payload = {"query": _SEARCH_QUERY % {"query": json.dumps(variant)}}
        try:
            result = _gql(payload)
        except (HTTPError, URLError, OSError) as exc:
            logger.warning("Twitch search network error for %r: %s", variant, exc)
            continue
        errors = result.get("errors")
        if errors:
            logger.warning("Twitch search failed for %r: %s", variant,
                           json.dumps(errors)[:400])
            continue
        items = (((result.get("data") or {}).get("searchFor") or {})
                 .get("videos", {}).get("items", []) or [])
        found = 0
        for item in items:
            if item.get("__typename") != "Video" or not item.get("id"):
                continue
            found += 1
            vod_id = str(item["id"])
            if vod_id in vods_by_id:
                continue
            vods_by_id[vod_id] = {
                "id": vod_id,
                "title": item.get("title") or "",
                "lengthSeconds": int(item.get("lengthSeconds") or 0),
                "publishedAt": item.get("publishedAt") or "",
                "viewCount": int(item.get("viewCount") or 0),
                "owner": (item.get("owner") or {}).get("login") or "",
                "game": (item.get("game") or {}).get("displayName") or "",
                "webpage_url": f"https://www.twitch.tv/videos/{vod_id}",
            }
        logger.info("Twitch search %r returned %d VODs (cumulative %d)",
                    variant, found, len(vods_by_id))
        if len(vods_by_id) >= limit:
            break
    return _rank_vods(list(vods_by_id.values()), query)[:limit]


def _rank_vods(vods: list[dict], query: str) -> list[dict]:
    for vod in vods:
        title = vod.get("title", "")
        metadata_text = " ".join(part for part in
                                 (title, vod.get("game", ""), vod.get("owner", "")) if part)
        vod["title_match_score"] = title_match_score(query, title, "")
        vod["metadata_score"] = lexical_score(query, metadata_text)
        vod["rawness_score"] = metadata_rawness_score(title, "", vod.get("owner"))
        vod["recap_score"] = recap_score(title, "")
        views = float(vod.get("viewCount") or 0)
        view_boost = 0.05 * min(1.0, math.log10(max(1.0, views)) / 9.0)
        vod["search_score"] = (
            0.40 * vod["metadata_score"]
            + 0.40 * vod["title_match_score"]
            + 0.20 * vod["rawness_score"]
            + view_boost
            - 0.30 * vod["recap_score"]
        )
    vods.sort(key=lambda item: item["search_score"], reverse=True)
    return vods


_TITLE_ENTITY_TOKENS = {"tyla", "kai", "cenat", "speed", "ispeed"}


def _query_core_tokens(query: str) -> list[str]:
    """Distinctive query terms (stopwords/action words removed)."""
    seen: set[str] = set()
    out: list[str] = []
    for token in re.findall(r"[a-z0-9']+", query.lower()):
        if len(token) > 1 and token not in STOPWORDS and token not in _ACTION_WORDS:
            if token not in seen:
                seen.add(token)
                out.append(token)
    return out


def select_analysis(vods: list[dict], query: str,
                    count: int | None = None) -> list[dict]:
    """Top, non-recap, correctly-sized VODs to analyse.

    Skips VODs that clearly cannot answer the query so we do not burn
    hours transcribing off-topic streams:
      - long VODs (>1h) whose title shares no query token are dropped;
      - if the query names a specific person (e.g. 'tyla'), long VODs must
        mention that person in the title; short clips stay exempt because
        they are cheap and often ARE the exact moment wanted.
    """
    count = count or config.TWITCH_VODS_TO_ANALYSE
    core = _query_core_tokens(query)
    named = [t for t in core if t in _TITLE_ENTITY_TOKENS]
    chosen: list[dict] = []
    for vod in vods:
        if vod.get("recap_score", 0.0) >= 0.7:
            continue
        length = float(vod.get("lengthSeconds") or 0)
        if not (config.MIN_VIDEO_LENGTH <= length <= config.TWITCH_MAX_VOD_SECONDS):
            continue
        title = (vod.get("title") or "").lower()
        short = length <= config.TWITCH_QUICK_CLIP_SECONDS
        if length > config.TWITCH_QUICK_CLIP_SECONDS and not short:
            if not any(tok in title for tok in core):
                logger.info("Skipping off-topic VOD %s (no query token in title): %s",
                            vod["id"], (vod.get("title") or "")[:60])
                continue
            if any(tok not in title for tok in named):
                logger.info("Skipping off-topic VOD %s (missing named entity '%s'): %s",
                            vod["id"], next(t for t in named if t not in title),
                            (vod.get("title") or "")[:60])
                continue
        chosen.append(vod)
        if len(chosen) >= count:
            break
    logger.info("Twitch analysis pool: %d VODs", len(chosen))
    for vod in chosen:
        logger.info("  - %s (match=%.2f recap=%.2f rawness=%.2f views=%s) %s",
                    vod["title"][:60], vod.get("title_match_score", 0.0),
                    vod.get("recap_score", 0.0), vod.get("rawness_score", 0.0),
                    vod.get("viewCount", 0), vod["webpage_url"])
    return chosen


# --------------------------------------------------------------------------
# VOD info + stream selection
# --------------------------------------------------------------------------
def get_vod_info(vod: dict, retries: int = 2) -> dict:
    """Fresh metadata + signed HLS formats via yt-dlp.

    yt-dlp's Twitch extractor intermittently reports "[twitch:stream] videos:
    videos does not exist" for VODs that resolve fine moments later, so a
    failed resolve is retried with a short backoff before being treated as
    dead.
    """
    if yt_dlp is None:
        raise RuntimeError("yt-dlp not available; cannot resolve Twitch VOD")
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                                   "skip_download": True, "noplaylist": True}) as ydl:
                return ydl.extract_info(vod["webpage_url"], download=False)
        except Exception as exc:
            last = exc
            if attempt < retries:
                logger.info("Twitch VOD resolve attempt %d/%d failed: %s",
                            attempt + 1, retries + 1, str(exc)[:120])
                time.sleep(2.0 * (attempt + 1))
    raise last


_VOD_AVAIL_TTL_SECONDS = 7 * 24 * 3600
_VOD_RESOLVE_MARKERS = ("videos does not exist", "video does not exist",
                        "no video is available", "not found")


def _availability_cache() -> dict:
    path = config.TWITCH_TRANSCRIPT_CACHE_DIR / "vod_availability.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_availability_cache(cache: dict) -> None:
    path = config.TWITCH_TRANSCRIPT_CACHE_DIR / "vod_availability.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def is_resolve_failure(exc: Exception) -> bool:
    """True when a yt-dlp exception means the VOD is (probably) gone for good."""
    text = str(exc)
    return any(marker in text.lower() for marker in _VOD_RESOLVE_MARKERS)


def vod_is_gone(vod: dict) -> bool:
    """Cached knowledge that a VOD previously failed to resolve as missing."""
    video_id = str(vod.get("id") or "")
    if not video_id:
        return False
    entry = _availability_cache().get(video_id)
    if not isinstance(entry, dict):
        return False
    stamp = entry.get("ts")
    return (isinstance(stamp, (int, float))
            and time.time() - stamp < _VOD_AVAIL_TTL_SECONDS)


def mark_vod_gone(vod: dict) -> None:
    video_id = str(vod.get("id") or "")
    if not video_id:
        return
    cache = _availability_cache()
    cache[video_id] = {"ts": time.time()}
    _save_availability_cache(cache)


def clear_vod_gone(vod: dict) -> None:
    video_id = str(vod.get("id") or "")
    if not video_id:
        return
    cache = _availability_cache()
    if video_id in cache:
        cache.pop(video_id, None)
        _save_availability_cache(cache)


def _video_formats(info: dict) -> list[dict]:
    return [f for f in (info.get("formats") or [])
            if f.get("url") and f.get("vcodec") not in (None, "none")]


def _audio_formats(info: dict) -> list[dict]:
    return [f for f in (info.get("formats") or [])
            if f.get("url") and f.get("acodec") not in (None, "none")
            and f.get("vcodec") in (None, "none")]


def get_frame_stream(info: dict) -> str | None:
    """Lowest-resolution HLS rendition for cheap, fast frame seeks."""
    pool = _video_formats(info)
    if not pool:
        return None
    pool.sort(key=lambda f: ((f.get("height") or 0), (f.get("tbr") or 0)))
    return pool[0]["url"]


def get_audio_stream(info: dict) -> str | None:
    """Audio-only HLS rendition, preferring a reasonable bitrate."""
    pool = _audio_formats(info)
    if not pool:
        return None
    pool.sort(key=lambda f: f.get("tbr") or 0)
    decent = [f for f in pool if 96 <= (f.get("tbr") or 0) <= 200]
    return (decent or pool)[-1]["url"]


# --------------------------------------------------------------------------
# Chunked, resumable transcription
# --------------------------------------------------------------------------
def _cache_path(vod_id: str) -> Path:
    return config.TWITCH_TRANSCRIPT_CACHE_DIR / f"{vod_id}.json"


def _load_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(path: Path, cache: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def transcribe_vod(info: dict, vod: dict, force: bool = False) -> list[dict]:
    """Resumable chunked Whisper transcription of a (possibly huge) Twitch VOD."""
    vod_id = str(info.get("id") or vod["id"])
    duration = float(info.get("duration") or vod.get("lengthSeconds") or 0)
    audio_url = get_audio_stream(info)
    if not audio_url or duration <= 0:
        logger.info("No usable audio stream for Twitch VOD %s", vod_id)
        return []

    path = _cache_path(vod_id)
    cache = _load_cache(path)
    if cache.get("merged") and not force:
        return cache["segments"]

    chunk_len = config.TWITCH_TRANSCRIPT_CHUNK_SECONDS
    chunk_count = max(1, math.ceil(duration / chunk_len))
    extra_args = stream_headers(info)
    done = 0
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        for index in range(chunk_count):
            key = str(index)
            if key in cache.get("chunks", {}):
                done += 1
                continue
            start = index * chunk_len
            length = min(chunk_len, duration - start)
            wav = Path(tmp) / f"chunk_{index}.wav"
            try:
                audio_transcriber._extract_audio(
                    audio_url, str(wav), extra_args=extra_args,
                    max_seconds=length, start_seconds=start,
                    timeout=config.TWITCH_CHUNK_EXTRACT_TIMEOUT)
            except Exception as exc:
                logger.warning("Chunk %d audio extract failed for %s: %s",
                               index, vod_id, exc)
                failed = True
                continue
            segments = audio_transcriber.transcribe_file(str(wav), model_size="tiny")
            for segment in segments:
                segment["start"] += start
                segment["end"] += start
            cache.setdefault("chunks", {})[key] = segments
            _save_cache(path, cache)
            done += 1
            logger.info("Twitch VOD %s transcribed %d/%d chunks (%.0fs of %.0fs)",
                        vod_id, done, chunk_count, start + length, duration)

    merged = [segment for chunk in cache.get("chunks", {}).values()
              if isinstance(chunk, list) for segment in chunk]
    merged.sort(key=lambda s: s["start"])
    cache["segments"] = merged
    if not failed:
        cache["merged"] = True
    cache["duration"] = duration
    _save_cache(path, cache)
    if failed:
        logger.warning("Twitch VOD %s has %d transcribed chunk(s) but some failed; "
                       "a later run will retry the missing chunks",
                       vod_id, len(cache.get("chunks", {})))
    logger.info("Twitch VOD %s transcribed: %d segments", vod_id, len(merged))
    return merged


# --------------------------------------------------------------------------
# Candidate analysis (mirrors run.analyse_video output shape)
# --------------------------------------------------------------------------
def _snippet_at(segments: list[dict], ts: float, window: float) -> str:
    start, end = ts, ts + window
    return " ".join(seg.get("text", "") for seg in segments
                    if seg.get("end", 0) >= start and seg.get("start", 0) <= end)


def _clean_video_id(raw: str) -> str:
    match = re.search(r"\d+", str(raw))
    return match.group(0) if match else str(raw)


def analyse_vod(vod: dict, query: str, info: dict | None = None) -> list[dict]:
    """Transcribe one Twitch VOD and propose clip moments (same shape as YouTube path)."""
    if info is None:
        info = get_vod_info(vod)
    video_id = _clean_video_id(info.get("id") or vod["id"])
    duration = float(info.get("duration") or vod.get("lengthSeconds") or 0)
    if duration < config.MIN_VIDEO_LENGTH or duration > config.TWITCH_MAX_VOD_SECONDS:
        logger.info("Skipping Twitch VOD %s: unsuitable length %.0fs", video_id, duration)
        return []

    segments = transcribe_vod(info, vod)
    if not segments:
        logger.info("No transcript for Twitch VOD %s", video_id)
        return []

    t_candidates = moment_proposer.transcript_candidates(
        segments, query, duration, config.CLIP_DURATION_SECONDS)
    if not t_candidates:
        logger.info("No transcript candidates in Twitch VOD %s", video_id)
        return []

    frame_stream = get_frame_stream(info)
    export_stream = get_best_stream(info)
    export_headers = stream_headers(info)
    extra_args = export_headers
    frames_dir = config.FRAMES_DIR / f"twitch_{video_id}"
    frames_per = config.FRAMES_PER_CANDIDATE if duration <= config.LONG_VIDEO_SECONDS else 1
    frames_work = False
    if frame_stream and t_candidates:
        first_ts = float(t_candidates[0]["start"]) + config.CLIP_DURATION_SECONDS / (frames_per + 1)
        frames_work = bool(sample_frames_at(frame_stream, [first_ts], frames_dir, extra_args))

    candidate_frames = []
    for tc in t_candidates:
        start = float(tc["start"])
        timestamps = [start + (i + 1) * config.CLIP_DURATION_SECONDS / (frames_per + 1)
                      for i in range(frames_per)]
        frames = (sample_frames_at(frame_stream, timestamps, frames_dir, extra_args)
                  if frame_stream and frames_work else [])
        candidate_frames.append((tc, frames))

    scored_frames = vision_ranker.score_frames(
        [frame for _, frames in candidate_frames for frame in frames], query)

    title = info.get("title") or vod.get("title") or ""
    description = ""
    rawness = metadata_rawness_score(title, description,
                                     info.get("uploader") or vod.get("owner"))
    title_match = title_match_score(query, title, description)
    metadata = lexical_score(query, " ".join(part for part in
                                             (title, vod.get("game", ""), vod.get("owner", "")) if part))

    candidates = []
    for tc, frames in candidate_frames:
        start = float(tc["start"])
        clip_dur = float(tc.get("duration", config.CLIP_DURATION_SECONDS))
        in_window = [sf for sf in scored_frames if start <= sf["timestamp"] <= start + clip_dur]
        vision_score = max((sf["similarity"] for sf in in_window), default=0.0)
        visual = visual_rawness(frames)
        evidence = tc.get("evidence", "")
        candidates.append({
            "start": start,
            "timestamp": start,
            "duration": clip_dur,
            "evidence": evidence,
            "semantic_score": float(tc.get("semantic_score", 0.0)),
            "lexical_score": float(tc.get("lexical_score", 0.0)),
            "entity_hits": tc.get("entity_hits", 0),
            "exact_phrase": tc.get("exact_phrase", 0.0),
            "vision_score": vision_score,
            "visual_raw": visual["visual_raw"],
            "overlay": visual["overlay"],
            "continuity": visual["continuity"],
            "webpage_url": vod.get("webpage_url") or f"https://www.twitch.tv/videos/{video_id}",
            "stream_url": frame_stream,
            "export_stream_url": export_stream,
            "export_headers": export_headers,
            "audio_url": get_audio_stream(info),
            "video_id": video_id,
            "query": query,
            "title": title,
            "description": description,
            "metadata_score": metadata,
            "title_match_score": title_match,
            "rawness_score": rawness,
            "editorial_score": evidence_editorial_score(evidence),
            "match_tier": tc.get("match_tier", "strict"),
            "snippet": _snippet_at(segments, start, config.CLIP_DURATION_SECONDS),
            "source": "twitch",
        })
    logger.info("Twitch VOD %s produced %d candidate clips", video_id, len(candidates))
    return candidates
