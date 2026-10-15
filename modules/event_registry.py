"""Curated event-footage registry (data/event_videos.json).

For multi-streamer live events (Streamer Games, Kai vs. X, ...) the raw moment
lives on the event's own broadcast channel and each participant's POV channel,
and YouTube search for the verbatim sentence mostly surfaces reactors and
compilations.  The registry pins the known original broadcasts/POVs so the
event pool is built from the actual event footage, not platform search noise.

``match(plan)`` returns the matching event entry when a query names the event
(or is an event with candidate streamers); ``candidates(entry, plan)`` resolves
each pinned video's live metadata and returns pool candidates marked
``event_pinned`` (they are the event, so no source-rejection applies to them).
"""
import json
import logging

import config
from modules import downloader

logger = logging.getLogger(__name__)


_TOKEN_RE = __import__("re").compile(r"[a-z0-9']+")
_STOPWORDS = {"a", "an", "the", "and", "or", "for", "with", "from", "that",
              "this", "these", "those", "who", "what", "when", "how", "why",
              "in", "on", "at", "to", "of", "is", "are", "was", "were", "his",
              "her", "their", "your", "our", "it", "its", "he", "she", "they",
              "you", "we", "i", "not", "no", "vs"}
# Low-information words that carry no event identity: generic action verbs and
# phrasing words.  These must not count toward a sibling-key merge, otherwise
# two DIFFERENT events that share a person + verb ("fousey fights bike owner"
# vs "fousey fights neon") cross-merge and pollute each other's raw pool.
_LOW_INFO = {"fight", "fights", "fighting", "fighter", "throw", "throws",
             "throwing", "slap", "slaps", "slapped", "crash", "crashed",
             "crashing", "crashout", "confrontation", "confronts",
             "confronted", "confront", "physical", "makes", "made", "make",
             "talk", "talks", "talked", "talking", "tries", "tried", "try",
             "gets", "get", "got", "hit", "hits", "kicks", "kicked", "says",
             "said", "goes", "went", "stream", "streams", "clip", "clips",
             "video", "videos", "live", "full", "footage", "reaction",
             "react", "reacts", "episode", "episodes", "day", "days"}


def _read_json(path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load() -> dict:
    """Merge the curated (data/event_videos.json) and auto-registered
    (data/auto_event_videos.json) event-footage pools into one registry.

    Auto entries sit on top of the curated defaults (merge order
    defaults <- auto), matching the convention in config.py.
    """
    merged = _read_json(config.EVENT_VIDEOS_FILE)
    merged.update(_read_json(config.AUTO_EVENT_VIDEOS_FILE))
    return merged


def _distinctive_tokens(text: str) -> set[str]:
    """Lowercased content words of a label/sentence (stopwords removed)."""
    normalized = str(text).lower().replace("'", " ")
    return {t for t in _TOKEN_RE.findall(normalized)
            if len(t) >= 3 and t not in _STOPWORDS}


def _merge_entries(entries: list[dict]) -> dict | None:
    """Merge registry entries that describe the SAME event into one pool.

    Auto-registry has historically split one event across several keys (label
    variants from different query phrasings: "chrisean kai cenat hole in the
    wall" vs "chrisean makes hole in kai cenat's wall").  Each key pins part of
    the raw footage, so returning only the first substring match strands the
    correct raw VOD under an unmatched sibling key.  Merging by shared
    distinctive tokens unions every overlapping key's videos into one pool.
    """
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0]
    videos: list[dict] = []
    seen_ids: set[str] = set()
    channels: list[str] = []
    for entry in entries:
        for video in entry.get("videos") or []:
            vid = str(video.get("id") or "").strip()
            if vid and vid not in seen_ids:
                seen_ids.add(vid)
                videos.append(video)
        for channel in entry.get("channels") or []:
            ch = str(channel).strip()
            if ch and ch.lower() not in {str(c).lower() for c in channels}:
                channels.append(ch)
    label = next((str(e.get("label") or "") for e in entries if e.get("label")), None)
    merged = {"label": label or entries[0].get("label") or entries[0].get("name") or "",
              "channels": channels,
              "videos": videos,
              "auto": any(e.get("auto") for e in entries),
              "_merged": True}
    logger.info("Event registry: merged %d overlapping entr(y/ies) -> %d video(s)",
                len(entries), len(videos))
    return merged


def match(plan: dict) -> dict | None:
    """Return the registry entry (or merged pool) for the query's event."""
    data = _load()
    if not data:
        return None
    sentence = str(plan.get("sentence") or plan.get("query") or "").lower()
    sentence_tokens = {t for t in _distinctive_tokens(sentence)
                       if t not in _LOW_INFO}
    matched: list[dict] = []
    for name, entry in data.items():
        label = str(entry.get("label") or name).lower()
        if name in sentence or (label and label in sentence):
            matched.append(entry)
            continue
        # Overlapping key for the same event (different label phrasing): merge
        # it in when it shares at least three distinctive content words with the
        # sentence (people + event words), so the raw VOD under the sibling key
        # enters the pool instead of being stranded.  Generic action/stream
        # words are ignored so distinct events that merely share a person + a
        # verb ("fousey fights bike owner" vs "fousey fights neon", or a shared
        # person pair like "fousey fights neon in jack's stream" vs "fousey
        # throws water on jack") stay apart.
        if sentence_tokens:
            label_tokens = {t for t in _distinctive_tokens(label)
                            if t not in _LOW_INFO}
            if len(sentence_tokens & label_tokens) >= 3:
                matched.append(entry)
    if matched:
        return _merge_entries(matched)
    # Event queries (planner flagged candidate streamers) that did not name the
    # event verbatim: use the registry's single event when there is exactly one.
    if plan.get("candidate_streamers"):
        if len(data) == 1:
            return next(iter(data.values()))
        logger.info("Multiple registry events; no sentence match for %r",
                    sentence)
    return None


def channel_allowlist(entry: dict) -> set[str] | None:
    """Lowercased exact channel names that ARE the event (official + participants)."""
    channels = {str(c).strip().lower() for c in entry.get("channels") or []
                if str(c).strip()}
    return channels or None


def candidates(entry: dict, plan: dict) -> list[dict]:
    """Resolve each pinned video's metadata into a pool candidate.

    Pinned videos are the event's own footage, so they bypass source rejection
    (``event_pinned``) and get the short-clip / exact-window treatment
    (``_exact_pool``) so short viral POV clips export whole and long broadcasts
    are sectioned to 30s windows.

    ``event_pinned`` only skips *rejection* for the event's trusted raw
    channels: a pinned video that resolves to a known reactor / aggregator /
    clip-shop channel and is NOT explicitly allowlisted in the entry's
    ``channels`` is dropped here (auto-registry has pinned such shops in the
    past, and the raw footage always lives on the trusted channels instead).
    """
    allowlist = channel_allowlist(entry)
    pool: list[dict] = []
    for item in entry.get("videos") or []:
        vid = str(item.get("id") or "").strip()
        if not vid:
            continue
        url = f"https://www.youtube.com/watch?v={vid}"
        info = downloader.get_video_metadata(url)
        if not info:
            logger.warning("Registry video %s unresolvable; skipping", url)
            continue
        if info.get("is_live") or info.get("live_status") == "is_live":
            logger.info("Registry video %s is currently live; skipping", url)
            continue
        channel = (info.get("uploader") or info.get("channel") or "").strip()
        lowered_channel = channel.lower()
        trusted = allowlist and lowered_channel in allowlist
        if not trusted and lowered_channel and (
                any(marker in lowered_channel for marker in config.REACTOR_CHANNEL_MARKERS)
                or any(marker in lowered_channel for marker in config.AGGREGATOR_CHANNEL_MARKERS)):
            logger.info("Registry video %s is on clip/reactor channel %r "
                        "(not allowlisted); skipping", url, channel)
            continue
        candidate = {
            "id": vid,
            "video_id": vid,
            "title": info.get("title") or item.get("title") or "",
            "description": info.get("description") or "",
            "channel": info.get("uploader") or info.get("channel") or "",
            "webpage_url": url,
            "duration": float(info.get("duration") or 0),
            "view_count": float(info.get("view_count") or 0),
            "upload_date": info.get("upload_date") or "",
            "source": "youtube",
            "channel_pinned": True,
            "event_pinned": True,
            "_info": info,
            "_exact_pool": True,
            "search_score": 1.0,
            "rawness_score": 1.0,
            "title_match_score": 0.5,
            "title": info.get("title") or item.get("title") or "",
        }
        pool.append(candidate)
    logger.info("Event registry pool: %d video(s)", len(pool))
    for video in pool:
        logger.info("  - [%s] %s %s", video.get("channel") or "?", video["title"], url)
    return pool
