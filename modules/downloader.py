"""YouTube discovery and metadata retrieval.  No media is downloaded here."""
import logging
import math
from typing import Any

try:
    import yt_dlp
except Exception:
    yt_dlp = None
    # Do not raise at import time; functions will detect and return helpful errors.

from modules.scoring import metadata_rawness_score, title_match_score
from modules.text_match import score as lexical_score
import config

logger = logging.getLogger(__name__)


def with_cookies(opts: dict) -> dict:
    """Attach the shared YouTube cookies.txt to yt-dlp options when it exists.

    YouTube gates metadata/downloads behind a 'Sign in to confirm you're not
    a bot' wall for cookie-less clients; supplying cookies bypasses it.
    Also forces ``noconfig`` so a user's global yt-dlp config cannot inject
    incompatible flags (e.g. a forced format like ``137/22/18``).
    Sets ``remote_components`` to enable YouTube JS challenge solving via deno.
    """
    opts.setdefault("noconfig", True)
    opts.setdefault("remote_components", "ejs:npm")
    cookie_file = getattr(config, "YT_COOKIES_FILE", None)
    if cookie_file is not None and cookie_file.is_file():
        opts["cookiefile"] = str(cookie_file)
    return opts


STREAM_SUFFIXES = ("stream", "live", "vod", "full stream", "live stream")
STREAM_HINTS = ("stream", "live", "vod", "full", "raw", "unedited", "irl", "twitch", "recording")


def _topic(query: str) -> str:
    """Content words of the query with filler/noise removed."""
    words = [word for word in query.split()
             if word.lower().strip(".,!?") not in _NOISE_TERMS]
    return " ".join(words)


_NOISE_TERMS = {
    "a", "an", "the", "who", "that", "with", "from", "for", "of", "in", "at",
    "to", "and", "or", "video", "clip",
    "what", "when", "how", "why", "did", "does", "is", "was", "this", "that's",
    "got",
}


def _query_variants(query: str) -> list[str]:
    """Build search variants: the plain query plus raw-stream hunting lane."""
    topic = _topic(query)
    variants = [query]
    if topic and topic.lower() != query.lower():
        variants.append(topic)
    for suffix in STREAM_SUFFIXES:
        variants.append(f"{topic} {suffix}")
    return list(dict.fromkeys(variant.strip() for variant in variants if variant.strip()))


def search_candidates(query: str, max_results: int = 15) -> list[dict[str, Any]]:
    if yt_dlp is None:
        logger.error("yt_dlp is not installed; install requirements to enable YouTube search")
        return []
    options = {"quiet": True, "no_warnings": True, "skip_download": True,
               "extract_flat": True, "noplaylist": True}
    videos_by_id: dict[str, dict] = {}
    per_variant = max(6, max_results // 2)
    with yt_dlp.YoutubeDL(with_cookies(options)) as ydl:
        for variant in _query_variants(query):
            try:
                result = ydl.extract_info(f"ytsearch{per_variant}:{variant}", download=False)
            except Exception as exc:
                logger.warning("YouTube search failed for %r: %s", variant, exc)
                continue
            entries = result.get("entries", []) or []
            logger.info("YouTube search %r returned %d results", variant, len(entries))
            for entry in entries:
                if not entry or not entry.get("id") or entry["id"] in videos_by_id:
                    continue
                title = entry.get("title") or ""
                description = entry.get("description") or ""
                channel = entry.get("uploader") or entry.get("channel") or ""
                videos_by_id[entry["id"]] = {
                    "id": entry["id"],
                    "title": title,
                    "description": description,
                    "channel": channel,
                    "webpage_url": entry.get("webpage_url") or f"https://www.youtube.com/watch?v={entry['id']}",
                    "upload_date": entry.get("upload_date") or "",
                    "metadata_score": lexical_score(query, f"{title} {description}"),
                    "rawness_score": metadata_rawness_score(title, description, channel),
                    "title_match_score": title_match_score(query, title, description),
                    "view_count": entry.get("view_count") or 0,
                }
    videos = list(videos_by_id.values())
    for video in videos:
        views = float(video.get("view_count") or 0)
        view_boost = 0.06 * min(1.0, math.log10(max(1.0, views)) / 9.0)
        video["search_score"] = (
            0.30 * video["metadata_score"]
            + 0.45 * video["title_match_score"]
            + 0.25 * video["rawness_score"]
            + view_boost
        )
    videos.sort(key=lambda item: item["search_score"], reverse=True)
    logger.info("Search returned %d unique candidates", len(videos))
    return videos


def search_candidates_exact(query: str, max_results: int = 30) -> list[dict[str, Any]]:
    """Search the query VERBATIM and keep the platform's own result order.

    Unlike ``search_candidates`` this runs the query as-is (no suffix/stream
    variant expansion) and does NOT re-rank by the metadata heuristic, so the
    video YouTube itself ranks #1 for the exact query stays #1.  This mirrors
    how a human finds the moment: search the sentence, check the top results.
    """
    if yt_dlp is None:
        logger.error("yt_dlp is not installed; install requirements to enable YouTube search")
        return []
    options = {"quiet": True, "no_warnings": True, "skip_download": True,
               "extract_flat": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(with_cookies(options)) as ydl:
        try:
            result = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
        except Exception as exc:
            logger.warning("Exact YouTube search failed for %r: %s", query, exc)
            return []
        entries = result.get("entries", []) or []
        logger.info("Exact YouTube search %r returned %d results", query, len(entries))
        videos: list[dict[str, Any]] = []
        for entry in entries:
            if not entry or not entry.get("id"):
                continue
            title = entry.get("title") or ""
            description = entry.get("description") or ""
            channel = entry.get("uploader") or entry.get("channel") or ""
            videos.append({
                "id": entry["id"],
                "title": title,
                "description": description,
                "channel": channel,
                "webpage_url": entry.get("webpage_url") or f"https://www.youtube.com/watch?v={entry['id']}",
                "upload_date": entry.get("upload_date") or "",
                "view_count": entry.get("view_count") or 0,
                "search_score": 1.0 - (0.01 * len(videos)),  # platform order, monotonically falling
            })
    return videos


def select_analysis_pool(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick the analysis pool: top generic candidates + a guaranteed rawness slice."""
    if not candidates:
        return []
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()

    generic = sorted(candidates, key=lambda item: item["search_score"], reverse=True)
    generic_slots = max(1, config.VIDEOS_TO_ANALYSE - config.RAWNESS_TOP_SLOTS)
    for candidate in generic[:generic_slots]:
        if candidate["id"] not in seen:
            pool.append(candidate)
            seen.add(candidate["id"])

    rawness = sorted(candidates, key=lambda item: item["rawness_score"], reverse=True)
    for candidate in rawness[: config.RAWNESS_TOP_SLOTS]:
        if candidate["id"] in seen or len(pool) >= config.VIDEOS_TO_ANALYSE:
            continue
        if float(candidate.get("title_match_score", 0.0)) < config.RAWNESS_SLOT_MIN_TITLE_MATCH:
            logger.info("Rawness slot skipped (no topical grounding): %s", candidate["title"])
            continue
        pool.append(candidate)
        seen.add(candidate["id"])

    logger.info("Analysis pool: %d videos", len(pool))
    for video in pool:
        logger.info("  - %s (rawness=%.2f, match=%.2f) %s",
                    video["title"], video["rawness_score"],
                    video["title_match_score"], video["webpage_url"])
    return pool


def get_video_metadata(url: str, *, flat: bool = False) -> dict[str, Any] | None:
    """Resolve metadata/subtitle URLs only; yt-dlp is never asked to download.

    With ``flat`` the extractor skips format resolution entirely (fast, works
    for Shorts where WEB-client format selection fails); the returned info
    carries id/title/duration/channel/webpage_url but no stream formats.
    """
    if yt_dlp is None:
        logger.error("yt_dlp is not installed; install requirements to enable metadata lookup")
        return None
    base = {"quiet": True, "no_warnings": True, "skip_download": True,
            "noplaylist": True, "writesubtitles": False, "writeautomaticsub": False}
    if flat:
        base["extract_flat"] = True
    # Try the default client list first: YouTube's WEB client currently
    # fails format selection for most videos ("Requested format is not
    # available"), wasting a full lookup per video.  WEB remains as the
    # fallback for videos the default clients cannot resolve.
    for clients in (None, ["web"]):
        options = base.copy()
        if clients is not None:
            options["extractor_args"] = {"youtube": {"player_client": clients}}
        try:
            with yt_dlp.YoutubeDL(with_cookies(options)) as ydl:
                info = ydl.extract_info(url, download=False)
            if info:
                return info
        except Exception as exc:
            logger.warning("Metadata lookup failed for %s (client=%s): %s",
                           url, clients, exc)
    return None
