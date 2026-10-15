"""Twitch-first lane: clips from the subject streamer's own channel.

Unlike the old pipeline, every candidate clip is downloaded (short audio),
transcribed with Whisper, and content-verified against the sentence before it
can be selected.  Title/views only rank which clips get verified first.
"""
import json
import logging
import math
import subprocess
import time
from pathlib import Path

import config
from modules import (audio_transcriber, content_verifier, event_vision,
                     moment_finder, streamers, text_match, twitch_source)
from modules.scoring import title_match_score
from modules.youtube_search import sample_frames_at

try:
    import requests
except Exception:
    requests = None

try:
    import yt_dlp
except Exception:
    yt_dlp = None

logger = logging.getLogger(__name__)

_MIN_SECONDS = config.CLIP_MIN_SECONDS
_MAX_SECONDS = config.CLIP_MAX_SECONDS

_CLIPS_QUERY = """query {
  user(login: "%(handle)s") {
    clips(first: 100%(after)s) {
      edges { node { slug title durationSeconds viewCount createdAt thumbnailURL } }
      pageInfo { hasNextPage endCursor }
    }
  }
}"""


# --------------------------------------------------------------------------
# Clip discovery
# --------------------------------------------------------------------------
def _fetch_clips(handle: str, sentence: str = "",
                 queries: list[str] | None = None) -> list[dict]:
    """Fetch ALL clips from the streamer's OWN channel.

    Twitch removed the global clip keyword search, so discovery is the full
    channel clip list via cursor-based pagination.  We fetch as many pages as
    possible (config.TWITCH_CLIP_PAGES) so old clips are not missed.
    No title-based ranking is applied — the caller ranks by thumbnail visual
    match instead.
    """
    cursor = None
    clips: dict[str, dict] = {}
    for page in range(config.TWITCH_CLIP_PAGES):
        after = f', after: "{cursor}"' if cursor else ""
        payload = {"query": _CLIPS_QUERY % {"handle": handle, "after": after}}
        try:
            result = twitch_source._gql(payload)
        except Exception as exc:
            logger.warning("Twitch clips fetch failed for %s: %s", handle, exc)
            break
        user = (result.get("data") or {}).get("user")
        if user is None:
            logger.info("No Twitch channel '%s' (or clips unavailable)", handle)
            return []
        data = user.get("clips") or {}
        edges = data.get("edges") or []
        for edge in edges:
            node = edge.get("node") or {}
            if node.get("slug"):
                clips[node["slug"]] = node
        page_info = data.get("pageInfo") or {}
        if not (page_info.get("hasNextPage") and page_info.get("endCursor")):
            break
        cursor = page_info["endCursor"]
    logger.info("Twitch lane for %s: %d clips fetched across %d pages",
                handle, len(clips), page + 1)
    return list(clips.values())


def _title_match(query: str, queries: list[str], title: str) -> float:
    best = title_match_score(query, title, "")
    for q in queries:
        best = max(best, title_match_score(q, title, ""))
    return best


def _rank_candidates(nodes: list[dict], sentence: str, queries: list[str],
                     plan: dict) -> list[dict]:
    candidates = []
    for node in nodes:
        duration = float(node.get("durationSeconds") or 0)
        if duration < _MIN_SECONDS or duration > _MAX_SECONDS:
            continue
        title = node.get("title") or ""
        match = _title_match(sentence, queries, title)
        views = float(node.get("viewCount") or 0)
        boost = 0.10 * min(1.0, math.log10(max(1.0, views)) / 9.0)
        recency = 0.04 if _recent(node.get("createdAt")) else 0.0
        action_hits, _ = content_verifier.action_match(title, plan)
        entity_hits, _ = content_verifier.entity_match(title, plan)
        object_hits, _ = content_verifier.object_match(title, plan)
        action_boost = 0.30 * min(1.0, action_hits / max(1, len(plan.get("action_terms") or [])))
        entity_boost = 0.15 * min(1.0, entity_hits / max(1, len(plan.get("core_entities") or [])))
        object_boost = 0.15 * min(1.0, object_hits / max(1, len(plan.get("object_terms") or [])))
        theme_sim = text_match.score(sentence, title)
        theme_boost = 0.15 * theme_sim
        slug = node["slug"]
        candidates.append({
            "id": slug,
            "video_id": slug,
            "title": title,
            "channel": "",
            "thumbnail": node.get("thumbnailURL") or "",
            "webpage_url": f"https://clips.twitch.tv/{slug}",
            "duration": duration,
            "start": 0.0,
            "view_count": views,
            "timestamp": node.get("createdAt"),
            "source": "twitch",
            "title_match_score": match,
            "theme_sim": theme_sim,
            "action_hits": action_hits,
            "entity_hits": entity_hits,
            "object_hits": object_hits,
            "rawness_score": 0.9,
            "clip_score": (0.30 * match + 0.25 * action_boost + 0.15 * entity_boost
                           + 0.15 * object_boost + theme_boost + boost + recency),
        })
    candidates.sort(key=lambda c: c["clip_score"], reverse=True)
    return candidates


def _recent(created_at) -> bool:
    try:
        age_days = (time.time() - float(created_at)) / 86400.0
        return age_days <= 365
    except Exception:
        return False


# --------------------------------------------------------------------------
# Download + transcribe (verified content)
# --------------------------------------------------------------------------
def _clip_cache_path(slug: str) -> Path:
    return Path(config.CLIPS_DIR) / ".cache" / f"{slug}.mp4"


def _file_decodes(path: Path) -> bool:
    """True when ffmpeg decodes a frame near the start AND mid-file.

    Stub mp4s (moov + init bytes, no real media) pass size and ffprobe checks
    but cannot be decoded past the first instant; such files are useless for
    transcription and verification, so they are purged rather than reused.
    """
    try:
        for timestamp in (0.5, max(1.0, min(30.0, float(path.stat().st_size) / 500000.0))):
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-ss", str(timestamp), "-i", str(path), "-frames:v", "1",
                 "-f", "null", "-"],
                capture_output=True, text=True)
            if result.returncode != 0:
                return False
    except Exception:
        return False
    return True


def _download_clip(clip: dict) -> Path | None:
    if yt_dlp is None:
        return None
    path = _clip_cache_path(clip["video_id"])
    if path.exists() and path.stat().st_size > 100_000:
        if _file_decodes(path):
            return path
        logger.info("twitch %s: cached clip undecodable; re-downloading",
                    clip["video_id"])
        path.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    try:
        class _TooBig(Exception):
            pass

        def _abort_hook(d: dict):
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total > 300 * 1024 * 1024:
                raise _TooBig()

        options = {
            "quiet": True, "no_warnings": True, "noplaylist": True,
            "format": "best", "outtmpl": str(tmp),
            "socket_timeout": 60, "progress_hooks": [_abort_hook],
        }
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([clip["webpage_url"]])
        if not tmp.exists() or tmp.stat().st_size < 100_000:
            tmp.unlink(missing_ok=True)
            return None
        if not _file_decodes(tmp):
            logger.info("twitch %s: downloaded clip undecodable; discarding",
                        clip["video_id"])
            tmp.unlink(missing_ok=True)
            return None
        tmp.replace(path)
        return path
    except Exception as exc:
        logger.info("Twitch clip download failed for %s: %s", clip["video_id"], exc)
        tmp.unlink(missing_ok=True)
        return None


def _transcript_cache_path(slug: str) -> Path:
    return config.TWITCH_CLIP_CACHE / f"{slug}.json"


def _transcribe_clip(clip: dict, path: Path) -> list[dict]:
    cache = _transcript_cache_path(clip["video_id"])
    if cache.exists():
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if payload.get("title") == clip.get("title") and payload.get("segments"):
                return payload["segments"]
        except Exception:
            pass
    segments = audio_transcriber.transcribe_file(
        str(path), model_size=config.WHISPER_MODEL_SIZE)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"title": clip.get("title"),
                                 "segments": segments, "saved_at": time.time()},
                                ensure_ascii=False), encoding="utf-8")
    logger.info("Twitch clip %s transcribed: %d segments",
                clip["video_id"], len(segments))
    return segments


# --------------------------------------------------------------------------
# Lane entry point
# --------------------------------------------------------------------------
def _thumbnail_probe(clip: dict, probe, anchors=None) -> None:
    """Score a clip's thumbnail against the event's CLIP scenes + reference anchors.

    The thumbnail frequently already shows the object (a police car, the
    helicopter), so a matching thumbnail ranks the clip for verification first
    even when its title is vague.

    When visual anchors are strong, also scores the thumbnail against the
    Google Images reference photos via anchors.frame_sims() — this is the
    primary visual signal for clip selection.
    """
    url = clip.get("thumbnail") or ""
    if not url or requests is None:
        return
    try:
        response = requests.get(url, timeout=config.IMAGE_NET_TIMEOUT)
        if response.status_code != 200 or not response.content:
            return
        path = config.FRAMES_DIR / f"twthumb_{clip.get('video_id', 'x')}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        # Scene-based scoring (existing behavior)
        scores = probe.scene_scores([str(path)]) if probe.active() else {}
        if scores:
            summary = probe.summarize(scores)
            clip["thumbnail_best"] = summary["best"]
            clip["thumbnail_fraction"] = summary["fraction"]
            clip["thumbnail_present"] = summary["present"]
            clip["event_visual_score"] = (1.0 if summary["present"]
                                           else summary["fraction"])
        # Anchor-based scoring (Google Images reference match)
        if anchors and getattr(anchors, "strong", False):
            anchor_sims = anchors.frame_sims_strict([str(path)])
            if anchor_sims:
                clip["thumbnail_anchor_sim"] = max(anchor_sims)
                # Boost event_visual_score when anchors agree
                anchor_boost = max(anchor_sims) * 0.3
                clip["event_visual_score"] = max(
                    float(clip.get("event_visual_score") or 0.0),
                    anchor_boost,
                )
    except Exception as exc:
        logger.info("Thumbnail probe failed for %s: %s", clip["video_id"], exc)


def run(sentence: str, plan: dict, need: int,
        anchors=None) -> list[dict]:
    """Return thumbnail-verified whole Twitch clips for the sentence (possibly 0).

    Every available clip's thumbnail is CLIP-scored against the Google Images
    reference anchors.  Clips whose thumbnail visually matches the event are
    selected directly — no transcript verification needed since Twitch clips
    are already short (15-240s) and the thumbnail proves the visual content.
    The export-time VLM and anchor gates on the downloaded clip act as safety.
    """
    streamer = plan.get("streamer") if isinstance(plan, dict) else None
    handle = streamers.twitch_handle(streamer)
    if not handle:
        return []
    nodes = _fetch_clips(handle)
    if not nodes:
        return []
    # Convert raw Twitch API nodes to proper candidate dicts with all fields
    # downstream code expects (video_id, start, duration, source, etc.).
    pool = []
    for n in nodes:
        if not (_MIN_SECONDS <= float(n.get("durationSeconds") or 0) <= _MAX_SECONDS):
            continue
        slug = n["slug"]
        pool.append({
            "id": slug,
            "video_id": slug,
            "title": n.get("title") or "",
            "channel": handle,
            "thumbnail": n.get("thumbnailURL") or "",
            "webpage_url": f"https://clips.twitch.tv/{slug}",
            "duration": float(n.get("durationSeconds") or 0),
            "start": 0.0,
            "view_count": float(n.get("viewCount") or 0),
            "timestamp": n.get("createdAt"),
            "source": "twitch",
            "query": sentence,
            "whole": True,
        })
    if not pool:
        logger.info("Twitch lane for '%s': no clips in duration range", sentence)
        return []
    probe = event_vision.probe_for(plan)
    has_anchors = anchors is not None and getattr(anchors, "strong", False)

    # Thumbnail probe EVERY clip: CLIP-score each thumbnail against the
    # Google Images reference anchors.  This is the primary selection signal —
    # the thumbnail that visually matches the event IS the right clip.
    # Probing is cheap (~50KB image download + CLIP forward pass per clip).
    for clip in pool:
        _thumbnail_probe(clip, probe, anchors)

    # Rank by visual match: anchor CLIP sim > scene present > clip_score.
    # Clips that visually match the reference anchors rise to the top
    # regardless of title — a generic-titled clip with a matching thumbnail
    # beats a perfectly-titled clip with a wrong thumbnail.
    pool.sort(key=lambda c: (
        float(c.get("thumbnail_anchor_sim") or 0.0),
        1.0 if c.get("thumbnail_present") else 0.0,
        c.get("clip_score", 0.0),
    ), reverse=True)

    # Select clips that pass the anchor threshold.  These clips' thumbnails
    # CLIP-matched the Google reference images — strong enough proof for short
    # clips.  No transcript download/verification needed.
    threshold = config.ANCHOR_SIM_FLOOR if has_anchors else 0.0
    verified: list[dict] = []
    for clip in pool:
        if len(verified) >= max(need, 3):
            break
        anchor_sim = float(clip.get("thumbnail_anchor_sim") or 0.0)
        has_scene = bool(clip.get("thumbnail_present"))
        if has_anchors:
            # With anchors: require thumbnail to match reference images
            if anchor_sim + 1e-6 < threshold:
                continue
        elif not has_scene:
            # Without anchors: require at least scene-probe match
            continue
        verified.append(clip)

    # If no clips passed the threshold, take the top-ranked ones anyway
    # (the exporter's VLM + anchor gates will catch wrong clips).
    if not verified and pool:
        logger.info("Twitch lane: no clips passed anchor threshold; "
                    "returning top %d by visual score for exporter gate",
                    min(3, len(pool)))
        verified = pool[:min(3, len(pool))]

    logger.info("Twitch lane for '%s': %d/%d clips thumbnail-verified",
                sentence, len(verified), len(pool))
    return verified


def _anchor_gate(clip: dict, path: Path, anchors, plan: dict | None = None) -> bool:
    """Pre-export visual filter for Twitch clips.

    For visual-event plans the EventProbe scene match is the primary evidence
    and a strong-anchor match is an independent alternative; a clip that fails
    BOTH is dropped early (the exporter re-checks strictly after download).
    Non-visual plans keep the old anchor-only behavior.
    """
    probe = event_vision.probe_for(plan)
    if probe.active():
        duration = float(clip.get("duration") or 0)
        count = config.EVENT_VISUAL_FRAMES
        if duration > 0 and count > 0:
            timestamps = [duration * (i + 1) / (count + 1) for i in range(count)]
            frames_dir = config.FRAMES_DIR / f"twitch_clip_{clip.get('video_id', 'x')}"
            frames = sample_frames_at(str(path), timestamps, frames_dir)
            if frames:
                scores = probe.scene_scores([f["path"] for f in frames])
                summary = probe.summarize(scores)
                clip["event_visual_score"] = (1.0 if summary["present"]
                                              else summary["fraction"])
                clip["event_visual_best"] = summary["best"]
                clip["event_visual_fraction"] = summary["fraction"]
                if summary["present"]:
                    logger.info("twitch %s event-probe scene match (frac=%.2f)",
                                clip["video_id"], summary["fraction"])
                    return True
    if anchors is None or not getattr(anchors, "strong", False):
        return True
    if not getattr(anchors, "enforceable", False):
        logger.info("twitch %s: anchors not event-specific; skipping anchor gate",
                    clip["video_id"])
        return True
    duration = float(clip.get("duration") or 0)
    count = config.ANCHOR_FRAMES_PER_WINDOW
    if duration <= 0 or count <= 0:
        return False
    timestamps = [duration * (i + 1) / (count + 1) for i in range(count)]
    frames_dir = config.FRAMES_DIR / f"twitch_clip_{clip.get('video_id', 'x')}"
    frames = sample_frames_at(str(path), timestamps, frames_dir)
    if not frames:
        logger.info("twitch %s: no frames sampled; skipping anchor gate",
                    clip["video_id"])
        return True
    anchor_sim = anchors.anchor_sim([frame["path"] for frame in frames])
    clip["anchor_sim"] = anchor_sim
    passes = anchor_sim + 1e-6 >= config.ANCHOR_SIM_FLOOR
    logger.info("twitch %s anchor_sim=%.2f (floor=%.2f) -> %s",
                clip["video_id"], anchor_sim, config.ANCHOR_SIM_FLOOR,
                "match" if passes else "no-match")
    if not passes and probe.active():
        logger.info("twitch %s: no probe match and no anchor match -> reject",
                    clip["video_id"])
        return False
    if not passes and not config.ANCHOR_GATE_STRICT:
        clip["anchor_failed"] = True
    return passes or not config.ANCHOR_GATE_STRICT


def _score_llm_moment(clip: dict, sentence: str, plan: dict,
                      transcript: list[dict]) -> None:
    """Attach the LLM moment-finder signal so exact-moment clips rank above
    merely on-topic ones.  The whole short clip is one cheap LLM read."""
    clip.setdefault("llm_moment_score", 0.0)
    clip.setdefault("llm_found", False)
    try:
        moments = moment_finder.find_moments(transcript, sentence, plan,
                                             video_id=clip["video_id"])
    except Exception as exc:
        logger.warning("moment_finder failed for twitch %s: %s", clip["video_id"], exc)
        return
    if not moments:
        return
    best = max(moments, key=lambda m: m.get("confidence", 0.0))
    confidence = float(best.get("confidence", 0.0))
    clip["llm_moment_score"] = confidence
    clip["llm_moment_reason"] = best.get("reason", "")
    full_text = content_verifier.transcript_text(transcript)
    if moment_finder.moment_is_grounded(best, full_text, plan):
        clip["llm_found"] = True
        logger.info("twitch %s llm_moment=%.2f (%s)", clip["video_id"], confidence,
                    best.get("reason", ""))
