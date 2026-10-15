"""Visual-first discovery: thumbnail CLIP scoring + stream frame-seek moment-finding.

Two-layer visual ranking using Google Images reference photos:

Layer 1 — Thumbnail CLIP (pre-metadata):
    Download YouTube thumbnails from img.youtube.com, CLIP-score against the
    query text.  Re-ranks the search pool so visually-matching videos get
    analysed first.  Costs ~100ms per thumbnail, no metadata needed.

Layer 2 — Stream frame-seek (post-metadata):
    Sample frames from the video stream via ffmpeg -ss (no full download),
    CLIP-score each against the Google Images reference anchors.  The
    timestamp with highest visual match = the likely moment.  Only that
    window is downloaded and verified.  Costs ~10s per video (20 frames).
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

import config
from modules import vision_ranker, youtube_search

logger = logging.getLogger(__name__)

_YOUTUBE_THUMB_URL = "https://img.youtube.com/vi/{vid}/{res}.jpg"
_THUMB_RESOLUTIONS = ("maxresdefault", "hqdefault", "mqdefault")


# --------------------------------------------------------------------------
# Layer 1: Thumbnail CLIP scoring (pre-metadata, cheap)
# --------------------------------------------------------------------------
def score_youtube_pool(candidates: list[dict], anchors, sentence: str) -> list[dict]:
    """Download YouTube thumbnails and CLIP-score each against the sentence.

    Adds ``thumbnail_score`` (0..1) to every candidate dict.  When visual
    anchors are strong, scores image-to-image against the anchor embeddings;
    otherwise falls back to text-to-image CLIP against the raw sentence.

    Returns candidates sorted by thumbnail_score descending.
    """
    if not candidates:
        return candidates
    paths = []
    valid = []
    for c in candidates:
        vid = c.get("video_id") or c.get("id")
        if not vid:
            continue
        thumb_path = _download_youtube_thumbnail(vid)
        if thumb_path:
            paths.append(str(thumb_path))
            valid.append(c)
        else:
            c["thumbnail_score"] = 0.0
    if not paths:
        return candidates
    if anchors and getattr(anchors, "strong", False):
        sims = anchors.frame_sims(paths)
        for c, sim in zip(valid, sims):
            c["thumbnail_score"] = sim
    else:
        scores = vision_ranker.image_query_scores(paths, sentence)
        for c, score in zip(valid, scores):
            c["thumbnail_score"] = score if score else 0.0
    scored = [c for c in candidates if "thumbnail_score" in c]
    unscored = [c for c in candidates if "thumbnail_score" not in c]
    for c in unscored:
        c["thumbnail_score"] = 0.0
    scored.sort(key=lambda c: c.get("thumbnail_score", 0.0), reverse=True)
    logger.info("Thumbnail CLIP scored %d YouTube candidates", len(scored))
    return scored + unscored


def _download_youtube_thumbnail(video_id: str) -> Path | None:
    """Download a YouTube thumbnail for the given video ID.

    Tries maxresdefault -> hqdefault -> mqdefault.  Returns the path on
    success or None on failure.
    """
    out_dir = config.FRAMES_DIR / "yt_thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{video_id}.jpg"
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    for res in _THUMB_RESOLUTIONS:
        url = _YOUTUBE_THUMB_URL.format(vid=video_id, res=res)
        try:
            resp = requests.get(url, timeout=config.VISUAL_THUMBNAIL_TIMEOUT,
                                stream=True)
            if resp.status_code == 200:
                content = resp.content
                if len(content) > 1024:
                    dest.write_bytes(content)
                    return dest
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------
# Layer 2: Stream frame-seek moment-finding (post-metadata, moderate cost)
# --------------------------------------------------------------------------
def visual_moment_localize(info: dict, anchors, sentence: str,
                           duration: float) -> list[dict]:
    """Coarse-to-fine frame scan to find the exact moment matching reference photos.

    Pass 1 (coarse): 100 frames spread across the entire video.  Find the
    120-second region with the highest mean anchor cosine.  This identifies
    WHERE the reference photos appear in the video.

    Pass 2 (fine): 30 frames within that region (1 per 4s).  Pinpoint the
    exact timestamp with the highest anchor cosine.

    Returns a list of candidate window dicts sorted by visual match score.
    Each dict has: start, score, frame_path.  Empty list on failure.

    For a 3-hour video this takes ~65s (vs ~10s for the old 20-frame scan)
    but raises the hit rate for a 30-second moment from ~0.3% to ~87.5%.
    """
    if not anchors or not getattr(anchors, "strong", False):
        return []
    if duration <= 0:
        return []
    stream_url = _get_stream_url(info)
    if not stream_url:
        return []
    video_id = str(info.get("id") or "")
    extra_args = youtube_search.stream_headers(info)
    out_dir = config.FRAMES_DIR / f"vmoment_{video_id}"

    # --- Pass 1: Coarse scan (40 frames across entire video) ---------------
    coarse_n = min(config.VISUAL_MOMENT_COARSE_FRAMES,
                   max(10, int(duration / 10)))
    coarse_stride = duration / (coarse_n + 1)
    coarse_ts = [coarse_stride * (i + 1) for i in range(coarse_n)]
    try:
        coarse_frames = youtube_search.sample_frames_at(
            stream_url, coarse_ts, out_dir, extra_args,
            skip_limit=3, video_id=video_id)
    except Exception as exc:
        logger.info("Visual moment coarse scan failed for %s: %s",
                    video_id, exc)
        return []
    if not coarse_frames:
        return []
    coarse_paths = [f["path"] for f in coarse_frames]
    coarse_sims = anchors.frame_sims_strict(coarse_paths)
    if not coarse_sims:
        return []

    # Find the best 120-second region via sliding window
    region_sec = config.VISUAL_MOMENT_REGION_SECONDS
    best_region_start = 0.0
    best_region_mean = 0.0
    for frame, sim in zip(coarse_frames, coarse_sims):
        center = frame["timestamp"]
        region_start = max(0.0, center - region_sec / 2)
        region_end = min(duration, region_start + region_sec)
        # Mean score of frames within this region
        region_scores = [
            sim2 for f2, sim2 in zip(coarse_frames, coarse_sims)
            if region_start <= f2["timestamp"] <= region_end
        ]
        if region_scores:
            mean_score = sum(region_scores) / len(region_scores)
            if mean_score > best_region_mean:
                best_region_mean = mean_score
                best_region_start = region_start

    logger.info("Coarse scan %s: best region @%.0fs-%.0fs (mean=%.3f, %d frames)",
                video_id, best_region_start,
                best_region_start + region_sec,
                best_region_mean, len(coarse_frames))

    if best_region_mean < config.VISUAL_MOMENT_COARSE_FLOOR:
        logger.info("Coarse scan %s: best mean %.3f below floor %.3f -> no visual match",
                    video_id, best_region_mean, config.VISUAL_MOMENT_COARSE_FLOOR)
        return []

    # --- Pass 2: Fine scan (30 frames within best region) -------------------
    fine_n = config.VISUAL_MOMENT_FINE_FRAMES
    fine_start = best_region_start
    fine_end = min(best_region_start + region_sec, duration)
    fine_span = fine_end - fine_start
    if fine_span <= 0:
        return []
    fine_stride = fine_span / (fine_n + 1)
    fine_ts = [fine_start + fine_stride * (i + 1) for i in range(fine_n)]
    fine_dir = out_dir / "fine"
    try:
        fine_frames = youtube_search.sample_frames_at(
            stream_url, fine_ts, fine_dir, extra_args,
            skip_limit=3, video_id=video_id)
    except Exception as exc:
        logger.info("Visual moment fine scan failed for %s: %s", video_id, exc)
        # Fall back to coarse results
        coarse_results = [
            {"start": f["timestamp"], "score": s, "frame_path": f["path"]}
            for f, s in zip(coarse_frames, coarse_sims)
        ]
        coarse_results.sort(key=lambda r: r["score"], reverse=True)
        return coarse_results[:5]
    if not fine_frames:
        return []
    fine_paths = [f["path"] for f in fine_frames]
    fine_sims = anchors.frame_sims_strict(fine_paths)
    if not fine_sims:
        return []

    # Build results from fine scan
    results = []
    for frame, sim in zip(fine_frames, fine_sims):
        results.append({
            "start": frame["timestamp"],
            "score": sim,
            "frame_path": frame["path"],
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0]
    logger.info("Fine scan %s: best=%.3f @%.0fs (%d frames in region %.0fs-%.0fs)",
                video_id, best["score"], best["start"],
                len(results), fine_start, fine_end)

    # --- Pass 3: Ultra-fine scan around top-3 fine moments ------------------
    # Pinpoint the exact frame with ~1s precision: scan 15 frames in ±15s
    # around each of the top-3 fine timestamps.  Disabled when
    # VISUAL_MOMENT_ULTRA_FRAMES=0 (too slow per video in visual-first flow).
    ultra_n = getattr(config, "VISUAL_MOMENT_ULTRA_FRAMES", 0)
    ultra_range = getattr(config, "VISUAL_MOMENT_ULTRA_RANGE", 15)
    ultra_ts = []
    ultra_map = {}  # timestamp -> source start for dedup
    for r in results[:3]:
        center = r["start"]
        lo = max(0.0, center - ultra_range)
        hi = min(duration, center + ultra_range)
        span = hi - lo
        if span <= 0:
            continue
        step = span / (ultra_n + 1)
        for i in range(ultra_n):
            ts = lo + step * (i + 1)
            if ts not in ultra_map:
                ultra_ts.append(ts)
                ultra_map[ts] = center
    if ultra_ts:
        ultra_dir = out_dir / "ultra"
        try:
            ultra_frames = youtube_search.sample_frames_at(
                stream_url, ultra_ts, ultra_dir, extra_args,
                skip_limit=3, video_id=video_id)
        except Exception as exc:
            logger.info("Ultra-fine scan failed for %s: %s", video_id, exc)
            ultra_frames = []
        if ultra_frames:
            ultra_paths = [f["path"] for f in ultra_frames]
            ultra_sims = anchors.frame_sims_strict(ultra_paths)
            if ultra_sims:
                for frame, sim in zip(ultra_frames, ultra_sims):
                    results.append({
                        "start": frame["timestamp"],
                        "score": sim,
                        "frame_path": frame["path"],
                    })
                results.sort(key=lambda r: r["score"], reverse=True)
                # Dedup: keep best per 3-second bucket
                deduped = []
                for r in results:
                    if all(abs(r["start"] - d["start"]) > 3.0 for d in deduped):
                        deduped.append(r)
                results = deduped
                best = results[0]
                logger.info("Ultra-fine scan %s: best=%.3f @%.0fs (%d moments)",
                            video_id, best["score"], best["start"], len(results))

    return results


def _get_stream_url(info: dict) -> str | None:
    """Get a seekable stream URL from yt-dlp info dict.

    Uses the progressive MP4 format that allows fast ffmpeg input-seeks.
    """
    try:
        fmt = youtube_search.get_frame_stream_format(info)
        if fmt and fmt.get("url"):
            return fmt["url"]
    except Exception as exc:
        logger.info("Stream URL extraction failed: %s", exc)
    return None


# --------------------------------------------------------------------------
# Combined: thumbnail + frame-seek scoring
# --------------------------------------------------------------------------
def score_and_localize(candidates: list[dict], anchors, sentence: str,
                       metadata_fn=None) -> list[dict]:
    """Full visual-first discovery pipeline.

    1. Thumbnail CLIP scores all candidates (cheap, pre-metadata).
    2. Sorts by visual match, takes top N.
    3. For top N: fetches metadata (if metadata_fn provided), then runs
       visual_moment_localize to find the exact timestamp.

    Returns candidates with added fields:
      - thumbnail_score: CLIP similarity of video thumbnail
      - visual_moment_start: best visual moment timestamp (if found)
      - visual_moment_score: CLIP similarity at that timestamp
      - visual_moment_frame: path to the best-matching frame

    Candidates are sorted by (visual_moment_score, thumbnail_score) descending.
    """
    if not candidates or not anchors:
        return candidates
    # Phase 1: thumbnail scoring (all candidates, no metadata needed)
    candidates = score_youtube_pool(candidates, anchors, sentence)
    # Phase 2: visual moment-finding on top candidates (needs metadata)
    top_n = candidates[: config.VISUAL_THUMBNAIL_TOP_N]
    for c in top_n:
        vid = c.get("video_id") or c.get("id")
        if not vid:
            continue
        info = c.get("_info")
        if not info and metadata_fn:
            try:
                info = metadata_fn(c.get("webpage_url", ""))
            except Exception:
                info = None
            if info:
                c["_info"] = info
        if not info:
            continue
        duration = float(info.get("duration") or c.get("duration") or 0)
        moments = visual_moment_localize(info, anchors, sentence, duration)
        if moments and moments[0]["score"] >= config.VISUAL_MOMENT_FLOOR:
            c["visual_moment_start"] = moments[0]["start"]
            c["visual_moment_score"] = moments[0]["score"]
            c["visual_moment_frame"] = moments[0].get("frame_path", "")
    candidates.sort(
        key=lambda c: (
            float(c.get("visual_moment_score") or 0.0),
            float(c.get("thumbnail_score") or 0.0),
        ),
        reverse=True,
    )
    return candidates
