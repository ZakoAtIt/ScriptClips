"""Kick VOD lane: the subject streamer's own Kick channel, HLS-window analysis.

Kick has no captions and no public search API, so discovery is the streamer's
own channel VOD list (``/api/v2/channels/{slug}/videos``) and the HLS source is
resolved from ``/api/v1/video/{uuid}``.  VODs are never fully downloaded: audio
chunks are extracted with ffmpeg input seeks over the HLS master (only the
needed segments are fetched), transcribed with Whisper, and the LLM moment
finder localizes the event.  Candidates are audio-verified exactly like the
YouTube lane windows.

The cache is resumable per chunk, so a crash re-does only the missing chunks.
"""
import json
import logging
import math
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import config
from modules import (audio_transcriber, content_verifier, event_vision,
                     moment_finder, moment_proposer, semantic_embedder,
                     streamers, vod_scan)
from modules.scoring import (evidence_editorial_score, metadata_rawness_score,
                             recap_score, title_match_score)
from modules.text_match import score as lexical_score
from modules.text_match import tokens
from modules.youtube_search import sample_frames_at

logger = logging.getLogger(__name__)

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


# --------------------------------------------------------------------------
# Kick API (no auth; Cloudflare-simple GETs)
# --------------------------------------------------------------------------
def _api_get(path: str) -> dict | list:
    request = Request(config.KICK_API_BASE + path, headers={
        "User-Agent": _UA,
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://kick.com/",
        "Origin": "https://kick.com",
    })
    with urlopen(request, timeout=40) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def list_channel_videos(slug: str) -> list[dict]:
    """VODs for a channel slug from the public list endpoint (newest first)."""
    try:
        data = _api_get(f"/v2/channels/{slug}/videos")
    except (HTTPError, URLError, OSError, ValueError) as exc:
        logger.warning("Kick VOD list failed for %s: %s", slug, exc)
        return []
    items = data if isinstance(data, list) else (data.get("data") or data.get("videos") or [])
    if not isinstance(items, list):
        return []
    vods: list[dict] = []
    for item in items[: config.KICK_CHANNEL_VIDEOS_LIMIT]:
        if not isinstance(item, dict):
            continue
        video = item.get("video") or {}
        uuid = video.get("uuid")
        vid = str(item.get("id") or video.get("id") or "")
        if not uuid:
            continue
        duration_ms = int(item.get("duration") or video.get("duration") or 0)
        vods.append({
            "id": vid,
            "uuid": uuid,
            "title": item.get("session_title") or "",
            "created_at": item.get("created_at") or "",
            "duration": duration_ms / 1000.0 if duration_ms else 0.0,
            "views": int(item.get("views") or 0),
            "language": item.get("language") or "",
            "slug": slug,
            "webpage_url": f"https://kick.com/{slug}/videos/{vid or uuid}",
        })
    logger.info("Kick channel %s: %d VODs listed", slug, len(vods))
    return vods


# --------------------------------------------------------------------------
# VOD selection
# --------------------------------------------------------------------------
def _title_match(sentence: str, queries: list[str], title: str) -> float:
    best = title_match_score(sentence, title, "")
    for query in queries:
        best = max(best, title_match_score(query, title, ""))
    return best


def _age_days(created_at: str) -> float:
    try:
        created = datetime.strptime(str(created_at)[:19], "%Y-%m-%d %H:%M:%S")
        return max(0.0, (datetime.now() - created).total_seconds() / 86400.0)
    except (TypeError, ValueError):
        return 0.0


def _rank_vods(vods: list[dict], sentence: str, queries: list[str]) -> list[dict]:
    for vod in vods:
        title = vod.get("title") or ""
        vod["title_match_score"] = _title_match(sentence, queries, title)
        vod["recap_score"] = recap_score(title, "")
        vod["rawness_score"] = metadata_rawness_score(title, "", vod.get("slug"))
        views = float(vod.get("views") or 0)
        view_boost = 0.08 * min(1.0, math.log10(max(1.0, views)) / 9.0)
        # IRL streams title themselves by theme ("TaKO TUESDAY IN COLOMBIA"),
        # never by the incident, so title-match is weak; recency dominates for
        # these recent events, views break ties for older ones.  Exponential
        # decay separates same-week VODs (linear is flat within ~8 days).
        age_days = _age_days(vod.get("created_at", ""))
        recency = 0.32 * math.exp(-age_days / 7.0)
        vod["search_score"] = (
            0.20 * vod["title_match_score"]
            + 0.15 * vod["rawness_score"]
            + view_boost
            + recency
            - 0.25 * vod["recap_score"]
        )
    vods.sort(key=lambda item: item["search_score"], reverse=True)
    return vods


def select_analysis(vods: list[dict], sentence: str, queries: list[str]) -> list[dict]:
    """Top, non-recap, correctly-sized VODs to transcribe.

    No query-token gate on long VODs: on a Kick channel the incident is never
    named in the title, so any recent stream could hold it.  Only clearly recap
    content and out-of-size VODs are skipped; correctness is left to the
    moment finder + audio verification (which reject irrelevant streams).
    """
    ranked = _rank_vods(vods, sentence, queries)
    chosen: list[dict] = []
    for vod in ranked:
        if vod.get("recap_score", 0.0) >= 0.7:
            continue
        duration = float(vod.get("duration") or 0)
        if not (config.MIN_VIDEO_LENGTH <= duration <= config.KICK_MAX_VOD_SECONDS):
            continue
        chosen.append(vod)
        if len(chosen) >= config.KICK_VODS_TO_ANALYSE:
            break
    logger.info("Kick analysis pool: %d VODs", len(chosen))
    for vod in chosen:
        logger.info("  - %s (match=%.2f recap=%.2f views=%s) %s",
                    (vod.get("title") or "")[:70], vod.get("title_match_score", 0.0),
                    vod.get("recap_score", 0.0), vod.get("views", 0), vod["webpage_url"])
    return chosen


# --------------------------------------------------------------------------
# VOD info + resumable chunked transcription
# --------------------------------------------------------------------------
def get_vod_info(vod: dict) -> dict:
    """Resolve the HLS master URL + duration for one Kick VOD."""
    data = _api_get(f"/v1/video/{vod['uuid']}")
    if not isinstance(data, dict):
        raise RuntimeError(f"Kick VOD info malformed for {vod.get('id')}")
    livestream = data.get("livestream") or {}
    duration_ms = int(data.get("duration") or livestream.get("duration") or vod.get("duration") or 0)
    return {
        "id": str(data.get("id") or vod.get("id") or ""),
        "uuid": vod.get("uuid"),
        "title": livestream.get("session_title") or vod.get("title") or "",
        "source": data.get("source") or "",
        "duration": duration_ms / 1000.0 if duration_ms else float(vod.get("duration") or 0),
    }


def _cache_path(vod_id: str) -> Path:
    return config.KICK_TRANSCRIPT_CACHE_DIR / f"{vod_id}.json"


def _load_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(path: Path, cache: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def transcribe_vod(info: dict, vod: dict, force: bool = False,
                   languages: list[str] | None = None) -> list[dict]:
    """Resumable chunked Whisper transcription of a Kick HLS VOD.

    Each chunk is a 30-minute ffmpeg input-seek over the HLS master (only the
    needed segments are fetched), transcribed (with the model best for the
    query's languages, see ``audio_transcriber.model_for``) and cached per
    chunk so interrupted runs resume without redoing finished chunks.
    """
    model_size = audio_transcriber.model_for(languages)
    vod_id = str(vod.get("id") or info.get("id") or vod.get("uuid") or "")
    duration = float(info.get("duration") or 0)
    hls_url = info.get("source") or ""
    if not hls_url or duration <= 0:
        logger.info("No usable HLS source for Kick VOD %s", vod_id)
        return []

    path = _cache_path(vod_id)
    cache = _load_cache(path)
    if cache.get("merged") and not force:
        return cache["segments"]

    chunk_len = config.KICK_TRANSCRIPT_CHUNK_SECONDS
    chunk_count = max(1, math.ceil(duration / chunk_len))
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
                    hls_url, str(wav), max_seconds=length, start_seconds=start,
                    timeout=config.KICK_CHUNK_EXTRACT_TIMEOUT)
            except Exception as exc:
                logger.warning("Kick chunk %d audio extract failed for %s: %s",
                               index, vod_id, exc)
                failed = True
                continue
            segments = audio_transcriber.transcribe_file(
                str(wav), model_size=model_size)
            for segment in segments:
                segment["start"] += start
                segment["end"] += start
            cache.setdefault("chunks", {})[key] = segments
            _save_cache(path, cache)
            done += 1
            logger.info("Kick VOD %s transcribed %d/%d chunks (%.0fs of %.0fs)",
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
        logger.warning("Kick VOD %s has %d transcribed chunk(s) but some failed; "
                       "a later run will retry the missing chunks",
                       vod_id, len(cache.get("chunks", {})))
    logger.info("Kick VOD %s transcribed: %d segments", vod_id, len(merged))
    return merged


# --------------------------------------------------------------------------
# Moment localization (LLM finder + heuristic windows)
# --------------------------------------------------------------------------
def _llm_windows(segments: list[dict], sentence: str, plan: dict, video_id: str,
                 duration: float, window_seconds: int) -> list[dict]:
    """Exact moments found by the LLM transcript reader, as window candidates."""
    try:
        moments = moment_finder.find_moments(
            segments, sentence, plan, video_id=video_id, duration=duration)
    except Exception as exc:
        logger.warning("moment_finder failed for Kick %s: %s", video_id, exc)
        return []
    windows: list[dict] = []
    for moment in moments:
        confidence = float(moment.get("confidence", 0.0))
        if confidence < config.MOMENT_FINDER_MIN_CONFIDENCE:
            continue
        start = max(0.0, float(moment.get("start", 0.0)))
        start = float(int(start))
        evidence = content_verifier._window_text(segments, start, window_seconds)
        if len(tokens(evidence)) < 4:
            logger.info("Kick moment @%.0fs has no audio in window %s", start, video_id)
            continue
        if not moment_finder.moment_is_grounded(moment, evidence, plan):
            logger.info("Kick dropping ungrounded moment @%.0fs in %s: %s",
                        start, video_id, moment.get("reason", ""))
            continue
        windows.append({
            "start": start,
            "duration": float(window_seconds),
            "evidence": evidence,
            "llm_found": True,
            "llm_moment_score": confidence,
            "llm_moment_reason": moment.get("reason", ""),
        })
    return windows


def _merge_windows(heuristic: list[dict], llm: list[dict], max_windows: int) -> list[dict]:
    combined = sorted(llm + heuristic,
                      key=lambda w: (bool(w.get("llm_found")),
                                     w.get("proposal_score", 0.0)),
                      reverse=True)
    kept: list[dict] = []
    for window in combined:
        if all(abs(window["start"] - item["start"]) >= config.MIN_CLIP_GAP_SECONDS
               for item in kept):
            kept.append(window)
        if len(kept) >= max_windows:
            break
    return kept


# --------------------------------------------------------------------------
# Per-VOD analysis
# --------------------------------------------------------------------------
def analyse_vod(vod: dict, sentence: str, plan: dict,
                info: dict | None = None, anchors=None) -> list[dict]:
    """Transcribe one Kick VOD, localize moments, and audio-verify candidates.

    ``anchors`` is the sentence's visual AnchorSet (or None); when it is strong,
    a window must have a sampled frame that CLIP-matches the anchors or it is
    rejected as a wrong event.
    """
    if info is None:
        info = get_vod_info(vod)
    video_id = str(info.get("id") or vod.get("id") or vod.get("uuid") or "")
    duration = float(info.get("duration") or 0)
    if duration < config.MIN_VIDEO_LENGTH or duration > config.KICK_MAX_VOD_SECONDS:
        logger.info("Skipping Kick VOD %s: unsuitable length %.0fs", video_id, duration)
        return []
    if not info.get("source"):
        logger.info("No HLS source for Kick VOD %s", video_id)
        return []

    languages = (plan.get("auto") or {}).get("languages") if isinstance(plan, dict) else None
    segments = transcribe_vod(info, vod, languages=languages)
    if not segments:
        logger.info("No transcript for Kick VOD %s", video_id)
        return []

    target = (plan.get("semantic_target") or sentence) if isinstance(plan, dict) else sentence
    window_seconds = int(plan.get("window_seconds") or config.CLIP_DURATION_SECONDS)
    max_windows = config.MAX_CANDIDATES_PER_VIDEO

    heuristic = moment_proposer.transcript_candidates(
        segments, target, duration, window_seconds)
    llm_windows = _llm_windows(segments, sentence, plan, video_id, duration,
                               window_seconds)
    for window in llm_windows:
        scores = semantic_embedder.similarity_scores(target, [window["evidence"]])
        semantic = float(scores[0]) if scores else 0.0
        action_hits, action_cov = content_verifier.action_match(window["evidence"], plan)
        confidence = float(window.get("llm_moment_score", 0.0))
        window["semantic_score"] = semantic
        window["action_coverage"] = action_cov
        window["proposal_score"] = 0.45 * confidence + 0.35 * semantic + 0.20 * action_cov
    windows = _merge_windows(heuristic, llm_windows, max_windows)
    probe = event_vision.probe_for(plan)
    if probe.active():
        # Visual event: captions may never mention the on-screen moment, and the
        # audio and video moments can be minutes apart.  Always coarsely scan
        # the VOD and CLIP-locate the strongest scene zones; only the pixels
        # decide where the event actually is.
        visual_windows = vod_scan.localize(
            info.get("source"), None, duration, probe, video_id, window_seconds,
            max_windows)
        if visual_windows:
            for window in visual_windows:
                window.setdefault("proposal_score", float(window.get("mean", 0.0)))
                window.setdefault("llm_found", False)
            windows = _merge_windows(windows, visual_windows, max_windows)
            logger.info("Kick VOD %s: visual scan added %d window(s) (total %d)",
                        video_id, len(visual_windows), len(windows))
    if not windows:
        return []

    title = info.get("title") or vod.get("title") or ""
    rawness = metadata_rawness_score(title, "", vod.get("slug"))
    title_match = _title_match(sentence, plan.get("search_queries") or [sentence], title)
    metadata = lexical_score(target, title)

    candidates: list[dict] = []
    for window in windows:
        start = float(window["start"])
        candidate = {
            "id": video_id,
            "video_id": video_id,
            "uuid": vod.get("uuid"),
            "duration": float(window_seconds),
            "start": start,
            "timestamp": start,
            "whole": False,
            "source": "kick",
            "hls_url": info.get("source"),
            "query": sentence,
            "title": title,
            "description": "",
            "channel": vod.get("slug") or "",
            "webpage_url": vod.get("webpage_url") or f"https://kick.com/video/{vod.get('uuid')}",
            "semantic_score": float(window.get("semantic_score", 0.0)),
            "action_hits": int(window.get("action_hits", 0)),
            "action_coverage": float(window.get("action_coverage", 0.0)),
            "entity_hits": int(window.get("entity_hits", 0)),
            "entity_coverage": float(window.get("entity_coverage", 0.0)),
            "lexical_score": float(window.get("lexical_score", 0.0)),
            "proposal_score": float(window.get("proposal_score", 0.0)),
            "evidence": window["evidence"],
            "snippet": window["evidence"],
            "llm_found": bool(window.get("llm_found", False)),
            "llm_moment_score": float(window.get("llm_moment_score", 0.0)),
            "llm_moment_reason": window.get("llm_moment_reason", ""),
            "editorial_score": evidence_editorial_score(window["evidence"]),
            "metadata_score": metadata,
            "title_match_score": title_match,
            "rawness_score": rawness,
            "visual_prompt": plan.get("visual_prompt") or sentence,
        }
        if probe.active():
            _event_probe_candidate(candidate, probe, info.get("source"), video_id)
        ok, reason = content_verifier.verify(candidate, target, plan, segments)
        candidate["verification_reason"] = reason
        logger.info("Kick window %s @%.0fs (sem=%.2f llm=%s) -> %s", video_id,
                    start, float(window.get("semantic_score", 0.0)),
                    bool(window.get("llm_found")), "OK" if ok else reason)
        if ok:
            if not _anchor_gate(candidate, info.get("source"), window_seconds,
                                video_id, anchors):
                logger.info("Kick window %s @%.0fs -> no visual anchor match",
                            video_id, start)
                continue
            candidates.append(candidate)
    logger.info("Kick VOD %s produced %d verified candidate(s)", video_id, len(candidates))
    return candidates


def _event_probe_candidate(candidate: dict, probe, hls_url: str,
                           video_id: str) -> None:
    """Score the window's frames against the event's CLIP scenes (ranking only).

    The strict pixel gate still runs at export time on the downloaded clip;
    this only lets windows whose frames show the event out-rank audio-only ones.
    """
    try:
        start = float(candidate.get("start") or 0)
        duration = float(candidate.get("duration") or config.CLIP_DURATION_SECONDS)
        if duration <= 0 or not hls_url:
            return
        count = config.EVENT_VISUAL_FRAMES
        timestamps = [start + duration * (i + 1) / (count + 1) for i in range(count)]
        frames_dir = config.FRAMES_DIR / f"kickprobe_{video_id}"
        frames = sample_frames_at(hls_url, timestamps, frames_dir)
        if not frames:
            return
        scores = probe.scene_scores([f["path"] for f in frames])
        if not scores:
            return
        summary = probe.summarize(scores)
        candidate["event_visual_score"] = (1.0 if summary["present"]
                                           else summary["fraction"])
        candidate["event_visual_best"] = summary["best"]
        candidate["event_visual_fraction"] = summary["fraction"]
    except Exception as exc:
        logger.info("Event-probe scoring failed for Kick %s: %s", video_id, exc)


def _anchor_gate(candidate: dict, hls_url: str, window_seconds: float,
                 video_id: str, anchors) -> bool:
    """Sample frames of a candidate window and reject it when no visual match.

    For visual-event plans the EventProbe scene match (already scored in
    ``_event_probe_candidate``) is the primary evidence and a strong-anchor
    match is an independent alternative; a window failing BOTH is dropped early
    (the exporter re-checks strictly after download).  Returns True when the
    window passes (or when there is no anchor set to enforce).
    """
    start = float(candidate.get("start") or 0)
    probe_used = candidate.get("event_visual_score") is not None
    if probe_used and candidate.get("event_visual_fraction", 0.0) >= \
            config.EVENT_VISUAL_FRACTION:
        logger.info("Kick window %s @%.0fs event-probe scene match -> PASS",
                    video_id, start)
        return True
    if anchors is None or not getattr(anchors, "strong", False):
        return True
    if not getattr(anchors, "enforceable", False):
        logger.info("Kick window %s @%.0fs: anchors not event-specific; "
                    "skipping anchor gate", video_id, start)
        return True
    count = config.ANCHOR_FRAMES_PER_WINDOW
    if window_seconds <= 0 or count <= 0 or not hls_url:
        return False
    timestamps = [start + window_seconds * (i + 1) / (count + 1)
                  for i in range(count)]
    frames_dir = config.FRAMES_DIR / f"kick_{video_id}"
    frames = sample_frames_at(hls_url, timestamps, frames_dir)
    if not frames:
        logger.info("Kick window %s @%.0fs: no frames sampled; skipping anchor gate",
                    video_id, start)
        return True
    anchor_sim = anchors.anchor_sim([frame["path"] for frame in frames])
    candidate["anchor_sim"] = anchor_sim
    passes = anchor_sim + 1e-6 >= config.ANCHOR_SIM_FLOOR
    logger.info("Kick window %s @%.0fs anchor_sim=%.2f (floor=%.2f) -> %s",
                video_id, start, anchor_sim, config.ANCHOR_SIM_FLOOR,
                "match" if passes else "no-match")
    if not passes and probe_used:
        logger.info("Kick window %s @%.0fs: no probe match and no anchor match "
                    "-> reject", video_id, start)
        return False
    if not passes and not config.ANCHOR_GATE_STRICT:
        candidate["anchor_failed"] = True
    return passes or not config.ANCHOR_GATE_STRICT


# --------------------------------------------------------------------------
# Lane entry point
# --------------------------------------------------------------------------
def run(sentence: str, plan: dict, need: int,
        anchors=None) -> list[dict]:
    """Content-verified Kick VOD windows for the sentence (possibly 0).

    ``anchors`` is the sentence's visual AnchorSet (or None) used to gate
    candidate windows on the event's actual pixels.
    """
    streamer = plan.get("streamer") if isinstance(plan, dict) else None
    slug = streamers.kick_handle(streamer)
    if not slug:
        return []
    vods = list_channel_videos(slug)
    if not vods:
        return []
    queries = plan.get("search_queries") or [sentence]
    pool = select_analysis(vods, sentence, queries)
    candidates: list[dict] = []
    for vod in pool:
        try:
            candidates.extend(analyse_vod(vod, sentence, plan, anchors=anchors))
        except Exception as exc:
            logger.warning("Error analysing Kick VOD %s: %s",
                           vod.get("webpage_url"), exc)
    candidates.sort(key=lambda c: (c.get("proposal_score", 0.0),
                                   c.get("semantic_score", 0.0)), reverse=True)
    logger.info("Kick lane for '%s' returned %d verified candidate(s)",
                sentence, len(candidates))
    return candidates
