"""Twitch raw-VOD lane: the subject streamer's own official VODs.

Visual events (a helicopter, a police-car escort, a gel-blaster hit, a fight)
live in the raw stream.  Twitch's official VODs are the most reliable source
for them: this lane searches the streamer's own channel VODs, transcribes them
in resumable 30-minute chunks (reusing ``twitch_source.transcribe_vod``), runs
the same window proposals + LLM moment finder + EventProbe visual localization
as the Kick lane, and content-verifies every candidate window.  The export-time
pixel gate then confirms the event is actually on screen.
"""
import logging

import config
from modules import (content_verifier, event_vision, moment_finder,
                     moment_proposer, semantic_embedder, streamers,
                     twitch_source, vod_scan)
from modules.scoring import (evidence_editorial_score, metadata_rawness_score,
                             title_match_score)
from modules.text_match import score as lexical_score
from modules.text_match import tokens
from modules.youtube_search import sample_frames_at, stream_headers

logger = logging.getLogger(__name__)


def _llm_windows(segments: list[dict], sentence: str, plan: dict, video_id: str,
                 duration: float, window_seconds: int) -> list[dict]:
    """Exact moments found by the LLM transcript reader, as window candidates."""
    try:
        moments = moment_finder.find_moments(
            segments, sentence, plan, video_id=video_id, duration=duration)
    except Exception as exc:
        logger.warning("moment_finder failed for Twitch VOD %s: %s", video_id, exc)
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
            continue
        if not moment_finder.moment_is_grounded(moment, evidence, plan):
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


def _merge_windows(heuristic: list[dict], llm: list[dict],
                   max_windows: int) -> list[dict]:
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


def _event_probe_candidate(candidate: dict, probe, stream_url: str,
                           extra_args: list[str], video_id: str) -> None:
    """Score the window's frames against the event's CLIP scenes (ranking only).

    The strict pixel gate still runs at export time on the downloaded clip;
    this only lets windows whose frames show the event out-rank audio-only ones.
    """
    try:
        start = float(candidate.get("start") or 0)
        duration = float(candidate.get("duration") or config.CLIP_DURATION_SECONDS)
        if duration <= 0 or not stream_url:
            return
        count = config.EVENT_VISUAL_FRAMES
        timestamps = [start + duration * (i + 1) / (count + 1) for i in range(count)]
        frames_dir = config.FRAMES_DIR / f"twvodprobe_{video_id}"
        frames = sample_frames_at(stream_url, timestamps, frames_dir, extra_args)
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
        logger.info("Event-probe scoring failed for Twitch VOD %s: %s", video_id, exc)


def _anchor_gate(candidate: dict, stream_url: str, window_seconds: float,
                 video_id: str, anchors, extra_args: list[str]) -> bool:
    """Probe-aware pre-export visual filter (mirrors the Kick lane).

    An EventProbe scene match passes immediately; a strong-anchor match is an
    independent alternative; a window failing BOTH is dropped early.  The
    exporter re-checks strictly after download.
    """
    start = float(candidate.get("start") or 0)
    probe_used = candidate.get("event_visual_score") is not None
    if probe_used and candidate.get("event_visual_fraction", 0.0) >= \
            config.EVENT_VISUAL_FRACTION:
        logger.info("Twitch VOD window %s @%.0fs event-probe scene match -> PASS",
                    video_id, start)
        return True
    if anchors is None or not getattr(anchors, "strong", False):
        return True
    if not getattr(anchors, "enforceable", False):
        logger.info("Twitch VOD window %s @%.0fs: anchors not event-specific; "
                    "skipping anchor gate", video_id, start)
        return True
    count = config.ANCHOR_FRAMES_PER_WINDOW
    if window_seconds <= 0 or count <= 0 or not stream_url:
        return False
    timestamps = [start + window_seconds * (i + 1) / (count + 1)
                  for i in range(count)]
    frames_dir = config.FRAMES_DIR / f"twvod_{video_id}"
    frames = sample_frames_at(stream_url, timestamps, frames_dir, extra_args)
    if not frames:
        logger.info("Twitch VOD window %s @%.0fs: no frames sampled; "
                    "skipping anchor gate", video_id, start)
        return True
    anchor_sim = anchors.anchor_sim([frame["path"] for frame in frames])
    candidate["anchor_sim"] = anchor_sim
    passes = anchor_sim + 1e-6 >= config.ANCHOR_SIM_FLOOR
    logger.info("Twitch VOD window %s @%.0fs anchor_sim=%.2f (floor=%.2f) -> %s",
                video_id, start, anchor_sim, config.ANCHOR_SIM_FLOOR,
                "match" if passes else "no-match")
    if not passes and probe_used:
        logger.info("Twitch VOD window %s @%.0fs: no probe match and no anchor "
                    "match -> reject", video_id, start)
        return False
    if not passes and not config.ANCHOR_GATE_STRICT:
        candidate["anchor_failed"] = True
    return passes or not config.ANCHOR_GATE_STRICT


def analyse_vod(vod: dict, sentence: str, plan: dict,
                anchors=None) -> list[dict]:
    """Transcribe one official Twitch VOD, localize moments, verify candidates."""
    target = (plan.get("semantic_target") or sentence) if isinstance(plan, dict) else sentence
    info = twitch_source.get_vod_info(vod)
    twitch_source.clear_vod_gone(vod)
    video_id = str(info.get("id") or vod.get("id") or "")
    duration = float(info.get("duration") or vod.get("lengthSeconds") or 0)
    if duration < config.MIN_VIDEO_LENGTH or duration > config.TWITCH_MAX_VOD_SECONDS:
        logger.info("Skipping Twitch VOD %s: unsuitable length %.0fs", video_id, duration)
        return []
    if not twitch_source.get_frame_stream(info):
        logger.info("No usable video stream for Twitch VOD %s", video_id)
        return []

    segments = twitch_source.transcribe_vod(info, vod)
    if not segments:
        logger.info("No transcript for Twitch VOD %s", video_id)
        return []

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
        stream = twitch_source.get_frame_stream(info)
        extra_args = stream_headers(info)
        visual_windows = vod_scan.localize(
            stream, extra_args, duration, probe, video_id, window_seconds,
            max_windows)
        if visual_windows:
            for window in visual_windows:
                window.setdefault("proposal_score", float(window.get("mean", 0.0)))
                window.setdefault("llm_found", False)
            windows = _merge_windows(windows, visual_windows, max_windows)
            logger.info("Twitch VOD %s: visual scan added %d window(s) (total %d)",
                        video_id, len(visual_windows), len(windows))
    if not windows:
        return []

    title = info.get("title") or vod.get("title") or ""
    owner = (info.get("uploader") or vod.get("owner") or "")
    rawness = metadata_rawness_score(title, "", owner)
    title_match = title_match_score(target, title, "")
    metadata = lexical_score(target, " ".join(part for part in
                                              (title, vod.get("game", ""), owner) if part))
    frame_stream = twitch_source.get_frame_stream(info)
    export_stream = twitch_source.get_best_stream(info)
    export_headers = stream_headers(info)
    extra_args = export_headers

    candidates: list[dict] = []
    for window in windows:
        start = float(window["start"])
        candidate = {
            "id": video_id,
            "video_id": video_id,
            "duration": float(window_seconds),
            "start": start,
            "timestamp": start,
            "whole": False,
            "source": "twitch",
            "query": sentence,
            "title": title,
            "description": "",
            "channel": owner or "",
            "webpage_url": f"https://www.twitch.tv/videos/{video_id}",
            "stream_url": frame_stream,
            "export_stream_url": export_stream,
            "export_headers": export_headers,
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
            _event_probe_candidate(candidate, probe, frame_stream, extra_args,
                                   video_id)
        ok, reason = content_verifier.verify(candidate, target, plan, segments)
        candidate["verification_reason"] = reason
        logger.info("Twitch VOD window %s @%.0fs (sem=%.2f llm=%s) -> %s",
                    video_id, start, float(window.get("semantic_score", 0.0)),
                    bool(window.get("llm_found")), "OK" if ok else reason)
        if ok:
            if not _anchor_gate(candidate, frame_stream, window_seconds,
                                video_id, anchors, extra_args):
                logger.info("Twitch VOD window %s @%.0fs -> no visual anchor match",
                            video_id, start)
                continue
            candidates.append(candidate)
    logger.info("Twitch VOD %s produced %d verified candidate(s)",
                video_id, len(candidates))
    return candidates


def run(sentence: str, plan: dict, need: int,
        anchors=None) -> list[dict]:
    """Content-verified Twitch raw-VOD windows for the sentence (possibly 0)."""
    streamer = plan.get("streamer") if isinstance(plan, dict) else None
    handle = streamers.twitch_handle(streamer)
    if not handle:
        return []
    queries = plan.get("search_queries") or [sentence]
    vods = twitch_source.search_vods(f"{handle} {' '.join(queries)}",
                                     limit=config.TWITCH_SEARCH_RESULTS)
    if not vods:
        return []
    owned = [v for v in vods if (v.get("owner") or "").lower() == handle.lower()]
    pool = twitch_source.select_analysis(owned, sentence,
                                         count=config.EXACT_TOP_RESULTS) if owned \
        else twitch_source.select_analysis(vods, sentence,
                                           count=config.EXACT_TOP_RESULTS)
    if not pool:
        return []
    candidates: list[dict] = []
    for vod in pool:
        try:
            if twitch_source.vod_is_gone(vod):
                logger.info("Skipping Twitch VOD %s: previously unresolvable",
                            vod.get("webpage_url"))
                continue
            candidates.extend(analyse_vod(vod, sentence, plan, anchors=anchors))
        except Exception as exc:
            logger.warning("Error analysing Twitch VOD %s: %s",
                           vod.get("webpage_url"), exc)
            if twitch_source.is_resolve_failure(exc):
                twitch_source.mark_vod_gone(vod)
    candidates.sort(key=lambda c: (c.get("proposal_score", 0.0),
                                   c.get("semantic_score", 0.0)), reverse=True)
    logger.info("Twitch VOD lane for '%s' returned %d verified candidate(s)",
                sentence, len(candidates))
    return candidates
