"""ScriptClips — queries.txt -> verified clips + images per query.

Each line of queries.txt is a detailed query naming the subjects, object and
action.  For each query:
  1. YouTube lane (FIRST priority, always run): the verbatim query is searched
     as-is, the first 12 16:9 (non-Short) results in YouTube's own ranking are
     analyzed for their best windows (any length — no long-form floor), and raw
     footage is scanned by caption and content verified.
  2. Twitch lane: clips from the subject streamer's own channel, content
     verified by transcribing each short clip's audio.
  3. Kick lane: HLS windows from the subject streamer's own Kick channel VODs
     (streamers who broadcast on Kick), transcribed in chunks and verified.
  4. LLM judge rejects reactors/re-uploads the metadata could not catch; the
     judge VETOES — rejected clips are never exported.
  5. Every exported clip is visually verified on its pixels (event-kind probe ->
     general scene probe -> audio corroboration) and anchor-gated; subject
     presence is the FIRST-PRIORITY selection signal, so clips that put every
     script subject on screen are chosen first.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import config
from modules import (auto_registry, clip_exporter, discovery_lane, image_lane,
                     kick_lane, llm_client, llm_judge, moment_proposer,
                     report_writer, run_report, runtime_stats, script_planner,
                     twitch_lane, twitch_vod_lane, visual_anchor, youtube_lane)
from modules.scoring import combine_score

logger = logging.getLogger("scriptclips")


# Lane priority for final selection.  YouTube's first-12 16:9 results reliably
# carry the best clip, so it outranks Twitch, which outranks Kick, at equal
# visual quality; within a source the combined score still decides.  Subject
# presence (all subjects on screen) outranks all of this at export time.
_SOURCE_PRIORITY = {"youtube": 3, "twitch": 2, "twitter": 2, "kick": 1, "other": 0}


def finalize(sentence: str, plan: dict, candidates: list[dict]) -> list[dict]:
    """LLM-judge the media clips and pick a backfill pool for export.

    The judge VETOES: clips it rejects are dropped from the pool entirely, so a
    reactor/re-upload/metadata-miss clip can never be exported — even if that
    leaves a query with nothing to ship.  Within the pool, subject presence is
    decided at export time (face_all first); here visual truth (event pixels /
    vision / anchors) outranks lane priority so a clip whose pixels confirm the
    moment beats a higher-priority lane's audio-only lookalike.
    """
    media = [c for c in candidates if c.get("source") != "image"]
    if media:
        media = llm_judge.judge(sentence, plan, media, pool_size=config.LLM_JUDGE_POOL)
    for candidate in media:
        candidate["combined_score"] = combine_score(candidate)
    rejected = [c for c in media if c.get("llm_reject")]
    if rejected and len(rejected) == len(media):
        logger.warning("LLM judge rejected EVERY candidate for %r; "
                       "nothing will be exported", sentence)
    media = [c for c in media if not c.get("llm_reject")]
    if not media:
        return []
    media_query = bool(plan.get("media_query"))
    # Only trust the CLIP scene scores in ranking when the plan has CONCRETE
    # scene prompts (a recognized visual-event kind or object terms).  For
    # generic/narration queries (e.g. "team worldwide wins streamer games") the
    # general scene prompts are vague, so a wrong window whose frames happen to
    # clear CLIP would outrank the real moment; there the pixel scores are
    # neutral and the spoken/LLM/keyword signals decide instead.
    concrete_visual = bool(plan.get("visual_event_kind")) or bool(plan.get("object_terms"))
    if concrete_visual:
        # The plan names a visual event/object, so the PIXELS are the truth:
        # VLM confirmation, keyword/LLM moment flags, then the scene/anchor
        # scores decide.  Spoken coverage is neutral here (the event footage may
        # be silent) and only breaks ties via the combined score.
        ranked = sorted(media, key=lambda c: (
            bool(media_query and c.get("whole")),
            bool(c.get("vlm_happening")),
            float(c.get("vlm_score") or 0.0),
            bool(c.get("keyword_located")),
            bool(c.get("llm_found")),
            float(c.get("llm_moment_score") or 0.0),
            float(c.get("event_visual_score") or 0.0),
            float(c.get("vision_score") or 0.0),
            float(c.get("anchor_sim") or 0.0),
            _SOURCE_PRIORITY.get(c.get("source"), 0),
            c.get("combined_score", 0.0),
        ), reverse=True)
    else:
        # Generic/narration queries ("X reacts to / watches Y"): every window
        # shares the same subject's face and the general scene probes fire on
        # any on-screen person, so the pixel/anchor scores are NEUTRAL.  The
        # SPOKEN transcript is the only signal that tells WHICH window is the
        # moment: plan action+entity coverage first (the one window whose
        # captions literally describe the event - e.g. iZPL9@198s "if you're
        # watching this, make a left" with both driver+watching hits), then
        # semantic and proposal.  VLM/keyword/LLM flags still break ties but
        # must NOT outrank coverage: an LLM moment finder that spots "a police
        # chase, wow" (coverage 0.5) must not beat the actual reaction window
        # the user asked for (coverage 1.0).
        ranked = sorted(media, key=lambda c: (
            bool(media_query and c.get("whole")),
            float(c.get("action_coverage") or 0.0),
            float(c.get("entity_coverage") or 0.0),
            float(c.get("semantic_score") or 0.0),
            float(c.get("proposal_score") or 0.0),
            bool(c.get("vlm_happening")),
            float(c.get("vlm_score") or 0.0),
            bool(c.get("keyword_located")),
            bool(c.get("llm_found")),
            float(c.get("llm_moment_score") or 0.0),
            float(c.get("anchor_sim") or 0.0),
            _SOURCE_PRIORITY.get(c.get("source"), 0),
            c.get("combined_score", 0.0),
        ), reverse=True)
    return moment_proposer.choose_diverse(
        ranked, config.EXPORT_TRY_LIMIT, config.MIN_CLIP_GAP_SECONDS,
        config.MAX_CLIPS_PER_VIDEO)


def process_sentence(plan: dict) -> None:
    sentence = plan["sentence"]
    need = config.CLIPS_PER_SENTENCE
    runtime_stats.maybe_gc(f"start:q{plan.get('index', '?')}")
    logger.info("=== Sentence %d: %s ===", plan.get("index", "?"), sentence)

    # Visual anchors: the actual event pixels per Google Images.  Clip lanes
    # hard-reject candidates whose frames do not match them, so a wrong event
    # that merely mentions the same words can never pass on audio/metadata alone.
    anchors = visual_anchor.build(sentence, plan)
    if anchors and anchors.strong:
        logger.info("Sentence %d: %d visual anchor image(s) ready",
                    plan.get("index", "?"), len(anchors.entries))
    try:
        from modules import event_vision
        sheet = event_vision.attach_contact_sheet(
            plan, anchors, int(plan.get("index") or 0))
        if sheet:
            logger.info("Sentence %d: contact sheet ready at %s",
                        plan.get("index", "?"), sheet)
    except Exception as exc:
        logger.warning("Contact sheet unavailable: %s", exc)

    # Web discovery lane: open-web search for the moment on ANY platform
    # (X mirrors, native Twitch clips, differently-titled uploads).  Runs
    # BEFORE the platform lanes so its thumbnails can enrich the anchors and
    # its long YouTube hits join the analysed pool.
    discovery_clips: list[dict] = []
    discovery_yt_pool: list[dict] = []
    try:
        discovery_clips, discovery_yt_pool, discovery_rejected = \
            discovery_lane.run(sentence, plan)
        run_report.log_discovery(sentence, discovery_clips,
                                 discovery_yt_pool, discovery_rejected)
        if discovery_clips or discovery_yt_pool:
            thumbs = [c["thumbnail"] for c in discovery_clips if c.get("thumbnail")]
            thumbs += [m.get("thumbnail") for m in discovery_yt_pool
                       if m.get("thumbnail")]
            prompt = plan.get("visual_prompt") or sentence
            anchors = visual_anchor.enrich(anchors, thumbs, prompt)
    except Exception as exc:
        logger.warning("Discovery lane failed for '%s': %s", sentence, exc)

    # YouTube is the FIRST-PRIORITY lane: the first 12 16:9 results in YouTube's
    # own ranking are analyzed for their best windows, then Twitch, then Kick.
    youtube_clips = youtube_lane.run(sentence, plan, need, anchors=anchors,
                                     extra_pool=discovery_yt_pool)
    logger.info("Sentence %d: %d verified YouTube candidate(s)", plan.get("index", "?"), len(youtube_clips))
    runtime_stats.maybe_gc(f"youtube:q{plan.get('index', '?')}")

    twitch_clips = []
    twitch_vod_clips = []
    kick_clips = []
    streamers_to_check = plan.get("candidate_streamers") or (
        [plan["streamer"]] if plan.get("streamer") else [])
    for streamer in streamers_to_check:
        sub_plan = dict(plan, streamer=streamer)
        streamer_name = streamer.get("name", "?")
        if streamer.get("twitch_handle"):
            got = twitch_lane.run(sentence, sub_plan, need, anchors=anchors)
            logger.info("Sentence %d: %d verified Twitch clip(s) for %s",
                        plan.get("index", "?"), len(got), streamer_name)
            twitch_clips += got

            if config.TWITCH_VOD_LANE_ENABLED:
                got = twitch_vod_lane.run(sentence, sub_plan, need, anchors=anchors)
                logger.info("Sentence %d: %d verified Twitch VOD candidate(s) for %s",
                            plan.get("index", "?"), len(got), streamer_name)
                twitch_vod_clips += got

        if streamer.get("kick") and (config.KICK_VOD_LANE_ENABLED
                                     or (plan.get("auto") or {}).get("kick", {}).get("enabled")):
            got = kick_lane.run(sentence, sub_plan, need, anchors=anchors)
            logger.info("Sentence %d: %d verified Kick candidate(s) for %s",
                        plan.get("index", "?"), len(got), streamer_name)
            kick_clips += got
    runtime_stats.maybe_gc(f"lanes:q{plan.get('index', '?')}")

    picked = finalize(sentence, plan,
                      youtube_clips + twitch_clips + twitch_vod_clips + kick_clips
                      + discovery_clips)
    for rank, clip in enumerate(picked, 1):
        logger.info("candidate %2d: [%s] %s dur=%.0fs match=%.2f sem=%.2f "
                    "vision=%.2f anchor=%.2f tier=%s %s",
                    rank, clip["source"], clip["video_id"], clip["duration"],
                    clip.get("title_match_score", 0.0),
                    clip.get("semantic_score", 0.0),
                    clip.get("vision_score", 0.0),
                    clip.get("anchor_sim", 0.0),
                    clip.get("verification_tier", "pending"),
                    clip["title"][:60])

    anchor_entries = anchors.entries if anchors else None
    images = image_lane.run(sentence, plan, config.IMAGES_PER_SENTENCE,
                            anchor_entries=anchor_entries)
    logger.info("Sentence %d: %d image candidate(s)", plan.get("index", "?"), len(images))

    exported = clip_exporter.export_media(picked, plan, Path(config.CLIPS_DIR), need,
                                          anchors=anchors)
    exported_images = clip_exporter.export_images(images, plan, Path(config.CLIPS_DIR))
    runtime_stats.maybe_gc(f"export:q{plan.get('index', '?')}")
    run_report.log_sentence_result(sentence, picked, exported, need)
    try:
        report_writer.write_query_report(sentence, picked + images, Path(config.REPORTS_DIR))
    except Exception as exc:
        logger.warning("Could not write reports: %s", exc)
    logger.info("=== Done: %d clips + %d images exported for sentence %d ===",
                len(exported), len(exported_images), plan.get("index", "?"))


def main() -> int:
    parser = argparse.ArgumentParser(description="ScriptClips — 5 verified clips per sentence")
    parser.add_argument("--sentence", help="Run a single sentence (substring match against script sentences)")
    parser.add_argument("--force-plan", action="store_true", help="Re-run the LLM planner (ignore cache)")
    args = parser.parse_args()

    # Auto-registry stage: discover per-batch people/events/VODs/lexicon BEFORE
    # planning so the curated-event machinery (event_videos pool, moment lexicon,
    # kick fallback, per-language Whisper) engages for brand-new queries.
    auto_hints = auto_registry.prepare()
    if auto_hints:
        logger.info("Auto-registry: %d per-query hint(s) ready", len(auto_hints))
    force_plan = bool(args.force_plan) or auto_registry.DIRTY

    if args.sentence:
        _, plans = script_planner.plan_script(force=force_plan)
        plans = [p for p in plans if args.sentence.lower() in p.get("sentence", "").lower()]
        if not plans:
            logger.error("No query matches %r; edit queries.txt and re-run without --sentence", args.sentence)
            return 1
    else:
        _, plans = script_planner.plan_script(force=force_plan)

    for plan in plans:
        plan["auto"] = auto_hints.get(plan.get("sentence") or "", {}) or {}

    if not llm_client.available():
        logger.warning("Ollama is UNREACHABLE (%s) — the LLM judge/planner will be "
                       "skipped and rule-based scoring used instead.", config.OLLAMA_HOST)
    elif not llm_client.vision_capable():
        logger.warning("Ollama is up but NO VISION MODEL is installed (need one of %s, "
                       "e.g. %s) — the judge will grade clips on transcript audio "
                       "alone. Run: ollama pull %s",
                       ", ".join(config.LLM_VISION_MODELS), config.LLM_VISION_MODEL,
                       config.LLM_VISION_MODEL)
    else:
        logger.info("Vision judge ready: %s sees real clip frames.",
                    llm_client._resolve_model(require_vision=True))
    logger.info("Visual verification: GENERAL_VISUAL=%s floor=%.2f | kind probe=%s | "
                "vision weight=%.2f",
                config.GENERAL_VISUAL_ENABLED, config.GENERAL_VISUAL_FLOOR,
                config.EVENT_VISUAL_ENABLED, config.SCORE_WEIGHTS.get("vision", 0.0))
    for index, plan in enumerate(plans, 1):
        plan["index"] = index
        try:
            process_sentence(plan)
        except Exception:
            logger.exception("Sentence failed: %s", plan.get("sentence", "?"))
    return 0


_WATCHDOG_RESTART_MAX = 3
_WATCHDOG_RESTART_SLEEP = 10


def _watchdog_loop(argv: list[str]) -> int:
    """Re-launch the run as a child and auto-restart on hard/native death.

    Transcripts, moment-finder caches and the export registry make re-runs
    cheap and resumable, so an intermittent native crash (silent process kill
    ~30 min in on this 16 GB box) no longer loses the whole run.  A clean exit
    (0) or an interrupt (130) stops; any other exit code restarts up to
    ``_WATCHDOG_RESTART_MAX`` times.
    """
    import subprocess
    import time

    env = dict(os.environ)
    env["SCRIPTCLIPS_WATCHDOG_CHILD"] = "1"
    last_code = 1
    for attempt in range(_WATCHDOG_RESTART_MAX + 1):
        logger.info("Watchdog: run attempt %d/%d",
                    attempt + 1, _WATCHDOG_RESTART_MAX + 1)
        try:
            proc = subprocess.run([sys.executable, "-u", __file__, *argv],
                                  env=env)
        except KeyboardInterrupt:
            logger.info("Watchdog: interrupted; stopping")
            return 130
        last_code = proc.returncode
        if last_code == 0 or last_code == 130:
            return last_code
        logger.error("Watchdog: attempt %d exited with code %s; restarting in %ds",
                     attempt + 1, last_code, _WATCHDOG_RESTART_SLEEP)
        if attempt < _WATCHDOG_RESTART_MAX:
            try:
                time.sleep(_WATCHDOG_RESTART_SLEEP)
            except KeyboardInterrupt:
                return 130
    logger.error("Watchdog: giving up after %d attempts (last code %s)",
                 _WATCHDOG_RESTART_MAX + 1, last_code)
    return last_code


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(errors="replace")
            except Exception:
                pass
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    _console = logging.StreamHandler()
    _console.setFormatter(_formatter)
    _file = logging.FileHandler(config.LOGS_DIR / "pipeline.log", encoding="utf-8")
    _file.setFormatter(_formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(_console)
    root.addHandler(_file)
    root.setLevel(logging.INFO)
    argv = sys.argv[1:]
    if "--no-watchdog" in argv:
        sys.argv = [sys.argv[0]] + [arg for arg in argv if arg != "--no-watchdog"]
        sys.exit(main())
    if os.environ.get("SCRIPTCLIPS_WATCHDOG_CHILD") == "1":
        try:
            sys.exit(main())
        except KeyboardInterrupt:
            sys.exit(130)
    sys.exit(_watchdog_loop(argv))
