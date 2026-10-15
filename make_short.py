"""make_short.py — voiceover-synced 9:16 YouTube Short from a folder of media.

Usage:
  python make_short.py --folder clips/spidertest \
      --voiceover "clips/spidertest/Untitled (56).mp3" --out spidertest

The voiceover is split into word-aligned beats (each visual stays on screen for
at most SHORT_BEAT_MAX_SECONDS), every beat is filled with the best-scoring
window trimmed from the input clips (multiple windows per clip allowed when
they clear the score threshold), images fill beats with no matching window, and
the result is stitched exactly in sync with the voiceover.

Use --preview-report to review the beat->window assignment without rendering.
"""
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import config
from modules.short_builder import (assigner, compositor, footage_index,
                                   voiceover_plan, window_scanner)

logger = logging.getLogger("make_short")


def _progress(done: int, total: int, note: str) -> None:
    print(f"  [{done:>3}/{total}] {note}", flush=True)


def _set_defaults(args: argparse.Namespace) -> None:
    if args.beat_min:
        config.SHORT_BEAT_MIN_SECONDS = args.beat_min
    if args.beat_max:
        config.SHORT_BEAT_MAX_SECONDS = args.beat_max
    if args.fps:
        config.SHORT_OUT_FPS = args.fps
    if args.no_semantic:
        config.SHORT_SEMANTIC_ENABLED = False
    if args.no_stack:
        config.SHORT_STACK_ENABLED = False
    if args.no_folder_match:
        config.SHORT_FOLDER_MATCH = False
    if args.keep_work:
        pass  # passed through to compositor


def _probe_output(path: Path) -> None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration,size:stream=codec_type,width,height",
             "-of", "json", str(path)],
            capture_output=True, text=True, check=True)
        import json
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration") or 0.0)
        size_mb = float(data.get("format", {}).get("size") or 0) / 1024 / 1024
        has_audio = any(s.get("codec_type") == "audio"
                        for s in (data.get("streams") or []))
        video = next((s for s in (data.get("streams") or [])
                      if s.get("codec_type") == "video"), None)
        width, height = video.get("width"), video.get("height")
        logger.info("Output verified: %.1fs, %.1f MiB, %sx%s, audio=%s",
                    duration, size_mb, width, height, has_audio)
    except Exception as exc:
        logger.warning("Output probe failed: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a voiceover-synced 9:16 short from a folder of media")
    parser.add_argument("--folder", default=str(config.SHORT_SCAN_DIR),
                        help="Folder containing clips and images")
    parser.add_argument("--voiceover", help="MP3 voiceover (default: first .mp3 in --folder)")
    parser.add_argument("--out", default="short",
                        help="Output name (written to output/shorts/<name>.mp4)")
    parser.add_argument("--min-score", type=float, default=None,
                        help="Window score threshold (default: auto ~top third)")
    parser.add_argument("--beat-min", type=float, default=None,
                        help="Merge beats rendering under this into neighbours")
    parser.add_argument("--beat-max", type=float, default=None,
                        help="Max on-screen seconds per clip window / image")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--preview-report", action="store_true",
                        help="Write the assignment report and exit without rendering")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Skip the CLIP semantic re-rank")
    parser.add_argument("--no-stack", action="store_true",
                        help="Render shots with the old center-crop instead of "
                             "the stacked top/bottom 16:9 duplicates")
    parser.add_argument("--no-folder-match", action="store_true",
                        help="Do not narrow each beat to its best topic folder")
    parser.add_argument("--keep-work", action="store_true",
                        help="Keep intermediate per-scene renders")
    parser.add_argument("--force-plan", action="store_true",
                        help="Re-transcribe the voiceover (ignore cached scene plan)")
    args = parser.parse_args()
    _set_defaults(args)

    folder = Path(args.folder)
    if not args.voiceover:
        mp3s = sorted(folder.glob("*.mp3"))
        if not mp3s:
            logger.error("No --voiceover given and no .mp3 found in %s", folder)
            return 1
        voiceover = mp3s[0]
    else:
        voiceover = Path(args.voiceover)
    if not voiceover.exists():
        logger.error("Voiceover not found: %s", voiceover)
        return 1

    start = time.time()
    print(f"Voiceover: {voiceover}")
    beats = voiceover_plan.build(voiceover, force=args.force_plan)
    if not beats:
        logger.error("No beats could be extracted from the voiceover")
        return 1
    for beat in beats:
        print(f"  beat {beat['index']:>2}  {beat['render_duration']:.1f}s  "
              f"{beat['text']}")

    print(f"Scanning {folder} for footage...")
    index = footage_index.scan(folder)
    if not index["videos"]:
        logger.warning("No usable video clips found; the short will be image-only")

    print(f"Scanning {len(index['videos'])} clip(s) for best windows...")
    candidates = window_scanner.scan_all(
        index["videos"],
        progress=lambda i, t, n: _progress(i + 1, t, f"scanned {n}"))
    print(f"  {len(candidates)} candidate window(s) scored")

    print("Assigning beats to windows...")
    assignments = assigner.assign(
        beats, candidates, index["images"],
        semantic=config.SHORT_SEMANTIC_ENABLED,
        min_score=args.min_score,
        folders=index["folders"])
    print(f"Assignment report: {config.SHORT_ASSIGNMENT_REPORT_TXT}")
    for assignment in assignments:
        kind = "CLIP" if assignment["kind"] == "video" else "IMG "
        if assignment["kind"] == "video":
            detail = (f"{assignment['source']} "
                      f"[{assignment['start']:.1f}s-{assignment['end']:.1f}s] "
                      f"score={assignment['score']:.3f}")
        else:
            detail = assignment["source"]
        print(f"  beat {assignment['scene_index']:>2} [{kind}] "
              f"{assignment['render_duration']:.1f}s  {detail}")

    if args.preview_report:
        print("Preview only; not rendering.")
        return 0

    config.SHORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.SHORTS_DIR / f"{args.out}.mp4"
    print(f"Rendering {len(assignments)} beat(s) -> {out_path} ...")
    compositor.render(assignments, voiceover, out_path,
                      keep_work=args.keep_work,
                      progress=lambda i, t, n: _progress(i, t, n))
    _probe_output(out_path)
    print(f"Done in {time.time() - start:.0f}s: {out_path}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(errors="replace")
            except Exception:
                pass
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    _console = logging.StreamHandler()
    _console.setFormatter(_formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(_console)
    root.setLevel(logging.INFO)
    sys.exit(main())
