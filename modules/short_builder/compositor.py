"""Compositor: render the assigned scenes and mux the voiceover.

Each scene is rendered to a 1080x1920 @30fps libx264 segment — a video window
is trimmed to its exact render duration, a still image gets a subtle Ken Burns
zoom — then all segments are concatenated (stream-copy) and the original
voiceover MP3 is muxed as the audio track.
"""
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def _ffmpeg(args: list[str], timeout: int) -> None:
    subprocess.run(args, check=True, timeout=timeout,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _cover_filter(fps: int) -> str:
    return (f"scale={config.SHORT_OUT_WIDTH}:{config.SHORT_OUT_HEIGHT}:"
            f"force_original_aspect_ratio=increase,"
            f"crop={config.SHORT_OUT_WIDTH}:{config.SHORT_OUT_HEIGHT},"
            f"fps={fps},format=yuv420p")


def _stack_filter(fps: int, duration: float) -> str:
    """Split layout: the same shot twice, full-bleed top and bottom halves.

    The source is scaled to cover the 1080x960 half (never letterboxed) and
    duplicated.  Each copy is a time-animated 1080x960 crop so the two halves
    show different zoom levels and camera positions over the shot — the bottom
    copy starts clearly more zoomed in than the top.  The halves are stacked
    into the 1080x1920 canvas with no black bars anywhere.
    """
    w = config.SHORT_OUT_WIDTH
    half = config.SHORT_OUT_HEIGHT // 2
    dur = max(1.0, float(duration))

    def crop_expr(zoom: tuple[float, float], pan: tuple[float, float]) -> str:
        z0, z1 = zoom
        dx, dy = pan
        growth = max(0.0, z1 - z0)
        ts = "if(isnan(t),0,t)"
        return (f"crop=w='{w}/({z0:.4f}+{growth:.4f}*{ts}/{dur:.3f})':"
                f"h='{half}/({z0:.4f}+{growth:.4f}*{ts}/{dur:.3f})':"
                f"x='(iw-ow)/2+{dx}*{ts}/{dur:.3f}':"
                f"y='(ih-oh)/2+{dy}*{ts}/{dur:.3f}'")

    return (
        f"fps={fps},setsar=1,"
        f"scale={w}:{half}:force_original_aspect_ratio=increase,"
        f"split=2[t][b];"
        f"[t]{crop_expr(config.SHORT_STACK_TOP_ZOOM, config.SHORT_STACK_TOP_PAN)},"
        f"scale={w}:{half}[t1];"
        f"[b]{crop_expr(config.SHORT_STACK_BOT_ZOOM, config.SHORT_STACK_BOT_PAN)},"
        f"scale={w}:{half}[b1];"
        f"[t1][b1]vstack=2,format=yuv420p[out]"
    )


def _render_video(assignment: dict, out_path: Path) -> None:
    duration = float(assignment["render_duration"])
    if config.SHORT_STACK_ENABLED:
        _ffmpeg([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", assignment["source"], "-ss", f'{assignment["start"]:.3f}',
            "-t", f"{duration:.3f}",
            "-filter_complex", _stack_filter(config.SHORT_OUT_FPS, duration),
            "-map", "[out]",
            "-c:v", "libx264", "-preset", config.SHORT_OUT_PRESET,
            "-crf", str(config.SHORT_OUT_CRF), "-an", str(out_path),
        ], timeout=int(60 + 60 * duration))
        return
    filter_chain = _cover_filter(config.SHORT_OUT_FPS)
    _ffmpeg([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", assignment["source"], "-ss", f'{assignment["start"]:.3f}',
        "-t", f"{duration:.3f}",
        "-vf", filter_chain,
        "-c:v", "libx264", "-preset", config.SHORT_OUT_PRESET,
        "-crf", str(config.SHORT_OUT_CRF), "-an", str(out_path),
    ], timeout=int(60 + 40 * duration))


def _render_image(assignment: dict, out_path: Path) -> None:
    duration = float(assignment["render_duration"])
    fps = config.SHORT_OUT_FPS
    total_frames = max(1, round(duration * fps))
    if config.SHORT_STACK_ENABLED:
        _ffmpeg([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-loop", "1", "-framerate", str(fps), "-t", f"{duration:.3f}",
            "-i", assignment["source"],
            "-filter_complex", _stack_filter(fps, duration),
            "-map", "[out]", "-frames:v", str(total_frames),
            "-c:v", "libx264", "-preset", config.SHORT_OUT_PRESET,
            "-crf", str(config.SHORT_OUT_CRF), "-an", str(out_path),
        ], timeout=int(60 + 60 * duration))
        return
    zoom = config.SHORT_KENBURNS_ZOOM
    increment = zoom / total_frames
    cover = (f"scale={config.SHORT_OUT_WIDTH}:{config.SHORT_OUT_HEIGHT}:"
             f"force_original_aspect_ratio=increase,"
             f"crop={config.SHORT_OUT_WIDTH}:{config.SHORT_OUT_HEIGHT}")
    filter_chain = (
        f"{cover},"
        f"zoompan=z='min(zoom+{increment:.6f},1.0+{zoom:.3f})':d=1:"
        f"fps={fps}:s={config.SHORT_OUT_WIDTH}x{config.SHORT_OUT_HEIGHT}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)',format=yuv420p")
    _ffmpeg([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-loop", "1", "-framerate", str(fps), "-t", f"{duration:.3f}",
        "-i", assignment["source"],
        "-vf", filter_chain, "-frames:v", str(total_frames),
        "-c:v", "libx264", "-preset", config.SHORT_OUT_PRESET,
        "-crf", str(config.SHORT_OUT_CRF), "-an", str(out_path),
    ], timeout=int(60 + 40 * duration))


def _render_scene(assignment: dict, index: int, work_dir: Path) -> Path:
    out_path = work_dir / f"scene_{index:03d}.mp4"
    if assignment["kind"] == "video":
        _render_video(assignment, out_path)
    else:
        _render_image(assignment, out_path)
    return out_path


def _concat(paths: list[Path], out_path: Path) -> None:
    list_path = out_path.with_suffix(".txt")
    list_path.write_text("\n".join(f"file '{str(p).replace(chr(39), chr(39)+chr(39)+chr(39))}'"
                                   for p in paths) + "\n", encoding="utf-8")
    try:
        _ffmpeg([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-c", "copy", str(out_path),
        ], timeout=120)
    except Exception:
        logger.info("Stream-copy concat failed; re-encoding concat")
        inputs: list[str] = []
        for path in paths:
            inputs += ["-i", str(path)]
        _ffmpeg([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            *inputs, "-filter_complex",
            f"concat=n={len(paths)}:v=1:a=0[outv]",
            "-map", "[outv]", "-c:v", "libx264",
            "-preset", config.SHORT_OUT_PRESET, "-crf", str(config.SHORT_OUT_CRF),
            str(out_path),
        ], timeout=300)


def render(assignments: list[dict], voiceover_path: Path, out_path: Path,
           keep_work: bool = False, progress=None) -> None:
    if not assignments:
        raise RuntimeError("Nothing to render: no scene assignments")
    work_dir = config.SHORT_WORK_DIR
    work_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    with ThreadPoolExecutor(max_workers=config.SHORT_WORKER_JOBS) as pool:
        futures = {}
        for index, assignment in enumerate(assignments):
            future = pool.submit(_render_scene, assignment, index, work_dir)
            futures[future] = (index, assignment)
        paths: list[Path] = [None] * len(assignments)  # type: ignore[list-item]
        for future in futures:
            index, _assignment = futures[future]
            paths[index] = future.result()
            if progress:
                progress(index + 1, len(assignments), f"scene {index + 1} rendered")

    concat_path = work_dir / "concat.mp4"
    _concat([p for p in paths if p is not None], concat_path)

    total = round(sum(float(a["render_duration"]) for a in assignments) + 0.05, 3)
    _ffmpeg([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(concat_path), "-i", str(voiceover_path),
        "-t", f"{total:.3f}",
        "-c:v", "copy", "-c:a", "aac",
        "-b:a", config.SHORT_OUT_AUDIO_BITRATE,
        "-movflags", "+faststart", str(out_path),
    ], timeout=300)

    if not keep_work:
        for path in work_dir.glob("scene_*.mp4"):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            concat_path.unlink(missing_ok=True)
        except OSError:
            pass
    logger.info("Rendered short: %s (%.1fs)", out_path, total)
