"""Window scanner: find the best 3-6s windows inside each input clip.

Frames are sampled at SHORT_SCAN_FPS and downscaled to SHORT_FRAME_WIDTH.  For
every candidate start (stride SHORT_WINDOW_STRIDE) and every length bucket the
window score combines per-clip-normalized motion, sharpness (gradient energy),
stability (low shake), shot coherence (no hard cut inside the window) and face
presence (YuNet on a coarse grid).  Windows overlapping an already-kept window
by more than ~0.5s are suppressed, so one motion burst yields a single window
while distinct good moments in the same clip all survive.

Per-clip results are cached (hash of path+size+mtime) under config.SHORT_SCAN_CACHE_DIR.
"""
import hashlib
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)

_face_grid_cache: dict[tuple[str, float], bool] = {}


def _clip_hash(clip: dict) -> str:
    path = Path(clip["path"])
    stat = path.stat()
    digest = hashlib.sha1(
        f"{config.SHORT_SCAN_VERSION}|{path}|{stat.st_size}|{stat.st_mtime_ns}"
        .encode()).hexdigest()
    return digest[:16]


def _extract_frames(video_path: str, fps: float, width: int):
    """Return [(ts, float32 gray array), ...] sampled from the clip."""
    with tempfile.TemporaryDirectory(prefix="scscan_") as tmpdir:
        pattern = str(Path(tmpdir) / "f_%06d.jpg")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", video_path, "-vf", f"fps={fps},scale={width}:-2",
               "-q:v", "4", pattern]
        subprocess.run(cmd, check=True, timeout=300,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        frames: list[tuple[float, np.ndarray]] = []
        i = 0
        while True:
            frame_path = Path(tmpdir) / f"f_{i + 1:06d}.jpg"
            if not frame_path.exists():
                break
            with Image.open(frame_path).convert("L") as im:
                arr = np.asarray(im, dtype=np.float32)
            frames.append((i / fps, arr))
            i += 1
        return frames


def _sharpness(arr: np.ndarray) -> float:
    gy, gx = np.gradient(arr)
    return float(np.mean(gx * gx + gy * gy))


def _extract_face_frame(video_path: str, ts: float) -> Image.Image | None:
    """Extract one frame at ts (input seek), retrying once with output seek."""
    with tempfile.TemporaryDirectory(prefix="scface_") as tmpdir:
        frame_path = Path(tmpdir) / "f.png"
        commands = (
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-ss", f"{ts:.2f}", "-i", video_path, "-frames:v", "1",
             "-vf", "scale=640:-2", str(frame_path)],
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", video_path, "-ss", f"{ts:.2f}", "-frames:v", "1",
             "-vf", "scale=640:-2", str(frame_path)],
        )
        for attempt, cmd in enumerate(commands):
            try:
                subprocess.run(cmd, check=True, timeout=45,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as exc:
                logger.info("Face frame extract attempt %d failed at %.2fs: %s",
                            attempt + 1, ts, exc)
                continue
            if frame_path.exists():
                try:
                    return Image.open(frame_path).convert("RGB")
                except Exception as exc:
                    logger.info("Face frame unreadable at %.2fs: %s", ts, exc)
            frame_path.unlink(missing_ok=True)
    return None


def _face_present(video_path: str, ts: float, duration: float, interval: float) -> bool:
    """Face presence cached on a coarse time grid (1 detection per grid cell)."""
    ts = max(0.0, min(ts, duration - 0.5))
    if ts < 0.1:
        return False
    grid = round(ts / interval) * interval
    key = (video_path, round(grid, 2))
    if key in _face_grid_cache:
        return _face_grid_cache[key]
    try:
        from modules import face_verify
    except Exception as exc:
        logger.warning("face_verify unavailable: %s", exc)
        _face_grid_cache[key] = False
        return False
    img = _extract_face_frame(video_path, grid)
    if img is None:
        _face_grid_cache[key] = False
        return False
    try:
        faces = face_verify._detect_faces(img)
        present = bool(faces)
    except Exception as exc:
        logger.info("Face detect failed at %.1fs: %s", grid, exc)
        present = False
    _face_grid_cache[key] = present
    return present


def scan_clip(clip: dict) -> dict:
    """Score all candidate windows of one clip; returns {candidates: [...]}."""
    cache = config.SHORT_SCAN_CACHE_DIR / f"{_clip_hash(clip)}.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("candidates"):
                return data
        except Exception:
            pass

    duration = float(clip["duration"])
    fps = config.SHORT_SCAN_FPS
    stride = config.SHORT_WINDOW_STRIDE
    weights = config.SHORT_SCORE_WEIGHTS

    try:
        frames = _extract_frames(clip["path"], fps, config.SHORT_FRAME_WIDTH)
    except Exception as exc:
        logger.warning("Could not scan clip %s: %s", clip["name"], exc)
        return {"candidates": []}
    if not frames:
        return {"candidates": []}
    ts = np.asarray([t for t, _ in frames], dtype=np.float64)
    grays = [arr for _, arr in frames]

    n = len(frames)
    motion = np.zeros(n, dtype=np.float64)
    sharp = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        motion[i] = float(np.mean(np.abs(grays[i] - grays[i - 1])))
    for i in range(n):
        sharp[i] = _sharpness(grays[i])

    m95 = float(np.percentile(motion[1:], 95)) if n > 2 else 0.0
    s95 = float(np.percentile(sharp, 95)) or 1.0
    m_scale = m95 if m95 > 1.0 else (float(np.max(motion)) or 1.0)
    if m_scale < 1.0:
        m_scale = 1.0

    motion_n = np.clip(motion / m_scale, 0.0, 1.0)
    sharp_n = np.clip(sharp / s95, 0.0, 1.0)

    median = float(np.median(motion[1:])) if n > 2 else 0.0
    mad = float(np.median(np.abs(motion[1:] - median))) if n > 2 else 0.0
    cut = motion > max(2.2 * median, median + 4.0 * mad, 30.0)

    buckets = [b for b in config.SHORT_LENGTH_BUCKETS
               if b <= config.SHORT_MAX_WINDOW]
    candidates: list[dict] = []
    start_t = 0.0
    face_interval = max(config.SHORT_FACE_SAMPLE_INTERVAL,
                        duration / max(12.0, 1.0))
    face_budget = 8
    face_hits = 0
    while start_t < duration - 0.4:
        lo = int(np.searchsorted(ts, start_t))
        center = min(start_t + config.SHORT_MAX_WINDOW / 2.0, duration - 0.5)
        face = None
        if face_hits < face_budget:
            face = _face_present(clip["path"], center, duration, face_interval)
            face_hits += 1
        else:
            face = False
        face_score = 1.0 if face else 0.0
        for bucket in buckets:
            end_t = start_t + bucket
            if end_t > duration - 0.1:
                continue
            hi = int(np.searchsorted(ts, end_t, side="right"))
            if hi - lo < 1:
                continue
            seg_motion = float(np.mean(motion_n[lo:hi]))
            seg_sharp = float(np.mean(sharp_n[lo:hi]))
            seg_raw = motion[lo:hi]
            mean_raw = float(np.mean(seg_raw))
            std_raw = float(np.std(seg_raw))
            stability = 1.0 - min(1.0, std_raw / (mean_raw + 1e-6)) if mean_raw > 0.05 else 0.5
            pad = max(1, int(0.3 * fps))
            cut_inside = bool(np.any(cut[lo + pad:max(lo + pad, hi - pad)]))
            shot_ok = 0.2 if cut_inside else 1.0
            score = (weights["motion"] * seg_motion
                     + weights["sharpness"] * seg_sharp
                     + weights["stability"] * stability
                     + weights["shot"] * shot_ok
                     + weights["face"] * face_score)
            candidates.append({
                "clip": clip["path"],
                "name": clip["name"],
                "folder": clip.get("folder", ""),
                "start": round(start_t, 2),
                "end": round(end_t, 2),
                "length": bucket,
                "score": round(score, 4),
                "motion": round(seg_motion, 4),
                "sharpness": round(seg_sharp, 4),
                "stability": round(stability, 4),
                "shot_ok": round(shot_ok, 2),
                "face": int(face_score),
                "clip_duration": duration,
            })
        start_t = round(start_t + stride, 2)

    candidates = _suppress_overlaps(candidates)
    candidates.sort(key=lambda c: c["score"], reverse=True)
    candidates = candidates[:config.SHORT_MAX_CANDIDATES_PER_CLIP]
    result = {"candidates": candidates}
    try:
        cache.write_text(json.dumps(result, indent=1), encoding="utf-8")
    except Exception as exc:
        logger.info("Could not cache scan for %s: %s", clip["name"], exc)
    return result


def _suppress_overlaps(candidates: list[dict]) -> list[dict]:
    """Non-maximum suppression: keep the best window per overlapping cluster."""
    kept: list[dict] = []
    for candidate in sorted(candidates, key=lambda c: -c["score"]):
        overlap = False
        for other in kept:
            lo = max(candidate["start"], other["start"])
            hi = min(candidate["end"], other["end"])
            if hi - lo >= 0.5:
                overlap = True
                break
        if not overlap:
            kept.append(candidate)
    return kept


def scan_all(clips: list[dict], progress=None) -> list[dict]:
    """Score every clip; returns a flat list of candidate windows."""
    results: list[dict] = []
    for index, clip in enumerate(clips):
        if progress:
            progress(index, len(clips), clip.get("name", ""))
        results.extend(scan_clip(clip)["candidates"])
    return results
