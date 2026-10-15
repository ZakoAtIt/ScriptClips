"""Footage index: recursively scan an input folder for usable clips and images."""
import logging
import re
import subprocess
from pathlib import Path

from PIL import Image

import config

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".flv", ".m4v", ".ts"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _probe_video(path: Path) -> dict | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=width,height,avg_frame_rate,codec_type",
             "-of", "json", str(path)],
            capture_output=True, text=True, check=True)
        import json
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration") or 0.0)
        streams = data.get("streams") or []
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        width = int(video.get("width") or 0) if video else 0
        height = int(video.get("height") or 0) if video else 0
        return {"duration": duration, "width": width, "height": height}
    except Exception:
        return None


def _probe_image(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", path.stem).strip("_")[:60]


def scan(folder: Path) -> dict:
    """Return {"videos": [clip dicts], "images": [image dicts], "folders": [names]}."""
    folder = Path(folder)
    if not folder.exists():
        logger.error("Input folder does not exist: %s", folder)
        return {"videos": [], "images": [], "folders": []}

    folders: list[str] = []
    for sub in sorted(folder.iterdir()):
        if sub.is_dir():
            folders.append(_safe_stem(sub))

    videos: list[dict] = []
    images: list[dict] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in VIDEO_EXTS:
            info = _probe_video(path)
            if not info or info["duration"] < 1.5:
                logger.info("Skipping unreadable/too-short clip: %s", path)
                continue
            rel = path.relative_to(folder)
            videos.append({
                "path": str(path),
                "name": path.name,
                "folder": str(rel.parts[0]) if len(rel.parts) > 1 else "",
                "duration": info["duration"],
                "width": info["width"],
                "height": info["height"],
            })
        elif suffix in IMAGE_EXTS:
            size = _probe_image(path)
            if not size or min(size) < config.IMAGE_MIN_SIDE:
                continue
            rel = path.relative_to(folder)
            images.append({
                "path": str(path),
                "name": path.name,
                "folder": str(rel.parts[0]) if len(rel.parts) > 1 else "",
                "width": size[0],
                "height": size[1],
            })

    videos.sort(key=lambda v: v["name"])
    images.sort(key=lambda im: im["name"])
    logger.info("Footage index: %d clip(s) (%.0f min total), %d image(s)",
                len(videos), sum(v["duration"] for v in videos) / 60, len(images))
    return {"videos": videos, "images": images, "folders": folders}
