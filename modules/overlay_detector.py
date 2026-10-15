"""Pixel-level editing detection: OCR text-overlay coverage + shot continuity."""
import hashlib
import logging
import os

import config

logger = logging.getLogger(__name__)

_reader = None
_read_attempted = False


def _ocr_reader():
    global _reader, _read_attempted
    if _read_attempted:
        return _reader
    _read_attempted = True
    if not config.OCR_ENABLED:
        return None
    try:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        logger.info("EasyOCR text detector loaded")
    except Exception as exc:
        logger.warning("EasyOCR unavailable; using continuity-only overlay scoring: %s", exc)
        _reader = None
    return _reader


def _cache_key(path: str) -> str:
    stat = os.stat(path)
    return hashlib.md5(f"{path}:{stat.st_size}:{int(stat.st_mtime)}".encode()).hexdigest()


def _coverage_cached(path: str):
    key = _cache_key(path)
    config.OCR_FRAME_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = config.OCR_FRAME_CACHE / f"{key}.json"
    if cache_file.exists():
        try:
            return float(cache_file.read_text().strip())
        except Exception:
            pass
    return None


def _write_coverage(path: str, value: float) -> None:
    key = _cache_key(path)
    config.OCR_FRAME_CACHE.mkdir(parents=True, exist_ok=True)
    (config.OCR_FRAME_CACHE / f"{key}.json").write_text(f"{value:.4f}")


def text_coverage(path: str):
    """Fraction of frame area covered by detected text boxes (0..1), or None if OCR is off."""
    reader = _ocr_reader()
    if reader is None:
        return None
    cached = _coverage_cached(path)
    if cached is not None:
        return cached
    try:
        from PIL import Image
        with Image.open(path) as image:
            width, height = image.size
        area = max(1, width * height)
        result = reader.readtext(path, detail=1)
        coverage = 0.0
        for box, text, confidence in result:
            if not text or confidence < 0.2:
                continue
            xs = [point[0] for point in box]
            ys = [point[1] for point in box]
            box_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            coverage += box_area / area
        coverage = min(1.0, coverage)
        _write_coverage(path, coverage)
        return float(coverage)
    except Exception as exc:
        logger.info("OCR failed for %s: %s", path, exc)
        return None


def visual_rawness(frames: list[dict]) -> dict:
    """Combine OCR overlay coverage and shot continuity into a visual rawness score.

    Returns {'overlay': float|None, 'continuity': float|None, 'visual_raw': float}
    """
    if not frames:
        return {"overlay": None, "continuity": None, "visual_raw": 0.5}

    from modules import vision_ranker

    overlays = [coverage for f in frames if (coverage := text_coverage(f["path"])) is not None]
    overlay = float(sum(overlays) / len(overlays)) if overlays else None

    paths = [f["path"] for f in frames]
    continuity = vision_ranker.pairwise_similarity(paths)

    overlay_term = 1.0 - (overlay if overlay is not None else 0.0)
    continuity_term = continuity if continuity is not None else 0.5
    visual_raw = max(0.0, min(1.0, overlay_term * continuity_term))
    return {"overlay": overlay, "continuity": continuity, "visual_raw": visual_raw}
