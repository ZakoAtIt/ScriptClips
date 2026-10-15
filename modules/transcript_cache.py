"""On-disk cache for fetched YouTube captions keyed by video id + language."""
import json
import logging
import time

import config

logger = logging.getLogger(__name__)


def cache_path(video_id: str, lang: str, kind: str):
    safe = f"{video_id}_{lang}_{kind}".replace("/", "_")
    return config.TRANSCRIPT_CACHE_DIR / f"{safe}.json"


def load(video_id: str, lang: str, kind: str, ttl_days: int | None = None):
    ttl_days = config.CAPTION_CACHE_TTL_DAYS if ttl_days is None else ttl_days
    path = cache_path(video_id, lang, kind)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if time.time() - payload.get("saved_at", 0) > ttl_days * 86400:
        return None
    return payload.get("segments")


def save(video_id: str, lang: str, kind: str, segments: list[dict]) -> None:
    path = cache_path(video_id, lang, kind)
    payload = {"saved_at": time.time(), "lang": lang, "kind": kind, "segments": segments}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    logger.info("Cached %d caption segments to %s", len(segments), path)
