"""Runtime memory watchdog.

The pipeline loads Whisper + CLIP + MiniLM + reranker + onnxruntime face models
into one ~16 GB process, and has died silently (no traceback) ~30 min into runs.
Log RSS/available RAM at pipeline boundaries and force a GC when the free floor
is breached so the native allocators get clean memory back.
"""
import gc
import logging

import config

logger = logging.getLogger(__name__)

_psutil = None
try:
    import psutil as _psutil
except Exception:
    _psutil = None


def _free_memory_mb() -> int:
    if _psutil is not None:
        try:
            return int(_psutil.virtual_memory().available // (1024 * 1024))
        except Exception:
            pass
    return -1


def _rss_mb() -> int:
    if _psutil is not None:
        try:
            return int(_psutil.Process().memory_info().rss // (1024 * 1024))
        except Exception:
            pass
    return -1


def log_memory(tag: str) -> None:
    """Log RSS + free RAM for a pipeline boundary."""
    rss = _rss_mb()
    free = _free_memory_mb()
    if rss >= 0 or free >= 0:
        logger.info("mem[%s]: rss=%sMB free=%sMB", tag,
                    rss if rss >= 0 else "?",
                    free if free >= 0 else "?")


def maybe_gc(tag: str) -> None:
    """Force a GC when free RAM is under the floor; always logs the snapshot."""
    free = _free_memory_mb()
    if 0 <= free < config.MEMORY_GC_THRESHOLD_MB:
        before = _rss_mb()
        collected = gc.collect()
        logger.info("mem[%s]: free=%sMB below %sMB floor; gc freed %d objects "
                    "(rss %sMB -> %sMB)", tag, free,
                    config.MEMORY_GC_THRESHOLD_MB, collected,
                    before, _rss_mb())
    else:
        log_memory(tag)
