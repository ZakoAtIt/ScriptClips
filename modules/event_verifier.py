"""Vision-LLM event verification: is the described moment ACTUALLY happening.

The CLIP face/scene probes confirm "the right people / the right object are in
the frame", but they cannot tell whether the MOMENT itself occurs (a gambling
stream with the same faces passes).  qwen2.5vl reads a contact sheet of 6 frames
evenly spread across a 30s window plus the window's spoken audio and answers
"is the target moment really happening here".  The verdict is a HARD gate at
export time (high-confidence "not happening" clips are dropped) and the primary
ranking signal, so a wrong event with the right faces can no longer ship.
Degrades to neutral (None) on any failure; the CLIP tiers remain the fallback.
"""
import base64
import hashlib
import json
import logging
from pathlib import Path

import config
from modules import llm_client

logger = logging.getLogger(__name__)

# Bump when the prompt/sheet/verdict schema changes so stale cached verdicts
# (different resolution, different questions) are never reused.
_SCHEMA_VERSION = "3"

# Source types that are NEVER the original live moment, even when the content
# is on-topic.  A high-confidence verdict of one of these hard-rejects.
_SOURCE_REJECT_TYPES = (
    "reaction", "react", "recap", "highlight", "replay", "compilation",
    "commentary", "review", "vlog",
)

_SYSTEM = (
    "You judge whether a livestream video window shows a MOMENT that matches a "
    "described event. You are shown ONE contact sheet of 6 frames sampled evenly "
    "across the window, the window's spoken audio, and the video's source context "
    "(uploader and title). "
    "MATCH RULES: "
    "(1) MATCH when the frames or audio show the described moment actually "
    "happening live — e.g. an arena/stage victory celebration with a trophy, "
    "confetti, big crowd, lights, or announcer audio that fits the described "
    "competition, even if you cannot read the exact team or person names. "
    "The source context is EVIDENCE the video is about this event, so weight it "
    "with the visuals, not against them. "
    "(2) NOT a match when the window only MENTIONS, DISCUSSES, RECAPS or REACTS "
    "to the moment, shows a replay/recording of it, or shows a totally different "
    "activity (gambling, an unrelated game, a slideshow, a green screen, people "
    "just talking). "
    "(3) If you genuinely cannot tell from the frames and audio, answer "
    "\"match\": \"uncertain\" (never guess a match or a reject). "
    "ALSO classify the window's SOURCE TYPE: original live broadcast/POV footage "
    "of the event is \"live\"; a person watching and commenting on it is "
    "\"reaction\"; a highlights/recap/replay edit is \"recap\"; anything else is "
    "\"other\". Reply only JSON."
)

_PROMPT = """Described event (the moment to match): "%(sentence)s"
Subjects: %(subjects)s
Action: %(action)s
Objects: %(objects)s

Window audio (auto-captions, may be empty):
"%(snippet)s"

Frame sheet 1: 6 frames evenly spread across this %(duration).0fs window.

Source context (the uploader and title this window comes from):
"%(context)s"

Return ONLY JSON, no prose:
{"match": "yes" or "no" or "uncertain", "confidence": 0.0-1.0, "source_type": "live" or "reaction" or "recap" or "other", "reason": "one sentence describing what the frames and audio actually show"}"""


def _sheet(paths: list[str], out_path) -> str | None:
    """Tile up to 6 frames into one 3x2 JPEG the VLM can read (384px cells)."""
    if not paths:
        return None
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        logger.info("VLM sheet rendering unavailable: %s", exc)
        return None
    cols, rows = 3, 2
    cell = 384
    thumbs: list[Image.Image] = []
    for p in paths[: config.VLM_VERIFY_MAX_SHEETS]:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((cell, cell))
                thumbs.append(im)
        except Exception:
            continue
    if not thumbs:
        return None
    while len(thumbs) < cols * rows:
        thumbs.append(thumbs[-1])
    cell_w = max(im.width for im in thumbs)
    cell_h = max(im.height for im in thumbs)
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows), (14, 14, 14))
    draw = ImageDraw.Draw(sheet)
    for i, im in enumerate(thumbs[: cols * rows]):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        sheet.paste(im, (x, y))
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1],
                       outline=(60, 60, 60))
    try:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out_path, "JPEG", quality=85)
        return str(out_path)
    except Exception as exc:
        logger.info("Could not save VLM sheet: %s", exc)
        return None


def _cache_key(sentence: str, video_id: str, start: float, duration: float,
               sheet_path: str | None) -> str:
    blob = hashlib.sha1(sheet_path.encode("utf-8")).hexdigest()[:8] if sheet_path else "x"
    digest = hashlib.sha1(
        f"{_SCHEMA_VERSION}|{sentence}|{video_id}|{start:.1f}|{duration:.1f}|{blob}".encode("utf-8")
    ).hexdigest()[:20]
    return config.VLM_CACHE / f"{digest}.json"


def _load_cache(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _save_cache(path: Path, verdict: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(verdict, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.info("Could not cache VLM verdict: %s", exc)


def verify_window(sentence: str, plan: dict, frame_paths: list[str],
                  snippet: str = "", video_id: str = "video",
                  start: float = 0.0, duration: float = 30.0,
                  cache: bool = True, context: str | None = None) -> dict | None:
    """Ask the VLM whether the moment happens in a 6-frame window sheet.

    Returns ``{"happening": bool, "confidence": float, "reason": str,
    "score": float, "source": "vlm"}`` or None when the VLM is unavailable /
    fails (caller must treat None as neutral).
    """
    if not config.VLM_VERIFY_ENABLED or not llm_client.vision_capable():
        return None
    paths = [p for p in frame_paths if p and Path(p).exists()]
    if not paths:
        return None
    if len(paths) > config.VLM_VERIFY_MAX_SHEETS:
        paths = paths[: config.VLM_VERIFY_MAX_SHEETS]
    sheet_path = _sheet(paths, config.VLM_CACHE / f"vlm_{video_id}_{start:.0f}.jpg")
    if not sheet_path:
        return None
    cache_path = _cache_key(sentence, video_id, start, duration, sheet_path)
    if cache:
        cached = _load_cache(cache_path)
        if cached:
            return cached
    subjects = ", ".join(str(e) for e in (plan.get("core_entities") or []))
    if isinstance(plan.get("streamer"), dict) and plan.get("streamer", {}).get("name"):
        subjects = plan["streamer"]["name"] + (f", {subjects}" if subjects else "")
    action = ", ".join(str(a) for a in (plan.get("action_terms") or []))
    objects = ", ".join(str(o) for o in (plan.get("object_terms") or []))
    try:
        with open(sheet_path, "rb") as fh:
            image = base64.b64encode(fh.read()).decode("ascii")
    except (OSError, ValueError) as exc:
        logger.info("Reading VLM sheet failed: %s", exc)
        return None
    prompt = _PROMPT % {
        "sentence": (sentence or "")[:400],
        "subjects": (subjects or "(unknown)")[:200],
        "action": (action or "(none)")[:200],
        "objects": (objects or "(none)")[:200],
        "snippet": (snippet or "(no audio)")[:900].replace("\n", " "),
        "duration": float(duration),
        "context": (context or "(no source info)")[:200],
    }
    result = llm_client.chat_json(prompt, system=_SYSTEM, images=[image])
    match = str(result.get("match") or "").strip().lower()
    if match not in ("yes", "no", "uncertain"):
        if match in ("true", "1", "true."):
            match = "yes"
        elif match in ("false", "0"):
            match = "no"
        else:
            match = "uncertain"
    if not result:
        logger.info("VLM returned no verdict for %s@%.0fs", video_id, start)
        return None
    try:
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    if match == "uncertain":
        happening = False
        confidence = min(confidence, 0.4)  # neutral, never above the reject floor
    else:
        happening = match == "yes"
    source_type = str(result.get("source_type") or "other").strip().lower()
    verdict = {
        "happening": bool(happening),
        "confidence": round(confidence, 3),
        "reason": str(result.get("reason", ""))[:300],
        "score": round(confidence, 3) if happening else 0.0,
        "source_type": source_type if source_type in _SOURCE_REJECT_TYPES or source_type == "live" else "other",
        "source": "vlm",
    }
    if cache:
        _save_cache(cache_path, verdict)
    return verdict


def attach(candidate: dict, plan: dict, frame_paths: list[str],
           snippet: str | None = None) -> dict | None:
    """Verify one candidate window and attach vlm_* fields to it."""
    sentence = plan.get("sentence") or (candidate.get("query") or "")
    context = None
    channel = candidate.get("channel")
    title = candidate.get("title")
    if channel or title:
        context = f"{channel or ''}: \"{title or ''}\"".strip()
    verdict = verify_window(
        sentence, plan,
        frame_paths or candidate.get("visual_frames") or [],
        snippet=snippet or candidate.get("snippet") or candidate.get("evidence") or "",
        video_id=str(candidate.get("video_id") or "video"),
        start=float(candidate.get("start") or 0.0),
        duration=float(candidate.get("duration") or config.EXACT_WINDOW_SECONDS),
        context=context)
    if not verdict:
        return None
    candidate["vlm_happening"] = verdict["happening"]
    candidate["vlm_confidence"] = verdict["confidence"]
    candidate["vlm_score"] = verdict["score"]
    candidate["vlm_reason"] = verdict["reason"]
    return verdict


def sample_and_verify(clip: dict, plan: dict, path: Path,
                      frames_dir: Path) -> dict | None:
    """Verify a DOWNLOADED clip on its real local pixels (export-time gate).

    When the clip is keyword-pinned (the captions literally spoke the moment's
    phrase) the frames are sampled concentrated around that phrase's timestamp,
    so the VLM sees the actual beat instead of a mostly-unrelated window.
    """
    duration = float(clip.get("duration") or 0)
    if duration <= 0:
        return None
    count = config.VLM_VERIFY_FRAMES
    hit_ts = float(clip.get("keyword_hit_ts") or -1.0)
    if hit_ts >= 0:
        local = max(0.0, min(duration, hit_ts - float(clip.get("start") or 0.0)))
        span = min(duration, 20.0)
        lo = max(0.0, local - span / 2.0)
        hi = min(duration, lo + span)
        if hi - lo < 6.0:
            lo = max(0.0, hi - 6.0)
        timestamps = [lo + (hi - lo) * (i + 1) / (count + 1) for i in range(count)]
    else:
        timestamps = [duration * (i + 1) / (count + 1) for i in range(count)]
    try:
        from modules.youtube_search import sample_frames_at
        frames = sample_frames_at(str(path), timestamps, frames_dir)
    except Exception as exc:
        logger.info("VLM frame sampling failed for %s: %s", path, exc)
        return None
    paths = [f["path"] for f in frames]
    return attach(clip, plan, paths)


def reject_verdict(verdict: dict | None) -> bool:
    """True when a high-confidence 'not happening' verdict must hard-reject."""
    if not verdict:
        return False
    confidence = float(verdict.get("confidence") or 0.0)
    if verdict.get("happening"):
        return False
    if confidence >= config.VLM_REJECT_CONFIDENCE:
        return True
    # A high-confidence classification as a reactor/recap/replay edit is never
    # the original live moment, regardless of on-topic content.
    if (str(verdict.get("source_type") or "").lower() in _SOURCE_REJECT_TYPES
            and confidence >= 0.5):
        return True
    return False
