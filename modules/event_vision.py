"""Local visual-event probing + contact-sheet rendering.

For queries whose truth is VISUAL (a police-car escort, a school bus, a
helicopter fly-by, a gel-blaster injury, a fight) cheap CLIP text scenes probe
local frames so candidates with the object/action on screen are ranked ahead of
audio-only lookalikes — and so the exporter's gate checks real pixels instead of
trusting a spoken prop name.  ``attach_contact_sheet`` tiles the best frames
into one JPEG the LLM judge can look at.
"""
import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# Scene specs per visual-event kind.  ``base`` is the primary CLIP prompt;
# ``variants`` widen the net (phrasing/scale/lighting); ``audio`` are spoken
# forms that corroborate the event in a transcript.
DEFAULT_VISUAL_EVENTS = {
    "police_car": {
        "base": "a police car",
        "variants": ["a police cruiser", "police vehicles with lights on",
                     "a patrol car", "a police car with lights"],
        "audio": {"police", "cops", "cop", "arrest", "escort"},
    },
    "school_bus": {
        "base": "a yellow school bus",
        "variants": ["a school bus on the road", "a yellow bus",
                     "people boarding a school bus"],
        "audio": {"school bus", "school buses", "bus", "buses"},
    },
    "helicopter": {
        "base": "a helicopter flying",
        "variants": ["a helicopter in the sky", "a helicopter landing",
                     "a police helicopter", "a helicopter overhead"],
        "audio": {"helicopter", "chopper", "rotor", "flying"},
    },
    "gel_blaster": {
        "base": "a toy gel blaster gun",
        "variants": ["a toy gun", "a gel water gun", "a nerf gun",
                     "a toy blaster", "someone shooting a toy gun"],
        "audio": {"gel blaster", "blaster", "gel gun", "toy gun"},
    },
    "fight": {
        "base": "two people fighting",
        "variants": ["people fighting each other", "a physical fight",
                     "people wrestling", "a brawl", "people grabbing each other",
                     "people fighting on the floor", "people wrestling on the ground",
                     "a brawl in a dorm room", "people pushing and shoving",
                     "people grabbing each other's shirts", "people yelling at each other"],
        "audio": {"fight", "fighting", "fought", "brawl", "scuffle",
                  "punch", "stop it", "get off", "let go"},
    },
    "pool_party": {
        "base": "a crowd at a pool party",
        "variants": ["people in a swimming pool", "a backyard pool party",
                     "people splashing in a pool"],
        "audio": {"pool party", "party", "pool", "jump in", "swimming", "splash"},
    },
}


class EventProbe:
    """Text-scene probes for one visual-event kind, applied to local frames."""

    def __init__(self, kind: str | None, subject: str = ""):
        self.kind = kind if kind in DEFAULT_VISUAL_EVENTS else None
        spec = DEFAULT_VISUAL_EVENTS.get(self.kind) if self.kind else None
        prompts = [spec["base"]] if spec else []
        prompts += (spec.get("variants") or []) if spec else []
        # Subject-injected prompts: "Kai Cenat in a helicopter" beats a bare
        # "a helicopter flying" for confirming the SUBJECT is part of the scene
        # (the fly-over is Kai's, not stock B-roll).  Kept as extra prompts so
        # the object-only scenes still have their own chance.
        if subject and prompts:
            prompts = list(prompts)
            prompts.insert(0, f"{subject} {spec['base']}")
            prompts.insert(1, f"{subject} in {spec['base']}")
        self.prompts = prompts
        self.audio = (spec or {}).get("audio") or set()

    def active(self) -> bool:
        return bool(self.prompts)

    @classmethod
    def from_prompts(cls, prompts: list[str], kind: str | None = None,
                     subject: str = "") -> "EventProbe":
        """Build a probe from arbitrary scene prompts (the GENERAL path).

        Every plan gets its own scene prompts (script_planner), so queries with
        no recognized event kind still get CLIP pixel grounding via this class.
        ``kind`` is a label used in evidence strings / logs ("scene" when None).
        """
        probe = cls.__new__(cls)
        probe.kind = kind
        probe.prompts = [str(p) for p in (prompts or []) if str(p).strip()]
        probe.audio = set()
        return probe

    def scene_scores(self, paths: list[str]) -> dict[str, float]:
        """Max over all scene prompts of the CLIP cosine for each frame path."""
        if not self.active() or not paths:
            return {}
        from modules import vision_ranker
        matrix = vision_ranker.text_image_scores(self.prompts, paths)
        if matrix is None:
            return {}
        out: dict[str, float] = {}
        for j, path in enumerate(paths):
            out[path] = max(float(matrix[i][j]) for i in range(len(self.prompts)))
        return out

    def summarize(self, scores: dict[str, float]) -> dict:
        vals = list(scores.values())
        if not vals:
            return {"present": False, "best": 0.0, "mean": 0.0, "fraction": 0.0}
        floor = config.EVENT_VISUAL_FLOOR
        best = max(vals)
        mean = sum(vals) / len(vals)
        fraction = sum(1 for v in vals if v >= floor) / len(vals)
        return {
            "present": fraction >= config.EVENT_VISUAL_FRACTION,
            "best": round(best, 3),
            "mean": round(mean, 3),
            "fraction": round(fraction, 3),
        }

    def probe_window(self, path: str, timestamps: list[float]) -> tuple[list[dict], dict]:
        """Sample local frames of a downloaded video and score them.

        Returns (frames, summary).  ``frames`` each carry path/timestamp/sim and
        are sorted best-first; ``summary`` has present/best/mean/fraction.
        """
        frames = _sample_local(path, timestamps)
        if not frames:
            return [], self.summarize({})
        paths = [f["path"] for f in frames]
        scores = self.scene_scores(paths)
        for frame in frames:
            frame["sim"] = scores.get(frame["path"], 0.0)
        frames.sort(key=lambda f: f["sim"], reverse=True)
        return frames, self.summarize(scores)


def probe_for(plan: dict) -> EventProbe:
    """Build an EventProbe from a plan's visual_event_kind (inactive if none)."""
    if not plan or not config.EVENT_VISUAL_ENABLED:
        return EventProbe(None)
    subject = ""
    if isinstance(plan, dict):
        subject = (plan.get("streamer") or {}).get("name") if isinstance(plan.get("streamer"), dict) else ""
        if not subject:
            subject = (plan.get("core_entities") or [""])[0]
        subject = str(subject or "").strip()
    return EventProbe((plan.get("visual_event_kind") or "").strip() or None,
                      subject=subject)


def general_probe(plan: dict) -> EventProbe:
    """The GENERAL visual scene probe for a plan (active for every query).

    Uses the plan's own scene_prompts (script_planner) so all queries get pixel
    grounding — not just the 6 whitelisted event kinds.  Inactive only when the
    planner produced no prompts or general vision is disabled.
    """
    if not plan or not config.GENERAL_VISUAL_ENABLED:
        return EventProbe(None)
    prompts = [str(p) for p in (plan.get("scene_prompts") or []) if str(p).strip()]
    if not prompts:
        return EventProbe(None)
    return EventProbe.from_prompts(prompts, kind="scene")


def audio_terms(plan: dict) -> set[str]:
    """Spoken terms that corroborate a visual event (empty for non-visual)."""
    probe = probe_for(plan)
    return probe.audio if probe.active() else set()


def _sample_local(path: str, timestamps: list[float]) -> list[dict]:
    """Sample downscaled frames of a local file at the given timestamps."""
    if not timestamps:
        return []
    out_dir = config.FRAMES_DIR / ("eventprobe_%s" % (abs(hash(path)) % 100000))
    try:
        from modules.youtube_search import sample_frames_at
        return sample_frames_at(path, timestamps, out_dir)
    except Exception as exc:
        logger.info("Event-probe frame sampling failed for %s: %s", path, exc)
        return []


# --- Contact sheets ----------------------------------------------------------
_contact_sheet_path: str | None = None


def current_contact_sheet() -> str | None:
    return _contact_sheet_path


def attach_contact_sheet(plan: dict, anchors, sentence_index: int = 0) -> str | None:
    """Tile the event's best anchor frames into one JPEG for the LLM judge.

    Uses a weak/mixed anchor set the exporter refuses to gate on, so the judge
    still gets a visual: the sheet is evidence, never a hard requirement.
    Returns the sheet path (or None when there is nothing worth tiling).
    """
    global _contact_sheet_path
    _contact_sheet_path = None
    entries = getattr(anchors, "entries", None) or []
    paths = [e.get("path") for e in entries
             if e.get("path") and Path(e["path"]).exists()]
    if not paths:
        return None
    if len(paths) > config.CONTACT_SHEET_ROWS * config.CONTACT_SHEET_COLS:
        paths = paths[: config.CONTACT_SHEET_ROWS * config.CONTACT_SHEET_COLS]
    sheet_path = config.EVENT_VISION_CACHE / f"contact_{sentence_index}.jpg"
    if build_contact_sheet(paths, sheet_path):
        _contact_sheet_path = str(sheet_path)
        return _contact_sheet_path
    return None


def build_contact_sheet(paths: list[str], out_path) -> object | None:
    """Tile images into a labeled grid JPEG; returns the path or None."""
    if not paths or not config.CONTACT_SHEET_ENABLED:
        return None
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        logger.info("Contact-sheet rendering unavailable: %s", exc)
        return None
    cols = config.CONTACT_SHEET_COLS
    thumb = config.CONTACT_SHEET_THUMB
    thumbs: list[Image.Image] = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb, thumb))
                thumbs.append(im)
        except Exception:
            continue
    if not thumbs:
        return None
    cap = config.CONTACT_SHEET_ROWS * cols
    if len(thumbs) > cap:
        thumbs = thumbs[:cap]
    cell_w = max(im.width for im in thumbs)
    cell_h = max(im.height for im in thumbs)
    sheet = Image.new("RGB", (cell_w * cols, cell_h * config.CONTACT_SHEET_ROWS),
                      (14, 14, 14))
    draw = ImageDraw.Draw(sheet)
    for i, im in enumerate(thumbs):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        sheet.paste(im, (x, y))
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1],
                       outline=(44, 44, 44))
    try:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out_path, "JPEG", quality=82)
        return out_path
    except Exception as exc:
        logger.info("Could not save contact sheet: %s", exc)
        return None
