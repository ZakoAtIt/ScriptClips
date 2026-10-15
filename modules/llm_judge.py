"""Human-like final judgment via the local LLM.

The ensemble gives a numeric score; the LLM is asked whether each top
candidate ACTUALLY shows the sentence's moment, and to reject reactors /
re-uploads / compilations that metadata could not catch.  Degrades silently to
a no-op when ollama is off.
"""
import base64
import json
import logging
from pathlib import Path

import config
from modules import content_verifier, llm_client, streamers

logger = logging.getLogger(__name__)

# Action verbs that must be HEARD/SPOKEN to count as happening.  Physical/visual
# actions (give, meet, receive, chase) are left lenient so on-topic footage is
# still accepted when the moment has little or no dialogue.
_SPEECH_ACTIONS = {
    "call", "answer", "pick", "guess", "nail", "ask", "react", "reveal",
    "announce", "say", "tell", "admit", "agree", "refuse", "play",
}
_SYSTEM = (
    "You are the final human judge for a short-clip finder. You must decide "
    "whether each candidate clip is usable footage for the script sentence. "
    "REJECT clips that are reactions by other people, compilations, recaps, "
    "re-uploads, or unrelated content. For narration/context sentences the "
    "line describes the subject's situation, so on-topic footage of the "
    "subject counts even if the exact words are not spoken. Reply only JSON."
)

_PROMPT = """Script sentence to match: "%(sentence)s"
Sentence type: %(sentence_type)s
Action words to look for in the audio: %(action)s

%(type_hint)s
%(visual_evidence)s
Candidate clips (transcript snippets from their actual audio):
%(candidates)s

For each candidate return JSON:
{"verdicts": [{"index": 0, "accept": true, "score": 0, "reason": "short reason"}]}
Score is 0-10, where 10 = the clip matches the sentence exactly, 5 = clearly on-topic footage of the subject, 0 = wrong/unrelated. Accept clips that are clearly on-topic; only reject clips that are reactions/compilations/re-uploads/unrelated or feature a different person.

Source rule: each candidate's channel_kind is listed. Twitch/original and official channels are the streamer's own footage. A "raw" channel is a third party that only re-hosts raw VODs. Candidates from THIRD-PARTY channels must be REJECTED even when the content is on-topic, because the product requires the ORIGINAL moment.
"""


def _channel_kind(c: dict, plan: dict | None) -> str:
    if c.get("source") == "twitch":
        return "twitch/original"
    streamer = plan.get("streamer") if isinstance(plan, dict) else None
    channel = c.get("channel") or c.get("uploader") or ""
    if streamers.is_official_channel(streamer, channel):
        return "official"
    if streamers.is_raw_footage_channel(
            channel, streamer,
            title_or_desc=f"{c.get('title') or ''} {c.get('description') or ''}"):
        return "raw"
    return "third-party"


def _render(candidates: list[dict], plan: dict | None,
            sheet_nums: dict[int, int] | None = None) -> str:
    sheet_nums = sheet_nums or {}
    lines = []
    for i, c in enumerate(candidates):
        text = (c.get("snippet") or c.get("evidence") or c.get("transcript") or "")[:1000].replace("\n", " ")
        moment = ""
        if c.get("llm_moment_score"):
            moment = (f" moment_finder_conf={float(c['llm_moment_score']):.2f}"
                      f" ({c.get('llm_moment_reason') or 'localized by transcript scan'})")
        frame_sheet = f"frame_sheet={sheet_nums.get(i, 'none')}"
        lines.append(
            f'[{i}] source={c.get("source")} channel_kind={_channel_kind(c, plan)} '
            f'{frame_sheet} '
            f'title="{c.get("title")}" '
            f'channel="{c.get("channel") or c.get("uploader") or ""}" '
            f'duration={c.get("duration")}s{moment}\n    audio: {text}')
    return "\n".join(lines)


def _forced_moments() -> dict:
    """User-forced moment keys (video_id:start) that bypass judge rejection."""
    try:
        payload = json.loads(config.FORCED_MOMENTS_FILE.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _forced_reason(video_id: str, start: float) -> str | None:
    """Forced-moment reason when the candidate is within 1s of a forced start."""
    for key, reason in _forced_moments().items():
        if not isinstance(key, str) or ":" not in key:
            continue
        vid, _, secs = key.partition(":")
        if vid != video_id:
            continue
        try:
            if abs(float(secs) - start) <= 1.0:
                return str(reason) if reason else None
        except (TypeError, ValueError):
            continue
    return None


def _frame_sheets(candidates: list[dict], fallback: str | None) -> list[tuple[int, str]]:
    """Tile each candidate's sampled frames into one labeled JPEG (judge only).

    Returns a list of (top_index, sheet_path) so the judge can see the ACTUAL
    pixels of each clip — not just the event reference sheet.  A candidate with
    no sampled frames falls back to the reference sheet so the visual is still
    provided; candidates with neither get no sheet.
    """
    from modules import event_vision
    sheets: list[tuple[int, str]] = []
    for i, c in enumerate(candidates):
        paths = [p for p in (c.get("visual_frames") or c.get("event_visual_frames") or [])
                 if p and Path(p).exists()]
        if not paths and fallback and Path(fallback).exists():
            paths = [fallback]
        if not paths:
            continue
        if len(paths) > config.CONTACT_SHEET_ROWS * config.CONTACT_SHEET_COLS:
            paths = paths[: config.CONTACT_SHEET_ROWS * config.CONTACT_SHEET_COLS]
        out = config.EVENT_VISION_CACHE / f"judge_{i}.jpg"
        try:
            if event_vision.build_contact_sheet(paths, out):
                sheets.append((i, str(out)))
        except Exception as exc:
            logger.info("Judge frame sheet failed for candidate %d: %s", i, exc)
    return sheets


def judge(sentence: str, plan: dict, candidates: list[dict],
          pool_size: int = 8) -> list[dict]:
    """Score the top candidates; attaches llm_score / llm_reason / llm_reject."""
    # Rank by relevance, not MiniLM alone: conversational Twitch audio scores
    # low semantically, so clips that actually speak the action/entities must
    # still reach the judge.
    def _relevance(c: dict) -> float:
        return (float(c.get("semantic_score", 0.0))
                + 0.6 * float(c.get("action_hits", 0))
                + 0.3 * float(c.get("entity_hits", 0))
                + 0.8 * float(c.get("llm_moment_score", 0.0)))

    top = sorted(candidates, key=_relevance, reverse=True)[:pool_size]
    if not top:
        return candidates
    sentence_type = (plan.get("sentence_type") or "narration").lower()
    narration = sentence_type in ("narration", "context")
    action_terms = plan.get("action_terms") or []
    # Physical/visual events (fight, escort, helicopter, gel blaster) happen on
    # screen, not in the transcript — never hold their audio to speech standards.
    visual_event = bool(plan.get("visual_event"))
    strict = bool(set(action_terms) & _SPEECH_ACTIONS) and not visual_event
    if narration:
        type_hint = (
            "This is a narration/context line: it describes the subject's situation "
            "rather than a specific moment. Clearly on-topic footage of the subject "
            "is acceptable (score 5-7); prefer footage that matches the topic."
        )
    elif strict:
        type_hint = (
            "This is a specific action/event line. The clip MUST actually show or "
            "speak the described moment (e.g. the call being made and answered, the "
            "guess being made and revealed). If the people are right but the action "
            "does not happen in the clip, REJECT it (score 0-2). Only accept "
            "(score 8-10) clips where the moment clearly occurs."
        )
    elif visual_event:
        type_hint = (
            "This is a VISUAL event line: it turns on what is on SCREEN (a police "
            "car, a helicopter, a school bus, a gel blaster, a fight). The pixels "
            "decide, not the audio — accept clips that show the object/action even "
            "if nobody speaks about it; reject clips that only mention it in words "
            "without showing it."
        )
    else:
        type_hint = (
            "This is a specific dialogue/event line: the clip should actually show "
            "or speak the described moment (score 8-10); on-topic-but-not-exact "
            "footage can still be accepted at 5-7."
        )
    try:
        from modules import event_vision
        sheet = event_vision.current_contact_sheet()
    except Exception:
        sheet = None
    images: list[str] = []
    sheet_nums: dict[int, int] = {}
    if llm_client.vision_capable():
        sheets = _frame_sheets(top, sheet)
        for image_num, (idx, path) in enumerate(sheets, start=1):
            sheet_nums[idx] = image_num
            try:
                with open(path, "rb") as fh:
                    images.append(base64.b64encode(fh.read()).decode("ascii"))
            except (OSError, ValueError) as exc:
                logger.info("Reading judge frame sheet failed: %s", exc)
                sheet_nums.pop(idx, None)
        if images:
            visual_evidence = (
                "Image sheets: attached frame sheets are numbered 1..N and the "
                "candidates' frame_sheet=N fields map each clip to its image. "
                "Judge the ACTUAL PIXELS: accept a clip only if its frames show "
                "the described object, person, or action on screen. Reject clips "
                "whose frames show something else, even if the audio mentions it."
            )
        else:
            visual_evidence = (
                f"No image sheets could be attached; the event reference stills "
                f"are at:\n{sheet}\nJudge on the transcript audio and metadata."
                if sheet
                else "No reference stills are available; judge on the transcript audio.")
    else:
        visual_evidence = (
            f"Reference stills (contact sheet) for this event are at:\n{sheet}\n"
            "If you can read images, compare the candidate frames' visual evidence to "
            "these stills before scoring." if sheet
            else "No reference stills are available; judge on the transcript audio.")
    result = llm_client.chat_json(
        _PROMPT % {"sentence": sentence, "sentence_type": sentence_type,
                   "action": ", ".join(action_terms) or "(none)",
                   "type_hint": type_hint,
                   "visual_evidence": visual_evidence,
                   "candidates": _render(top, plan, sheet_nums)},
        system=_SYSTEM, images=images)
    verdicts = result.get("verdicts") or []
    by_index: dict[int, dict] = {}
    for v in verdicts:
        try:
            idx = int(v.get("index"))
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict) and (v.get("score") is not None or v.get("accept") is not None):
            by_index[idx] = v
    for idx, candidate in enumerate(top):
        v = by_index.get(idx)
        if not v:
            continue
        score = 0.0
        try:
            score = max(0.0, min(10.0, float(v.get("score", 0))))
        except (TypeError, ValueError):
            score = 0.0
        candidate["llm_score"] = score / 10.0
        candidate["llm_reason"] = str(v.get("reason", ""))[:300]
        reject = bool(v.get("accept") is False or (v.get("accept") is not True and score < 1.0))
        key = "%s:%.1f" % (candidate.get("video_id", ""), float(candidate.get("start", -1.0)))
        reason = _forced_reason(candidate.get("video_id", ""), float(candidate.get("start", -1.0)))
        if reject and reason:
            reject = False
            candidate["llm_reason"] = ("%s [user-forced: %s]" %
                                       (candidate.get("llm_reason", ""),
                                        reason)).strip()[:320]
            logger.info("Judge rejection overridden for forced moment %s", key)
        if reject:
            trusted, treason = content_verifier.transcript_verified(candidate, plan)
            if trusted:
                reject = False
                candidate["trusted"] = True
                candidate["llm_reason"] = ("%s [transcript-verified: %s]" %
                                           (candidate.get("llm_reason", ""),
                                            treason)).strip()[:320]
                logger.info("Judge rejection overridden for transcript-verified "
                            "moment %s", key)
        candidate["llm_reject"] = reject
    rejected = sum(1 for c in candidates if c.get("llm_reject"))
    if rejected:
        logger.info("LLM judge rejected %d candidate(s)", rejected)
    return candidates
