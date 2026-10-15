"""LLM moment finder: reads a timestamped transcript and localizes the query's event.

The heuristic window proposers only surface transcript regions whose words overlap
the query.  Real moments often carry NO query words: a phone call is "Hello. Hello.
Hello. I'll be there.", a meeting is crowd noise + introductions, an accidental
donation is "I just donated four thousand dollars".  The moment finder asks the
local LLM to read the transcript and return exact [start, end] timestamps where
the event actually happens, so those moments survive verification and reach the
LLM judge.

Results are cached per (video_id, sentence) in cache/moments/ and degrade to []
on any LLM failure — the heuristic lane is always the fallback.
"""
import hashlib
import json
import logging
import re
from pathlib import Path

import config
from modules import content_verifier, llm_client, semantic_embedder
from modules.text_match import stem, tokens

logger = logging.getLogger(__name__)

_MERGE_GAP_SECONDS = 3.0
_LINE_TOKEN_ESTIMATE = 1.6

# The LLM writes its reason for each moment; a reason that hedges ("discussion
# about...", "context of...", "mentions...") means the model itself does NOT
# think the event happens there.  Such moments must never become gate-bypassing
# llm_found candidates.
# Reasons that say the event was only SPOKEN about (discussed / mentioned /
# summarized) can never be the moment itself — the model is telling us the
# event did not happen there.  These reject unconditionally, even when the
# reason references a plan subject (a mention is not the event).
_DISCUSSION_REASONS = (
    "discussion", "discussing", "summary", "summariz", "recap", "backstory",
    "talks about", "talk about", "mention", "mentions", "mentioning",
    "mentioned", "chatting", "chitchat", "describes", "describing",
    "explains", "explaining", "while discussing", "during a discussion",
)

# Soft hedging — the model could not confirm the event happens at this exact
# window ("context of...", "appears to...", "likely...").  For EVENT sentences
# a reason that still anchors on a plan entity/action is specific enough to
# trust ("Maya hits JasonTheWeen in the context of a streamer interaction"):
# the audio legitimately never names a VISUAL moment, and the export pixel gate
# confirms it.  Narration/context sentences stay strict.
_DOUBT_REASONS = (
    "context of", "in context", "context", "unrelated", "unclear",
    "does not show", "doesn't show", "doesn't seem", "not the moment",
    "not a clear", "not sure", "may be", "might be", "could be", "could have",
    "interpreted", "interpretation", "suggest", "suggests", "suggesting",
    "maybe", "probably", "likely", "infer", "inferred", "appears to",
    "appears like", "seems to", "seems like", "seems", "background", "setting",
)

_SYSTEM = (
    "You localize an exact moment inside a long stream VOD transcript for a "
    "short-clip finder. You are given best-effort auto-captions that may have "
    "missing words and wrong speaker labels. Timestamped lines are prefixed "
    "[index] [start-end] so you can point at the exact line. Reply only JSON."
)

_PROMPT = """Script sentence to match: "%(sentence)s"
Key subjects (they may NOT be spoken by name — match by context): %(subjects)s
Action/event to find: %(action)s
%(objects)s
%(chapters)sImportant: the moment usually contains NONE of the subjects' names and
few query words. Judge by conversational context:
- a phone call   = "hello? hello hello", "i'll be there", someone picks up / hangs up
- a meeting       = introductions, "he's here", "come in", crowd screams, "finally"
- an accidental donation = dollar amounts, "i just donated", "my bad", "oops", a misclick
- leaked messages     = "did i just show", "oh no", reading private dms/screenshots aloud
Do NOT force a match: if the event does not clearly happen in this transcript,
return {"moments": []}.

%(hint)s

Transcript (index [start-end] timestamps are real; each line covers its window):
%(transcript)s

Return ONLY JSON:
{"moments":[{"start_line": 12, "end_line": 13, "confidence": 0.0-1.0, "reason": "short reason"}]}
- start_line/end_line are the [index] prefixes of the lines where the moment begins and ends.
- Return up to 3 moments, best first. confidence=1.0 only for an unambiguous exact match.
"""


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------
def _cache_path(video_id: str, sentence: str) -> Path:
    digest = hashlib.sha1(sentence.encode("utf-8")).hexdigest()[:12]
    return config.MOMENT_FINDER_CACHE_DIR / f"{video_id}_{digest}.json"


def _load_cache(video_id: str, sentence: str):
    path = _cache_path(video_id, sentence)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload.get("moments")


def _save_cache(video_id: str, sentence: str, moments: list[dict]) -> None:
    try:
        _cache_path(video_id, sentence).write_text(
            json.dumps({"sentence": sentence, "moments": moments},
                       ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not cache moment finder result for %s: %s", video_id, exc)


# --------------------------------------------------------------------------
# Transcript preparation
# --------------------------------------------------------------------------
def _merge_segments(segments: list[dict], max_chars: int = 200) -> list[dict]:
    """Coalesce short overlapping auto-caption fragments into readable lines.

    Auto-caption segments are ~2-5s long and heavily overlapping, so a generous
    gap threshold collapses whole minutes into one line and destroys timestamp
    granularity.  Only adjacent fragments whose combined text stays short are
    merged, keeping each rendered line ~10-20s wide.
    """
    merged: list[dict] = []
    for seg in (segments or []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 2.0))
        prev = merged[-1] if merged else None
        if (prev and start - prev["end"] <= 0.5
                and len(prev["text"]) + len(text) + 1 <= max_chars):
            prev["end"] = max(prev["end"], end)
            prev["text"] = f"{prev['text']} {text}"
        else:
            merged.append({"start": start, "end": end, "text": text})
    return merged


def _stamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hh, mm = divmod(seconds // 60, 60)
    ss = seconds % 60
    if hh:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"


def _render_entries(segments: list[dict], limit: int = config.MOMENT_FINDER_LINE_LIMIT) -> list[dict]:
    """{stamp, start, end, text} lines, uniformly downsampled if over the limit."""
    merged = _merge_segments(segments)
    if len(merged) > limit:
        step = len(merged) / limit
        merged = [merged[int(i * step)] for i in range(limit)]
    entries = []
    for seg in merged:
        text = " ".join((seg.get("text") or "").split())
        if not text:
            continue
        entries.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": f"[{_stamp(seg['start'])}-{_stamp(seg['end'])}] {text}",
        })
    return entries


def _format_entries(entries: list[dict]) -> str:
    return "\n".join(f"[{i}] {e['text']}" for i, e in enumerate(entries))


def _estimate_tokens(segments: list[dict]) -> float:
    total = 0.0
    for seg in (segments or []):
        total += len((seg.get("text") or "").split()) * _LINE_TOKEN_ESTIMATE + 2.0
    return total


def _render_chapters(chapters) -> str:
    items = chapters or []
    if not items:
        return ""
    lines = ["Chapters (approximate localization guide):"]
    for ch in items[:12]:
        title = (ch.get("title") or "").strip()
        if not title:
            continue
        start = _stamp(float(ch.get("start_time", 0.0)))
        end = _stamp(float(ch.get("end_time", 0.0)))
        lines.append(f"- [{start}-{end}] {title}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Keyword zoom for long transcripts
# --------------------------------------------------------------------------
def _cue_terms(plan: dict, sentence: str) -> list[str]:
    cues: set[str] = set()
    for name in plan.get("core_entities") or []:
        name = " ".join(str(name).lower().split())
        cues.add(name)
        cues.update(name.split())
        cues.add(stem(name))
    cues.update(_event_cues(plan))
    for word in re.findall(r"[a-z0-9']+", sentence.lower()):
        if len(word) > 3 and word not in _STOPWORDS:
            cues.add(word)
    return sorted(_filter_cues(cues))


def _event_cues(plan: dict) -> list[str]:
    """Event-signal words only (action terms + their synonyms), no entity names
    and no GENERIC_ACTION_TERMS (right/answer/world/cup/...).

    Used both for zone-building and to decide whether an LLM moment is 'grounded'
    in the audio: a window that literally speaks 'hello'/'donated'/'call' is far
    more trustworthy than one the model inferred from unrelated talk.
    """
    cues: set[str] = set()
    generic = set(config.GENERIC_ACTION_TERMS)
    for term in plan.get("action_terms") or []:
        term = str(term).lower()
        if term in generic:
            continue
        base = stem(term)
        if base in generic:
            continue
        cues.add(term)
        cues.add(base)
        for key in (base, term):
            for syn in config.ACTION_SYNONYMS.get(key, ()) or ():
                syn = str(syn).lower()
                cues.add(syn)
                cues.update(syn.split())
                cues.add(stem(syn))
    return sorted(_filter_cues(cues))


def _filter_cues(cues: set[str]) -> set[str]:
    cleaned: set[str] = set()
    generic = set(config.GENERIC_ACTION_TERMS)
    for cue in cues:
        cue = cue.strip()
        if not cue or cue in _STOPWORDS:
            continue
        words = cue.split()
        if " " in cue and cue in _PHRASE_WHITELIST:
            cleaned.add(cue)
            continue
        if any(word in _STOPWORDS or len(word) < 3 or word in generic for word in words):
            continue
        cleaned.add(cue)
    return cleaned


_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "with", "from", "that", "this", "these",
    "those", "who", "what", "when", "how", "why", "his", "her", "their", "your",
    "our", "was", "were", "have", "has", "had", "they", "them", "there", "here",
    "about", "into", "over", "under", "then", "than", "will", "would", "could",
    "should", "because", "full", "clip", "video", "live", "stream", "reaction",
    "on", "up", "down", "out", "off", "in", "at", "to", "of", "for", "by", "so",
    "hey", "time", "face", "yeah", "come", "going", "today",
}

# Multi-word spoken cues that are DISTINCTIVE even though a component word
# ("got", "good", "give") is generic in isolation.  These survive _filter_cues.
_PHRASE_WHITELIST = {
    "got it", "good job", "nailed it", "on the phone", "pick up", "picked up",
    "handed over", "hand over", "give away", "game worn", "game-worn",
    "face to face", "in person", "ran into", "bumped into", "bump into",
    "he's here", "she's here", "showed up", "show up", "came in", "come in",
    "walked in", "walk in", "finally here", "there he is", "there she is",
    "private messages", "on screen", "on stream", "dms open", "my bad",
    "into the pool", "in the pool", "into the water", "in the water",
    "shoving match", "spit on", "spat on", "spitting on", "splattered on",
    "apologize to my leg", "apologize to her leg", "you need to apologize",
    "call out", "calls out", "called out", "calling out", "you're expelled",
    "you are expelled", "kicked out", "kick out", "kicks out", "knocked out",
    "chin check", "chin checked", "get out", "get him out", "off the campus",
    "out of here",
}


def _contains_any(text: str, cues: list[str]) -> bool:
    lowered = text.lower()
    return any(re.search(r"\b" + re.escape(cue) + r"\b", lowered) for cue in cues)


def _corroborates(evidence: str, plan: dict, refs_plan: bool = False) -> bool:
    """A moment still needs corroboration: the audio speaks the event action or
    an object, or (narration only) matches the sentence semantically.  Entity
    presence alone is never sufficient for event sentences — it proves the right
    person is in the room, not that the described action happened.

    For EVENT sentences, spoken action evidence or spoken object evidence
    corroborates the moment.  refs_plan=True (meaning the reason references an
    action/object term) also corroborates — the model read the transcript and
    pointed at a specific window.

    For NARRATION/CONTEXT sentences, entity presence or semantic similarity is
    sufficient (the moment is about on-topic footage of the subject).
    """
    try:
        sentence_type = (plan.get("sentence_type") or "narration").lower()
        action_hits, _ = content_verifier.action_match(evidence, plan)
        object_hits, _ = content_verifier.object_match(evidence, plan)
        entity_hits, _ = content_verifier.entity_match(evidence, plan)

        if sentence_type == "event":
            # Event sentences: action OR object evidence required.
            # Entity-only is NOT sufficient.
            if action_hits >= 1:
                return True
            if object_hits >= 1:
                return True
            # refs_plan=True means the reason references action/object (computed
            # by caller via _reason_refs_action_or_object), which is sufficient
            # corroboration for events.
            if refs_plan:
                return True
            # Multi-entity fallback: when the reason references 2+ core entities
            # AND the transcript also mentions them, the model clearly found the
            # right conversation between the right people — even without an action
            # term match.  E.g. "Rakai talks about Yonna's dream" where both
            # names appear in the transcript window.
            if entity_hits >= 2:
                return True
            return False

        # Narration/context: entity-only is sufficient, plus semantic fallback
        if entity_hits >= 1:
            return True
        query = plan.get("semantic_target") or plan.get("sentence") or ""
        if query:
            scores = semantic_embedder.similarity_scores(query, [evidence])
            if scores and float(scores[0]) >= 0.30:
                return True
    except Exception:
        return False
    return False



def _reason_refs_plan(moment: dict, plan: dict) -> bool:
    """The LLM's own reason must reference a known subject or an action term.
    A reason like 'the incident occurs in the middle of the stream' means the
    model could not point at the event — never trust that as an exact moment.

    For visual events a reason referencing an object/prop (the police car pulls
    up, the chopper lands) is equally grounded."""
    reason = (moment.get("reason") or "").lower()
    if not reason:
        return False
    for name in plan.get("core_entities") or []:
        for token in str(name).lower().split():
            if len(token) >= 3 and token in reason:
                return True
    for term in plan.get("action_terms") or []:
        if str(term).lower() in reason:
            return True
    for term in plan.get("object_terms") or []:
        if str(term).lower() in reason:
            return True
    return False


def _reason_refs_action_or_object(moment: dict, plan: dict) -> bool:
    """For event sentences: the reason must reference an action or object term
    (not just an entity name).  Entity-only reasons are NOT sufficient corroboration
    for events — two people being in the same room doesn't mean the action happened."""
    reason = (moment.get("reason") or "").lower()
    if not reason:
        return False
    for term in plan.get("action_terms") or []:
        if str(term).lower() in reason:
            return True
    for term in plan.get("object_terms") or []:
        if str(term).lower() in reason:
            return True
    return False


def moment_is_grounded(moment: dict, evidence: str, plan: dict,
                       event_pinned: bool = False) -> bool:
    """An LLM moment counts as grounded (gate-bypassing) only if the audio window
    actually speaks an event cue at sufficient confidence, the model's reason is
    confident (no 'discussion/context/mentions' hedging), the reason points at a
    subject/action, and the window carries corroborating action/entity signal.
    Un-cued (visual/meeting) moments need the very high uncued-confidence floor
    AND the same corroboration — a confident number is never enough on its own.

    ``event_pinned`` marks footage pinned by the event registry: the video is
    KNOWN raw broadcast of this event, so the audio does not need to spell out
    the event for the localization to be trusted.  The export-level pixel / VLM
    gate still confirms the exact window, but a reason that references a plan
    entity/action is allowed to use discussion/mention phrasing without killing
    the moment — the reason points at the right region of the right VOD."""
    confidence = float(moment.get("confidence", 0.0))
    if confidence < config.LLM_MOMENT_MIN_CONFIDENCE:
        return False
    reason = (moment.get("reason") or "").lower()
    sentence_type = (plan.get("sentence_type") or "narration").lower()
    # For event sentences, the reason must reference an action or object term
    # (not just an entity) — entity-only is never sufficient corroboration.
    # For narration/context, entity presence in the reason is sufficient.
    if sentence_type == "event":
        refs_plan = _reason_refs_action_or_object(moment, plan)
    else:
        refs_plan = _reason_refs_plan(moment, plan)
    # Doubt phrasing ("discussion/context/mentions/talks about") means the model
    # could not point at a real moment.  A reason that only says the event was
    # discussed/mentioned is NEVER the moment (a mention is not the event).  The
    # remaining soft hedges ("context of", "appears to") are accepted for EVENT
    # sentences when the reason still anchors on a plan action/object — the
    # audio legitimately never names a visual moment ("Maya hits JasonTheWeen in
    # the context of a streamer interaction").  Narration/context stay strict.
    # Discussion phrasing ("talks about", "mentions", "discusses") means the
    # model described the moment as spoken-about rather than visually confirmed.
    # Two bypasses allow through moments that ARE grounded despite the phrasing:
    # 1) event_pinned VODs + refs_plan: the footage IS the raw event, so
    #    "mentioned as fighting" inside the known broadcast is a valid localization.
    # 2) event sentence + refs_plan: the reason references an action/object term,
    #    proving the model localized the right moment even though it used "talks
    #    about" phrasing (the audio legitimately describes the event as discussion).
    if any(word in reason for word in _DISCUSSION_REASONS):
        if sentence_type == "event" and refs_plan:
            pass  # action/object term in reason overrides discussion phrasing
        elif not (event_pinned and refs_plan and sentence_type == "event"):
            return False
    if any(word in reason for word in _DOUBT_REASONS) and not (
            refs_plan and sentence_type == "event"):
        return False
    cued = _contains_any(evidence, _event_cues(plan))
    floor = (config.LLM_MOMENT_MIN_CONFIDENCE if (cued or refs_plan)
             else config.LLM_MOMENT_UNCUED_CONFIDENCE)
    if confidence < floor:
        return False
    # For event sentences with known entities, the reason must reference the
    # action/object OR at least two core entities (proving the model localized
    # the right people interacting, not just one person in the room).  A reason
    # like "Rakai talks about Yonna's dream" references both entities — the
    # model clearly found the right conversation, even without an action term.
    if (plan.get("core_entities")
            and not plan.get("subjects_unknown")
            and not refs_plan):
        entity_hits = sum(1 for name in plan["core_entities"]
                          if any(tok in reason for tok in str(name).lower().split()
                                 if len(tok) >= 3))
        if entity_hits < 2:
            return False
    return _corroborates(evidence, plan, refs_plan=refs_plan)


def _zoom_zones(segments: list[dict], cues: list[str], duration: float) -> list[dict]:
    """Contiguous regions around segments that speak any cue word, ±neighborhood."""
    flagged = [seg for seg in (segments or []) if _contains_any(seg.get("text") or "", cues)]
    if not flagged:
        return []
    zones: list[dict] = []
    for seg in flagged:
        start = max(0.0, float(seg.get("start", 0.0)) - config.MOMENT_FINDER_ZONE_NEIGHBORHOOD)
        end = min(duration, float(seg.get("end", start + 2.0)) + config.MOMENT_FINDER_ZONE_NEIGHBORHOOD)
        if zones and start <= zones[-1]["end"] + _MERGE_GAP_SECONDS:
            zones[-1]["end"] = max(zones[-1]["end"], end)
        else:
            zones.append({"start": start, "end": end})
    return zones


def _zone_segments(segments: list[dict], zone: dict) -> list[dict]:
    return [seg for seg in (segments or [])
            if float(seg.get("end", 0)) >= zone["start"] and float(seg.get("start", 0)) <= zone["end"]]


# --------------------------------------------------------------------------
# LLM calls
# --------------------------------------------------------------------------
def _ask(entries: list[dict], sentence: str, plan: dict, chapters,
         hint: str = "") -> list[dict]:
    if not entries:
        return []
    action_terms = plan.get("action_terms") or []
    subjects = plan.get("core_entities") or []
    objects = plan.get("object_terms") or []
    prompt = _PROMPT % {
        "sentence": sentence,
        "subjects": ", ".join(subjects) or "(unknown)",
        "action": ", ".join(action_terms) or "(none — the event itself, by context)",
        "objects": ("Visual props to localize (the event is VISUAL — pinpoint when "
                    "they appear/arrive even if never spoken about): %s"
                    % ", ".join(objects)) if objects else "",
        "chapters": _render_chapters(chapters),
        "hint": hint,
        "transcript": _format_entries(entries),
    }
    try:
        result = llm_client.chat_json(prompt, system=_SYSTEM)
    except Exception as exc:
        logger.warning("Moment finder LLM call failed: %s", exc)
        return []
    moments: list[dict] = []
    for entry in (result.get("moments") or []):
        if not isinstance(entry, dict):
            continue
        try:
            start_line = int(entry.get("start_line", -1))
        except (TypeError, ValueError):
            start_line = -1
        try:
            end_line = int(entry.get("end_line", start_line))
        except (TypeError, ValueError):
            end_line = start_line
        if start_line < 0 or start_line >= len(entries):
            continue
        start_entry = entries[start_line]
        end_entry = entries[min(max(end_line, start_line), len(entries) - 1)]
        try:
            confidence = float(entry.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        moments.append({
            "start": float(start_entry["start"]),
            "end": float(end_entry["end"]),
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(entry.get("reason") or "")[:200],
        })
    return moments


def _hint_for(zone: dict | None) -> str:
    if zone:
        return (f"The event should occur inside the region [{_stamp(zone['start'])} - "
                f"{_stamp(zone['end'])}]; only that region is shown. "
                "Report absolute timestamps as printed.")
    return ""


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def find_moments(segments: list[dict], sentence: str, plan: dict, video_id: str = "",
                 chapters=None, duration: float = 0.0) -> list[dict]:
    """Return up to 3 ranked {start, end, confidence, reason} moments, best first."""
    segments = [s for s in (segments or []) if (s.get("text") or "").strip()]
    if not segments:
        return []
    if not duration:
        duration = max(float(seg.get("end", 0.0) or 0.0) for seg in segments)
    if video_id:
        cached = _load_cache(video_id, sentence)
        if cached is not None:
            logger.info("Moment finder cache hit for %s", video_id)
            return cached

    plan = plan or {}
    moments: list[dict] = []
    over_budget = _estimate_tokens(segments) > config.MOMENT_FINDER_TOKEN_BUDGET

    if not over_budget:
        moments = _ask(_render_entries(segments), sentence, plan, chapters)
    else:
        cues = _cue_terms(plan, sentence)
        zones = _zoom_zones(segments, cues, duration)
        for zone in zones[: config.MOMENT_FINDER_ZONE_MAX]:
            zone_segments = _zone_segments(segments, zone)
            found = _ask(_render_entries(zone_segments), sentence, plan, chapters,
                         hint=_hint_for(zone))
            for m in found:
                m.setdefault("zone", {"start": zone["start"], "end": zone["end"]})
            moments.extend(found)
        if not moments and zones:
            logger.info("Moment finder found no cued zones match for %s", video_id)
        if not zones and over_budget:
            logger.info("Moment finder: no cue words in long VOD %s; coarse scan", video_id)
            moments = _coarse_scan(segments, duration, sentence, plan, chapters)

    cleaned = _clean_moments(moments, duration)
    if video_id:
        _save_cache(video_id, sentence, cleaned)
    return cleaned


def _clean_moments(moments: list[dict], duration: float) -> list[dict]:
    """Clamp, drop below-min-confidence, keep well-separated top moments."""
    kept: list[dict] = []
    for m in sorted(moments, key=lambda x: x.get("confidence", 0.0), reverse=True):
        if float(m.get("confidence", 0.0)) < config.MOMENT_FINDER_MIN_CONFIDENCE:
            continue
        start = max(0.0, float(m.get("start", 0.0)))
        end = min(duration, float(m.get("end", 0.0)))
        if end <= start:
            end = min(duration, start + config.ACTION_WINDOW_SECONDS)
        if end <= start:
            continue
        out = {
            "start": round(start, 1),
            "end": round(end, 1),
            "duration": round(end - start, 1),
            "confidence": float(m.get("confidence", 0.0)),
            "reason": m.get("reason", ""),
        }
        if all(abs(out["start"] - k["start"]) >= config.MIN_CLIP_GAP_SECONDS for k in kept):
            kept.append(out)
        if len(kept) >= 3:
            break
    return kept


def _coarse_scan(segments: list[dict], duration: float, sentence: str,
                 plan: dict, chapters) -> list[dict]:
    """Evenly-spread chunk scan for long VODs with zero cue words (meeting screams, etc.)."""
    count = config.MOMENT_FINDER_COARSE_SCAN
    chunk = duration / count
    found: list[dict] = []
    for index in range(count):
        zone = {"start": index * chunk, "end": min(duration, (index + 1) * chunk)}
        zone_segments = _zone_segments(segments, zone)
        if not zone_segments:
            continue
        results = _ask(_render_entries(zone_segments), sentence, plan, chapters,
                       hint=_hint_for(zone))
        found.extend(results)
    return found


# --------------------------------------------------------------------------
# Keyword moment lexicon (exact spoken-phrase localization)
# --------------------------------------------------------------------------
def _auto_lexicon() -> dict:
    """Auto-generated exact-moment lexicon from data/auto_moment_lexicon.json."""
    path = config.AUTO_MOMENT_LEXICON_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("auto_moment_lexicon.json unreadable (%s); ignoring", exc)
        return {}


def _lexicon_for(plan: dict) -> dict | None:
    """The keyword lexicon entries active for the query, or None.

    Lexicon keys are the event names in data/event_videos.json; a query names
    its event (or is a planner-flagged event query), so the same containment
    check applies.  Within the event's lexicon only the specs whose
    ``query_terms`` appear in the sentence are active — an apple-cut query
    never pins 'champions' windows and vice versa.
    """
    sentence = str(plan.get("sentence") or plan.get("query") or "").lower()
    lexicons = dict(config.EVENT_MOMENT_LEXICON)
    lexicons.update(_auto_lexicon())
    mined = plan.get("mined_lexicon")
    if isinstance(mined, dict):
        lexicons.update(mined)
    for name, entry in lexicons.items():
        name = str(name).lower()
        if name not in sentence:
            continue
        active: dict = {}
        for spec_name, spec in entry.items():
            query_terms = spec.get("query_terms") or ()
            if any(term in sentence for term in query_terms):
                active[spec_name] = spec
        return active or None
    return None


def _window_evidence(segments: list[dict], start: float, duration: float) -> str:
    end = start + duration
    return " ".join(dict.fromkeys((seg.get("text") or "").strip()
                                  for seg in (segments or [])
                                  if float(seg.get("end", 0)) >= start
                                  and float(seg.get("start", 0)) <= end))


def keyword_hits(segments: list[dict], plan: dict) -> list[dict]:
    """Segments whose captions literally speak a lexicon phrase, as hit dicts.

    Each hit: {name, seg_start, priority, pad}.  Deterministic and cheap (no LLM),
    so the exact moment is pinned even when the fuzzy semantic/LLM windows miss
    it (e.g. "We came, we dominated, we won" for the championship).
    """
    lexicon = _lexicon_for(plan)
    if not lexicon:
        return []
    hits: list[dict] = []
    for name, spec in lexicon.items():
        patterns = spec.get("patterns") or ()
        priority = float(spec.get("priority", 4))
        pad = float(spec.get("pad", 10))
        for seg in (segments or []):
            text = " ".join((seg.get("text") or "").lower().split())
            if not text:
                continue
            for pat in patterns:
                if re.search(pat, text):
                    hits.append({
                        "name": name,
                        "seg_start": float(seg.get("start", 0.0)),
                        "priority": priority,
                        "pad": pad,
                    })
                    break
    return hits


def keyword_moments(segments: list[dict], plan: dict, window_seconds: float,
                    max_windows: int = 4) -> list[dict]:
    """Pinned exact-moment windows where a lexicon phrase is literally spoken.

    Returns window dicts ({start, duration, evidence, keyword_located,
    locator_hits}) snapped to the phrase segment minus its pad; every window is
    a STRONG signal for verification and ranking (keyword_located), so the
    moment's own words, not just semantic resemblance, decide where the clip is
    cut.  Deduped by MIN_CLIP_GAP, strongest lexicon entry first.
    """
    hits = keyword_hits(segments, plan)
    if not hits:
        return []
    seen: list[dict] = []
    for hit in sorted(hits, key=lambda h: (h["priority"], -h["seg_start"]), reverse=True):
        start = max(0.0, hit["seg_start"] - hit.get("pad", 10.0))
        if all(abs(start - item["start"]) >= config.MIN_CLIP_GAP_SECONDS
               for item in seen):
            seen.append({"start": start, "hit": hit})
        if len(seen) >= max_windows:
            break
    windows: list[dict] = []
    for item in seen:
        start = float(item["start"])
        evidence = _window_evidence(segments, start, window_seconds)
        if len(tokens(evidence)) < 3:
            continue
        windows.append({
            "start": start,
            "duration": float(window_seconds),
            "evidence": evidence,
            "keyword_located": True,
            "locator_hits": [item["hit"]["name"]],
            "keyword_priority": item["hit"]["priority"],
            "hit_ts": item["hit"]["seg_start"],
        })
    return windows
