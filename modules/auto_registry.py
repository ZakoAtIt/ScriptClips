"""Automatic per-batch registry builder.

Runs BEFORE ``plan_script`` so the whole curated-event machinery engages for
every new batch of queries, with no manual edits:

  * **Triage** (LLM): names people, classifies the event, and extracts the exact
    spoken phrases that mark the moment in the captions.
  * **People**: resolves identities into ``auto_streamers.json`` /
    ``auto_people.json`` (manual entries always win).
  * **Events**: writes ``auto_events.json`` (event phrase -> participants) so
    every participant's channel is scanned.
  * **VODs**: searches + LLM-reranks candidate videos against the query's
    context and pins the best raw full-stream candidates in
    ``auto_event_videos.json`` (the exclusive pool for that event, exactly like
    the curated "streamer games" registry).
  * **Lexicon**: writes ``auto_moment_lexicon.json`` so the keyword moment
    locator can pin the exact moment from the captions.

All auto files are cached (``data/auto_registry.json``) and only re-discovered
when queries.txt changes.  ``prepare()`` returns per-query hints used by the
lanes (languages for Whisper, Kick-fallback flags).
"""
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import config
from modules import downloader, kick_lane, llm_client, streamers
from modules.scoring import metadata_rawness_score
from modules.text_match import STOPWORDS

logger = logging.getLogger(__name__)

# Set to True when prepare() wrote anything; run.py forces a re-plan then.
DIRTY = False

_MAX_PEOPLE_PER_QUERY = 6
_MAX_HOOKS = 6
_MAX_METADATA = 24
_MAX_RERANK = 20
_MAX_PICKED = 6
_PICK_FLOOR = 7.0


# --------------------------------------------------------------------------
# Small JSON helpers (atomic writes)
# --------------------------------------------------------------------------
def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("%s unreadable (%s); treating as empty", path.name, exc)
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:64] or "event"


def _cache_hash(queries: list[str]) -> str:
    return hashlib.sha1("\n".join(queries).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Query triage (LLM)
# --------------------------------------------------------------------------
_TRIAGE_SYSTEM = (
    "You are a media researcher preparing a video-clip pipeline. You extract "
    "structured facts from a query about a livestream moment. Reply ONLY valid "
    "JSON matching the requested schema. Be precise and complete."
)
_TRIAGE_PROMPT = """You extract facts from a livestream-moment query.

QUERY: "%(query)s"

Return JSON with EXACTLY these fields:
{
 "is_event": bool,
 "event_phrase": string,
 "people": [{"name": string, "aliases": [string], "platform": string, "is_streamer": bool}],
 "related_streamers": [string],
 "search_hints": [string],
 "moment_cues": [string],
 "is_news_drama": bool,
 "language": string
}

RULES:
1. people: EVERY distinct person or streamer named in the query is one object.
   NEVER merge two names into one (e.g. "RaKai Yonna" is TWO people, not one).
   Do not invent names. If a person is not named, leave them out.
2. is_event = true when the query names a specific real-life moment/incident
   with footage (a slap, a confrontation, a dream confession).
3. event_phrase = the contiguous substring of the query (verbatim, lowercase)
   naming that moment. STRIP filler words from it: clip, clips, livestream,
   stream, controversial, generated, ai. Example: for "TreyLiving slaps Burt
   Bronx BAU house livestream" use "treyliving slaps burt bronx".
4. moment_cues: 2-4 exact spoken phrases (2+ words) that someone would LITERALLY
   say at the exact moment in the captions, e.g. "wet dream about", "you slapped
   me", "you need to apologize". Never generic words like "clip" or "stream".
5. is_news_drama = true ONLY when the query is about news/reporting/discussion
   of a controversy (e.g. "streamers backlash over policy") rather than raw
   footage of a specific incident.
6. language = "en", "ja" or "other" (dominant spoken language of the footage).
7. related_streamers: real streamer names (not named in the query) who likely
   hosted the moment on their channel. Empty if none. Never platform names
   (kick, twitch, youtube) or countries/cities.
8. search_hints: 2-4 short phrases most likely to surface the RAW full-stream
   VOD of the exact moment (not clips/compilations/reactions).

EXAMPLES:
QUERY: "RaKai talks about Yonna dream livestream clip AI generated"
{"is_event": true, "event_phrase": "rakai talks about yonna dream",
 "people": [{"name": "RaKai", "aliases": ["rakai"], "platform": "twitch", "is_streamer": true},
            {"name": "Yonna", "aliases": ["yonna"], "platform": "unknown", "is_streamer": false}],
 "related_streamers": ["Kai Cenat"], "search_hints": ["rakai yonna relationship"],
 "moment_cues": ["wet dream about", "i had a dream about", "you were in my dream"],
 "is_news_drama": false, "language": "en"}

QUERY: "Twitch streamers Amazon AI training setting backlash livestream"
{"is_event": false, "event_phrase": "",
 "people": [], "related_streamers": [],
 "search_hints": [], "moment_cues": [],
 "is_news_drama": true, "language": "en"}

Now answer for QUERY: "%(query)s" """


def _triage(query: str) -> dict:
    """LLM classification of one query; {} on failure (degrade gracefully)."""
    try:
        result = llm_client.chat_json(
            _TRIAGE_PROMPT % {"query": query}, system=_TRIAGE_SYSTEM)
    except Exception as exc:
        logger.info("Auto-registry triage failed for %r: %s", query, exc)
        return {}
    if not result:
        logger.info("Auto-registry triage returned nothing for %r", query)
        return {}
    people = [p for p in (result.get("people") or [])
              if isinstance(p, dict) and str(p.get("name") or "").strip()]
    result["people"] = _merge_detected_people(query, people)
    return _clean_triage(result, query)


_EVENT_FILLER = {"clip", "clips", "livestream", "stream", "controversial",
                 "generated", "ai", "viral", "japan", "japanese", "vtuber",
                 "streamer", "streamers"}
_EVENT_KEYWORDS = {"slap", "slaps", "slapped", "dream", "confrontation",
                   "argument", "fight", "incident", "apologize", "police",
                   "arrest", "physical"}


def _fallback_phrase(query: str, people: list[dict]) -> str:
    """Build a contiguous event phrase from person names + event keywords."""
    words = query.lower().split()
    positions: dict[str, int] = {}
    for idx, word in enumerate(words):
        positions.setdefault(word, idx)
    anchors: list[int] = []
    for person in people:
        for part in str(person.get("name") or "").lower().split():
            if part in positions:
                anchors.append(positions[part])
    for word in words:
        if word in _EVENT_KEYWORDS:
            anchors.append(positions[word])
    if not anchors:
        return ""
    lo, hi = min(anchors), max(anchors)
    span = [w for w in words[lo:hi + 1] if w not in _EVENT_FILLER]
    return " ".join(span)
_NON_STREAMER_WORDS = {
    "kick", "twitch", "youtube", "japan", "amazon", "united", "states",
    "news", "media", "police", "stream", "live", "clip", "clips", "backlash",
    "setting", "training", "house", "bau",
}
_CUE_FILLER = {"clip", "clips", "stream", "livestream", "reaction", "video",
               "moment", "footage", "controversial"}


def _clean_triage(result: dict, query: str) -> dict:
    """Sanitize LLM fields: filler-free event phrase, real streamer names."""
    lowered = query.lower()

    def _clean_phrase(phrase: str) -> str:
        phrase = " ".join(str(phrase or "").lower().split())
        phrase = " ".join(w for w in phrase.split() if w not in _EVENT_FILLER)
        phrase = re.sub(r"\s+", " ", phrase).strip(" .,;")
        return phrase

    phrase = _clean_phrase(result.get("event_phrase"))
    if phrase and phrase not in lowered:
        phrase = " ".join(str(result.get("event_phrase") or "").lower().split())
    if phrase and phrase not in lowered:
        phrase = ""
    if not phrase and result.get("is_event"):
        phrase = _fallback_phrase(query, result.get("people") or [])
    result["event_phrase"] = phrase

    related = [str(s).strip() for s in (result.get("related_streamers") or [])
               if str(s).strip()
               and str(s).lower().split()[0] not in _NON_STREAMER_WORDS
               and " " not in str(s).strip()]
    result["related_streamers"] = list(dict.fromkeys(related))[:5]

    cues = [str(c).strip() for c in (result.get("moment_cues") or [])
            if str(c).strip() and len(str(c).split()) >= 2
            and not any(w in _CUE_FILLER for w in str(c).lower().split())]
    result["moment_cues"] = list(dict.fromkeys(cues))[:5]

    hints = [str(h).strip() for h in (result.get("search_hints") or [])
             if str(h).strip() and len(str(h).strip()) > 4]
    result["search_hints"] = list(dict.fromkeys(hints))[:5]

    lang = str(result.get("language") or "en").lower()
    result["language"] = lang if lang in ("en", "ja", "other") else "other"
    result.setdefault("is_event", False)
    result.setdefault("is_news_drama", False)
    return result


_NAME_PAIR_RE = re.compile(r"\b([A-Z][a-zA-Z']+)\s+([A-Z][a-zA-Z']+)\b")
_NON_PERSON_LAST = {
    "university", "college", "campus", "house", "street", "city", "world",
    "stream", "streams", "streamer", "streamers", "live", "vod", "clip",
    "clips", "amazon", "japan", "twitch", "kick", "youtube", "police",
    "backlash", "setting", "training", "game", "games", "day", "live",
}


def _merge_detected_people(query: str, people: list[dict]) -> list[dict]:
    """Deterministic safety net: add capitalized names the LLM missed.

    A consecutive-capitalized pair covers both words when it was already
    returned as one person (e.g. "Burt Bronx").  Otherwise add each missing
    single name (e.g. "Nanatty" when only "Ozemi" was returned).
    """
    known = {str(p.get("name") or "").lower() for p in people}
    known_words: set[str] = set()
    for name in known:
        known_words.update(name.split())
    pairs = [(m.group(1), m.group(2))
             for m in _NAME_PAIR_RE.finditer(query)]

    def add(name: str) -> None:
        if name.lower() in known:
            return
        known.add(name.lower())
        people.append({"name": name, "aliases": [name.lower()],
                       "platform": "unknown", "is_streamer": False})

    for first, second in pairs:
        f, s = first.lower(), second.lower()
        if f in _NON_PERSON_LAST or s in _NON_PERSON_LAST:
            continue
        if f in STOPWORDS or s in STOPWORDS:
            continue
        if len(first) < 3 or len(second) < 3:
            continue
        if f"{f} {s}" in known:
            continue
        piece_of_known_f = f in known_words and f not in known
        piece_of_known_s = s in known_words and s not in known
        if f in known and not piece_of_known_s:
            add(second)
        elif s in known and not piece_of_known_f:
            add(first)
        elif not piece_of_known_f and not piece_of_known_s:
            add(first)
            add(second)
    return people


# --------------------------------------------------------------------------
# Identity resolution
# --------------------------------------------------------------------------
def _find_kick_slug(name: str) -> str | None:
    """Best-effort Kick channel slug: try the plain slug forms of the name."""
    candidates = {
        re.sub(r"[\s'\-_.]+", "", name.lower()),
        re.sub(r"[\s'\-_.]+", "_", name.lower()),
        name.lower(),
    }
    for slug in candidates:
        if not slug or len(slug) < 3:
            continue
        try:
            vods = kick_lane.list_channel_videos(slug)
        except Exception as exc:
            logger.info("Kick probe failed for %s: %s", slug, exc)
            vods = []
        if vods:
            logger.info("Resolved Kick channel: %s", slug)
            return slug
    return None


def _canonical_kb_name(name: str) -> str | None:
    """Canonical name if a manual registry already knows this person."""
    entry = streamers.lookup(name)
    if entry:
        return str(entry.get("display_name") or entry.get("name") or entry.get("handle") or "")
    return None


def _resolve_people(triage: dict, notes: list[str]) -> tuple[list[dict], list[dict]]:
    """Resolve triaged people into (streamers_to_add, people_to_add) dicts.

    Streamers are added to the auto-streamers KB (category streamer) with any
    Twitch/Kick identity found; non-streamers go to the auto people registry.
    Manual registry entries are never duplicated.
    """
    streamer_entries: dict[str, dict] = {}
    person_entries: dict[str, dict] = {}

    for person in (triage.get("people") or [])[:_MAX_PEOPLE_PER_QUERY]:
        name = str(person.get("name") or "").strip()
        if not name:
            continue
        aliases = [str(a).strip() for a in (person.get("aliases") or [])
                   if str(a).strip() and str(a).strip().lower() != name.lower()]
        platform = str(person.get("platform") or "unknown").lower()
        is_streamer = bool(person.get("is_streamer"))

        existing_entry = _canonical_kb_name(name)
        if existing_entry:
            existing = streamers.lookup(name)
            if existing and (existing.get("twitch") or existing.get("kick")):
                notes.append(f"person '{name}' already known ({existing_entry}); skipped")
                continue
            notes.append(f"person '{name}' known but no platform identity; re-resolving")

        twitch_handle = None
        kick_slug = None
        if is_streamer or platform in ("twitch", "kick", "unknown"):
            if platform in ("twitch", "unknown"):
                try:
                    resolved = streamers.resolve(name)
                except Exception as exc:
                    logger.info("Twitch resolve failed for %r: %s", name, exc)
                    resolved = None
                if resolved:
                    twitch_handle = str(resolved.get("twitch_handle")
                                        or resolved.get("twitch")
                                        or resolved.get("handle") or "")
                    aliases += [str(a) for a in (resolved.get("aliases") or [])
                                if a and str(a).lower() != name.lower()]
            if platform in ("kick", "unknown") and (is_streamer or platform == "kick"):
                kick_slug = _find_kick_slug(name)
                if kick_slug:
                    notes.append(f"person '{name}' -> kick/{kick_slug}")

        if is_streamer:
            entry = {
                "display_name": name,
                "aliases": list(dict.fromkeys(aliases))[:8],
                "category": "streamer",
            }
            if twitch_handle:
                entry["twitch"] = twitch_handle
            else:
                entry["twitch"] = None
            if kick_slug:
                entry["kick"] = kick_slug
            entry["auto"] = True
            streamer_entries[_slug(name)] = entry
            notes.append(f"auto streamer: {name} (twitch={twitch_handle} kick={kick_slug})")
        else:
            person_entries[_slug(name)] = {
                "display_name": name,
                "aliases": list(dict.fromkeys(aliases))[:8],
                "category": "person",
            }
            notes.append(f"auto person: {name}")

    return list(streamer_entries.items()), list(person_entries.items())


# --------------------------------------------------------------------------
# VOD discovery + LLM rerank
# --------------------------------------------------------------------------
def _discovery_filter(candidate: dict) -> bool:
    """Light metadata filter: drop obvious clip shops before the LLM rerank."""
    title = str(candidate.get("title") or "").lower()
    channel = str(candidate.get("channel") or "").lower()
    if not candidate.get("id") or not title:
        return False
    if "/shorts/" in str(candidate.get("webpage_url") or ""):
        return False
    from modules.source_reject import _HARD_BLOCK
    if any(marker in title for marker in _HARD_BLOCK):
        return False
    if any(marker in channel for marker in config.REACTOR_CHANNEL_MARKERS):
        return False
    if any(marker in channel for marker in config.AGGREGATOR_CHANNEL_MARKERS):
        return False
    return True


def _search_hooks(query: str, triage: dict) -> list[str]:
    people = [str(p.get("name") or "").strip()
              for p in (triage.get("people") or []) if str(p.get("name") or "").strip()]
    related = [str(s).strip() for s in (triage.get("related_streamers") or [])
               if str(s).strip()]
    phrase = str(triage.get("event_phrase") or "").strip()
    hints = [str(h).strip() for h in (triage.get("search_hints") or [])
             if str(h).strip() and len(str(h).strip()) > 4]

    hooks: list[str] = []
    for i, p in enumerate(people[:2]):
        if phrase:
            hooks.append(f"{p} {phrase}")
        for q in people[i + 1:i + 2]:
            hooks.append(f"{p} {q}")
    for rs in related[:3]:
        hooks.append(f"{rs} {phrase}" if phrase else rs)
    if phrase:
        hooks.append(phrase)
    hooks += hints[:4]
    if people and not phrase:
        hooks.append(" ".join(people[:3]))

    deduped: list[str] = []
    for hook in hooks:
        key = " ".join(hook.lower().split())
        if key and key not in deduped:
            deduped.append(key)
    return deduped[:_MAX_HOOKS]


def _collect_candidates(hooks: list[str]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for hook in hooks:
        for fetch in (downloader.search_candidates_exact(hook, max_results=10),
                      downloader.search_candidates(hook, max_results=8)):
            for entry in fetch:
                candidate = dict(entry)
                candidate.setdefault("channel", entry.get("uploader") or entry.get("channel") or "")
                if _discovery_filter(candidate) and candidate["id"] not in by_id:
                    by_id[candidate["id"]] = candidate
    ranked = sorted(by_id.values(), key=lambda c: (
        c.get("title_match_score", 0.0) + 0.5 * c.get("rawness_score", 0.0)), reverse=True)
    logger.info("Auto-registry discovery: %d raw candidate(s) before metadata",
                len(ranked))
    return ranked[:_MAX_METADATA]


def _resolve_candidates(raw: list[dict], notes: list[str]) -> list[dict]:
    """Fetch full metadata (duration/captions) for the top raw candidates."""
    out: list[dict] = []
    for item in raw:
        url = item.get("webpage_url") or f"https://www.youtube.com/watch?v={item['id']}"
        info = downloader.get_video_metadata(url)
        if not info:
            continue
        if info.get("is_live") or info.get("live_status") == "is_live":
            continue
        duration = float(info.get("duration") or 0)
        if duration < config.CLIP_MIN_SECONDS or duration > config.MAX_VIDEO_LENGTH:
            continue
        captions = bool(info.get("automatic_captions") or info.get("subtitles"))
        out.append({
            "id": str(item["id"]),
            "title": info.get("title") or item.get("title") or "",
            "description": (info.get("description") or "")[:400],
            "channel": info.get("uploader") or info.get("channel") or item.get("channel") or "",
            "duration": duration,
            "views": float(info.get("view_count") or 0),
            "upload_date": info.get("upload_date") or "",
            "has_captions": captions,
            "webpage_url": url,
            "rawness_score": metadata_rawness_score(
                info.get("title") or "", info.get("description") or "",
                info.get("uploader") or ""),
        })
        if len(out) >= _MAX_METADATA:
            break
    logger.info("Auto-registry metadata: %d candidate(s) with duration/captions "
                "(captioned=%d)", len(out),
                sum(1 for c in out if c["has_captions"]))
    return out


_RERANK_SYSTEM = (
    "You pick which YouTube videos most plausibly CONTAIN the raw footage of a "
    "described livestream moment. A video CONTAINS the moment when the original "
    "broadcast footage of the exact incident is inside it. Prefer official "
    "channels and multi-hour RAW full-stream uploads / POV broadcasts with "
    "captions. Strongly penalize compilations, recap/edited channels, reaction "
    "channels and clickbait clip shops. Reply only JSON."
)
_RERANK_PROMPT = """Moment/query: "%(query)s"
Event phrase: %(event_phrase)s
People: %(people)s
Spoken cues expected at the moment: %(cues)s

Candidate videos (index: title | channel | duration_mins | views | captions | upload_date):
%(rows)s

Score each 0-10 for how likely the RAW FOOTAGE of the exact moment is inside it.
10 = very likely the exact raw VOD, 7+ = strong candidate raw footage,
4-6 = possibly related but not the moment, 0-3 = clip shop / compilation /
reaction / unrelated.
Return JSON:
{"picks":[{"index": i, "score": 0-10, "note": "one short reason"}],
 "kick_channel": "kick.com slug where the raw footage lives, or null",
 "kick_footage": true or false}"""


def _pin_quality_reject(candidate: dict) -> str | None:
    """Deterministic quality gate for registry pins: a candidate that is a
    reactor / aggregator / clip shop / comedy channel must NEVER be pinned as
    "raw event footage", no matter what the rerank LLM believes from the title.

    Auto-registry has pinned Bach Vlogs (a comedy skit), AJSGB, Crcclips and
    Super Clips as raw VODs before; the raw footage always lives elsewhere.
    """
    channel = str(candidate.get("channel") or candidate.get("uploader") or "").lower()
    title = str(candidate.get("title") or "").lower()
    if not channel and not title:
        return None
    from modules.source_reject import _HARD_BLOCK
    if any(marker in title for marker in _HARD_BLOCK):
        return f"edited/packaged title marker in '{title[:60]}'"
    if any(marker in channel for marker in config.REACTOR_CHANNEL_MARKERS):
        return f"reactor channel '{channel}'"
    if any(marker in channel for marker in config.AGGREGATOR_CHANNEL_MARKERS):
        return f"aggregator channel '{channel}'"
    # Clip / clip-shop channels (Super Clips, Crcclips, ...,Clips) carry edited
    # recaps, not the raw broadcast.  Reactor markers above already cover
    # "reaction" channels; this catches the plain "clips"/"replays" variants.
    if re.search(r"\b(clips|clip|highlights|replays|reacts)\b", channel):
        return f"clip-shop channel '{channel}'"
    return None


def _rerank_vods(candidates: list[dict], query: str, triage: dict) -> tuple[list[dict], dict]:
    if not candidates:
        return [], {"kick_channel": None, "kick_footage": False}
    people = ", ".join(str(p.get("name") or "") for p in (triage.get("people") or [])
                       if str(p.get("name") or ""))
    cues = ", ".join(str(c) for c in (triage.get("moment_cues") or [])) or "unknown"
    # Drop clip shops / reactors / aggregators before the LLM rerank: their
    # titles name the moment verbatim, which over-trusts the small local LLM.
    candidates = [c for c in candidates if not _pin_quality_reject(c)]
    if not candidates:
        logger.info("Auto-registry rerank: all candidates failed the pin "
                    "quality gate; nothing pinned")
        return [], {"kick_channel": None, "kick_footage": False}
    rows = []
    for i, c in enumerate(candidates):
        rows.append(
            f"[{i}] {c['title'][:90]} | {c['channel'][:50]} | "
            f"{c['duration']/60.0:.0f}min | {int(c['views']):,} | "
            f"{'yes' if c['has_captions'] else 'no'} | {c['upload_date'][:8]}")
    result = llm_client.chat_json(
        _RERANK_PROMPT % {
            "query": query,
            "event_phrase": triage.get("event_phrase") or "",
            "people": people or "(none named)",
            "cues": cues,
            "rows": "\n".join(rows),
        },
        system=_RERANK_SYSTEM)
    meta = {
        "kick_channel": result.get("kick_channel") or None,
        "kick_footage": bool(result.get("kick_footage")),
    }
    picks = [p for p in (result.get("picks") or [])
             if isinstance(p, dict) and p.get("index") is not None]
    scored = []
    for p in picks:
        try:
            idx = int(p["index"])
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < len(candidates)):
            continue
        entry = dict(candidates[idx])
        entry["_score"] = float(p.get("score") or 0.0)
        entry["_note"] = str(p.get("note") or "")[:200]
        scored.append(entry)
    if not scored:
        scored = [dict(c, _score=5.0, _note="LLM rerank failed; best-effort")
                  for c in candidates[:_MAX_PICKED]]
    scored.sort(key=lambda c: (c["_score"], 1.0 if c["has_captions"] else 0.0),
                reverse=True)
    strong = [c for c in scored if c["_score"] >= _PICK_FLOOR]
    chosen = strong[:_MAX_PICKED]
    if len(chosen) < 2 and scored:
        chosen = scored[:max(len(chosen), min(2, len(scored)))]
    # Prefer caption-bearing picks when trimming to the cap.
    chosen.sort(key=lambda c: (1.0 if c["has_captions"] else 0.0, c["_score"]),
                reverse=True)
    chosen = chosen[:_MAX_PICKED]
    logger.info("Auto-registry rerank: %d pick(s) (strong=%d)",
                len(chosen), len(strong))
    for c in chosen:
        logger.info("  pick %.1f [%s] captions=%s %s", c["_score"], c["id"],
                    c["has_captions"], c["title"][:70])
    return chosen, meta


# --------------------------------------------------------------------------
# Distinctive words for lexicon query_terms
# --------------------------------------------------------------------------
def _distinctive_words(sentence: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9']+", sentence.lower())
             if len(w) >= 3 and w not in STOPWORDS
             and w not in config.GENERIC_ACTION_TERMS]
    return list(dict.fromkeys(words))[:8]


# --------------------------------------------------------------------------
# Auto-file writers
# --------------------------------------------------------------------------
def _merge_add(path: Path, items: list[tuple[str, dict]]) -> None:
    if not items:
        return
    data = _read_json(path)
    for key, entry in items:
        data[key] = entry
    _write_json(path, data)
    global DIRTY
    DIRTY = True
    logger.info("Auto-registry: wrote %d entr(y/ies) to %s", len(items), path.name)


def _register_event(sentence: str, triage: dict, picked: list[dict],
                    streamer_keys: list[str], notes: list[str]) -> dict:
    """Write auto_events / auto_event_videos / auto_moment_lexicon entries.

    Returns per-query hints (languages, kick fallback, vods).
    """
    phrase = str(triage.get("event_phrase") or "").strip().lower()
    query = " ".join(sentence.lower().split())
    event_key = phrase or _slug(sentence)
    languages = [str(triage.get("language") or "en").lower()]

    # events.json: phrase -> channeled participants (only those with identity).
    if phrase and streamer_keys:
        _merge_add(config.AUTO_EVENTS_FILE,
                   [(phrase, [k for k in streamer_keys if k])])

    # event_videos.json: the pinned raw VOD pool (the exclusive pool downstream).
    if picked:
        videos = [{
            "id": c["id"],
            "title": c["title"],
            "note": (c.get("_note") or f"LLM score {c['_score']:.1f}")[:220],
            "has_captions": bool(c["has_captions"]),
        } for c in picked]
        channels = list(dict.fromkeys(str(c["channel"]).strip() for c in picked
                                      if str(c["channel"]).strip()))
        _merge_add(config.AUTO_EVENT_VIDEOS_FILE, [(
            event_key, {
                "label": phrase or sentence,
                "channels": channels,
                "videos": videos,
                "auto": True,
            })])
        for c in picked:
            notes.append(f"  VOD pick score={c['_score']:.1f} captions={c['has_captions']} "
                         f"[{c['id']}] {c['title'][:80]}")

    # Lexicon: exact spoken cues -> keyword moment locator patterns.
    cues = [str(c).strip().lower() for c in (triage.get("moment_cues") or [])
            if str(c).strip() and len(str(c).split()) >= 2]
    patterns = [re.escape(cue) for cue in cues if cue]
    if patterns and event_key:
        query_terms = _distinctive_words(query) or [w for w in query.split()[:4] if len(w) >= 3]
        lexicon = _read_json(config.AUTO_MOMENT_LEXICON_FILE)
        spec = lexicon.setdefault(event_key, {})
        spec[_slug("_".join(cues[:1]) or "moment")] = {
            "query_terms": query_terms,
            "patterns": patterns,
            "pad": 12,
            "priority": 5,
        }
        _write_json(config.AUTO_MOMENT_LEXICON_FILE, lexicon)
        global DIRTY
        DIRTY = True
        notes.append(f"  lexicon cues: {cues}")

    return {
        "event_key": event_key,
        "languages": languages,
        "vods": [c["id"] for c in picked],
        "captioned_vods": [c["id"] for c in picked if c["has_captions"]],
    }


def _kick_meta(triage: dict, rerank_meta: dict, hints: dict) -> dict:
    """Kick fallback decision: only when no captioned YouTube VOD won."""
    platform_kick = any(str(p.get("platform") or "").lower() == "kick"
                        for p in (triage.get("people") or []))
    slug = rerank_meta.get("kick_channel") or (
        triage.get("_kick_slug") if isinstance(triage.get("_kick_slug"), str) else None)
    kick_footage = bool(rerank_meta.get("kick_footage")) or bool(slug) or platform_kick
    captioned_won = bool(hints.get("captioned_vods"))
    enabled = kick_footage and (platform_kick or not captioned_won)
    if slug is None and enabled and platform_kick:
        # Resolve the kick slug from the named kick-streamer directly.
        for person in (triage.get("people") or []):
            if str(person.get("platform") or "").lower() == "kick":
                slug = _find_kick_slug(str(person.get("name") or ""))
                if slug:
                    break
    return {"slug": slug, "enabled": enabled}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _already_registered(query: str) -> bool:
    lowered = query.lower()
    for key, entry in _read_json(config.AUTO_EVENT_VIDEOS_FILE).items():
        label = str(entry.get("label") or key).lower()
        if key in lowered or (label and label in lowered):
            return True
    return False


def prepare() -> dict:
    """Discover registries for the current queries.txt; returns hints by sentence.

    Uses the cached result when queries.txt is unchanged.  Fast and safe when
    the LLM is down: failures degrade to the pre-auto behaviour.
    """
    global DIRTY
    queries = [q.strip() for q in config.QUERIES_FILE.read_text(encoding="utf-8").splitlines()
               if q.strip()] if config.QUERIES_FILE.exists() else []
    if not queries:
        return {}
    digest = _cache_hash(queries)
    cache = _read_json(config.AUTO_REGISTRY_CACHE)
    cached_sentences = {s.get("sentence") for s in cache.get("queries") or []}
    if cache.get("queries_hash") == digest and cached_sentences >= set(queries):
        logger.info("Auto-registry: cache hit (%d query hint(s))", len(cached_sentences))
        DIRTY = False
        return {s["sentence"]: {k: v for k, v in s.items() if k != "sentence"}
                for s in cache.get("queries") or []}

    logger.info("Auto-registry: discovering registries for %d quer(y/ies)", len(queries))
    if not llm_client.available():
        logger.warning("Ollama unreachable; auto-registry disabled for this run "
                       "(pipeline falls back to platform search)")
        return {}

    report_queries: list[dict] = []
    hints: dict[str, dict] = {}
    for query in queries:
        entry: dict = {"sentence": query}
        notes: list[str] = []
        if _already_registered(query):
            notes.append("already registered in event registry; skipped")
            hints[query] = {"registered": True, "notes": notes}
            report_queries.append({**entry, "hints": hints[query]})
            continue

        triage = _triage(query)
        entry["triage"] = {k: v for k, v in triage.items() if k != "people"}
        entry["people"] = triage.get("people") or []
        if not triage:
            notes.append("LLM triage failed; no registries written")
            hints[query] = {"registered": False, "notes": notes}
            report_queries.append({**entry, "hints": hints[query]})
            continue
        if triage.get("is_news_drama"):
            notes.append("news/drama topic (no raw stream footage); routed to image/news handling")
            hints[query] = {"registered": False, "news_drama": True,
                            "languages": [str(triage.get("language") or "en").lower()],
                            "notes": notes}
            report_queries.append({**entry, "hints": hints[query]})
            continue

        streamer_items, person_items = _resolve_people(triage, notes)
        _merge_add(config.AUTO_STREAMERS_FILE, streamer_items)
        _merge_add(config.AUTO_PEOPLE_FILE, person_items)

        hints_for_query: dict = {"registered": True, "languages": []}
        rerank_meta: dict = {"kick_channel": None, "kick_footage": False}
        if triage.get("is_event"):
            hooks = _search_hooks(query, triage)
            raw = _collect_candidates(hooks)
            resolved = _resolve_candidates(raw, notes)
            picked, rerank_meta = _rerank_vods(resolved, query, triage)
            hints_for_query = _register_event(
                query, triage, picked,
                [k for k, _ in streamer_items], notes)
            if picked:
                notes.append(f"event registered under '{hints_for_query.get('event_key')}'")
            elif notes:
                notes.append("no VOD picks survived discovery; platform search will run")
        else:
            notes.append("not flagged as an event; platform search will run")

        kick = _kick_meta(triage, rerank_meta, hints_for_query)
        triage["_kick_slug"] = kick.get("slug")
        hints_for_query["kick"] = kick
        hints_for_query["registered"] = True
        hints_for_query["notes"] = notes
        hints[query] = hints_for_query
        report_queries.append({**entry, "hints": hints_for_query})

    cache["queries_hash"] = digest
    cache["queries"] = report_queries
    _write_json(config.AUTO_REGISTRY_CACHE, cache)
    _write_report(report_queries)
    DIRTY = True
    return hints


def _write_report(report_queries: list[dict]) -> None:
    """Human-readable review report of everything the stage auto-registered."""
    try:
        report = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": str(config.QUERIES_FILE),
            "queries": report_queries,
        }
        path = config.AUTO_REPORT_DIR / "auto_registry_report.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        logger.info("Auto-registry review report written to %s", path)
    except Exception as exc:
        logger.warning("Could not write auto-registry report: %s", exc)


def hints() -> dict:
    """Cached per-query hints (fast path for run.py), keyed by sentence."""
    cache = _read_json(config.AUTO_REGISTRY_CACHE)
    return {s.get("sentence", ""): {k: v for k, v in s.items() if k != "sentence"}
            for s in cache.get("queries") or []}
