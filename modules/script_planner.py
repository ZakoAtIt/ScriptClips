"""Turn queries.txt into a plan per query (no LLM).

Each non-empty line of queries.txt is a well-detailed query that names the
subjects, object and action.  Subject names are resolved offline against the
streamers KB (data/streamers.json) plus a people registry (data/people.json),
so the face gate knows exactly which people must appear on screen.  Each line
produces up to CLIPS_PER_SENTENCE clips + IMAGES_PER_SENTENCE images.

Plans are cached in data/sentences.json; edit the queries (or the plan) and
re-run the lanes without re-parsing by deleting the cache.
"""
import json
import logging
import re
from pathlib import Path

import config
from modules import streamers

logger = logging.getLogger(__name__)


def load_queries() -> list[str]:
    """Non-empty, whitespace-collapsed lines of queries.txt."""
    if not config.QUERIES_FILE.exists():
        return []
    return [q.strip() for q in config.QUERIES_FILE.read_text(encoding="utf-8").splitlines()
            if q.strip()]


def _load_people() -> list[dict]:
    entries: list[dict] = []

    def _ingest(path, prefer_manual: bool):
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8")) or {}
            for handle, entry in data.items():
                e = dict(entry)
                e.setdefault("handle", handle)
                e.setdefault("aliases", [])
                canon = _canonical(e).lower()
                # Manual registries win over auto ones (same canonical name).
                if prefer_manual and any(_canonical(x).lower() == canon for x in entries):
                    continue
                entries.append(e)
        except Exception as exc:
            logger.warning("%s unreadable (%s); using empty registry", path.name, exc)

    _ingest(config.PEOPLE_FILE, prefer_manual=True)
    _ingest(config.AUTO_PEOPLE_FILE, prefer_manual=False)
    return entries


def _load_topics() -> list[dict]:
    """Non-person subjects (movies, franchises, event names) from topics.json."""
    entries: list[dict] = []
    path = config.TOPICS_FILE
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8")) or {}
            for handle, entry in data.items():
                e = dict(entry)
                e.setdefault("handle", handle)
                e.setdefault("name", e.get("display_name") or handle)
                e.setdefault("aliases", [])
                entries.append(e)
        except Exception as exc:
            logger.warning("topics.json unreadable (%s); using empty registry", exc)
    return entries


def _match_topics(text: str) -> list[dict]:
    """Topic registry entries whose name / alias appears in the text.

    Topics are matched separately from people/streamers so they never enter
    core_entities (which drives the face gate — a character like Spider-Man
    must not force a face match) while still feeding search and verification.
    """
    lowered = text.lower()
    matched: list[dict] = []
    for entry in _load_topics():
        names = [entry.get("name") or "", entry.get("display_name") or ""]
        names += entry.get("aliases") or []
        for name in names:
            name = " ".join(str(name).strip().lower().split())
            if not name or len(name) < 3:
                continue
            if re.search(r"\b" + re.escape(name) + r"\b", lowered):
                if not any(_canonical(m) == _canonical(entry) for m in matched):
                    matched.append(entry)
                break
    return matched


def _registry() -> list[dict]:
    kb = streamers._load_kb()
    entries = [dict(entry, handle=handle) for handle, entry in kb.items()]
    entries.extend(_load_people())
    return entries


def _canonical(entry: dict | None) -> str:
    if not entry:
        return ""
    return str(entry.get("display_name") or entry.get("name")
               or entry.get("handle") or "").strip()


def _load_events() -> dict:
    """data/events.json + data/auto_events.json: event phrase -> participant handles."""
    merged: dict = {}

    def _ingest(path, prefer_manual: bool):
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("%s unreadable (%s); no event expansion", path.name, exc)
            return
        for phrase, names in (data.items() if isinstance(data, dict) else {}):
            phrase = str(phrase).lower()
            if prefer_manual or phrase not in merged:
                merged[phrase] = names

    _ingest(config.EVENTS_FILE, prefer_manual=True)
    _ingest(config.AUTO_EVENTS_FILE, prefer_manual=False)
    return merged


def _event_candidate_streamers(query: str) -> list[dict]:
    """Participants of an event named by the query, as streamer entries.

    An event (e.g. "Streamer Games") is hosted by many participants, each with
    their own POV footage on Twitch/Kick.  When the query names no primary
    streamer, the event map expands it into one sub-plan per participant so
    every channel that could carry the moment is scanned.  Only participants
    with a resolvable Twitch/Kick identity qualify.
    """
    candidates: list[dict] = []
    seen: set[str] = set()
    lowered = query.lower()
    for phrase, names in _load_events().items():
        if phrase.lower() not in lowered:
            continue
        for name in names:
            try:
                entry = streamers.lookup(name) or streamers.resolve(name)
            except Exception:
                entry = None
            if not entry:
                continue
            handle = _twitch_handle(entry)
            kick = streamers.kick_handle(entry)
            if not handle and not kick:
                continue
            key = (handle or kick or "").lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "name": _canonical(entry),
                "twitch_handle": handle,
                "kick": kick,
                "youtube": entry.get("youtube") or [],
                "aliases": entry.get("aliases") or [],
                "category": entry.get("category"),
            })
    return candidates


def _twitch_handle(entry: dict) -> str | None:
    """Twitch handle for a registry entry.  People are never streamer-channel
    identities — their registry handle must not leak in as a Twitch login."""
    if (entry.get("category") or "").lower() == "person":
        return None
    if entry.get("twitch_handle"):
        return entry["twitch_handle"]
    if "twitch" in entry:
        return entry.get("twitch")  # explicit null stays null (Kick-only streamers)
    return entry.get("handle")


def _match_subjects(text: str) -> list[dict]:
    """Registry entries whose display name / handle / alias appears in text."""
    lowered = text.lower()
    matched: list[dict] = []
    for entry in _registry():
        names = [entry.get("display_name") or "", entry.get("handle") or ""]
        names += entry.get("aliases") or []
        for name in names:
            name = " ".join(str(name).strip().lower().split())
            if not name or len(name) < 3:
                continue
            if re.search(r"\b" + re.escape(name) + r"\b", lowered):
                if not any(_canonical(m) == _canonical(entry) for m in matched):
                    matched.append(entry)
                break
    return matched


# Common nouns that follow a capitalized word but are NOT a person's last name;
# "Streamer University" / "TaKO Tuesday" must never become a subject entity.
_NON_PERSON_NOUNS = {
    "university", "college", "campus", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "monday", "cup", "game", "league", "park",
    "street", "city", "world", "stream", "live", "vod", "clip", "clips",
    "party", "presents", "class", "challenge", "videos",
}

_NAME_PAIR_RE = re.compile(r"\b([A-Z][a-zA-Z']+)\s+([A-Z][a-zA-Z']+)\b")


def _topic_blocks_pair(pair: str, topics: list[dict]) -> bool:
    """True when a capitalized pair is a word-subset of a known topic.

    Prevents "Brand New", "New Day" or "The Odyssey" from being resolved as
    Twitch channels (which would otherwise be persisted into streamers.json as
    junk identities).  Hyphens and spaces are ignored so "Spider-Man" also
    blocks "Spider Man".
    """
    norm_pair = re.sub(r"[\s\-']", "", pair.lower())
    if not norm_pair or len(norm_pair) < 3:
        return True
    for topic in topics:
        names = [topic.get("name") or "", topic.get("display_name") or ""]
        names += topic.get("aliases") or []
        for name in names:
            norm_name = re.sub(r"[\s\-']", "", str(name).lower())
            if norm_name and norm_pair in norm_name:
                return True
    return False


def _resolve_named_entities(text: str, matched: list[dict],
                            topics: list[dict] | None = None) -> list[dict]:
    """Fallback: capitalized proper-name pairs not already in the registry,
    resolved against Twitch search so brand-new streamers still become
    entities.  Degrades silently (network/name failures) to [].
    Pairs that are already known subjects (people/streamers) or word-subsets of
    a topic are never searched — real-world celebrities would otherwise resolve
    to junk Twitch channels and get persisted into the KB."""
    topics = topics or []
    out: list[dict] = [dict(m, _resolved=False) for m in matched]
    known = {_canonical(m).lower() for m in out}
    seen_pairs: set[str] = set()
    for match in _NAME_PAIR_RE.finditer(text):
        pair = f"{match.group(1)} {match.group(2)}"
        key = pair.lower()
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        if match.group(2).lower() in _NON_PERSON_NOUNS:
            continue
        if key in known or _topic_blocks_pair(pair, topics):
            continue
        try:
            entry = streamers.resolve(pair)
        except Exception:
            entry = None
        if entry and _canonical(entry).lower() not in known:
            known.add(_canonical(entry).lower())
            out.append(dict(entry, _resolved=True))
    return out


def _pick_streamer(text: str, matched: list[dict]) -> dict | None:
    """Most-mentioned entity with ANY channel identity (Twitch first, then
    Kick, then YouTube-only).  Kick-only streamers (e.g. Rampage Jackson)
    must still become the primary streamer so their channel gets pinned."""
    def _channel_weight(entry: dict) -> int:
        if _twitch_handle(entry):
            return 2
        if streamers.kick_handle(entry):
            return 1
        return 1 if entry.get("youtube") else 0

    best = None
    best_key = (-1, -1)
    lowered = text.lower()
    for entry in matched:
        if not _channel_weight(entry):
            continue
        name = _canonical(entry)
        aliases = [name, entry.get("handle") or ""] + (entry.get("aliases") or [])
        hits = 0
        for alias in aliases:
            alias = " ".join(str(alias).lower().split())
            if len(alias) >= 3:
                hits += len(re.findall(r"\b" + re.escape(alias) + r"\b", lowered))
        if hits > best_key[0]:
            best = {
                "name": name,
                "twitch_handle": _twitch_handle(entry),
                "kick": streamers.kick_handle(entry),
                "youtube": entry.get("youtube") or [],
                "aliases": entry.get("aliases") or [],
                "category": entry.get("category"),
            }
            best_key = (hits, _channel_weight(entry))
    return best


def _dedupe_words(text: str) -> str:
    out: list[str] = []
    for word in text.split():
        if not out or out[-1].lower() != word.lower():
            out.append(word)
    return " ".join(out)


def _has_entity(query: str, name: str) -> bool:
    if name.lower() in query.lower():
        return True
    resolved = streamers.lookup(name)
    if resolved:
        for alias in (resolved.get("aliases") or []):
            if len(alias) > 2 and alias.lower() in query.lower():
                return True
    return False


def _inject_entities(query: str, names: list[str]) -> str:
    missing = [n for n in names if not _has_entity(query, n)]
    return _dedupe_words(" ".join(missing + [query]))


def _compact_query(query: str) -> str:
    """Stopword-stripped compact variant that keeps subject/object/action words."""
    from modules.text_match import STOPWORDS
    tokens = [w for w in re.findall(r"[A-Za-z0-9']+", query)
              if w.lower() not in STOPWORDS and len(w) > 2]
    return " ".join(dict.fromkeys(tokens))[:160]


# Query filler that adds NO search/semantic signal ("livestream clip AI
# generated").  Stripped from the semantic target and search hooks so the
# matching/scoring never keys on them; the full sentence stays for reports.
_FILLER_PHRASES = (
    "ai generated", "controversial clip", "stream clip", "livestream clip",
    "live stream clip", "viral clip", "clip ai generated", "controversial",
)
_FILLER_TOKENS = {"clip", "clips", "livestream", "generated"}


def _strip_filler(text: str) -> str:
    """Return the query with generic clip/noise filler words removed."""
    lowered = " ".join(str(text).lower().split())
    if not lowered:
        return str(text)
    out = lowered
    for phrase in _FILLER_PHRASES:
        out = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", out)
    out = " ".join(word for word in out.split() if word not in _FILLER_TOKENS)
    out = _dedupe_words(out).strip()
    return out if out else str(text)


# --------------------------------------------------------------------------
# Action extraction (rule-based, offline)
# --------------------------------------------------------------------------
_ACTION_VERBS = {
    "call": "call", "calls": "call", "called": "call", "calling": "call",
    "answer": "answer", "answers": "answer", "answered": "answer", "answering": "answer",
    "pick": "pick", "picks": "pick", "picked": "pick", "picking": "pick",
    "guess": "guess", "guesses": "guess", "guessed": "guess", "guessing": "guess",
    "nail": "nail", "nails": "nail", "nailed": "nail",
    "give": "give", "gives": "give", "gave": "give", "given": "give", "giving": "give",
    "receive": "receive", "receives": "receive", "received": "receive",
    "meet": "meet", "meets": "meet", "met": "meet", "meeting": "meet",
    "chase": "chase", "chases": "chase", "chased": "chase", "chasing": "chase",
    "ask": "ask", "asks": "ask", "asked": "ask", "asking": "ask",
    "play": "play", "plays": "play", "played": "play", "playing": "play",
    "open": "open", "opens": "open", "opened": "open", "opening": "open",
    "reveal": "reveal", "reveals": "reveal", "revealed": "reveal", "revealing": "reveal",
    "donate": "donate", "donates": "donate", "donated": "donate", "donating": "donate",
    "spend": "spend", "spends": "spend", "spent": "spend", "spending": "spend",
    "leak": "leak", "leaks": "leak", "leaked": "leak", "leaking": "leak",
    "shove": "shove", "shoves": "shove", "shoved": "shove", "shoving": "shove",
    "push": "push", "pushes": "push", "pushed": "push", "pushing": "push",
    "fight": "fight", "fights": "fight", "fought": "fight", "fighting": "fight",
    "punch": "punch", "punches": "punch", "punched": "punch", "punching": "punch",
    "slap": "slap", "slaps": "slap", "slapped": "slap", "slapping": "slap",
    "hit": "hit", "hits": "hit", "hitting": "hit",
    "attack": "attack", "attacks": "attack", "attacked": "attack", "attacking": "attack",
    "throw": "throw", "throws": "throw", "threw": "throw", "throwing": "throw",
    "confront": "confront", "confronts": "confront", "confronted": "confront",
    "confronting": "confront", "confrontation": "confront",
    "spit": "spit", "spits": "spit", "spat": "spit", "spitting": "spit",
    "expel": "expel", "expels": "expel", "expelled": "expel", "expelling": "expel",
    "escort": "escort", "escorts": "escort", "escorted": "escort", "escorting": "escort",
    "fly": "fly", "flies": "fly", "flew": "fly", "flying": "fly", "flown": "fly",
    "drive": "drive", "drives": "drive", "drove": "drive", "driving": "drive",
    "ride": "ride", "rides": "ride", "rode": "ride", "riding": "ride",
    "bust": "bust", "busts": "bust", "busted": "bust", "busting": "bust",
    "arrest": "arrest", "arrests": "arrest", "arrested": "arrest", "arresting": "arrest",
    "shoot": "shoot", "shoots": "shoot", "shot": "shoot", "shooting": "shoot",
    "injure": "injure", "injures": "injure", "injured": "injure", "injuring": "injure",
}

# Verbs that are too weak to be standalone action signals.  "talk"/"say"/"mention"
# are generic streamer behavior; "react" is a content category, not a specific event.
# These are NOT added to _ACTION_VERBS — they're skipped by _action_terms().
_NON_ACTION_WORDS = {
    "talk", "talks", "talked", "talking",
    "say", "says", "said", "saying",
    "mention", "mentions", "mentioned", "mentioning",
    "react", "reacts", "reacted", "reacting",
    "watch", "watches", "watched", "watching",
    "look", "looks", "looked", "looking",
}

# Function words and generic filler that carry no action signal for matching.
_ACTION_GENERIC = {
    "stream", "streams", "live", "livestream", "video", "videos", "clip", "clips",
    "streamer", "streamers", "moment", "meme", "somehow", "almost", "finally",
    "really", "stuff", "things",
    "thing", "his", "her", "their", "your", "our", "its", "been", "being", "that",
    "this", "these", "those", "there", "here", "was", "were", "had", "have", "has",
    "will", "would", "could", "should", "can", "may", "might", "them", "they",
    "you", "we", "also", "even", "then", "when", "where", "what", "which", "until",
    "while", "game", "games", "years", "year", "day", "days", "spent", "spend",
    "money", "million", "people", "person", "everyone", "everybody",
    "the", "a", "an", "of", "to", "for", "with", "on", "at", "in", "and", "or",
    "not", "no", "from", "by", "into", "over", "under", "so", "but",
    "if", "than", "do", "does", "did", "get", "gets", "got", "going", "go", "goes",
    "full", "reaction", "reactions", "ever", "randomly", "impossible", "right",
    "wrong", "private", "messages", "message", "huge", "worn", "world", "cup",
    "him", "her", "she", "he", "actually", "literally", "extremely", "super", "very",
    "during", "trying", "try", "out", "university", "water", "pool",
}

# Adjectives/adverbs that are fine in a full query but noisy as a search hook.
_ACTION_ADJ = {"viral", "crazy", "insane", "wild", "massive", "huge", "accidentally",
               "finally", "almost", "really", "eventually", "instantly", "randomly"}

# Physical objects / props that an event turns on (police car, school bus,
# helicopter, gel blaster...).  These are confirmed by the VISUAL anchor gate on
# the exported clip, never by a mere spoken mention, so they are kept out of the
# audio action signal (content_verifier) while still feeding search hooks and
# the LLM localizer.
_OBJECT_NOUNS = {
    "police", "car", "cars", "bus", "buses", "helicopter", "helicopters",
    "blaster", "gel", "eye", "eyes", "dorm", "campus", "party", "pool",
    "water", "boots", "phone", "phones", "burger", "burgers", "money",
    "cash", "dms", "messages", "screenshots", "mic", "microphone", "cake",
    "cakes", "ring", "necklace", "chain", "chains", "trophy", "jet",
    "yacht", "lamborghini", "ferrari", "porsche", "rolex", "watch",
    "snow", "paint", "eggs", "lego", "ball", "balloons", "pizza",
    "paintball", "golf cart", "roller coaster",
    "leg", "legs", "ankle", "ankles", "bow", "bows", "arrow", "arrows",
    "target", "bow and arrow", "archery target",
}
_OBJECT_PHRASES = {
    "police car", "school bus", "gel blaster", "private jet", "squad car",
    "party bus", "snow machine", "school bus",
    "bow and arrow", "archery target",
}

# Event words that imply concrete props CLIP can actually recognize, mapped to
# the object terms that make the query a visual event (e.g. "archery" -> the
# bow and arrow on screen, even though the word itself is a sport, not a prop).
_OBJECT_KEYWORDS = {
    "archery": ("bow and arrow", "bow", "arrow"),
}


def _visual_event_kind(query: str) -> str | None:
    """Map a query to its visual-event scene kind (for EventProbe probing).

    These kinds drive the local CLIP scene probes that confirm the event's
    pixels are actually on screen before/while a clip is exported.  Order
    matters: the most specific prop is checked first.
    """
    q = " ".join(query.lower().split())
    if "gel blaster" in q or ("blaster" in q and "injur" in q):
        return "gel_blaster"
    if "helicopter" in q:
        return "helicopter"
    if "police car" in q or ("police" in q and any(w in q for w in ("car", "cops", "cruiser", "escort"))):
        return "police_car"
    if "school bus" in q:
        return "school_bus"
    if any(w in q for w in ("fighting", "fought", "brawl", "scuffle", "punched", "wrestling")):
        return "fight"
    if "pool party" in q or ("party" in q and "pool" in q):
        return "pool_party"
    return None


def _object_terms(query: str) -> list[str]:
    """Physical-object/prop nouns the query's event turns on, phrases first.

    Kept as-is (multi-word phrases intact) so downstream phrase matching and the
    anchor/LLM prompts see exactly what the event is about."""
    lowered = " ".join(query.lower().split())
    found: list[str] = []
    seen: set[str] = set()
    for phrase in sorted(_OBJECT_PHRASES, key=len, reverse=True):
        if phrase in lowered and phrase not in seen:
            seen.add(phrase)
            found.append(phrase)
    for word in re.findall(r"[a-z0-9']+", lowered):
        if word in _OBJECT_NOUNS and word not in seen:
            seen.add(word)
            found.append(word)
    for keyword, props in _OBJECT_KEYWORDS.items():
        if keyword in lowered:
            for prop in props:
                if prop not in seen:
                    seen.add(prop)
                    found.append(prop)
    return found[:8]


# Specific CLIP scene phrases for distinctive event words that generic
# subject/action generation cannot express ("archery" is a sport, not a scene;
# "leg" alone matches every talking-head frame).  "{subject}" is replaced by the
# query's subject when one is known.  These phrases dominate the general probe
# because scene_scores takes the max over all prompts.
_VISUAL_SCENE_PHRASES = {
    "archery": [
        "a bow and arrow",
        "shooting a bow and arrow",
        "an arrow hitting someone",
        "an archery target",
        "{subject} shooting a bow",
        "people doing archery",
    ],
    "bow": ["a bow and arrow", "{subject} with a bow"],
    "arrow": ["an arrow hitting someone", "a bow and arrow"],
    "leg": [
        "someone grabbing someone's leg",
        "grabbing her leg",
        "a leg injury",
        "grabbing an ankle",
        "holding her leg in pain",
    ],
    "ankle": ["grabbing her ankle", "an ankle injury"],
}


def _scene_prompts(full: str, compact: str, entities: list[str],
                   streamer: dict | None, object_terms: list[str],
                   action_terms: list[str], topics: list[str]) -> list[str]:
    """Short CLIP scene descriptions for the query — the general visual target.

    Only the 6 whitelisted event kinds used to get pixel grounding; this gives
    EVERY query its own scene prompts so the visual ranking/gates have something
    concrete to score frames against (a lamborghini, a jet, a yacht, a fight).
    CLIP matches short concrete phrases best, so keep them terse and physical.
    """
    prompts: list[str] = []
    seen: set[str] = set()

    def _add(p: str) -> None:
        p = " ".join(str(p).lower().split())[:90]
        if p and p not in seen and len(p) >= 4:
            seen.add(p)
            prompts.append(p)

    subject = ""
    if isinstance(streamer, dict):
        subject = str(streamer.get("name") or "").strip()
    if not subject and entities:
        subject = str(entities[0] or "").strip()

    lowered = " ".join(str(full).lower().split())
    for keyword, phrases in _VISUAL_SCENE_PHRASES.items():
        if keyword in lowered:
            for phrase in phrases:
                if "{subject}" in phrase:
                    if not subject:
                        continue
                    phrase = phrase.replace("{subject}", subject)
                _add(phrase)

    objects = [str(o).lower().strip() for o in object_terms if str(o).strip()]
    skip: set[str] = set(objects)
    skip.update(str(e).lower() for e in entities)
    if isinstance(streamer, dict):
        skip.update((streamer.get("name") or "").lower().split())
    actions: list[str] = []
    for a in dict.fromkeys(action_terms):
        a = str(a).lower().strip()
        if a in skip or len(a) < 3:
            continue
        # action_terms is a loose stem list (it includes nouns like "fans");
        # keep only real verbs so prompts stay physical ("ride", not "fans").
        if a in _ACTION_VERBS.values() or a.endswith(("ing", "ed")):
            actions.append(a)

    # Body parts and generic props are useless as standalone CLIP prompts
    # ("a leg on screen" matches any talking-head frame), so they only enter
    # through the specific lexicon phrases above or an action pairing.
    weak_objects = {"leg", "legs", "ankle", "ankles", "hand", "hands", "arm",
                    "arms", "eye", "eyes", "head", "face"}
    for obj in objects:
        if obj in weak_objects:
            continue
        _add(f"a {obj} on screen")
        _add(f"the {obj} in the scene")
    if subject:
        for obj in objects[:1]:
            if obj not in weak_objects:
                _add(f"{subject} with a {obj}")
        for act in actions[:1]:
            _add(f"{subject} {act}")
    for act in actions:
        if objects:
            first = objects[0]
            if act == "hit" and any(w in first for w in ("arrow", "bow")):
                _add("someone hit with an arrow")
                _add("an arrow hitting someone")
            elif first not in weak_objects:
                _add(f"{act} with a {first}")
                _add(f"{act} the {first}")
        # Audio-primary verbs (react, answer, call) are spoken events, not visual
        # scenes — "someone reacting" matches any talking-head frame.  Weak verbs
        # (spend, leak, donate, guess, receive) are too generic for CLIP.  Physical
        # action verbs (hit, injure, fight, chase) DO get the "someone <verb>"
        # prompt because they read as concrete visual scenes.
        if act in ("react", "reacts", "answer", "call",
                   "spend", "leak", "donate", "guess", "receive"):
            continue
        _add(f"someone {act}")
    for topic in topics[:3]:
        _add(f"the {topic} scene")
        _add(f"a {topic}")
    _add(compact or full)
    if len(prompts) > 8:
        prompts = prompts[:8]
    return prompts


def _action_terms(query: str, entities: list[str], streamer: dict | None) -> list[str]:
    """Distinct action-relevant word stems from the query, entity names removed."""
    skip: set[str] = set()
    for entry in _registry():
        names = [entry.get("display_name") or "", entry.get("handle") or ""]
        names += entry.get("aliases") or []
        for name in names:
            name = " ".join(str(name).lower().split())
            if name:
                skip.add(name)
                skip.update(name.split())
    for topic in _load_topics():
        names = [topic.get("name") or "", topic.get("display_name") or ""]
        names += topic.get("aliases") or []
        for name in names:
            name = " ".join(str(name).lower().split())
            if name:
                skip.add(name)
                skip.update(name.split())
    if streamer:
        skip.add((streamer.get("name") or "").lower())
        skip.add((streamer.get("twitch_handle") or "").lower())

    terms: list[str] = []
    lowered = query.lower()
    for word in re.findall(r"[a-z0-9']+", lowered):
        if len(word) <= 2 or word in skip:
            continue
        if word in _NON_ACTION_WORDS:
            continue
        if word in _ACTION_GENERIC:
            continue
        base = _ACTION_VERBS.get(word, word)
        if base in _ACTION_GENERIC:
            continue
        is_verb = (word in _ACTION_VERBS
                   or word.endswith(("ing", "ed", "es"))
                   or base != word)
        if not is_verb:
            continue
        if base not in terms:
            terms.append(base)
    return terms


def _action_search_queries(entities: list[str], terms: list[str]) -> list[str]:
    """Clip-style YouTube search hooks from entities + action terms.

    Each action verb is also emitted in its inflected forms (call/calls/called/
    calling, guess/guesses/guessed/guessing...) because clip-style videos title
    themselves with the inflected verb ('Caseoh Guesses Burgers') that a bare
    stem query never surfaces.  For multi-subject events it also emits the raw
    entity-pair hooks ('IShowSpeed Cristiano Ronaldo') so the actual meeting
    VOD has a chance to enter the pool even when no action verb is in its title.
    """
    hooks = [t for t in terms if t not in _ACTION_ADJ]
    if not hooks:
        hooks = terms
    verbs = [t for t in hooks if t in _ACTION_VERBS.values()]
    non_verbs = [t for t in hooks if t not in _ACTION_VERBS.values()]

    inflection_forms: list[str] = []
    if verbs:
        base = verbs[0]
        inflection_forms = [base]
        for word, mapped in _ACTION_VERBS.items():
            if mapped == base and word != base:
                inflection_forms.append(word)
        inflection_forms = inflection_forms[:4]

    qs: list[str] = []

    def _push(head, tail):
        q = _dedupe_words(" ".join(str(w) for w in list(head) + list(tail) if str(w)))
        if q and q not in qs:
            qs.append(q)

    for head in ([entities, entities[:1]] if entities else [[]]):
        _push(head, hooks)
        if verbs:
            _push(head, [inflection_forms[0]] + non_verbs)
            for form in inflection_forms[1:]:
                _push(head, [form] + non_verbs)

    if len(entities) >= 2:
        _push(entities[:2], [])
        _push(entities[:2], verbs[:1])
        _push([entities[0], verbs[0] if verbs else (non_verbs[0] if non_verbs else ""), entities[1]], [])
    return qs[:8]


def _entity_detail(entry: dict) -> dict:
    return {
        "name": _canonical(entry),
        "aliases": entry.get("aliases") or [],
        "twitch": _twitch_handle(entry),
        "kick": streamers.kick_handle(entry),
        "youtube": entry.get("youtube") or [],
        "category": entry.get("category"),
        "resolved": bool(entry.get("_resolved", False)),
    }


def _build_plan(query: str) -> dict:
    matched = _match_subjects(query)
    topics = _match_topics(query)
    matched = _resolve_named_entities(query, matched, topics)
    entities_detail = [_entity_detail(e) for e in matched]
    entities = list(dict.fromkeys(d["name"] for d in entities_detail if d["name"]))[:4]
    topic_names = list(dict.fromkeys(_canonical(t) for t in topics if _canonical(t)))
    streamer = _pick_streamer(query, matched)
    candidate_streamers = _event_candidate_streamers(query)
    if not streamer and candidate_streamers:
        streamer = candidate_streamers[0]
    full = _dedupe_words(query)
    clean_full = _strip_filler(full)
    compact = _compact_query(clean_full)
    semantic_target = compact if len(clean_full.split()) > 16 else clean_full
    search_queries = [_inject_entities(clean_full, entities)]
    if compact and compact.lower() != clean_full.lower():
        search_queries.append(_inject_entities(compact, entities))
    search_queries = search_queries[:2]
    action_terms = _action_terms(full, entities, streamer)
    # Visual-event queries hinge on physical objects/props (a police car, a
    # helicopter, a gel blaster) that only VISUAL evidence can confirm, so they
    # must never be granted by a spoken mention alone.
    object_terms = _object_terms(full)
    visual_event_kind = _visual_event_kind(full)
    visual_event = bool(object_terms) or bool(visual_event_kind)
    scene_prompts = _scene_prompts(
        full, compact, entities, streamer, object_terms, action_terms, topic_names)
    # Media queries (premieres, trailers, interviews, red carpets) carry no
    # streamer identity; they get the relaxed media policy downstream.
    media_query = bool(streamer is None and (matched or topics))
    is_event = bool(action_terms) or bool(topics)
    window_seconds = (config.ACTION_WINDOW_SECONDS
                      if (is_event or media_query) else config.CLIP_DURATION_SECONDS)
    return {
        "streamer": streamer,
        "candidate_streamers": candidate_streamers or None,
        "subject": (streamer or {}).get("name") or (entities[0] if entities else None),
        "speaker": None,
        "core_entities": entities,
        "topics": topic_names,
        "media_query": media_query,
        "entities_detail": entities_detail,
        "subjects_unknown": bool(is_event and not entities),
        "action": " ".join(action_terms),
        "action_terms": action_terms,
        "object_terms": object_terms,
        "visual_event": visual_event,
        "visual_event_kind": visual_event_kind,
        "scene_prompts": scene_prompts,
        "action_search_queries": _action_search_queries(entities, action_terms),
        "window_seconds": window_seconds,
        "context_terms": [],
        "sentence_type": "event" if is_event else "narration",
        "visual_prompt": full,
        "search_queries": search_queries,
        "semantic_target": semantic_target,
        "required_terms": list(dict.fromkeys(topic_names + entities)) or [full],
        "excluded": ([] if media_query
                     else ["reaction", "compilation", "recap", "edited"]),
        "note": "parsed from queries.txt",
        "sentence": full,
        "actions": [],
    }


def plan_script(force: bool = False) -> tuple[dict, list[dict]]:
    """Return (subject, per-query plans).  Uses the cache unless force."""
    path = config.SENTENCES_CACHE
    people_text = (config.PEOPLE_FILE.read_text(encoding="utf-8")
                   if config.PEOPLE_FILE.exists() else "")
    if path.exists() and not force:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if (data.get("queries") == config.QUERIES_FILE.read_text(encoding="utf-8")
                    and data.get("people") == people_text
                    and all(isinstance(p.get("scene_prompts"), list)
                            for p in data.get("sentences") or [])):
                return data.get("subject", {}), data["sentences"]
        except Exception as exc:
            logger.info("Sentences cache unusable (%s); re-planning", exc)

    queries = load_queries()
    if not queries:
        raise RuntimeError(f"No queries found at {config.QUERIES_FILE}")
    subject = {"source": "queries.txt", "queries": queries}
    plans = [_build_plan(q) for q in queries]

    config.SENTENCES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    config.SENTENCES_CACHE.write_text(
        json.dumps({"queries": config.QUERIES_FILE.read_text(encoding="utf-8"),
                    "people": people_text,
                    "subject": subject, "sentences": plans},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    return subject, plans


def load_plans() -> list[dict]:
    """Load cached plans only (no parsing)."""
    path = config.SENTENCES_CACHE
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("sentences", [])
        except Exception:
            pass
    return []
