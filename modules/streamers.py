"""Streamer knowledge base + name resolution.

Seeded aliases cover the streamers most likely to appear in scripts; unknown
names are resolved against Twitch's public searchFor index (ranked by
followers) so brand-new streamers still work without editing the KB.
"""
import json
import logging
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import config
from modules.twitch_source import _gql

logger = logging.getLogger(__name__)

_DEFAULT_STREAMERS = {
    "n3on": {
        "display_name": "N3on",
        "aliases": ["n3on", "neon", "n3on the streamer", "n3on streamer"],
        "twitch": "n3on",
        "youtube": ["N3on", "N3ON", "N3on Clips"],
        "category": "streamer",
    },
    "kaicenat": {
        "display_name": "Kai Cenat",
        "aliases": ["kai", "kai cenat", "cenat"],
        "twitch": "kaicenat",
        "youtube": ["Kai Cenat Live", "Kai Cenat Live Today", "Kai Cenat", "Kai Cenat Clips", "Kai Cenat Shorts"],
        "category": "streamer",
    },
    "ishowspeed": {
        "display_name": "IShowSpeed",
        "aliases": ["speed", "ispeed", "ishowspeed"],
        "twitch": "ishowspeed",
        "youtube": ["IShowSpeed", "IShowSpeed Live", "Live Speedy", "IShowSpeed Clips", "IShowSpeed Shorts"],
        "category": "streamer",
    },
    "adinross": {
        "display_name": "Adin Ross",
        "aliases": ["adin", "adin ross"],
        "twitch": "adinross",
        "youtube": ["Adin Ross", "Adin Live", "Adin Ross Live"],
        "category": "streamer",
    },
    "xqc": {
        "display_name": "xQC",
        "aliases": ["xqc", "xqcow"],
        "twitch": "xqc",
        "youtube": ["xQc", "xQcOW", "xQc Streams"],
        "category": "streamer",
    },
    "pokimane": {
        "display_name": "Pokimane",
        "aliases": ["pokimane", "poki"],
        "twitch": "pokimane",
        "youtube": ["Pokimane"],
        "category": "streamer",
    },
    "ludwig": {
        "display_name": "Ludwig",
        "aliases": ["ludwig", "ludwig ahgren"],
        "twitch": "ludwig",
        "youtube": ["Ludwig", "Ludwig Clips"],
        "category": "streamer",
    },
    "mrbeast": {
        "display_name": "MrBeast",
        "aliases": ["mrbeast", "jimmy", "jimmy donaldson", "mr beast"],
        "twitch": None,
        "youtube": ["MrBeast", "MrBeast Gaming"],
        "category": "youtuber",
    },
}

_CHANNEL_SEARCH = """query {
  searchFor(userQuery: %(query)s, platform: "web", target: { index: ALL }) {
    channels {
      items {
        __typename
        ... on User {
          id
          login
          displayName
          description
          followers { totalCount }
        }
      }
    }
  }
}"""


def _load_kb() -> dict:
    merged = dict(_DEFAULT_STREAMERS)
    # Auto-discovered identities (modules/auto_registry.py) come first; the
    # manual file wins when a key appears in both.
    if config.AUTO_STREAMERS_FILE.exists():
        try:
            auto = json.loads(config.AUTO_STREAMERS_FILE.read_text(encoding="utf-8"))
            merged.update(auto or {})
        except Exception as exc:
            logger.warning("Auto-streamers file unreadable (%s); skipping", exc)
    if config.STREAMERS_FILE.exists():
        try:
            user = json.loads(config.STREAMERS_FILE.read_text(encoding="utf-8"))
            merged.update(user or {})
            return merged
        except Exception as exc:
            logger.warning("Streamers file unreadable (%s); using defaults+auto", exc)
    return merged


def _save_kb(handle: str, entry: dict) -> None:
    try:
        data = _load_kb()
        data[handle] = entry
        config.STREAMERS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not persist streamer %s: %s", handle, exc)


def lookup(name: str) -> dict | None:
    """Knowledge-base only lookup (no network). Used for term expansion."""
    if not name:
        return None
    query = " ".join(str(name).lower().strip().split())
    if not query:
        return None
    kb = _load_kb()
    for handle, entry in kb.items():
        aliases = [a.lower() for a in entry.get("aliases", [])] + [handle.lower()]
        if query in aliases or any(query in a for a in aliases):
            return dict(entry, handle=handle)
    return None


def resolve(name: str) -> dict | None:
    """Resolve a display name / alias to a streamer identity, or None."""
    found = lookup(name)
    if found:
        return found
    return search_channel(name) if _name_worth_searching(name) else None


def _name_worth_searching(name: str) -> bool:
    """Skip network lookup for obvious non-names (topic phrases, numbers)."""
    if not name:
        return False
    cleaned = re.sub(r"[^a-z ]", "", name.lower()).strip()
    words = cleaned.split()
    if not words:
        return False
    if len(words) > 4:
        return False
    return all(len(w) >= 3 for w in words)


def search_channel(name: str) -> dict | None:
    """Best-matching Twitch channel by display name, ranked by followers."""
    if not config.TWITCH_FOLLOWERS_SORT:
        return None
    payload = {"query": _CHANNEL_SEARCH % {"query": json.dumps(name.lower())}}
    try:
        result = _gql(payload)
    except (HTTPError, URLError, OSError) as exc:
        logger.info("Twitch channel search failed for %r: %s", name, exc)
        return None
    items = (((result.get("data") or {}).get("searchFor") or {})
             .get("channels", {}).get("items", []) or [])
    if not items:
        return None
    candidates = []
    for item in items:
        if item.get("__typename") != "User" or not item.get("login"):
            continue
        display = (item.get("displayName") or "").lower()
        score = 1.0 if name.lower() in display else (0.3 if name.lower() in (item.get("login") or "").lower() else 0.0)
        if score <= 0.0:
            continue
        candidates.append({
            "handle": item["login"],
            "display_name": item.get("displayName") or item["login"],
            "description": item.get("description") or "",
            "followers": int(((item.get("followers") or {}).get("totalCount") or 0)),
            "score": score,
            "aliases": [name.lower()],
            "youtube": [],
            "category": "streamer",
        })
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c["score"], c["followers"]), reverse=True)
    best = candidates[0]
    entry = {k: v for k, v in best.items() if k != "score"}
    logger.info("Resolved streamer '%s' -> twitch/%s (%d followers)",
                name, entry["handle"], entry["followers"])
    _save_kb(entry["handle"], entry)
    return entry


def youtube_channels(streamer: dict | None) -> list[str]:
    if not streamer:
        return []
    channels = [c for c in (streamer.get("youtube") or []) if c]
    if channels:
        return channels
    entry = lookup(streamer.get("name") or "")
    if entry:
        return [c for c in (entry.get("youtube") or []) if c]
    return []


def _names_for(streamer: dict | None) -> set[str]:
    """Subject-name strings used to decide whether a channel belongs to the
    streamer: plan streamer dicts only carry name + twitch_handle, so the KB
    (aliases / display_name / youtube handles) is resolved for the rest."""
    names: set[str] = set()
    if not streamer:
        return names
    handle = twitch_handle(streamer)
    if handle:
        names.add(handle.lower())
    kick = kick_handle(streamer)
    if kick:
        names.add(str(kick).lower())
    display = streamer.get("display_name") or streamer.get("name") or ""
    if display:
        names.add(str(display).lower())
    for alias in streamer.get("aliases") or []:
        if alias:
            names.add(str(alias).lower())
    entry = lookup(streamer.get("name") or "")
    if entry:
        if entry.get("display_name"):
            names.add(str(entry["display_name"]).lower())
        for alias in entry.get("aliases") or []:
            if alias:
                names.add(str(alias).lower())
        resolved_handle = twitch_handle(entry)
        if resolved_handle:
            names.add(resolved_handle.lower())
    return {n for n in names if n and len(n) >= 3}


def twitch_handle(streamer: dict | None) -> str | None:
    if not streamer:
        return None
    if streamer.get("twitch_handle"):
        return streamer["twitch_handle"]
    if "twitch" in streamer:
        return streamer.get("twitch")  # explicit null stays null (Kick-only streamers)
    return streamer.get("handle")


def kick_handle(streamer: dict | None) -> str | None:
    """Kick channel slug for a streamer entry (events that stream on Kick)."""
    if not streamer:
        return None
    return streamer.get("kick") or streamer.get("kick_handle")


def is_official_channel(streamer: dict | None, channel: str) -> bool:
    """True if a YouTube channel is the streamer's own (raw footage allowed).

    Explicit youtube list entries match exactly; beyond that only the bare
    identity channel (handle / display name) or 'handle live|streams|vod' counts.
    Loose substring matching is gone, so "xQc Reacts" is no longer treated as
    xQc's own channel.
    """
    if not streamer or not channel:
        return False
    lowered = channel.lower().strip()
    if not lowered:
        return False
    for name in youtube_channels(streamer):
        if name and lowered == name.lower().strip():
            return True
    handle = twitch_handle(streamer)
    display = (streamer.get("display_name") or "").lower().strip()
    if handle and lowered == handle.lower():
        return True
    if display and lowered == display:
        return True
    if handle:
        handle = handle.lower()
        for suffix in ("live", "streams", "vod", "vods", "raw"):
            if lowered == f"{handle} {suffix}":
                return True
    return False


def is_raw_footage_channel(channel: str, subject_streamer: dict | None = None,
                           title_or_desc: str | None = None) -> bool:
    """Third-party channels are allowed only if they look like raw VOD hosts:
    they name the subject streamer AND use a strict raw marker AND carry no
    reactor / aggregator markers.  "Daily Dose of LivestreamFail" (aggregator,
    no subject name) and "xQc Clips" (reactor clip shop) both fail.

    ``title_or_desc`` lets a raw-marker channel that does NOT carry the streamer
    in its channel name qualify when its VIDEO's own title/description names the
    subject ("Twitch Game Vods" hosting "CRAZY MERCEDES S-CLASS POLICE CHASE -
    Kai Cenat"): that upload is a 1:1 mirror of the original stream, i.e. the
    original footage on a re-host channel, which the product rule permits.  The
    downstream transcript/VLM gates still confirm the moment itself.
    """
    lowered = (channel or "").lower()
    if not lowered:
        return False
    if any(marker in lowered for marker in config.REACTOR_CHANNEL_MARKERS):
        return False
    if any(marker in lowered for marker in config.AGGREGATOR_CHANNEL_MARKERS):
        return False
    names = _names_for(subject_streamer)
    if names:
        blob = f"{lowered} {str(title_or_desc or '').lower()}"
        if not any(name in blob for name in names):
            return False
    return any(marker in lowered for marker in config.STREAMER_RAW_CHANNEL_MARKERS)
