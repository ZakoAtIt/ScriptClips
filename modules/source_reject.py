"""Hard rejection of edited / reaction / re-uploaded content.

The user's product rule: clips must be the ORIGINAL moment from the source
streamer — never a reactor's take, never an edited recap, never a third-party
re-upload.  This module encodes that at the metadata level; the LLM judge and
content verification double-check at the content level.
"""
import re

import config
from modules.scoring import contains_keyword
from modules import streamers

_HARD_BLOCK = (
    "compilation", "montage", "supercut", "top 10", "best moments", "best of",
    "meme", "edited", "recap", "story time", "storytime", "rewind", "explained",
    "documentary", "in 60 seconds", "in 10 minutes", "watch party", "trailer",
    "highlights", "highlight reel", "review", "analysis", "breakdown",
    "summary", "lecture", "tutorial", "guide", "reuploaded",
)
# Media queries (premieres, trailers, interviews, red carpets, box office)
# legitimately come from official trailers, interviews, reactions and news
# channels, so only true junk is blocked; quality is decided downstream by
# verification + the LLM judge + the face gate.
_MEDIA_HARD_BLOCK = (
    "compilation", "montage", "supercut", "top 10", "best moments", "best of",
    "meme", "edited", "story time", "storytime", "reuploaded",
    "in 60 seconds", "in 10 minutes", "lecture", "tutorial", "guide",
)
_MEDIA_CHANNEL_MARKERS = ("compilation", "compilations", "montage", "supercut",
                          "best of", "meme")
_REACTION_MARKERS = (
    "reaction", "reacting", "reacts", "react to", "reacts to", "watching",
    "watches",
)
_RAW_MARKERS = ("stream", "live", "raw", "twitch", "irl", "vod", "unedited",
                "original", "recording", "full stream", "unfiltered")

_NAME_RE = re.compile(r"[A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+)?")


def _blocked_title(title: str, block: tuple[str, ...] | None = None) -> str | None:
    lowered = (title or "").lower()
    for marker in block or _HARD_BLOCK:
        if contains_keyword(lowered, marker):
            return f"edited/packaged content marker '{marker}'"
    return None


def _reaction_title(title: str) -> bool:
    lowered = (title or "").lower()
    return any(contains_keyword(lowered, marker) for marker in _REACTION_MARKERS)


_EVENT_STRONG_BLOCK = (
    "compilation", "montage", "supercut", "top 10", "best moments", "best of",
    "meme", "story time", "storytime", "rewind", "documentary", "in 60 seconds",
    "in 10 minutes", "watch party", "trailer", "reuploaded", "edited",
)


def reject(candidate: dict, plan: dict | None = None,
           allowed_channels: set[str] | None = None,
           event_terms: set[str] | None = None) -> str | None:
    """Return a rejection reason, or None if the candidate may pass.

    ``allowed_channels`` (lowercased channel names) lets EVENT queries keep the
    event's own official / participant / VOD channels in the pool: the "Streamer
    Games" official channel or a participant streamer's channel would otherwise
    be dropped as a "third-party re-upload" even though it IS the original
    event footage.

    ``event_terms`` (lowercased topic terms / bigrams from the query sentence)
    relaxes the originality policy for MULTI-STREAMER events: the winning
    moment's raw footage lives on the organizer's channel, the sponsor's
    channel and every participant's POV channel, so the top-N pool must admit
    any on-topic upload of the event itself.  Reactor / aggregator channels and
    clearly-edited recap shops are still rejected; correctness is decided
    downstream by the LLM judge and the VLM pixel gate, so over-inclusion is
    safe here and over-exclusion is the bug (an empty pool).
    """
    title = candidate.get("title") or ""
    description = candidate.get("description") or ""
    channel = candidate.get("channel") or candidate.get("uploader") or ""
    source = candidate.get("source", "youtube")
    streamer = plan.get("streamer") if plan else None
    media = bool(plan and plan.get("media_query"))

    allowed = bool(allowed_channels) and bool(channel) \
        and channel.lower().strip() in allowed_channels

    # Event-query relaxation: an on-topic upload of the event itself enters the
    # pool from any channel, as long as the channel is not a reactor/aggregator
    # and the title is not a hard recap/compilation shop.  "On-topic" is STRICT:
    # the blob must carry a distinctive bigram of the event sentence (e.g. the
    # event name "streamer games"), or several single words AND come from an
    # event-allowlisted channel.  A vague "STREAMERS SHOCKED" narration title
    # from an unrelated channel (that merely mentions event words) must NOT enter.
    if event_terms and source != "twitch":
        blob = f"{title} {description}".lower()
        matches = [term for term in event_terms if term in blob]
        bigram_matches = [term for term in matches if " " in term]
        lowered_channel = (channel or "").lower()
        reactor_or_agg = (any(marker in lowered_channel for marker in config.REACTOR_CHANNEL_MARKERS)
                          or any(marker in lowered_channel for marker in config.AGGREGATOR_CHANNEL_MARKERS))
        strong_edited = any(contains_keyword(title, marker)
                            for marker in _EVENT_STRONG_BLOCK)
        on_topic = bool(bigram_matches) or (
            len(matches) >= 2 and allowed_channels
            and lowered_channel.strip() in {str(a).lower().strip() for a in allowed_channels})
        if on_topic and not reactor_or_agg and not strong_edited:
            return None

    blocked = _blocked_title(title, _MEDIA_HARD_BLOCK if media else _HARD_BLOCK)
    if blocked and not allowed:
        return blocked

    # Media queries (official trailers, interviews, reactions, red-carpet and
    # news footage) come from third-party and official channels alike; only
    # pure compilation/aggregator shops are rejected there.  Streamer queries
    # keep the strict original-footage policy below.
    if media:
        lowered_channel = (channel or "").lower()
        if any(contains_keyword(lowered_channel, marker)
               for marker in _MEDIA_CHANNEL_MARKERS):
            if not allowed:
                return f"aggregator/editorial channel '{channel}'"
        return None

    # Channel-level editorial/aggregator shops are rejected regardless of title
    # ("Daily Dose of LivestreamFail", "Best Twitch, YT and Kick Clips" hosts).
    lowered_channel = (channel or "").lower()
    if any(marker in lowered_channel for marker in config.AGGREGATOR_CHANNEL_MARKERS):
        if not allowed:
            return f"aggregator/editorial channel '{channel}'"

    # Twitch channel clips are the streamer's own raw stream footage.
    if source == "twitch":
        return None

    official = streamers.is_official_channel(streamer, channel)
    raw_host = streamers.is_raw_footage_channel(
        channel, streamer, title_or_desc=f"{title} {description}")

    reaction_meta = (_reaction_title(title)
                     or any(marker in lowered_channel for marker in config.REACTOR_CHANNEL_MARKERS))
    if config.REJECT_REACTION and reaction_meta and not allowed:
        if not (official or raw_host):
            return "reaction video from a third-party channel"
        if not official and raw_host:
            return "reaction-style title on a re-upload channel"
        return None

    if config.REJECT_REUPLOAD and not official and not raw_host:
        if not allowed:
            return f"third-party re-upload channel '{channel}'"

    return None
