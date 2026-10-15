"""YouTube fill lane: raw footage scanned by caption, sectioned to 10s windows.

Runs only when the Twitch lane cannot supply enough verified clips.  Searches
the sentence + planner queries, hard-rejects edited/reaction/re-upload content,
then for each candidate video reads its transcript (captions — no download; or
Whisper for short caption-less clips) and proposes semantically-matched windows.
"""
import hashlib
import json
import logging
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import config
from modules import (audio_transcriber, content_verifier, downloader,
                     event_registry, event_vision, llm_client, moment_finder,
                     overlay_detector, semantic_embedder, source_reject, streamers,
                     thumbnail_rank, vision_ranker, vod_scan, youtube_search)
from modules.scoring import (evidence_editorial_score, metadata_rawness_score,
                             title_match_score)
from modules.text_match import score as lexical_score
from modules.text_match import STOPWORDS, stem, tokens

logger = logging.getLogger(__name__)


def _forced_start(video_id: str) -> tuple[float, str] | None:
    """User-forced moment (``video_id:start``) for a video, or None.

    A forced moment is authoritative: the user pinned the exact second the
    moment happens, so the lane must surface a candidate there even when the
    transcript (degraded / re-captioned) can no longer keyword-locate it.
    """
    try:
        payload = json.loads(config.FORCED_MOMENTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    for key, reason in payload.items():
        if not isinstance(key, str) or ":" not in key:
            continue
        vid, _, secs = key.partition(":")
        if vid != video_id:
            continue
        try:
            start = float(secs)
        except (TypeError, ValueError):
            continue
        if start >= 0:
            return start, (str(reason) if reason else "user-forced moment")
    return None


# Extra config toggle for frame/CLIP/OCR verification (CPU cost).
# Disabled: remote-stream ffmpeg seeks time out on this setup; the caption/
# content verification + metadata rejection is the active guard.
VISUAL_VERIFY = False


# --------------------------------------------------------------------------
# LLM search-query expansion (raw-source hooks)
# --------------------------------------------------------------------------
_SEARCH_SYSTEM = (
    "You expand a query into YouTube search hooks that surface the ORIGINAL raw "
    "footage of a livestream moment from the streamer's own channel or raw VOD "
    "uploads. Reply only JSON."
)
_SEARCH_PROMPT = """Sentence to find raw footage for: "%(sentence)s"
Streamer / subjects: %(subjects)s

Return 8 short YouTube search queries most likely to surface the original clip
of THIS EXACT moment. Prefer the short punchy titles real clips use, e.g.
"caseoh guess the burger", "speed meets ronaldo", "xqc accidental donation".
Describe the EVENT itself, not just the sentence — a driver watching a stream
during a police chase is searched as "police chase live", "fan live streams
police chase while watching kai", "driver flees cops on instagram live" — the
words real uploads use, even when the sentence omits them.
Must not contain the words: reaction, compilation, clip, best, highlights.
Reply only JSON: {"queries": ["...", "..."]}"""

# Media queries (premieres, trailers, interviews, red carpets, box office)
# legitimately live on official / editorial / reaction channels, so the hooks
# must allow those words instead of banning them like the raw-footage prompt.
_SEARCH_SYSTEM_MEDIA = (
    "You expand a query into YouTube search hooks that surface the best official "
    "and editorial videos about a movie, premiere, trailer, interview or red "
    "carpet event. Reply only JSON."
)
_SEARCH_PROMPT_MEDIA = """Sentence to find the best videos for: "%(sentence)s"
Subjects: %(subjects)s

Return 8 short YouTube search queries most likely to surface the best clip of
THIS EXACT subject. Prefer the titles real videos use, e.g. "Spider-Man Brand
New Day official trailer", "Tom Holland Zendaya Brand New Day premiere
interview", "Brand New Day red carpet", "audience reaction Brand New Day".
Reply only JSON: {"queries": ["...", "..."]}"""


def _expand_search_queries(sentence: str, plan: dict) -> list[str]:
    """LLM-generated raw-source search hooks, cached per sentence.  The planner's
    hooks miss fan-style titles ('guess the burger'), so the ideal video never
    enters the pool; this closes that gap.  Degrades to [] on failure."""
    digest = hashlib.sha1(sentence.encode("utf-8")).hexdigest()[:12]
    cache_dir = config.CACHE_DIR / "search_expand"
    cache_path = cache_dir / f"{digest}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    streamer = plan.get("streamer") if isinstance(plan, dict) else None
    subjects = ", ".join(plan.get("core_entities") or [])
    if isinstance(streamer, dict) and streamer.get("name"):
        subjects = streamer["name"] + (f", {subjects}" if subjects else "")
    media = bool(isinstance(plan, dict) and plan.get("media_query"))
    system = _SEARCH_SYSTEM_MEDIA if media else _SEARCH_SYSTEM
    prompt = (_SEARCH_PROMPT_MEDIA if media else _SEARCH_PROMPT)
    try:
        result = llm_client.chat_json(
            prompt % {"sentence": sentence, "subjects": subjects or "(unknown)"},
            system=system)
        queries = [str(q).strip() for q in (result.get("queries") or [])
                   if str(q).strip() and len(str(q).strip()) > 4]
    except Exception as exc:
        logger.info("Search-query expansion failed: %s", exc)
        queries = []
    queries = list(dict.fromkeys(queries))[:8]
    if queries:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(queries, ensure_ascii=False),
                                  encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not cache search expansion: %s", exc)
    return queries


# Extra search hooks per streamer handle that reliably surface the known
# original moments (fan-style titles the planner never generates).
_STREAMER_EXTRA_HOOKS = {
    "caseoh": ("guess that burger", "guess the burger", "whose burger is that",
               "burger guessing game", "fast food guessing game"),
    "ishowspeed": ("boots reaction", "golden boots", "ronaldo gave him boots",
                   "showing the boots", "unboxing boots"),
}
# Event-description hooks keyed by action stem: a sentence that names the action
# ("chase", "arrest") but not the incident gets these generic descriptions of
# the moment type injected, so the raw footage (not the sentence's own words)
# enters the pool.  Mirrors how a human re-searches the EVENT, not the sentence.
_EVENT_HOOK_BANK = {
    "chase": ("police chase live", "fleeing police live", "police chase streamer",
              "driver live police chase", "running from police live"),
    "arrest": ("arrested live", "police arrest streamer", "handcuffed live",
               "detained live", "getting arrested live"),
    "fight": ("fight live", "brawl live", "scuffle live stream",
              "gets jumped live", "punching live"),
    "confront": ("confrontation live", "confronted live", "confronts live",
                 "argument live", "face off live"),
    "expel": ("kicked out live", "expelled live", "removed from venue live",
              "kicked out of campus live"),
}
def _event_channel_allowlist(plan: dict) -> set[str] | None:
    """Channels that ARE the event for event queries (official/VOD/participant).

    For a multi-streamer event (Streamer Games, Kai vs. X event, ...) the raw
    footage lives on the event's official channel and each participant's
    channel, yet ``source_reject`` would drop every one of them as a
    "third-party re-upload" of the primary subject.  This builds the set of
    lowercased channel names that must stay in the top-N pool: every candidate
    streamer's known channels/names, plus the event topic phrase (a channel
    whose name contains it, e.g. "Streamer Games", is the event host).  Returns
    None for single-streamer / narration queries (no relaxation there).
    """
    is_event = bool(plan.get("candidate_streamers")) or bool(plan.get("subjects_unknown"))
    if not is_event:
        return None
    allowed: set[str] = set()
    streamers_all = list(plan.get("candidate_streamers") or [])
    primary = plan.get("streamer")
    if isinstance(primary, dict) and primary not in streamers_all:
        streamers_all.append(primary)
    for s in streamers_all:
        if not isinstance(s, dict):
            continue
        for channel in streamers.youtube_channels(s):
            if channel:
                allowed.add(str(channel).lower().strip())
        for key in ("name", "handle", "twitch_handle", "kick", "display_name"):
            value = s.get(key)
            if value and len(str(value)) >= 3:
                allowed.add(str(value).lower().strip())
    sentence = str(plan.get("sentence") or plan.get("query") or "")
    words = [w for w in re.findall(r"[a-z0-9']+", sentence.lower())
             if len(w) > 2 and w not in STOPWORDS]
    if words:
        bigrams = {f"{a} {b}" for a, b in zip(words, words[1:])}
        allowed.update(bigrams)
    return allowed or None


def _event_terms(plan: dict) -> set[str] | None:
    """Topic terms (significant words + bigrams) of an event query's sentence.

    Used by ``source_reject`` to admit on-topic uploads of the event itself
    into the top-N pool from any channel (organizer / sponsor / participant
    POV).  None for non-event queries.
    """
    if not (bool(plan.get("candidate_streamers")) or bool(plan.get("subjects_unknown"))):
        return None
    sentence = str(plan.get("sentence") or plan.get("query") or "")
    words = [w for w in re.findall(r"[a-z0-9']+", sentence.lower())
             if len(w) > 2 and w not in STOPWORDS]
    terms: set[str] = set(words)
    if words:
        terms.update(f"{a} {b}" for a, b in zip(words, words[1:]))
    return terms or None


def _build_candidate(entry: dict, plan: dict, channel_pinned: bool = False,
                     allowed_channels: set[str] | None = None,
                     event_terms: set[str] | None = None) -> dict | None:
    vid = entry.get("id")
    if not vid:
        return None
    candidate = {
        "id": vid,
        "video_id": vid,
        "title": entry.get("title", ""),
        "description": entry.get("description", ""),
        "channel": entry.get("channel", ""),
        "webpage_url": entry.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}",
        "duration": float(entry.get("duration") or entry.get("view_count") or 0),
        "view_count": float(entry.get("view_count") or 0),
        "upload_date": entry.get("upload_date") or "",
        "source": "youtube",
        "channel_pinned": channel_pinned,
        "title_match_score": float(entry.get("title_match_score", 0.0)),
        "rawness_score": float(entry.get("rawness_score", 0.0)),
        "metadata_score": float(entry.get("metadata_score", 0.0)),
        "search_score": float(entry.get("search_score", 0.0)),
    }
    candidate["reject_reason"] = source_reject.reject(candidate, plan,
                                                      allowed_channels,
                                                      event_terms)
    return candidate


def _search_candidates(sentence: str, plan: dict) -> list[dict]:
    is_event = (plan.get("sentence_type") or "").lower() == "event"
    media = bool(plan.get("media_query"))
    use_wide = is_event or media
    query_limit = config.YOUTUBE_QUERY_LIMIT_EVENT if use_wide else 6
    max_results = (config.YOUTUBE_MAX_RESULTS_EVENT if use_wide
                   else config.YOUTUBE_MAX_RESULTS)
    base = list(dict.fromkeys((plan.get("search_queries") or [sentence])
                              + (plan.get("action_search_queries") or [])))
    expanded = _expand_search_queries(sentence, plan)
    streamer = plan.get("streamer") if isinstance(plan, dict) else None
    handle = streamers.twitch_handle(streamer)
    extra = list(_STREAMER_EXTRA_HOOKS.get(handle or "", ()))
    # Visual-event queries (helicopter / police car / gel blaster / fight) hinge
    # on the physical OBJECT: raw VOD titles rarely contain the full sentence, so
    # inject short object-term hooks ("kai cenat helicopter", "helicopter streamer
    # university") that the full-sentence search drowns out.  Without these the
    # actual event footage never enters the pool.
    if (plan.get("visual_event")
            and plan.get("object_terms")
            and not plan.get("media_query")):
        entities = [str(e) for e in (plan.get("core_entities") or [])]
        objects = [str(o).lower() for o in plan.get("object_terms") or []
                   if str(o).strip()]
        obj_hooks = [" ".join(objects)]
        for entity in entities[:1]:
            obj_hooks.append(f"{entity} {' '.join(objects)}")
        obj_hooks.append(" ".join(objects + list(plan.get("topics") or [])))
        base = base + [q for q in obj_hooks if q not in base]
    # Action-described moments (chase/arrest/fight/...) rarely carry the action
    # word in raw VOD titles, and the sentence itself may not name the incident
    # ("kai cenat driver live watching stream" never says "police chase").  Inject
    # generic event-description hooks so the actual footage enters the pool.
    action_stems = {stem(str(t)) for t in (plan.get("action_terms") or [])}
    for stem_key, hooks in _EVENT_HOOK_BANK.items():
        if stem_key in action_stems:
            base = base + [h for h in hooks if h not in base]
    queries = (base + [q for q in expanded if q not in base] + extra)[:query_limit]
    allowed = _event_channel_allowlist(plan)
    event_terms = _event_terms(plan)
    by_id: dict[str, dict] = {}
    for query in queries:
        for entry in downloader.search_candidates(query, max_results=max_results):
            candidate = _build_candidate(entry, plan, allowed_channels=allowed,
                                         event_terms=event_terms)
            if candidate and candidate["id"] not in by_id:
                by_id[candidate["id"]] = candidate
    # Channel-scoped pass: search the streamer's own YouTube channels directly so
    # their raw VOD / clips (MoreCaseOh, Live Speedy, ...) are guaranteed a pool
    # slot instead of being buried by third-party ranking.
    for channel in streamers.youtube_channels(streamer):
        core = " ".join((plan.get("search_queries") or [sentence])[:1])
        for entry in downloader.search_candidates(f"{core} {channel}", max_results=6):
            candidate = _build_candidate(entry, plan, channel_pinned=True,
                                         allowed_channels=allowed,
                                         event_terms=event_terms)
            if candidate and candidate["id"] not in by_id:
                by_id[candidate["id"]] = candidate
    return [c for c in by_id.values() if not c.get("reject_reason")]


def _recency_bonus(upload_date: str) -> float:
    try:
        if not upload_date:
            return 0.0
        uploaded = datetime.strptime(str(upload_date)[:8], "%Y%m%d").date()
        days = (date.today() - uploaded).days
        if days < 0:
            return 0.0
        return 0.06 * max(0.0, 1.0 - days / 730.0)
    except Exception:
        return 0.0


def _is_short(entry: dict) -> bool:
    """True when the entry is a 9:16 YouTube Short ("/shorts/" URL), not a
    regular 16:9 video ("/watch?v=" URL)."""
    url = str(entry.get("webpage_url") or "")
    return "/shorts/" in url


def _search_pool_only(sentence: str, plan: dict) -> tuple[list[dict], list[dict]]:
    """Build the exact-query pool WITHOUT metadata fetch (fast, ~5s).

    Returns (pool, rejected_shorts) where pool candidates have basic search
    metadata only (no _info dict).  The expensive get_video_metadata() call
    is deferred to _fetch_metadata_pool() so visual thumbnail scoring can
    filter the pool first.
    """
    entries = downloader.search_candidates_exact(
        sentence, max_results=config.EXACT_SEARCH_FETCH)
    allowed = _event_channel_allowlist(plan)
    event_terms = _event_terms(plan)
    pool: list[dict] = []
    rejected_shorts: list[dict] = []
    for entry in entries:
        if config.EXACT_SHORTS_EXCLUDE and _is_short(entry):
            logger.info("Exact-query result skipped (9:16 Short): %s",
                        entry.get("webpage_url") or entry.get("id"))
            continue
        candidate = _build_candidate(entry, plan, allowed_channels=allowed,
                                     event_terms=event_terms)
        if not candidate or candidate.get("reject_reason"):
            rejected_shorts.append(entry)
            continue
        candidate["_exact_pool"] = True
        pool.append(candidate)
        if len(pool) >= config.EXACT_SEARCH_FETCH:
            break
    logger.info("Exact-query pool (pre-metadata): %d candidate(s)", len(pool))
    return pool, rejected_shorts


def _fetch_metadata_pool(pool: list[dict]) -> list[dict]:
    """Fetch full metadata for each candidate in parallel (~6s total vs ~60s sequential).

    Filters out Shorts and live streams discovered after resolve.  Returns
    only candidates with valid _info dicts.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_one(candidate: dict) -> dict | None:
        info = downloader.get_video_metadata(candidate["webpage_url"])
        if not info:
            return None
        if config.EXACT_SHORTS_EXCLUDE and _is_short(info):
            logger.info("Exact-query result skipped (9:16 Short after resolve): %s",
                        candidate["webpage_url"])
            return None
        if info.get("is_live") or info.get("live_status") == "is_live":
            return None
        candidate["_info"] = info
        return candidate

    resolved: list[dict] = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_one, c): c for c in pool}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    resolved.append(result)
            except Exception as exc:
                logger.info("Metadata fetch failed: %s", exc)
    logger.info("Metadata resolved: %d / %d candidates", len(resolved), len(pool))
    return resolved


def _search_candidates_exact(sentence: str, plan: dict) -> tuple[list[dict], list[dict]]:
    """Verbatim-query, platform-ordered, raw 16:9 top-N pool.

    Mirrors how a human finds the moment: search the sentence AS-IS, keep
    YouTube's own ranking (no metadata re-rank), keep only raw/original footage
    (``source_reject`` unchanged) at ANY length, and return the top
    ``EXACT_TOP_RESULTS`` 16:9 videos in platform order — Shorts (9:16) are
    excluded so the first twelve results are the regular videos YouTube itself
    ranks for the query, and no long-form floor filters short viral clips out of
    the pool.  The streamer's own raw VODs that never rank for the verbatim
    sentence are still supplied by the channel-pinned pass inside
    ``_search_candidates``.

    The rejected entries are ALSO returned (as ``(pool, rejected_shorts)``) so
    the signature-mining step can transcribe the short ones: their transcripts
    are where the event's payoff dialogue lives verbatim.
    """
    pool, rejected_shorts = _search_pool_only(sentence, plan)
    pool = _fetch_metadata_pool(pool)
    logger.info("Exact-query pool: %d long-form raw video(s)", len(pool))
    for video in pool:
        logger.info("  - %s %s", (video.get("title") or "")[:70],
                    video.get("webpage_url"))
    return pool, rejected_shorts


def _search_channel_pinned(sentence: str, plan: dict,
                           allowed_override: set[str] | None = None,
                           strict: bool = False) -> list[dict]:
    """The subject's own raw-VOD YouTube channels, searched directly.

    The subject's long VOD uploads ("Rakai Live" archive VODs, etc.) rarely
    rank in YouTube's top results for the verbatim sentence, but they are the
    most reliable source of the actual raw footage, so they are guaranteed a
    pool slot (``channel_pinned``) ahead of any third-party upload.  For event
    queries every candidate streamer's channels are searched, not just the
    primary subject's.

    ``strict`` (event queries with a curated registry): only the registry's own
    channels and the candidate streamers' own channels are admitted — no
    on-topic relaxation, so unrelated narration/reaction channels never slip in.
    """
    streamers_all = list(plan.get("candidate_streamers") or [])
    primary = plan.get("streamer")
    if isinstance(primary, dict) and primary not in streamers_all:
        streamers_all.append(primary)
    channels: list[str] = []
    for s in streamers_all:
        for channel in streamers.youtube_channels(s):
            if channel and channel not in channels:
                channels.append(channel)
    if not channels:
        return []
    core = " ".join((plan.get("search_queries") or [sentence])[:1])
    if strict:
        allowed = set(allowed_override or set())
        for s in streamers_all:
            for channel in streamers.youtube_channels(s):
                if channel:
                    allowed.add(str(channel).lower().strip())
        event_terms = None
    else:
        allowed = _event_channel_allowlist(plan)
        event_terms = _event_terms(plan)
    by_id: dict[str, dict] = {}
    for channel in channels:
        for entry in downloader.search_candidates(f"{core} {channel}", max_results=6):
            candidate = _build_candidate(entry, plan, channel_pinned=True,
                                         allowed_channels=allowed,
                                         event_terms=event_terms)
            if candidate and candidate["id"] not in by_id:
                by_id[candidate["id"]] = candidate
    return [c for c in by_id.values() if not c.get("reject_reason")]


def _select_pool(candidates: list[dict], sentence: str, plan: dict) -> list[dict]:
    """Rank search results by which video plausibly CONTAINS the moment:
    semantic (title+desc vs query) + action terms present in the title + rawness.
    This surfaces clip-style videos (e.g. 'CaseOh Guess The Burger') that a bare
    title-word overlap would leave outside the analysis pool.  Channel-pinned
    (the streamer's own channel) videos and recent uploads get a boost."""
    for c in candidates:
        title = c.get("title", "")
        description = c.get("description", "")
        blob = " ".join(part for part in (title, description) if part)
        c["title_match_score"] = title_match_score(sentence, title, description)
        c["rawness_score"] = metadata_rawness_score(title, description, c.get("channel"))
        video_semantic = 0.0
        try:
            scores = semantic_embedder.similarity_scores(sentence, [blob[:800]])
            video_semantic = float(scores[0]) if scores else 0.0
        except Exception:
            video_semantic = 0.0
        c["video_semantic"] = video_semantic
        action_hits, _ = content_verifier.action_match(blob, plan)
        entity_hits, _ = content_verifier.entity_match(blob, plan)
        object_hits, _ = content_verifier.object_match(blob, plan)
        c["action_title_hits"] = action_hits
        c["object_title_hits"] = object_hits
        action_signal = min(1.0, action_hits / max(1, len(plan.get("action_terms") or [])))
        entity_signal = min(1.0, entity_hits / max(1, len(plan.get("core_entities") or [])))
        object_signal = min(1.0, object_hits / max(1, len(plan.get("object_terms") or [])))
        views = c.get("view_count", 0)
        boost = 0.05 * min(1.0, (views / 1_000_000) ** 0.2)
        media = bool(plan.get("media_query"))
        if media:
            # Media footage is judged on how well the video matches the subject,
            # not on how "raw" it looks (premieres/trailers/interviews are edited).
            c["search_score"] = (0.45 * video_semantic + 0.25 * action_signal
                                 + 0.15 * entity_signal + 0.10 * c["title_match_score"]
                                 + 0.05 * c["rawness_score"] + boost
                                 + _recency_bonus(c.get("upload_date", ""))
                                 + (0.15 if c.get("channel_pinned") else 0.0))
        else:
            # Visual-event queries lean on object terms in the metadata (a title
            # mentioning 'police car / helicopter' is a strong containment signal).
            c["search_score"] = (0.35 * video_semantic + 0.20 * action_signal
                                 + 0.15 * entity_signal + 0.10 * object_signal
                                 + 0.20 * c["rawness_score"]
                                 + boost + _recency_bonus(c.get("upload_date", ""))
                                 + (0.15 if c.get("channel_pinned") else 0.0))
    ranked = sorted(candidates, key=lambda c: c["search_score"], reverse=True)
    use_wide = (plan.get("sentence_type") or "").lower() == "event" or bool(plan.get("media_query"))
    pool = ranked[: (config.VIDEOS_TO_ANALYSE_EVENT if use_wide else config.VIDEOS_TO_ANALYSE)]
    pinned = [c for c in ranked if c.get("channel_pinned")]
    pool_ids = {c["id"] for c in pool}
    for c in pinned:
        if c["id"] not in pool_ids:
            pool[-1] = c
            pool_ids.add(c["id"])
    logger.info("YouTube lane analysis pool: %d videos", len(pool))
    for video in pool:
        logger.info("  - %s (raw=%.2f vsem=%.2f act=%d pin=%s) %s",
                    (video.get("title") or "")[:70], video["rawness_score"],
                    video.get("video_semantic", 0.0), video.get("action_title_hits", 0),
                    bool(video.get("channel_pinned")), video["webpage_url"])
    return pool


# --------------------------------------------------------------------------
# Window proposal (semantic-first, narration-friendly)
# --------------------------------------------------------------------------
def _propose_windows(transcript: list[dict], query: str, duration: float,
                     clip_duration: int, plan: dict | None = None,
                     max_windows: int = 8) -> list[dict]:
    """Slide windows over the caption and rank by a hybrid score.

    Semantic-only scoring picks any on-topic region (a basketball dribble
    scored 0.47 for 'calls LeBron'); the hybrid blends the embedding with
    action-term and entity coverage so the actual moment surfaces first.
    """
    stride = config.WINDOW_STRIDE_SECONDS
    windows: dict[int, list[str]] = defaultdict(list)
    for segment in transcript:
        seg_start = float(segment["start"])
        seg_end = float(segment.get("end", seg_start + 2.0))
        first = max(0, int((seg_end - clip_duration) // stride) * stride)
        last = min(int(duration), int(seg_start))
        for start in range(first, last + 1, stride):
            windows[start].append(segment["text"])

    entries: list[dict] = []
    for start, pieces in windows.items():
        evidence = " ".join(dict.fromkeys(pieces))[:1200]
        if len(tokens(evidence)) < 4:
            continue
        entries.append({"start": float(start), "evidence": evidence})

    scores = semantic_embedder.similarity_scores(query, [e["evidence"] for e in entries])
    for entry, semantic in zip(entries, scores or [0.0] * len(entries)):
        entry["semantic_score"] = float(semantic)

    for entry in entries:
        semantic = entry["semantic_score"]
        if isinstance(plan, dict):
            action_hits, action_cov = content_verifier.action_match(entry["evidence"], plan)
            entity_hits, entity_cov = content_verifier.entity_match(entry["evidence"], plan)
            object_hits, object_cov = content_verifier.object_match(entry["evidence"], plan)
        else:
            action_hits, action_cov = 0, 0.0
            entity_hits, entity_cov = 0, 0.0
            object_hits, object_cov = 0, 0.0
        lex = lexical_score(query, entry["evidence"])
        entry.update({
            "action_hits": action_hits,
            "action_coverage": action_cov,
            "entity_hits": entity_hits,
            "entity_coverage": entity_cov,
            "object_hits": object_hits,
            "object_coverage": object_cov,
            "lexical_score": lex,
            # Object coverage only shifts visual events (object_match returns 0
            # for regular queries) so the escort/chopper windows rise to the top.
            # Action coverage is the primary audio signal — a window that speaks
            # the event action outranks one that merely mentions the entity.
            "proposal_score": 0.30 * semantic + 0.35 * action_cov
                              + 0.15 * entity_cov + 0.10 * object_cov + 0.10 * lex,
        })

    entries.sort(key=lambda e: e["proposal_score"], reverse=True)
    kept: list[dict] = []
    for entry in entries:
        if all(abs(entry["start"] - item["start"]) >= config.MIN_CLIP_GAP_SECONDS for item in kept):
            kept.append(entry)
        if len(kept) >= max_windows:
            break
    return kept


# --------------------------------------------------------------------------
# LLM moment-finder windows
# --------------------------------------------------------------------------
def _window_evidence(segments: list[dict], start: float, duration: float) -> str:
    end = start + duration
    return " ".join(dict.fromkeys((seg.get("text") or "").strip()
                                  for seg in (segments or [])
                                  if seg.get("end", 0) >= start and seg.get("start", 0) <= end))


def _llm_windows(segments: list[dict], sentence: str, plan: dict, info: dict,
                 video_id: str, duration: float, window_seconds: int,
                 event_pinned: bool = False) -> list[dict]:
    """Exact moments found by the LLM transcript reader, as window candidates.

    Each found moment becomes a candidate window snapped to the event onset; the
    transcript text inside it is used for ranking/verification exactly like the
    heuristic windows.
    """
    try:
        moments = moment_finder.find_moments(
            segments, sentence, plan, video_id=video_id,
            chapters=info.get("chapters") or [], duration=duration)
    except Exception as exc:
        logger.warning("moment_finder failed for %s: %s", video_id, exc)
        return []
    windows: list[dict] = []
    for moment in moments:
        confidence = float(moment.get("confidence", 0.0))
        if confidence < config.MOMENT_FINDER_MIN_CONFIDENCE:
            continue
        moment_start = max(0.0, float(moment.get("start", 0.0)))
        # Snap export window: 8s lead-in, 22s tail (asymmetric padding).
        # The LLM identified the exact moment — give 8s of context before it
        # and 22s after, instead of centering symmetrically.
        start = max(0.0, moment_start - 8)
        window_duration = config.ACTION_WINDOW_SECONDS
        start = float(int(start))
        evidence = _window_evidence(segments, start, window_duration)
        if len(tokens(evidence)) < 4:
            logger.info("moment_finder moment @%.0fs has no audio in window %s",
                        start, video_id)
            continue
        if not moment_finder.moment_is_grounded(moment, evidence, plan,
                                                event_pinned=event_pinned):
            logger.info("moment_finder dropping ungrounded moment @%.0fs in %s: %s",
                        start, video_id, moment.get("reason", ""))
            continue
        windows.append({
            "start": start,
            "duration": float(window_duration),
            "evidence": evidence,
            "llm_found": True,
            "llm_moment_score": confidence,
            "llm_moment_reason": moment.get("reason", ""),
        })
    return windows


def _merge_windows(heuristic: list[dict], llm: list[dict], max_windows: int) -> list[dict]:
    """Keyword-pinned and LLM-found moments first, then heuristic windows, deduped.

    An exact spoken-phrase keyword hit is ground truth for the moment, so it
    outranks fuzzy LLM/visual windows even when those carry llm_found=True.
    """
    combined = sorted(llm + heuristic,
                      key=lambda w: (bool(w.get("keyword_located")),
                                     bool(w.get("llm_found")),
                                     w.get("keyword_priority", 0.0),
                                     w.get("proposal_score", 0.0)),
                      reverse=True)
    kept: list[dict] = []
    for window in combined:
        if all(abs(window["start"] - item["start"]) >= config.MIN_CLIP_GAP_SECONDS
               for item in kept):
            kept.append(window)
        if len(kept) >= max_windows:
            break
    return kept


def _keyword_windows(segments: list[dict], plan: dict, window_seconds: int,
                     max_windows: int) -> list[dict]:
    """Exact moments pinned by the spoken-phrase lexicon, as candidate windows.

    Keyword windows carry ``keyword_located`` + ``locator_hits`` — the captions
    literally speak the moment's phrase — and are scored like the LLM windows so
    they rank by their own audio evidence.
    """
    try:
        moments = moment_finder.keyword_moments(
            segments, plan, window_seconds, max_windows=max_windows)
    except Exception as exc:
        logger.warning("keyword_moments failed: %s", exc)
        return []
    target = (plan.get("semantic_target") or plan.get("sentence") or "")
    for window in moments:
        scores = semantic_embedder.similarity_scores(target, [window["evidence"]])
        window["semantic_score"] = float(scores[0]) if scores else 0.0
        action_hits, action_cov = content_verifier.action_match(window["evidence"], plan)
        entity_hits, entity_cov = content_verifier.entity_match(window["evidence"], plan)
        object_hits, object_cov = content_verifier.object_match(window["evidence"], plan)
        lex = lexical_score(target, window["evidence"])
        priority = float(window.get("keyword_priority", 0.0))
        window.update({
            "action_hits": action_hits,
            "action_coverage": action_cov,
            "entity_hits": entity_hits,
            "entity_coverage": entity_cov,
            "object_hits": object_hits,
            "object_coverage": object_cov,
            "lexical_score": lex,
            "proposal_score": 0.50 * priority + 0.30 * window["semantic_score"]
                              + 0.20 * action_cov,
        })
    return moments


def _sample_candidate_frames(candidate: dict, info: dict,
                             frames_dir: Path) -> list[dict]:
    """Sample a bounded set of frames from the candidate window (best-effort).

    Remote ffmpeg seeks are slow/timeout-prone on this setup, so this is ONE
    cheap sampling pass reused by the general scene probe, the event probe, the
    anchor match, and the LLM judge's contact sheet.  Failures return [] and the
    candidate is simply scored on audio/metadata alone.
    """
    try:
        stream = youtube_search.get_frame_stream(info)
        if not stream:
            return []
        extra_args = youtube_search.stream_headers(info)
        start = float(candidate["start"])
        duration = float(candidate.get("duration") or config.CLIP_DURATION_SECONDS)
        count = max(config.GENERAL_VISUAL_FRAMES, config.EVENT_VISUAL_FRAMES)
        timestamps = [start + duration * (i + 1) / (count + 1)
                      for i in range(count)]
        return youtube_search.sample_frames_at(
            stream, timestamps, frames_dir, extra_args,
            video_id=candidate.get("video_id"))
    except Exception as exc:
        logger.info("Candidate frame sampling failed for %s: %s",
                    candidate.get("video_id"), exc)
        return []


def _visual_check(candidate: dict, info: dict, frames_dir: Path,
                  plan: dict | None = None, anchors=None) -> None:
    """Attach vision/overlay/anchor signals; failures degrade to neutral.

    The GENERAL scene probe (plan.scene_prompts) is scored for EVERY candidate,
    not just the 6 whitelisted event kinds, so pixel truth feeds ranking
    (vision_score) and the export gate for all queries.  Frames are sampled once
    and reused across every consumer.
    """
    candidate.setdefault("visual_raw", 0.5)
    frames = _sample_candidate_frames(candidate, info, frames_dir)
    paths = [f["path"] for f in frames]
    if paths:
        candidate["visual_frames"] = paths
        if VISUAL_VERIFY:
            scored = vision_ranker.score_frames(frames, candidate.get("visual_prompt", ""))
            candidate["vision_score"] = max((sf["similarity"] for sf in scored), default=0.0)
            visual = overlay_detector.visual_rawness(frames)
            candidate["overlay"] = visual["overlay"]
            candidate["continuity"] = visual["continuity"]
            candidate["visual_raw"] = visual["visual_raw"]
    general = event_vision.general_probe(plan)
    if general.active() and paths:
        try:
            scores = general.scene_scores(paths)
            if scores:
                summary = general.summarize(scores)
                candidate["vision_score"] = 1.0 if summary["present"] else summary["fraction"]
                candidate["general_visual_best"] = summary["best"]
                candidate["general_visual_fraction"] = summary["fraction"]
        except Exception as exc:
            logger.info("General scene probe failed for %s: %s",
                        candidate.get("video_id"), exc)
    _attach_event_probe(candidate, plan, paths)
    if anchors is not None and getattr(anchors, "strong", False) and paths:
        try:
            sims = anchors.frame_sims_strict(paths)
            if sims:
                candidate["anchor_sim"] = max(sims)
                candidate["anchor_fraction"] = round(
                    sum(1 for s in sims if s + 1e-6 >= config.ANCHOR_SIM_FLOOR) / len(sims), 3)
        except Exception as exc:
            logger.info("Anchor match failed for %s: %s",
                        candidate.get("video_id"), exc)


def _attach_event_probe(candidate: dict, plan: dict | None,
                        frame_paths: list[str] | None) -> None:
    """Score sampled frames against the event's CLIP scenes (visual events only)."""
    probe = event_vision.probe_for(plan)
    if not probe.active():
        return
    paths = frame_paths or []
    if not paths:
        return
    scores = probe.scene_scores(paths)
    if not scores:
        return
    summary = probe.summarize(scores)
    candidate["event_visual_score"] = 1.0 if summary["present"] else summary["fraction"]
    candidate["event_visual_best"] = summary["best"]
    candidate["event_visual_fraction"] = summary["fraction"]
    candidate["event_visual_frames"] = paths


# --------------------------------------------------------------------------
# Per-video analysis
# --------------------------------------------------------------------------
def _analyse_video(meta: dict, sentence: str, plan: dict, anchors=None) -> list[dict]:
    target = (plan.get("semantic_target") or sentence) if isinstance(plan, dict) else sentence
    video_sem = float(meta.get("video_semantic") or 0.0)
    if 0.0 < video_sem < config.VIDEO_SEMANTIC_SKIP_FLOOR:
        logger.info("Skipping %s: video_semantic=%.2f below floor %.2f",
                    meta.get("webpage_url", meta.get("video_id")), video_sem,
                    config.VIDEO_SEMANTIC_SKIP_FLOOR)
        return []
    info = meta.get("_info") or downloader.get_video_metadata(meta["webpage_url"])
    if not info:
        return []
    if info.get("age_limit"):
        logger.info("Skipping age-restricted video: %s", meta["webpage_url"])
        return []
    duration = float(info.get("duration") or 0)
    # Exact-pool 16:9 clips are checked at ANY length (no 120s floor, and short
    # viral clips down to CLIP_MIN_SECONDS are kept too); other lanes keep the
    # general MIN_VIDEO_LENGTH floor.
    min_length = (config.CLIP_MIN_SECONDS if meta.get("_exact_pool")
                  else config.MIN_VIDEO_LENGTH)
    if duration < min_length or duration > config.MAX_VIDEO_LENGTH:
        return []
    if info.get("is_live") or info.get("live_status") == "is_live":
        return []
    video_id = str(info.get("id") or meta["video_id"])
    frames_dir = config.FRAMES_DIR / f"sentence_{video_id}"

    window_seconds = (config.EXACT_WINDOW_SECONDS if meta.get("_exact_pool")
                      else int(plan.get("window_seconds") or config.CLIP_DURATION_SECONDS))
    # Media clips (trailers, interviews, reactions, official clips) export WHOLE
    # up to CLIP_MAX_SECONDS — the whole trailer/interview is the best clip.  A
    # longer premiere/red-carpet video is still sectioned to best moments.
    # Exact-pool 16:9 clips at ANY length now enter the pool (no long-form
    # floor), so short viral clips (30-240s) are also exported WHOLE — they ARE
    # the moment; only longer VODs are sectioned to windows.
    exact_short = bool(meta.get("_exact_pool")) and not plan.get("media_query")
    whole_limit = (config.CLIP_MAX_SECONDS if (plan.get("media_query") or exact_short)
                   else window_seconds)
    whole = config.CLIP_MIN_SECONDS <= duration <= whole_limit

    segments = youtube_search.get_caption_segments(info)
    if not segments and (whole or duration <= config.MAX_TRANSCRIPTION_SECONDS):
        audio_url = youtube_search.get_best_audio_stream(info)
        if audio_url:
            logger.info("No captions; transcribing audio for %s", video_id)
            languages = (plan.get("auto") or {}).get("languages") if isinstance(plan, dict) else None
            segments = audio_transcriber.transcribe(
                audio_url, extra_args=youtube_search.stream_headers(info),
                model_size=audio_transcriber.model_for(languages),
                max_seconds=None if whole else duration)
    if not segments:
        return []

    if whole:
        candidate = dict(meta)
        candidate.update({
            "video_id": video_id,
            "duration": duration,
            "start": 0.0,
            "whole": True,
            "source": "youtube",
            "query": sentence,
            "title": meta.get("title", ""),
            "description": meta.get("description", ""),
            "channel": meta.get("channel", ""),
        })
        if meta.get("llm_scan") and not candidate.get("llm_found"):
            try:
                found_moments = moment_finder.find_moments(
                    segments, sentence, plan, video_id=video_id,
                    chapters=info.get("chapters") or [], duration=duration)
                full_text = content_verifier.transcript_text(segments)
                grounded = [m for m in found_moments
                            if moment_finder.moment_is_grounded(m, full_text, plan)]
            except Exception as exc:
                logger.warning("moment_finder failed for whole %s: %s", video_id, exc)
                grounded = []
            if grounded:
                best = max(grounded, key=lambda m: m.get("confidence", 0.0))
                candidate["llm_found"] = True
                candidate["llm_moment_score"] = float(best.get("confidence", 0.0))
                candidate["llm_moment_reason"] = best.get("reason", "")
        keyword_hits = moment_finder.keyword_hits(segments, plan)
        if keyword_hits:
            candidate["keyword_located"] = True
            candidate["locator_hits"] = list(dict.fromkeys(
                h["name"] for h in keyword_hits))[:3]
        ok, reason = content_verifier.verify(candidate, target, plan, segments)
        candidate["verification_reason"] = reason
        logger.info("youtube whole-clip %s (dur=%.0fs) -> %s", video_id, duration,
                    "OK" if ok else reason)
        if ok:
            candidate["transcript"] = content_verifier.transcript_text(segments)
            _visual_check(candidate, info, frames_dir, plan, anchors)
            _corroborate(candidate, plan)
            return [candidate]
        return []

    # Event-registry (pinned) videos are multi-hour raw broadcasts.  The
    # full-stride semantic scan over 12k+ windows is both slow and mostly
    # noise; the cheap keyword locator already nails the confirmed moments
    # (plus a few extra candidates), so skip the whole-VOD scan for them.
    if meta.get("event_pinned"):
        heuristic = []
    else:
        heuristic = _propose_windows(segments, target, duration, window_seconds,
                                     plan=plan,
                                     max_windows=config.MAX_CANDIDATES_PER_VIDEO)
    llm_windows = (_llm_windows(segments, sentence, plan, info, video_id, duration,
                                window_seconds,
                                event_pinned=bool(meta.get("event_pinned")))
                   if meta.get("llm_scan") else [])
    for window in llm_windows:
        scores = semantic_embedder.similarity_scores(target, [window["evidence"]])
        semantic = float(scores[0]) if scores else 0.0
        action_hits, action_cov = content_verifier.action_match(window["evidence"], plan)
        entity_hits, entity_cov = content_verifier.entity_match(window["evidence"], plan)
        object_hits, object_cov = content_verifier.object_match(window["evidence"], plan)
        lex = lexical_score(target, window["evidence"])
        confidence = float(window.get("llm_moment_score", 0.0))
        window.update({
            "semantic_score": semantic,
            "action_hits": action_hits,
            "action_coverage": action_cov,
            "entity_hits": entity_hits,
            "entity_coverage": entity_cov,
            "object_hits": object_hits,
            "object_coverage": object_cov,
            "lexical_score": lex,
            "proposal_score": 0.45 * confidence + 0.35 * semantic + 0.20 * action_cov,
        })
    keyword_windows = _keyword_windows(segments, plan, window_seconds,
                                       config.MAX_CANDIDATES_PER_VIDEO + 4)
    windows = _merge_windows(heuristic, llm_windows + keyword_windows,
                             config.MAX_CANDIDATES_PER_VIDEO)
    # Visual localization runs for EVERY query (not just the 6 hardcoded event
    # kinds): captions rarely say "a helicopter is landing" or "Kai backflips",
    # so the transcript windows miss exactly the moments a visual query needs.
    # Always coarsely scan the video and CLIP-locate the strongest scene zones
    # using the query's own scene prompts (the general probe), THEN merge them
    # with the transcript windows — the pixels decide where the moment actually
    # is.  A recognized visual-event kind uses its more specific probe when
    # available.  sample_frames_at early-aborts on remote-seek timeouts, so the
    # cost stays bounded.
    probe = event_vision.probe_for(plan)
    grid_probe = probe if probe.active() else event_vision.general_probe(plan)
    if grid_probe.active() and not plan.get("media_query"):
        stream = youtube_search.get_frame_stream(info)
        if stream:
            extra_args = youtube_search.stream_headers(info)
            visual_windows = vod_scan.localize(
                stream, extra_args, duration, grid_probe, video_id, window_seconds,
                config.MAX_CANDIDATES_PER_VIDEO,
                max_frames=config.YOUTUBE_VISUAL_SCAN_FRAMES,
                min_stride=config.YOUTUBE_VISUAL_MIN_STRIDE)
            if visual_windows:
                for window in visual_windows:
                    window.setdefault("proposal_score", float(window.get("mean", 0.0)))
                    window.setdefault("llm_found", False)
                    window["visual_zone"] = True
                windows = _merge_windows(windows, visual_windows,
                                         config.MAX_CANDIDATES_PER_VIDEO)
                logger.info("Video %s: visual scan added %d window(s) (total %d)",
                            video_id, len(visual_windows), len(windows))
    forced = _forced_start(video_id)
    forced_candidate = None
    if forced:
        forced_start, forced_reason = forced
        forced_window = _window_evidence(segments, forced_start, window_seconds)
        if len(tokens(forced_window)) < 4:
            forced_window = " ".join((seg.get("text") or "").strip()
                                     for seg in (segments or []))[:1200]
        forced_candidate = dict(meta)
        forced_candidate.update({
            "video_id": video_id,
            "duration": float(window_seconds),
            "start": forced_start,
            "timestamp": forced_start,
            "whole": False,
            "source": "youtube",
            "query": sentence,
            "title": meta.get("title", ""),
            "description": meta.get("description", ""),
            "channel": meta.get("channel", ""),
            "semantic_score": 0.0,
            "action_hits": 0,
            "action_coverage": 0.0,
            "entity_hits": 0,
            "entity_coverage": 0.0,
            "object_hits": 0,
            "object_coverage": 0.0,
            "lexical_score": 0.0,
            "proposal_score": 1.0,
            "evidence": forced_window,
            "snippet": forced_window,
            "llm_found": False,
            "llm_moment_score": 0.0,
            "llm_moment_reason": "",
            "keyword_located": True,
            "locator_hits": ["user-forced"],
            "keyword_priority": 1.0,
            "keyword_hit_ts": forced_start,
            "visual_zone": False,
            "editorial_score": evidence_editorial_score(forced_window),
            "visual_prompt": plan.get("visual_prompt") or sentence,
            "verified": True,
            "verification_reason": f"user-forced moment: {forced_reason}",
            "forced_moment": True,
        })
        logger.info("youtube forced moment %s @%.0fs (%s)",
                    video_id, forced_start, forced_reason)
    if not windows and not forced_candidate:
        return []
    for window in windows:
        window.setdefault("semantic_score", 0.0)
        window.setdefault("action_hits", 0)
        window.setdefault("action_coverage", 0.0)
        window.setdefault("entity_hits", 0)
        window.setdefault("entity_coverage", 0.0)
        window.setdefault("object_hits", 0)
        window.setdefault("object_coverage", 0.0)
        window.setdefault("lexical_score", 0.0)
        window.setdefault("llm_found", False)
        window.setdefault("llm_moment_score", 0.0)
        window.setdefault("llm_moment_reason", "")
        window.setdefault("keyword_located", False)
        window.setdefault("locator_hits", [])
        window.setdefault("keyword_priority", 0.0)
        window.setdefault("proposal_score", float(window.get("mean", 0.0)))
    candidates = []
    for window in windows:
        start = float(window["start"])
        candidate = dict(meta)
        candidate.update({
            "video_id": video_id,
            "duration": float(window_seconds),
            "start": start,
            "timestamp": start,
            "whole": False,
            "source": "youtube",
            "query": sentence,
            "title": meta.get("title", ""),
            "description": meta.get("description", ""),
            "channel": meta.get("channel", ""),
            "semantic_score": float(window["semantic_score"]),
            "action_hits": int(window["action_hits"]),
            "action_coverage": float(window["action_coverage"]),
            "entity_hits": int(window["entity_hits"]),
            "entity_coverage": float(window["entity_coverage"]),
            "object_hits": int(window["object_hits"]),
            "object_coverage": float(window["object_coverage"]),
            "lexical_score": float(window["lexical_score"]),
            "proposal_score": float(window["proposal_score"]),
            "evidence": window["evidence"],
            "snippet": window["evidence"],
            "llm_found": bool(window.get("llm_found", False)),
            "llm_moment_score": float(window.get("llm_moment_score", 0.0)),
            "llm_moment_reason": window.get("llm_moment_reason", ""),
            "keyword_located": bool(window.get("keyword_located", False)),
            "locator_hits": list(window.get("locator_hits") or []),
            "keyword_priority": float(window.get("keyword_priority", 0.0)),
            "keyword_hit_ts": float(window.get("hit_ts") or -1.0),
            "visual_zone": bool(window.get("visual_zone", False)),
            "editorial_score": evidence_editorial_score(window["evidence"]),
            "visual_prompt": plan.get("visual_prompt") or sentence,
        })
        # Skip windows with zero transcript match AND no keyword/LLM locator signal.
        # These are completely unrelated windows that waste verification time.
        _sem = float(window["semantic_score"])
        _act = int(window["action_hits"])
        _kw = bool(window.get("keyword_located", False))
        _llm = bool(window.get("llm_found", False))
        _viz = bool(window.get("visual_zone", False))
        if _sem == 0.0 and _act == 0 and not _kw and not _llm and not _viz:
            logger.info("youtube window %s @%.0fs (sem=0.00 act=0) -> skipped "
                        "(no transcript/keyword/visual signal)", video_id, start)
            continue
        _visual_check(candidate, info, frames_dir, plan, anchors)
        _corroborate(candidate, plan)
        ok, reason = content_verifier.verify(candidate, target, plan, segments)
        candidate["verification_reason"] = reason
        logger.info("youtube window %s @%.0fs (sem=%.2f act=%d) -> %s", video_id,
                    start, float(window["semantic_score"]), int(window["action_hits"]),
                    "OK" if ok else reason)
        if ok:
            candidates.append(candidate)
    if forced_candidate is not None:
        already = any(abs(float(c["start"]) - forced_candidate["start"]) <= 1.0
                      for c in candidates)
        if not already:
            _visual_check(forced_candidate, info, frames_dir, plan, anchors)
            candidates.append(forced_candidate)
    # VLM window verification: the export-time VLM gate is authoritative,
    # so analysis-time VLM is skipped to save ~3-8s per video.
    candidates.sort(key=lambda c: (
        bool(c.get("vlm_happening")),
        float(c.get("vlm_score") or 0.0),
        float(c.get("proposal_score", c.get("semantic_score", 0.0))),
    ), reverse=True)
    logger.info("Video %s produced %d verified window(s)", video_id, len(candidates))
    return candidates


# --------------------------------------------------------------------------
# Signature-phrase mining from rejected short clips
# --------------------------------------------------------------------------
def _normalize_text(text: str) -> str:
    """Lowercased, whitespace-collapsed text used for phrase matching."""
    return " ".join(str(text or "").lower().split())


def _clip_transcript(info: dict, plan: dict | None = None) -> list[dict] | None:
    """Caption segments for a short clip; Whisper fallback (never a download)."""
    try:
        segments = youtube_search.get_caption_segments(info)
        if segments:
            return segments
        duration = float(info.get("duration") or 0)
        if duration > config.MAX_TRANSCRIPTION_SECONDS:
            return None
        audio_url = youtube_search.get_best_audio_stream(info)
        if not audio_url:
            return None
        languages = (plan.get("auto") or {}).get("languages") if isinstance(plan, dict) else None
        return audio_transcriber.transcribe(
            audio_url, extra_args=youtube_search.stream_headers(info),
            model_size=audio_transcriber.model_for(languages), max_seconds=duration)
    except Exception as exc:
        logger.info("Signature-mining transcript failed for %s: %s",
                    info.get("id") or info.get("webpage_url"), exc)
        return None


def _clip_relevance(entry: dict, sentence: str) -> float:
    """Lexical title/desc relevance used to pick which rejected shorts to mine."""
    title = entry.get("title", "")
    description = entry.get("description", "")
    blob = " ".join(part for part in (title, description) if part)
    return title_match_score(sentence, title, description)


_MINER_SYSTEM = (
    "You extract the exact spoken phrases that mark a specific live-stream moment "
    "from its transcript. Reply only JSON."
)
_MINER_PROMPT = """Transcript of a short clip about: "%(sentence)s"
Subjects: %(subjects)s

The clip shows a specific moment (a police chase, a confrontation, an accident,
a confession). Return 2-4 short verbatim phrases (each 3-10 words) that someone
LITERALLY says at the moment — the signature lines that identify THIS incident
in any recording of it. Each phrase MUST appear VERBATIM inside a SINGLE line
of the numbered transcript below (the auto-caption lines are short, so phrases
that cross lines are useless). Never invent phrases; never return generic chat
filler.

Transcript (one caption line per row):
%(transcript)s

Reply only JSON: {"phrases": ["...", "..."]}"""


def _mine_signature_phrases(sentence: str, plan: dict,
                            rejected_shorts: list[dict]) -> list[str]:
    """Mine the moment's signature spoken phrases from rejected short clips.

    The exact-query pass rejects short clips on title/channel policy (reaction-
    titled uploads of the driver's own IG live, fan clips), yet those transcripts
    are the guaranteed-verbatim source of the payoff dialogue ("I'm watching bro,
    I'm off this").  Extract the signature phrases and return them normalized; the
    caller feeds them back as keyword-lexicon patterns that pin long VODs at the
    exact second.  Degrades to [] on any failure.
    """
    if not rejected_shorts:
        return []
    ranked = [e for e in rejected_shorts if _clip_relevance(e, sentence) > 0.0]
    ranked = sorted(ranked, key=lambda e: _clip_relevance(e, sentence), reverse=True)
    candidates = ranked[: config.SIGNATURE_MINING_MAX_CLIPS]
    subjects = ", ".join(plan.get("core_entities") or [])
    streamer = plan.get("streamer")
    if isinstance(streamer, dict) and streamer.get("name"):
        subjects = streamer["name"] + (f", {subjects}" if subjects else "")
    phrases: list[str] = []
    for entry in candidates:
        if len(phrases) >= config.SIGNATURE_MINING_MAX_PHRASES:
            break
        info = downloader.get_video_metadata(entry["webpage_url"])
        if not info:
            continue
        duration = float(info.get("duration") or 0)
        if duration < config.CLIP_MIN_SECONDS or duration > config.SIGNATURE_MINING_MAX_SECONDS:
            continue
        if info.get("is_live") or info.get("live_status") == "is_live":
            continue
        segments = _clip_transcript(info, plan)
        if not segments:
            continue
        per_segment = [_normalize_text(seg.get("text")) for seg in segments]
        per_segment = [t for t in per_segment if t]
        full = " ".join(per_segment)
        if not full:
            continue
        sim = 0.0
        # The auto-caption text alone under-scores (fragmented, censored tokens);
        # the cleanest same-event signal is the clip's own title ("guy goes on
        # instagram live during police chase while watching kai cenat"), so the
        # relevance check blends title+description+transcript.
        try:
            blob = " ".join(part for part in
                            (entry.get("title"), entry.get("description"), full) if part)
            scores = semantic_embedder.similarity_scores(sentence, [blob[:1500]])
            sim = float(scores[0]) if scores else 0.0
        except Exception:
            sim = 0.0
        if sim < config.SIGNATURE_MINING_MIN_SIM:
            logger.info("Signature mining: short %s not the event (sem=%.2f)",
                        info.get("id"), sim)
            continue
        try:
            result = llm_client.chat_json(
                _MINER_PROMPT % {
                    "sentence": sentence,
                    "subjects": subjects or "(unknown)",
                    "transcript": "\n".join(f"[{i}] {t}" for i, t in enumerate(per_segment[:120])),
                },
                system=_MINER_SYSTEM)
        except Exception as exc:
            logger.info("Signature mining LLM failed for %s: %s", info.get("id"), exc)
            continue
        for phrase in (result or {}).get("phrases") or []:
            norm = _normalize_text(phrase)
            # Keep only phrases that appear VERBATIM inside a single caption
            # segment (keyword_hits matches per-segment, not across the join).
            if len(norm.split()) < 3:
                continue
            if not any(norm in seg_text for seg_text in per_segment):
                continue
            if norm in phrases:
                continue
            phrases.append(norm)
            if len(phrases) >= config.SIGNATURE_MINING_MAX_PHRASES:
                break
    if phrases:
        logger.info("Mined %d signature phrase(s) from rejected shorts: %s",
                    len(phrases), "; ".join(phrases))
    return phrases[: config.SIGNATURE_MINING_MAX_PHRASES]


def _apply_mined_lexicon(plan: dict, sentence: str,
                         mined_phrases: list[str]) -> None:
    """Inject mined phrases into the plan as keyword-lexicon patterns.

    ``keyword_hits`` / ``keyword_moments`` read the plan through
    ``moment_finder._lexicon_for``; keying the mined lexicon by the sentence
    itself guarantees the containment check activates.
    """
    if not mined_phrases:
        return
    key = str(plan.get("sentence") or sentence).lower()
    plan["mined_phrases"] = mined_phrases
    specs: dict = {}
    for i, phrase in enumerate(mined_phrases):
        specs[f"mined{i}"] = {
            "query_terms": (key,),
            "patterns": (re.escape(phrase),),
            "pad": 8.0,
            "priority": 5,
        }
    plan["mined_lexicon"] = {key: specs}


def _corroborate(candidate: dict, plan: dict | None) -> None:
    """Cross-source corroboration: a mined signature phrase heard in the window.

    A phrase mined from a SHORT clip of the event that also appears in a LONG-VOD
    window is independent confirmation the window is the same moment ("two
    witnesses"), so the window gets a ranking lift and a verification flag.
    """
    if not config.CORROBORATION_ENABLED or not isinstance(plan, dict):
        return
    mined = plan.get("mined_phrases") or []
    if not mined:
        return
    evidence = _normalize_text(candidate.get("snippet")
                               or candidate.get("evidence")
                               or candidate.get("transcript")
                               or "")
    hit = next((p for p in mined if p in evidence), None)
    if not hit:
        return
    candidate["corroborated"] = True
    candidate["corroboration_phrase"] = hit
    candidate["proposal_score"] = float(candidate.get("proposal_score") or 0.0) \
        + config.CORROBORATION_LIFT
    logger.info("youtube window %s @%.0fs corroborated by mined phrase %r",
                candidate.get("video_id"), float(candidate.get("start") or 0.0), hit)


def _build_visual_candidate(meta: dict, sentence: str, plan: dict,
                            anchors=None) -> dict | None:
    """Build a candidate from a visual-moment-found video.

    Pipeline: visual match discovers the exact timestamp → transcript refines
    the optimal 30-second window around it.  The visual moment is the anchor;
    the transcript only searches ±60s around it to find the best audio window.

    Returns None when the visual match is too weak or metadata is missing.
    """
    info = meta.get("_info")
    if not info:
        return None
    video_id = str(info.get("id") or meta.get("video_id") or "")
    duration = float(info.get("duration") or meta.get("duration") or 0)
    visual_start = float(meta.get("visual_moment_start") or 0.0)
    visual_score = float(meta.get("visual_moment_score") or 0.0)
    if visual_start < 0 or (duration > 0 and visual_start >= duration):
        return None

    window_seconds = config.EXACT_WINDOW_SECONDS
    target = (plan.get("semantic_target") or sentence) if isinstance(plan, dict) else sentence

    # --- Step 1: Build base candidate dict ----------------------------------
    candidate = dict(meta)
    candidate.update({
        "video_id": video_id,
        "duration": duration,
        "start": visual_start,
        "end": min(visual_start + window_seconds, duration) if duration > 0 else visual_start + window_seconds,
        "whole": False,
        "source": "youtube",
        "query": sentence,
        "title": meta.get("title", ""),
        "description": meta.get("description", ""),
        "channel": meta.get("channel", ""),
        "visual_moment_start": visual_start,
        "visual_moment_score": visual_score,
        "visual_event": True,
    })
    candidate["proposal_score"] = max(
        float(candidate.get("proposal_score") or 0.0),
        visual_score,
    )
    candidate["vision_score"] = visual_score
    candidate["anchor_sim"] = visual_score

    # --- Step 2: Fetch transcript -------------------------------------------
    segments = youtube_search.get_caption_segments(info)
    if not segments and duration <= config.MAX_TRANSCRIPTION_SECONDS:
        audio_url = youtube_search.get_best_audio_stream(info)
        if audio_url:
            logger.info("No captions; transcribing audio for visual candidate %s", video_id)
            languages = (plan.get("auto") or {}).get("languages") if isinstance(plan, dict) else None
            segments = audio_transcriber.transcribe(
                audio_url, extra_args=youtube_search.stream_headers(info),
                model_size=audio_transcriber.model_for(languages),
                max_seconds=duration)

    if not segments:
        # No transcript — visual-only candidate
        candidate["verification_reason"] = (
            f"visual moment @ {visual_start:.0f}s (score={visual_score:.3f}, no transcript)")
        candidate["verified"] = True
        logger.info("Visual candidate %s @%.0fs score=%.3f (no transcript, visual-only)",
                    video_id, visual_start, visual_score)
        return candidate

    # --- Step 3: Transcript refinement — find the best 30s window -----------
    # The visual moment is the anchor.  Search ±60s around it for the window
    # whose transcript best matches the query (semantic + action + entity).
    search_pad = 60.0
    search_lo = max(0.0, visual_start - search_pad)
    search_hi = min(duration, visual_start + search_pad + window_seconds)

    # Build candidate windows at 5-second strides within the search range
    stride = 5
    best_window = None
    best_score = -1.0
    best_text = ""

    for w_start_int in range(int(search_lo), int(search_hi) - window_seconds + 1, stride):
        w_start = float(w_start_int)
        w_end = w_start + window_seconds
        # Extract transcript text for this window
        text = " ".join(
            (seg.get("text") or "").strip()
            for seg in segments
            if seg.get("end", 0) >= w_start and seg.get("start", 0) <= w_end
        ).strip()
        if len(text) < 8:
            continue

        # Score this window
        semantic_scores = semantic_embedder.similarity_scores(target, [text])
        semantic = float(semantic_scores[0]) if semantic_scores else 0.0
        action_hits, action_cov = content_verifier.action_match(text, plan)
        entity_hits, entity_cov = content_verifier.entity_match(text, plan)
        object_hits, object_cov = content_verifier.object_match(text, plan)
        lex = lexical_score(target, text)

        # Visual proximity bonus: windows closer to the visual moment score higher
        dist = abs(w_start - visual_start)
        proximity = max(0.0, 1.0 - dist / search_pad)

        combined = (0.30 * semantic
                    + 0.30 * action_cov
                    + 0.15 * entity_cov
                    + 0.10 * object_cov
                    + 0.10 * lex
                    + 0.05 * proximity)

        if combined > best_score:
            best_score = combined
            best_window = w_start
            best_text = text

    # Fall back to visual-moment-centered window if no transcript window wins
    if best_window is None or best_score <= 0:
        best_window = visual_start
        best_text = content_verifier._window_text(segments, visual_start, window_seconds)

    # --- Step 4: Populate all candidate fields ------------------------------
    semantic_scores = semantic_embedder.similarity_scores(target, [best_text])
    semantic = float(semantic_scores[0]) if semantic_scores else 0.0
    action_hits, action_cov = content_verifier.action_match(best_text, plan)
    entity_hits, entity_cov = content_verifier.entity_match(best_text, plan)
    object_hits, object_cov = content_verifier.object_match(best_text, plan)
    lex = lexical_score(target, best_text)

    candidate.update({
        "start": best_window,
        "end": min(best_window + window_seconds, duration) if duration > 0 else best_window + window_seconds,
        "semantic_score": semantic,
        "action_hits": action_hits,
        "action_coverage": action_cov,
        "entity_hits": entity_hits,
        "entity_coverage": entity_cov,
        "object_hits": object_hits,
        "object_coverage": object_cov,
        "lexical_score": lex,
        "evidence": best_text[:1200],
        "snippet": best_text[:1200],
        "transcript": content_verifier.transcript_text(segments),
        "proposal_score": max(
            float(candidate.get("proposal_score") or 0.0),
            0.45 * best_score + 0.35 * semantic + 0.20 * action_cov,
        ),
    })

    # --- Step 5: Verify via content_verifier --------------------------------
    ok, reason = content_verifier.verify(candidate, target, plan, segments)
    candidate["verification_reason"] = reason

    if not ok:
        logger.info("Visual candidate %s @%.0fs score=%.3f -> REJECTED: %s",
                    video_id, best_window, visual_score, reason)
        return None

    logger.info("Visual candidate %s @%.0fs score=%.3f sem=%.2f act=%d ent=%d -> %s",
                video_id, best_window, visual_score, semantic,
                action_hits, entity_hits, "OK" if ok else reason)

    # --- Step 6: VLM check if frames available ------------------------------
    if config.VLM_VERIFY_ENABLED:
        frames_dir = config.FRAMES_DIR / f"sentence_{video_id}"
        _visual_check(candidate, info, frames_dir, plan, anchors)

    return candidate


# --------------------------------------------------------------------------
# Lane entry point
# --------------------------------------------------------------------------
def run(sentence: str, plan: dict, need: int,
        anchors=None, extra_pool: list[dict] | None = None) -> list[dict]:
    """Return content-verified YouTube window candidates for the sentence.

    Simplified flow: YouTube's own top-10 results for the verbatim query
    (platform ranking is the best signal) + the streamer's own raw-VOD
    channels.  Each video is fully analysed (coarse visual scan → fine frame
    sampling → transcript windows → LLM moment finder → content verify).
    No LLM search expansion, no thumbnail pre-scoring, no separate visual
    moment path — the platform's ranking + the existing analysis pipeline.
    """
    # --- Phase 1: Build pool from YouTube's top-10 + channel-pinned ---------
    # YouTube's own ranking for the verbatim query is the best signal.
    # No re-ranking, no LLM expansion, no thumbnail pre-filtering.
    pool, rejected_shorts = _search_pool_only(sentence, plan)
    channel_pool = _search_channel_pinned(sentence, plan)
    exact_ids = {c["id"] for c in pool}
    pool = pool + [c for c in channel_pool if c["id"] not in exact_ids]

    # Event registry: curated event broadcasts/POVs (data/event_videos.json)
    registry = event_registry.match(plan)
    if registry:
        registry_pool = event_registry.candidates(registry, plan)
        existing_ids = {c["id"] for c in pool}
        for c in registry_pool:
            if c["id"] not in existing_ids:
                pool.append(c)

    # Discovery lane: long YouTube hits found via open-web search
    for c in (extra_pool or []):
        if c["id"] not in {p["id"] for p in pool}:
            pool.append(c)

    if not pool:
        logger.info("YouTube lane: nothing found for '%s'", sentence)
        return []

    logger.info("YouTube lane pool: %d videos (top-10 + channel-pinned + registry)",
                len(pool))

    # --- Phase 2: Fetch metadata for all pool candidates --------------------
    pool = _fetch_metadata_pool(pool)
    if not pool:
        logger.info("YouTube lane: no metadata resolved for '%s'", sentence)
        return []

    # --- Phase 3: Analyse each video fully ----------------------------------
    # Every video gets the full analysis: transcript windows + LLM moment
    # finder + coarse visual scan + fine frame sampling + content verify.
    # No thumbnail pre-filtering — the analysis pipeline handles everything.
    candidates: list[dict] = []
    for index, meta in enumerate(pool):
        meta["llm_scan"] = index < config.MOMENT_FINDER_TOP_VIDEOS
        try:
            candidates.extend(_analyse_video(meta, sentence, plan, anchors))
        except Exception as exc:
            logger.warning("Error analysing %s: %s", meta.get("webpage_url"), exc)

    candidates.sort(key=lambda c: (
        bool(c.get("vlm_happening")),
        float(c.get("vlm_score") or 0.0),
        (c.get("proposal_score") if "proposal_score" in c
         else c.get("semantic_score", 0.0)),
    ), reverse=True)

    if anchors is not None and getattr(anchors, "strong", False):
        for candidate in candidates:
            candidate["visual_event"] = True

    logger.info("YouTube lane for '%s' returned %d verified candidate(s)",
                sentence, len(candidates))
    return candidates
