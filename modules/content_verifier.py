"""Content-level verification: does a clip's ACTUAL audio match the sentence?

Twitch clips are ranked by title+views in the old pipeline without ever
checking what they say — that is why unrelated clips (Walking Dead, etc.)
slipped through.  Every candidate here must be verified against its own
transcript before it may be exported.
"""
import logging
import re

import config
from modules import llm_reranker, semantic_embedder, streamers
from modules.text_match import score as lexical_score
from modules.text_match import stem

logger = logging.getLogger(__name__)

_STRIP = re.compile(r"[^a-z0-9']")


def normalize(text: str) -> str:
    return _STRIP.sub(" ", (text or "").lower())


def term_variants(plan: dict) -> list[str]:
    """Expand required terms with streamer aliases so 'N3on' matches 'neon'."""
    variants: list[str] = []
    streamer = plan.get("streamer") if isinstance(plan, dict) else None
    if streamer:
        name = (streamer.get("name") or "")
        handle = (streamer.get("twitch_handle") or "")
        if name:
            variants.append(name)
        if handle and handle.lower() != name.lower():
            variants.append(handle)
    terms = plan.get("required_terms") or plan.get("core_entities") or []
    for term in terms:
        variants.append(term)
        resolved = streamers.lookup(term) if term and len(term) > 2 else None
        if resolved:
            variants.append(resolved.get("display_name", "") or "")
            for alias in resolved.get("aliases", []):
                variants.append(alias)
    return list(dict.fromkeys(v.strip() for v in variants if v.strip()))


def _topic_required(plan: dict) -> list[str]:
    """Entity terms that must be audible. Narration/context only require the
    streamer's name (the topic phrase is rarely spoken verbatim). Media queries
    always require the people/topics (the subject words ARE the content).
    Visual-event queries (police car / helicopter / gel blaster / fight) never
    require a spoken name: the event is confirmed by pixels, and on-screen
    footage of the moment rarely has anyone say the streamer's name."""
    if plan.get("media_query"):
        return term_variants(plan)
    if plan.get("visual_event"):
        return []
    streamer = plan.get("streamer") if isinstance(plan, dict) else None
    sentence_type = (plan.get("sentence_type") or "narration").lower()
    if sentence_type in ("narration", "context"):
        base: list[str] = []
        if streamer:
            name = streamer.get("name") or ""
            handle = streamer.get("twitch_handle") or ""
            for value in (name, handle):
                if value and value.lower() not in base:
                    base.append(value)
        return [v for v in base if v]
    return term_variants(plan)


def transcript_text(segments: list[dict]) -> str:
    return " ".join((seg.get("text") or "").strip() for seg in (segments or []))


def _window_text(transcript: list[dict], start: float, duration: float) -> str:
    end = start + duration
    return " ".join((seg.get("text") or "").strip()
                    for seg in (transcript or [])
                    if seg.get("end", 0) >= start and seg.get("start", 0) <= end)


def _expanded_action_groups(plan: dict) -> list[set[str]]:
    """Per action term: {term, stem, synonyms (stemmed too)} used for matching.

    GENERIC_ACTION_TERMS (right/answer/money/world...) are excluded from the
    SYNONYM sets so filler chatter never counts as the event happening.  The
    term itself stays — a query word literally spoken is real signal.

    Visual-event object terms (police, bus, helicopter...) are EXCLUDED entirely
    from the audio signal: a spoken prop name is not evidence the event happened,
    the exported clip is visually verified against the event's anchors instead.
    """
    groups: list[set[str]] = []
    generic = set(config.GENERIC_ACTION_TERMS)
    objects = set()
    for obj in (plan.get("object_terms") or []):
        objects.add(stem(str(obj).lower()))
        objects.update(stem(word) for word in str(obj).lower().split())
    for term in plan.get("action_terms") or []:
        base = stem(term.lower())
        if base in objects:
            continue
        vals = {term.lower(), base}
        for key in (base, term.lower()):
            for syn in config.ACTION_SYNONYMS.get(key, ()) or ():
                syn = syn.lower()
                if syn in generic or stem(syn) in generic:
                    continue
                vals.add(syn)
                vals.add(stem(syn))
        groups.append(vals)
    return groups


def action_match(text: str, plan: dict) -> tuple[int, float]:
    """(distinct action terms hit, coverage 0..1) in normalized text.

    Text is matched both raw and stem-normalized so a term like 'donate' (stem
    'donat') hits transcript 'donated' — a word-boundary regex against raw text
    alone would never match inflected forms.
    """
    groups = _expanded_action_groups(plan) if isinstance(plan, dict) else []
    if not groups:
        return 0, 0.0
    normalized = normalize(text)
    stemmed = " ".join(stem(word) for word in normalized.split())
    hits = 0
    for vals in groups:
        if (any(re.search(r"\b" + re.escape(v) + r"\b", normalized) for v in vals)
                or any(re.search(r"\b" + re.escape(v) + r"\b", stemmed) for v in vals)):
            hits += 1
    return hits, hits / len(groups)


def entity_match(text: str, plan: dict) -> tuple[int, float]:
    """(variants hit, coverage vs distinct core entities) in normalized text."""
    if not isinstance(plan, dict):
        return 0, 0.0
    variants = term_variants(plan)
    if not variants:
        return 0, 0.0
    entities = [e for e in (plan.get("core_entities") or []) if e]
    normalized = normalize(text)
    hits = sum(1 for v in variants if re.search(r"\b" + re.escape(v) + r"\b", normalized))
    return hits, min(1.0, hits / max(1, len(entities)))


def object_match(text: str, plan: dict) -> tuple[int, float]:
    """(distinct visual-event object/prop terms hit, coverage 0..1).

    Only counts for visual-event plans (plan['visual_event']); regular queries
    return (0, 0.0) so nothing downstream shifts for them."""
    if not isinstance(plan, dict) or not plan.get("visual_event"):
        return 0, 0.0
    objects = [str(o).lower() for o in (plan.get("object_terms") or [])]
    if not objects:
        return 0, 0.0
    normalized = normalize(text)
    hits = 0
    for term in objects:
        if re.search(r"\b" + re.escape(term) + r"\b", normalized):
            hits += 1
    return hits, min(1.0, hits / len(objects))


def _token_coverage(text: str, plan: dict) -> float:
    """Fraction of core entities whose ANY name token appears in the text.

    Token-tolerant on purpose: raw VOD audio almost never speaks full multi-word
    names ("Kai", not "Kai Cenat"), so any single name token counts as that
    entity being heard.
    """
    entities = [e for e in (plan.get("core_entities") or []) if e]
    if not entities or not text:
        return 0.0
    tokens = {t for t in normalize(text).split() if t.isalnum() and len(t) > 2}
    hit = 0
    for entity in entities:
        etoks = {t for t in normalize(entity).split() if t.isalnum() and len(t) > 2}
        if etoks and etoks & tokens:
            hit += 1
    return min(1.0, hit / len(entities))


def transcript_verified(clip: dict, plan: dict) -> tuple[bool, str]:
    """Transcript-trust: the caption LITERALLY speaks the moment.

    True when the clip is keyword-located, its audio heard the action and a
    core entity's name, and it comes from an official/raw channel.  This is the
    safety net for moments the vision stack keeps false-rejecting (a real moment
    whose frames don't clearly show it); it never gates anything, it only ADDS
    exports.  Never mutates the clip.
    """
    if not isinstance(plan, dict) or not isinstance(clip, dict):
        return False, ""
    if not config.TRANSCRIPT_TRUST_ENABLED:
        return False, ""
    if config.TRANSCRIPT_TRUST_REQUIRE_KEYWORD and not clip.get("keyword_located"):
        return False, "not keyword-located"
    if int(clip.get("action_hits") or 0) < config.TRANSCRIPT_TRUST_MIN_ACTION_HITS:
        return False, "no action heard"
    text = (clip.get("snippet") or clip.get("evidence") or clip.get("transcript") or "")
    # The window audio alone rarely speaks full names ("You put a hole in my
    # wall" — nobody says "Kai Cenat").  The video's OWN title/description naming
    # the subject is legitimate association: a VOD titled "Kai Cenat & Blueface
    # FULL STREAM!" whose audio literally speaks the keyword + action IS the
    # moment.  The keyword+action+channel gates already anchor the exact window.
    meta = " ".join(str(v) for v in
                    (clip.get("title"), clip.get("description"), clip.get("video_title"))
                    if isinstance(v, str) and v.strip())
    coverage = _token_coverage(" ".join((text, meta)), plan)
    if coverage < config.TRANSCRIPT_TRUST_MIN_ENTITY_COVERAGE:
        return False, "entity not heard (coverage=%.2f)" % coverage
    streamer = plan.get("streamer")
    channel = clip.get("channel") or clip.get("uploader") or ""
    if streamers.is_official_channel(streamer, channel):
        kind = "official"
    elif streamers.is_raw_footage_channel(channel, streamer, title_or_desc=meta):
        kind = "raw"
    else:
        kind = "third-party"
    if kind not in config.TRANSCRIPT_TRUST_ACCEPT_CHANNELS:
        return False, "channel is %s" % kind
    return True, ("transcript-verified: keyword + %d action(s) + %.0f%% entities"
                  % (int(clip.get("action_hits") or 0), coverage * 100))


def verify(candidate: dict, sentence: str, plan: dict,
           transcript: list[dict] | None) -> tuple[bool, str]:
    """Attach semantic/lexical/rerank/entity scores; return (verified, reason)."""
    if not transcript:
        return False, "no transcript available"
    start = float(candidate.get("start", 0.0))
    duration = float(candidate.get("duration", config.CLIP_DURATION_SECONDS))
    # Pixels can carry a silent visual moment (an archery hit, a leg grab, a
    # police car passing) when nobody speaks the event.  A candidate whose
    # sampled frames already cleared the general scene probe is accepted here;
    # the export-level pixel gate re-confirms it on the downloaded clip.
    pixel_ok = (
        float(candidate.get("general_visual_best") or 0.0) >= config.GENERAL_VISUAL_FLOOR
        or float(candidate.get("event_visual_best") or 0.0) >= config.EVENT_VISUAL_FLOOR
        or float(candidate.get("event_visual_score") or 0.0) >= 1.0
    )
    if candidate.get("source") == "twitch" or candidate.get("whole", False):
        text = transcript_text(transcript)
    else:
        text = _window_text(transcript, start, duration)
    text = text.strip()
    # Visual-event candidates are verified by the export-level pixel gate
    # (EventProbe / anchor scenes), never by audio chatter: the object can be
    # on screen while nobody says a word (a passing police car, a helicopter
    # landing).  Accept the candidate here so the visual gate decides — but drop
    # audio that carries NO signal at all: a candidate whose window text shares
    # nothing topical (no object, no action, no entity, no semantic pull) is
    # just filler and floods the judge/export pool with wrong-event lookalikes.
    if plan.get("visual_event"):
        semantic = semantic_embedder.similarity_scores(sentence, [text])
        semantic_score = float(semantic[0]) if semantic else 0.0
        action_hits, action_cov = action_match(text, plan)
        entity_hits, entity_cov = entity_match(text, plan)
        object_hits, object_cov = object_match(text, plan)
        candidate.update({
            "semantic_score": semantic_score,
            "action_hits": action_hits,
            "action_coverage": action_cov,
            "entity_hits": entity_hits,
            "entity_coverage": entity_cov,
            "object_hits": object_hits,
            "object_coverage": object_cov,
            "snippet": text[:1200],
            "verified": True,
        })
        has_signal = (action_hits > 0 or entity_hits > 0 or object_hits > 0
                      or bool(candidate.get("locator_hits"))
                      or semantic_score >= config.RELEVANCE_GATE_SEMANTIC
                      or pixel_ok)
        if not has_signal:
            candidate["verified"] = False
            return False, (f"visual-event window with no topical audio signal "
                           f"(sem={semantic_score:.2f})")
        return True, ("verified (visual-event: audio is corroboration only, "
                      f"sem={semantic_score:.2f})")
    if len(text) < 8:
        return False, "transcript too short"

    semantic = semantic_embedder.similarity_scores(sentence, [text])
    semantic_score = float(semantic[0]) if semantic else 0.0

    sentence_type = (plan.get("sentence_type") or "narration").lower()
    media = bool(plan.get("media_query"))
    semantic_min = (config.VERIFY_SEMANTIC_MIN_NARRATION
                    if media or sentence_type in ("narration", "context")
                    else config.VERIFY_SEMANTIC_MIN)

    rerank_score = 0.0
    try:
        rerank = llm_reranker._reranker_predict([(sentence, text)])
        rerank_score = float(rerank[0]) if rerank else 0.0
    except Exception:
        rerank_score = 0.0

    normalized = normalize(text)
    required = _topic_required(plan)
    entity_hits = sum(1 for term in required if normalize(term) in normalized)
    # Hearing the required entities is a score BOOST, never a hard gate: a clip
    # of the actual moment (e.g. Ronaldo handing boots to Speed) is on-screen
    # footage where both names are rarely spoken, so audio must not reject it.
    entity_boost = min(0.10, 0.04 * entity_hits)
    entity_hits_raw, entity_coverage = entity_match(text, plan)

    action_hits, action_coverage = action_match(text, plan)
    action_boost = 0.0
    if sentence_type == "event":
        action_boost = min(0.12, 0.06 * action_hits)
    object_hits, object_coverage = object_match(text, plan)
    object_boost = 0.0
    if plan.get("visual_event"):
        # Object/prop terms (police, bus, chopper) are the visual event's only
        # audio trace; count them as corroborating signal and a small semantic
        # lift, since the export-level visual gate confirms the pixels anyway.
        object_boost = min(0.06, 0.02 * object_hits)
    effective_semantic = semantic_score + entity_boost + action_boost + object_boost

    lexical = lexical_score(sentence, text)

    candidate.update({
        "semantic_score": semantic_score,
        "effective_semantic": effective_semantic,
        "entity_boost": entity_boost,
        "action_boost": action_boost,
        "action_hits": action_hits,
        "action_coverage": action_coverage,
        "object_hits": object_hits,
        "object_coverage": object_coverage,
        "entity_hits": entity_hits,
        "entity_coverage": entity_coverage,
        "lexical_score": lexical,
        "rerank_score": rerank_score,
        "snippet": text[:1200],
        "verified": True,
    })

    is_livestream = candidate.get("source") in ("twitch", "kick")
    # An LLM-grounded moment lowers the semantic floor (the finder read the full
    # transcript), but never removes it entirely, and an event moment must carry
    # corroborating audio signal (an action or an entity actually spoken).  A
    # hallucinated 'llm_found' on unrelated chatter must not verify.
    llm_grounded = (bool(candidate.get("llm_found"))
                    and float(candidate.get("llm_moment_score", 0.0)) >= config.LLM_MOMENT_MIN_CONFIDENCE)
    # A keyword-located moment (the captions LITERALLY speak the lexicon phrase)
    # is the strongest audio signal there is — as strong as LLM grounding.
    keyword_grounded = bool(candidate.get("keyword_located")) and bool(candidate.get("locator_hits"))
    grounded = llm_grounded or keyword_grounded
    has_signal = (entity_hits > 0 or action_hits >= config.ACTION_MIN_HITS
                  or object_hits > 0 or bool(candidate.get("locator_hits")))
    if is_livestream:
        # Conversational clip/window audio rarely clears the MiniLM bar
        # (0.03-0.22); accept when the entities or the action are actually
        # spoken, exactness is left to the LLM judge.  Only garbage without any
        # signal is dropped.
        if effective_semantic < 0.10 and not has_signal and not grounded and not pixel_ok:
            candidate["verified"] = False
            return False, f"low semantic match {effective_semantic:.2f}"
        if grounded and not has_signal and effective_semantic < 0.15 and not pixel_ok:
            candidate["verified"] = False
            return False, "grounded livestream window has no audio signal"
    else:
        if effective_semantic < semantic_min and not grounded and not pixel_ok:
            candidate["verified"] = False
            return False, f"low semantic match {effective_semantic:.2f}"
        if grounded and effective_semantic < semantic_min and not has_signal and not pixel_ok:
            candidate["verified"] = False
            return False, "grounded moment has no audio signal"
    # ms-marco rerank is informational only: it scores real conversational
    # audio ~0.00 even when on-topic, so it must never gate verification.
    if (config.VERIFY_REQUIRE_TOPIC and required
            and entity_hits < config.VERIFY_MIN_ENTITY_HITS and not grounded
            and not pixel_ok):
        candidate["verified"] = False
        return False, f"no required entity in audio (hits={entity_hits})"
    # --- action/object gate for event sentences ---
    # For event sentences, entity presence alone is never sufficient — the clip
    # must show the described action or have visual proof.  High combined_score
    # from vlm/vision weights does NOT compensate for missing action/object evidence.
    if sentence_type == "event" and not pixel_ok:
        has_action_or_object = (
            action_hits >= config.RELEVANCE_GATE_ACTION_OR_OBJECT
            or object_hits >= config.RELEVANCE_GATE_ACTION_OR_OBJECT
            or bool(candidate.get("event_visual_verified"))
            or bool(candidate.get("vlm_happening"))
        )
        if not has_action_or_object:
            # Visual verification hasn't run yet — flag for confirmation
            # instead of hard-rejecting.  The export-level pixel gate will decide.
            if not grounded:
                candidate["needs_visual_confirmation"] = True
            else:
                candidate["verified"] = False
                return False, (f"event sentence: no action/object/visual evidence "
                               f"(action={action_hits}, object={object_hits})")
    grounding = " llm_moment=%.2f" % float(candidate.get("llm_moment_score", 0.0)) if grounded else ""
    if keyword_grounded:
        grounding = " keyword=%s" % ",".join(candidate.get("locator_hits") or [])
    pixel_note = (" pixel=%.2f" % float(candidate.get("general_visual_best") or 0.0)
                  if pixel_ok else "")
    return True, (f"verified (sem={effective_semantic:.2f} "
                  f"rerank={rerank_score:.2f} entities={entity_hits} "
                  f"action={action_hits}{grounding}{pixel_note})")
