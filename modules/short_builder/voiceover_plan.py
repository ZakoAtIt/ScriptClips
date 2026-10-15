"""Voiceover plan: split the MP3 into beats with exact timestamps.

Uses faster-whisper (already loaded by audio_transcriber) with word-level
timestamps.  Words are grouped into near-fixed-length beats of ~
SHORT_BEAT_TARGET_SECONDS of speech (uniform cuts every ~2s), instead of the
uneven sentence fragments whisper produces.  Every beat also gets a short
CLIP-friendly label (LLM condensed, raw text as fallback) derived from the
words actually spoken during that beat.  The plan is cached per audio file
(hashed) and versioned.
"""
import hashlib
import json
import logging
import re
from pathlib import Path

import config
from modules import audio_transcriber, llm_client

logger = logging.getLogger(__name__)

_SENTENCE_ENDINGS = (".", "!", "?", "…")


def _transcribe_words(mp3_path: Path) -> list[dict]:
    try:
        model = audio_transcriber._get_model(config.WHISPER_MODEL_SIZE)
    except Exception as exc:
        logger.warning("Could not load Whisper model: %s", exc)
        return []
    segments, _info = model.transcribe(
        str(mp3_path), beam_size=1, vad_filter=True, word_timestamps=True)
    words: list[dict] = []
    for segment in segments:
        segment_words = getattr(segment, "words", None)
        if segment_words:
            for word in segment_words:
                text = (word.word or "").strip()
                if text:
                    words.append({"text": text, "start": float(word.start),
                                  "end": float(word.end)})
        else:
            text = (segment.text or "").strip()
            if text:
                words.append({"text": text, "start": float(segment.start),
                              "end": float(segment.end)})
    return words


def _split_sentences(words: list[dict]) -> list[list[dict]]:
    sentences: list[list[dict]] = []
    buffer: list[dict] = []
    for i, word in enumerate(words):
        buffer.append(word)
        next_word = words[i + 1] if i + 1 < len(words) else None
        gap = (next_word["start"] - word["end"]) if next_word else 0.0
        duration = word["end"] - buffer[0]["start"]
        is_ending = any(word["text"].endswith(p) for p in _SENTENCE_ENDINGS)
        forced_break = duration >= config.SHORT_SENTENCE_MAX_SECONDS
        if is_ending or gap > config.SHORT_SENTENCE_PAUSE or forced_break:
            sentences.append(buffer)
            buffer = []
    if buffer:
        sentences.append(buffer)
    return [s for s in sentences if s]


def _split_bundle(bundle: list[dict], max_seconds: float) -> list[list[dict]]:
    """Split a word bundle into sub-bundles each <= max_seconds of speech."""
    duration = bundle[-1]["end"] - bundle[0]["start"]
    if duration <= max_seconds:
        return [bundle]
    best_gap, best_index = 0.0, -1
    for i in range(len(bundle) - 1):
        gap = bundle[i + 1]["start"] - bundle[i]["end"]
        if gap > best_gap:
            best_gap, best_index = gap, i
    if best_index >= 0 and best_gap >= 0.25:
        return (_split_bundle(bundle[:best_index + 1], max_seconds)
                + _split_bundle(bundle[best_index + 1:], max_seconds))
    if len(bundle) >= 2:
        mid = len(bundle) // 2
        return (_split_bundle(bundle[:mid], max_seconds)
                + _split_bundle(bundle[mid:], max_seconds))
    token = bundle[0]
    pieces = token["text"].split()
    if len(pieces) >= 2:
        per = duration / len(pieces)
        step = max(1, int(round(max_seconds / per)))
        parts: list[list[dict]] = []
        for i in range(0, len(pieces), step):
            chunk = pieces[i:i + step]
            start = token["start"] + i * per
            end = token["start"] + min(len(pieces), i + step) * per
            parts.append([{"text": " ".join(chunk),
                           "start": start, "end": end}])
        return parts
    return [bundle]


def _segments_by_target(words: list[dict], target: float) -> list[list[dict]]:
    """Group words into near-fixed-length beats (~target seconds of speech each).

    Whisper sentence/pause grouping produces uneven 0.8-2.5s fragments ("First,",
    "fans."), so beats are re-chunked by wall-clock target for uniform cuts.
    """
    segments: list[list[dict]] = []
    cur: list[dict] = []
    for word in words:
        if not cur:
            cur = [word]
            continue
        cur.append(word)
        if word["end"] - cur[0]["start"] >= target:
            segments.append(cur)
            cur = []
    if cur:
        segments.append(cur)
    return segments


def _merge_short_beats(beats: list[dict], audio_duration: float) -> list[dict]:
    """Fold beats rendering under ~1.4s into a neighbour so every cut stays ~2s."""
    minimum = config.SHORT_BEAT_MIN_SECONDS * 0.7
    maximum = config.SHORT_BEAT_TARGET_SECONDS + 1.0
    merged: list[dict] = []
    for i, beat in enumerate(beats):
        render = beat["render_duration"]
        if render >= minimum:
            merged.append(beat)
            continue
        nxt = beats[i + 1] if i + 1 < len(beats) else None
        if nxt is not None and render + nxt["render_duration"] <= maximum:
            nxt["start"] = beat["start"]
            nxt["words"] = beat.get("words", []) + nxt.get("words", [])
            nxt["text"] = " ".join(w["text"] for w in nxt["words"]).strip()
            nxt["label"] = nxt["text"]
            continue
        if merged and merged[-1]["render_duration"] + render <= maximum:
            prev = merged[-1]
            prev["words"] = prev.get("words", []) + beat.get("words", [])
            prev["text"] = " ".join(w["text"] for w in prev["words"]).strip()
            prev["label"] = prev["text"]
            prev["end"] = beat["end"]
            continue
        merged.append(beat)
    return merged


def _beat_from_words(bundle: list[dict], index: int, scene_index: int) -> dict:
    text = " ".join(w["text"] for w in bundle).strip()
    return {
        "index": index,
        "scene_index": scene_index,
        "text": text,
        "label": text,
        "start": float(bundle[0]["start"]),
        "end": float(bundle[-1]["end"]),
        "words": bundle,
    }


def _render_durations(beats: list[dict], audio_duration: float) -> None:
    for i, beat in enumerate(beats):
        next_start = beats[i + 1]["start"] if i + 1 < len(beats) else audio_duration
        beat["duration"] = round(beat["end"] - beat["start"], 3)
        render = next_start - beat["start"]
        beat["render_duration"] = round(max(0.05, render), 3)


def _enforce_render_cap(beats: list[dict], audio_duration: float) -> None:
    """Split any beat whose on-screen render exceeds the cap at its largest pause."""
    cap = max(config.SHORT_BEAT_MAX_SECONDS,
              config.SHORT_BEAT_TARGET_SECONDS + 0.7)
    guard = 0
    while guard < 20:
        guard += 1
        rebuilt: list[dict] = []
        changed = False
        for beat in beats:
            bundle = beat.get("words") or []
            if beat["render_duration"] > cap and len(bundle) >= 2:
                best_gap, best_index = 0.0, -1
                for i in range(len(bundle) - 1):
                    gap = bundle[i + 1]["start"] - bundle[i]["end"]
                    if gap > best_gap:
                        best_gap, best_index = gap, i
                if best_index < 0:
                    best_index = len(bundle) // 2 - 1
                if best_index >= 0:
                    rebuilt.append(_beat_from_words(
                        bundle[:best_index + 1], 0, beat["scene_index"]))
                    rebuilt.append(_beat_from_words(
                        bundle[best_index + 1:], 0, beat["scene_index"]))
                    changed = True
                    continue
            rebuilt.append(beat)
        if not changed:
            break
        beats[:] = rebuilt
        _render_durations(beats, audio_duration)


def _merge_short(beats: list[dict]) -> list[dict]:
    """Merge beats rendering under the min into a same-sentence neighbour within the cap."""
    merged: list[dict] = []
    maximum = config.SHORT_BEAT_MAX_SECONDS
    for i, beat in enumerate(beats):
        render = beat["render_duration"]
        if render >= config.SHORT_BEAT_MIN_SECONDS:
            merged.append(dict(beat))
            continue
        if i + 1 >= len(beats):
            merged.append(dict(beat))
            continue
        next_beat = beats[i + 1]
        if merged:
            prev = merged[-1]
            if (prev["scene_index"] == beat["scene_index"]
                    and (prev["render_duration"] + render) <= maximum):
                prev["end"] = beat["end"]
                prev["text"] = f"{prev['text']} {beat['text']}".strip()
                prev["label"] = prev["text"]
                prev["render_duration"] = round(prev["render_duration"] + render, 3)
                continue
        if (next_beat["scene_index"] == beat["scene_index"]
                and (next_beat["render_duration"] + render) <= maximum):
            next_beat["start"] = beat["start"]
            next_beat["text"] = f"{beat['text']} {next_beat['text']}".strip()
            next_beat["label"] = next_beat["text"]
            next_beat["render_duration"] = round(next_beat["render_duration"] + render, 3)
            continue
        merged.append(dict(beat))
    return merged


def _llm_labels(beats: list[dict]) -> None:
    if not config.SHORT_LLM_LABELS:
        return
    body = "\n".join(f"{b['index']}: {b['text']}" for b in beats)
    prompt = (
        "Below are the spoken lines of a short video, in order.  For each one "
        "return a concise 2-7 word VISUAL caption describing what should be on "
        "screen, as a JSON object:\n"
        '{"labels":[{"index":0,"label":"..."}]}\n'
        f"Lines:\n{body}")
    payload = llm_client.chat_json(prompt, system=(
        "You write short on-screen visual captions. Reply with JSON only."))
    labels = payload.get("labels")
    if not isinstance(labels, list):
        logger.info("LLM label cleanup produced nothing; using raw beat text")
        return
    by_index = {int(item.get("index", -1)): str(item.get("label", "")).strip()
                for item in labels if isinstance(item, dict)}
    if not by_index:
        return

    def tokens(text: str) -> set[str]:
        return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 1}

    for beat in beats:
        text_tokens = tokens(beat["text"])
        label = by_index.get(beat["index"])
        if (label and 2 <= len(label.split()) <= 7
                and tokens(label) & text_tokens):
            beat["label"] = label
            continue
        words = beat["text"].split()
        if len(words) > 7:
            beat["label"] = " ".join(words[:7])
    logger.info("LLM condensed %d/%d beat label(s)", len(by_index), len(beats))


def _plan_cache_path(mp3_path: Path) -> Path:
    """Per-audio cache key so different voiceovers never share a beat plan."""
    try:
        stat = mp3_path.stat()
        seed = f"{mp3_path}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        seed = f"{mp3_path}"
    digest = hashlib.sha1(seed.encode("utf-8", "replace")).hexdigest()[:12]
    return config.SHORT_CACHE_DIR / f"scenes_{digest}.json"


def build(mp3_path: Path, force: bool = False) -> list[dict]:
    """Return beats [{index, scene_index, text, label, start, end, duration, render_duration}]."""
    cache = _plan_cache_path(mp3_path)
    if cache.exists() and not force:
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if (isinstance(data, dict)
                    and data.get("version") == config.SHORT_PLAN_VERSION
                    and isinstance(data.get("beats"), list) and data["beats"]):
                return data["beats"]
        except Exception:
            pass

    words = _transcribe_words(mp3_path)
    if not words:
        logger.warning("Voiceover transcription produced no words")
        return []

    target = config.SHORT_BEAT_TARGET_SECONDS
    words = [sub[0] for w in words for sub in _split_bundle([w], target)]
    scene_of: dict[int, int] = {}
    for sidx, sentence in enumerate(_split_sentences(words)):
        for word in sentence:
            scene_of[id(word)] = sidx
    segments = _segments_by_target(words, target)
    beats = [
        _beat_from_words(segment, i, scene_of.get(id(segment[0]), 0))
        for i, segment in enumerate(segments)
    ]

    audio_duration = _probe_duration(mp3_path) or 0.0
    if audio_duration <= 0:
        audio_duration = beats[-1]["end"] + 1.0

    _render_durations(beats, audio_duration)
    _enforce_render_cap(beats, audio_duration)
    beats = _merge_short_beats(beats, audio_duration)
    for i, beat in enumerate(beats):
        beat["index"] = i
        beat.pop("words", None)
    _render_durations(beats, audio_duration)

    _llm_labels(beats)
    cache.write_text(
        json.dumps({"version": config.SHORT_PLAN_VERSION, "beats": beats},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    total = round(sum(b["render_duration"] for b in beats), 2)
    logger.info("Voiceover plan: %d beat(s), total %.1fs (audio %.1fs)",
                len(beats), total, audio_duration)
    return beats


def _probe_duration(path: Path) -> float | None:
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True)
        out = result.stdout.strip()
        return float(out) if out else None
    except Exception:
        return None
