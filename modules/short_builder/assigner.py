"""Assigner: map each voiceover scene to its best qualifying window (or image).

For every scene in voiceover order the highest-scoring candidate window whose
length bucket covers the scene duration and whose segment does not overlap an
already-used segment of the same clip is chosen (a clip can contribute several
windows — each distinct moment counts).  When CLIP refinement is enabled the
per-scene shortlist is re-ranked first.  If nothing qualifies, an unused image
(fallback-only, per user choice) is picked.  A reviewable assignment report is
written before any rendering happens.
"""
import json
import logging
import re
import subprocess

import numpy as np

import config
from modules.short_builder.semantic_refine import (_safe_stem, folder_person,
                                                   folders_for_person)

logger = logging.getLogger(__name__)

_STIMULUS_MARKERS = ("montage", "swinging", "action")


def _overlaps(used: list[tuple[float, float]], start: float, end: float) -> bool:
    return any(lo < end - 0.3 and start < hi - 0.3 for lo, hi in used)


def _auto_min_score(candidates: list[dict], explicit: float | None) -> float:
    if explicit is not None and explicit > 0:
        return explicit
    if not candidates:
        return config.SHORT_MIN_SCORE_FLOOR
    scores = np.asarray([c["score"] for c in candidates], dtype=np.float64)
    return max(float(np.percentile(scores, config.SHORT_MIN_SCORE_PERCENTILE)),
               config.SHORT_MIN_SCORE_FLOOR)


def _pick_window(candidates: list[dict], family: set[str] | None, scene: dict,
                 used: dict[str, list[tuple[float, float]]], semantic: bool,
                 min_score: float, floor: float,
                 prefer_stimulating: bool = False) -> dict | None:
    """Best window inside `family` (or the whole pool), relaxing constraints in
    stages so a beat never silently leaks into another topic's footage.

    Identity/topic comes first: the family is fixed by the folder matcher (and
    person pinning), so later stages only relax score/overlap/length — never
    the family.
    """
    render = float(scene["render_duration"])
    needed = min(render, config.SHORT_MAX_WINDOW)
    in_family = [c for c in candidates if family is None or c.get("folder") in family]
    if prefer_stimulating:
        stim = [c for c in in_family
                if any(m in _safe_stem(c.get("folder", "")) for m in _STIMULUS_MARKERS)]
        if stim:
            in_family = stim

    def usable(pool: list[dict], score_min: float,
               allow_overlap: bool, allow_short: bool) -> list[dict]:
        out = []
        for c in pool:
            if c["score"] < score_min:
                continue
            if not allow_short:
                if c["length"] < needed:
                    continue
                if c["start"] + render > float(c["clip_duration"]) - 0.1:
                    continue
            if not allow_overlap and _overlaps(
                    used.get(c["clip"], []), c["start"], c["start"] + render):
                continue
            out.append(c)
        return out

    stages = (
        (min_score, False, False),
        (floor, False, False),
        (floor, True, False),
        (0.0, True, True),
    )
    for score_min, allow_overlap, allow_short in stages:
        pool = usable(in_family, score_min, allow_overlap, allow_short)
        if not pool:
            continue
        if semantic and len(pool) > 1:
            from modules.short_builder import semantic_refine
            pool = semantic_refine.rerank(scene.get("label") or scene["text"], pool)
        return pool[0]
    return None


def assign(scenes: list[dict], candidates: list[dict], images: list[dict],
           semantic: bool = True, min_score: float | None = None,
           folders: list[str] | None = None) -> list[dict]:
    threshold = _auto_min_score(candidates, min_score)
    logger.info("Window score threshold: %.3f (%d candidate window(s) considered)",
                threshold, len(candidates))

    from modules.short_builder import semantic_refine

    persons: dict[str, tuple[str, ...]] = {}
    folder_map: dict[int, str | None] = {}
    contexts: dict[int, dict] = {}
    person_folders: dict[str, set[str]] = {}
    if config.SHORT_FOLDER_MATCH and folders:
        persons = semantic_refine.detect_persons(folders)
        contexts = semantic_refine.beat_contexts(folders, scenes)
        folder_map = {i: ctx["folder"] for i, ctx in contexts.items()}
        person_folders = {
            person: {_safe_stem(f)
                     for f in folders_for_person(folders, person, persons)}
            for person in persons
        }
        matched = sum(1 for f in folder_map.values() if f)
        logger.info("Folder matching: %d/%d beat(s) tied to a topic folder; "
                    "persons=%s", matched, len(scenes), ", ".join(persons))

    used_segments: dict[str, list[tuple[float, float]]] = {}
    used_images: set[str] = set()
    assignments: list[dict] = []
    active_person: str | None = None
    carry = 0

    for scene in scenes:
        render = float(scene["render_duration"])
        idx = scene["index"]
        ctx = contexts.get(idx, {})
        scene_folder = folder_map.get(idx)
        topic_score = float(ctx.get("score", 0.0))

        person = ctx.get("person")
        if person:
            active_person, carry = person, 0
        elif active_person:
            folder_pid = folder_person(scene_folder or "", persons)
            strong_topic = bool(scene_folder) and topic_score >= config.SHORT_PERSON_CARRY_TOPIC_STRENGTH
            text_low = (scene.get("label") or scene["text"]).lower()
            pronoun = bool(re.search(
                r"\b(he|his|him|himself|she|her|hers|herself|their|theirs)\b",
                text_low))
            opens_story = any(m in text_low
                              for m in config.SHORT_PERSON_CARRY_TRANSITIONS)
            if strong_topic and not folder_pid and not pronoun:
                active_person = None
            elif opens_story and not folder_pid:
                active_person = None
            else:
                carry = 0 if pronoun else carry + 1
                if carry > config.SHORT_PERSON_CARRY_BEATS:
                    active_person = None
        person = active_person

        family: set[str] | None = None
        if person:
            family = person_folders.get(person)
            if not family:
                logger.warning("Scene %d: person '%s' has no footage folder; "
                               "falling back to neutral footage", idx, person)
        elif scene_folder:
            family = {_safe_stem(scene_folder)}

        floor = (config.SHORT_PERSON_MIN_SCORE if person
                 else config.SHORT_FOLDER_MIN_SCORE)

        neutral = [c for c in candidates
                   if not folder_person(c.get("folder", ""), persons)]
        chosen = None
        image = None
        if family:
            chosen = _pick_window(candidates, family, scene, used_segments,
                                  semantic, threshold, floor)
            if chosen is None and scene_folder and images:
                image = semantic_refine.best_image(
                    images, scene.get("label") or scene["text"], used_images,
                    folder=scene_folder)
        if chosen is None and image is None:
            chosen = _pick_window(neutral, None, scene, used_segments,
                                  semantic, threshold, floor,
                                  prefer_stimulating=True)
        if chosen is None and image is None:
            chosen = _pick_window(candidates, None, scene, used_segments,
                                  semantic, threshold, floor,
                                  prefer_stimulating=True)
        if chosen is None and image is None:
            chosen = _pick_window(neutral, None, scene, used_segments,
                                  semantic, threshold, floor)
        if chosen is None and image is None:
            chosen = _pick_window(candidates, None, scene, used_segments,
                                  semantic, threshold, floor)

        if image is not None:
            used_images.add(image["path"])
            assignment = {
                "scene_index": scene["index"],
                "text": scene["text"],
                "label": scene.get("label") or scene["text"],
                "scene_start": scene["start"],
                "scene_duration": scene["duration"],
                "render_duration": render,
                "kind": "image",
                "source": image["path"],
                "start": 0.0,
                "end": render,
                "window_start": 0.0,
                "window_end": render,
                "length": render,
                "score": 0.0,
                "clip_sim": image.get("clip_sim"),
                "combined": None,
                "folder": image.get("folder", ""),
                "person": person or None,
            }
            assignments.append(assignment)
            continue
        elif chosen:
            chosen = dict(chosen)
            used_segments.setdefault(chosen["clip"], []).append(
                (chosen["start"], chosen["start"] + render))
            assignment = {
                "scene_index": scene["index"],
                "text": scene["text"],
                "label": scene.get("label") or scene["text"],
                "scene_start": scene["start"],
                "scene_duration": scene["duration"],
                "render_duration": render,
                "kind": "video",
                "source": chosen["clip"],
                "start": chosen["start"],
                "end": chosen["start"] + render,
                "window_start": chosen["start"],
                "window_end": chosen["end"],
                "length": chosen["length"],
                "score": chosen["score"],
                "clip_sim": chosen.get("clip_sim"),
                "combined": chosen.get("combined"),
                "folder": chosen.get("folder", ""),
                "person": person or None,
            }
        else:
            image = semantic_refine.best_image(
                images, scene.get("label") or scene["text"], used_images,
                folder=scene_folder)
            if image is None and images:
                image = dict(images[0])
            if image:
                used_images.add(image["path"])
                assignment = {
                    "scene_index": scene["index"],
                    "text": scene["text"],
                    "label": scene.get("label") or scene["text"],
                    "scene_start": scene["start"],
                    "scene_duration": scene["duration"],
                    "render_duration": render,
                    "kind": "image",
                    "source": image["path"],
                    "start": 0.0,
                    "end": render,
                    "window_start": 0.0,
                    "window_end": render,
                    "length": render,
                    "score": 0.0,
                    "clip_sim": image.get("clip_sim"),
                    "combined": None,
                    "folder": image.get("folder", ""),
                    "person": person or None,
                }
            else:
                logger.warning("Scene %d: no clip window and no image available",
                               scene["index"])
                continue
        assignments.append(assignment)

    video_count = sum(1 for a in assignments if a["kind"] == "video")
    image_count = sum(1 for a in assignments if a["kind"] == "image")
    logger.info("Assignment: %d scene(s) -> %d clip window(s), %d image fallback(s)",
                len(assignments), video_count, image_count)
    _write_report(scenes, assignments)
    return assignments


def _write_report(scenes: list[dict], assignments: list[dict]) -> None:
    try:
        config.SHORT_ASSIGNMENT_REPORT.write_text(
            json.dumps(assignments, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not write assignment JSON: %s", exc)
    lines = [
        "Beat | person | kind | dur | score | source | [window]",
        "-----|--------|------|-----|-------|--------|---------",
    ]
    for a in assignments:
        who = a.get("person") or "-"
        if a["kind"] == "video":
            detail = (f"{a['source']} [{a['start']:.1f}s-{a['end']:.1f}s] "
                      f"score={a['score']:.3f} sim={a['clip_sim'] if a['clip_sim'] is not None else '-'}")
        else:
            detail = f"IMAGE {a['source']}"
        lines.append(f"{a['scene_index']:>5} | {who:<6} | {a['kind']:<5} | "
                     f"{a['render_duration']:.1f}s | {a['score']:.3f} | {detail}")
    try:
        config.SHORT_ASSIGNMENT_REPORT_TXT.write_text(
            "\n".join(lines), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not write assignment report: %s", exc)
