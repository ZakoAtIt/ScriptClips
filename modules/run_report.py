"""Structured run report: JSON-lines confidence scores + missing matches.

Appended to ``config.RUN_REPORT_FILE`` (pipeline.log) after every stage so a
run can be audited without reading the full python log:

  * one line per discovery hit / final candidate carrying ALL scoring signals
  * a per-sentence summary line with the export quota, the exported count and
    an explicit ``missing_matches`` list naming WHICH gate rejected each
    un-exported candidate (llm judge / content verify / anchor gate / VLM /
    source policy), so a shortfall is never silent.
"""
import datetime
import json
import logging

import config

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _write(record: dict) -> None:
    try:
        record = {"ts": _now(), **record}
        with open(config.RUN_REPORT_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.warning("run_report write failed: %s", exc)


def log_stage(stage: str, sentence: str, **fields) -> None:
    """Generic stage marker (pipeline start, discovery done, export done...)."""
    _write({"event": "stage", "stage": stage, "sentence": sentence, **fields})


def log_discovery(sentence: str, clips: list[dict], yt_pool: list[dict],
                  rejected: list[dict]) -> None:
    """Discovery-lane outcome: kept whole-clips, YouTube pool adds, rejects."""
    for clip in clips:
        _write({"event": "discovery", "sentence": sentence,
                "kept": True,
                "source": clip.get("source"),
                "url": clip.get("webpage_url"),
                "title": (clip.get("title") or "")[:120],
                "channel": clip.get("channel"),
                "duration_s": clip.get("duration"),
                "llm_score": clip.get("discovery_llm_score")})
    for meta in yt_pool:
        _write({"event": "discovery", "sentence": sentence,
                "kept": True, "routed_to": "youtube_pool",
                "source": meta.get("source"),
                "url": meta.get("webpage_url"),
                "title": (meta.get("title") or "")[:120]})
    for item in rejected:
        _write({"event": "discovery", "sentence": sentence,
                "kept": False,
                "reason": item.get("reason"),
                "url": item.get("url") or item.get("webpage_url"),
                "title": (item.get("title") or "")[:120]})


def _rejection_reasons(clip: dict) -> list[str]:
    """Which gates evidence says this candidate failed (may be several)."""
    reasons: list[str] = []
    if clip.get("llm_reject"):
        reasons.append("llm_judge")
    verify_ok = clip.get("verified")
    if verify_ok is False:
        reasons.append("content_verify:" + str(clip.get("verification_reason") or "?"))
    if clip.get("anchor_unverified"):
        reasons.append("anchor_gate")
    if clip.get("vlm_happening") is False and not clip.get("trusted"):
        reasons.append("vlm_reject")
    if clip.get("source_rejected"):
        reasons.append("source_policy:" + str(clip.get("source_reject_reason") or "?"))
    if not reasons and not clip.get("exported"):
        reasons.append("ranked_out")
    return reasons


def log_sentence_result(sentence: str, picked: list[dict],
                        exported: list, need: int) -> None:
    """Per-candidate confidence lines + a missing-matches summary line."""
    exported_keys = {str(p) for p in (exported or [])}
    for rank, clip in enumerate(picked, 1):
        _write({
            "event": "candidate",
            "sentence": sentence,
            "rank": rank,
            "source": clip.get("source"),
            "video_id": clip.get("video_id"),
            "url": clip.get("webpage_url"),
            "title": (clip.get("title") or "")[:140],
            "channel": clip.get("channel"),
            "start_s": clip.get("start"),
            "duration_s": clip.get("duration"),
            "confidence": {
                "semantic": clip.get("semantic_score"),
                "rerank": clip.get("rerank_score"),
                "vision": clip.get("vision_score"),
                "anchor_sim": clip.get("anchor_sim"),
                "event_visual": clip.get("event_visual_score"),
                "vlm": clip.get("vlm_score"),
                "vlm_happening": clip.get("vlm_happening"),
                "llm_moment": clip.get("llm_moment_score"),
                "keyword_located": clip.get("keyword_located"),
                "title_match": clip.get("title_match_score"),
                "combined": clip.get("combined_score"),
                "tier": clip.get("verification_tier"),
            },
            "exported": bool(clip.get("_exported_path")
                             or any(str(clip.get("video_id")) in k
                                    for k in exported_keys)),
            "rejections": _rejection_reasons(clip),
        })
    exported_count = len(exported or [])
    missing: list[dict] = []
    if exported_count < need:
        for clip in picked:
            reasons = _rejection_reasons(clip)
            if reasons and reasons != ["ranked_out"]:
                missing.append({"url": clip.get("webpage_url"),
                                "title": (clip.get("title") or "")[:100],
                                "rejected_by": reasons})
        _write({
            "event": "missing_matches",
            "sentence": sentence,
            "quota": need,
            "exported": exported_count,
            "shortfall": need - exported_count,
            "candidates_considered": len(picked),
            "missing": missing[:12],
        })
    _write({"event": "sentence_summary", "sentence": sentence,
            "quota": need, "exported": exported_count,
            "candidates": len(picked)})
