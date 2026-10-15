"""Write JSON/HTML/CSV reports for a query, including YouTube timestamped links."""
import csv
import html
import json
from collections import Counter
from pathlib import Path

import config


def _stem(query: str) -> str:
    return "_".join("".join(c if c.isalnum() else " " for c in query).split())[:80] or "query"


def _clip_link(clip: dict) -> str:
    if clip.get("source") == "twitch":
        return clip.get("webpage_url", "")
    if clip.get("source") == "image":
        return clip.get("webpage_url", "")
    video_id = clip.get("video_id", "")
    start = float(clip.get("start", 0))
    end = start + float(clip.get("duration", 10))
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}&t={int(start)}&end={int(end)}"
    return clip.get("webpage_url", "")


def raw_label(clip: dict) -> str:
    """Classify a clip as RAW, likely edited, or mixed based on the fused signals."""
    if not clip.get("verified", True):
        return "unverified"
    rawness = float(clip.get("rawness_score", 1.0))
    visual_raw = float(clip.get("visual_raw", 0.5))
    overlay = clip.get("overlay")
    editorial = float(clip.get("editorial_score", 1.0))
    if rawness >= 0.7 and visual_raw >= 0.6 and editorial >= 0.7:
        return "RAW"
    if overlay is not None and overlay > 0.25:
        return "likely edited"
    if rawness < 0.4 or editorial < 0.4:
        return "likely edited"
    return "mixed"


def _platform_stats(clips: list[dict]) -> dict:
    """Per-platform breakdown of clips: count, avg score, source distribution."""
    platforms: dict[str, list[dict]] = {}
    for clip in clips:
        src = clip.get("source") or "youtube"
        platforms.setdefault(src, []).append(clip)
    stats = {}
    for src, group in platforms.items():
        scores = [float(c.get("combined_score", 0)) for c in group]
        stats[src] = {
            "count": len(group),
            "avg_score": sum(scores) / max(1, len(scores)),
            "tiers": dict(Counter(c.get("verification_tier", "unknown") for c in group)),
        }
    return stats


def write_query_report(query: str, clips: list[dict], report_dir: Path,
                       search_meta: dict | None = None) -> tuple[Path, Path, Path]:
    """Write JSON, HTML, and CSV reports for a single query/sentence.

    ``search_meta`` is an optional dict with search transparency info:
      - search_queries: list of queries sent to each platform
      - per_platform_top10: {platform: [clip titles/ids]} showing what each
        platform returned before ranking
      - analysis_pool_sizes: {platform: count} of videos analysed

    Returns (json_path, html_path, csv_path).
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem(query)
    data_path = report_dir / f"{stem}.json"
    html_path = report_dir / f"{stem}.html"
    csv_path = report_dir / f"{stem}.csv"

    enriched = []
    for index, clip in enumerate(clips, 1):
        item = dict(clip)
        item["rank"] = index
        item["end"] = float(clip.get("start", 0)) + float(clip.get("duration", 10))
        item["clip_url"] = _clip_link(clip)
        item["raw_label"] = raw_label(clip)
        item["verification_tier"] = clip.get("verification_tier") or (
            "event_probe" if clip.get("event_visual_verified") else "unverified")
        item["evidence"] = item.get("evidence") or item.get("snippet") or item.get("transcript", "")
        enriched.append(item)

    report_data = {"query": query, "clips": enriched}
    if search_meta:
        report_data["search_transparency"] = search_meta
    report_data["platform_stats"] = _platform_stats(enriched)
    data_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    # --- search transparency section ---
    transparency_html = ""
    if search_meta:
        queries_html = "".join(
            f"<li>{html.escape(q)}</li>"
            for q in (search_meta.get("search_queries") or [])
        )
        pool_items = ""
        for platform, count in (search_meta.get("analysis_pool_sizes") or {}).items():
            pool_items += f"<li>{html.escape(str(platform))}: {count} videos analysed</li>"
        top10_items = ""
        for platform, items in (search_meta.get("per_platform_top10") or {}).items():
            entries = "".join(
                f"<li>{html.escape(str(i+1))}. {html.escape(str(item))}</li>"
                for i, item in enumerate(items[:10])
            )
            top10_items += (
                f"<h4>{html.escape(platform)} top 10</h4><ol>{entries}</ol>"
            )
        transparency_html = (
            "<div class='transparency'>"
            "<h2>Search Transparency</h2>"
            "<h3>Search Queries</h3>"
            f"<ul>{queries_html}</ul>"
            "<h3>Analysis Pool</h3>"
            f"<ul>{pool_items}</ul>"
            f"{top10_items}"
            "</div>"
        )

    # --- platform stats ---
    pstats = _platform_stats(enriched)
    platform_html = "<h2>Per-Platform Breakdown</h2><ul>"
    for src, info in pstats.items():
        tiers = ", ".join(f"{k}: {v}" for k, v in info["tiers"].items())
        platform_html += (
            f"<li><strong>{html.escape(src)}</strong>: {info['count']} clip(s), "
            f"avg score {info['avg_score']:.3f} — {html.escape(tiers)}</li>"
        )
    platform_html += "</ul>"

    cards = "".join(
        f'<li><a href="{html.escape(c["clip_url"])}">{html.escape(c.get("video_id") or "video")}</a> '
        f'<span class="badge src_{html.escape(c.get("source") or "youtube")}">{html.escape(c.get("source") or "youtube")}</span>'
        f'<span class="badge {(c["raw_label"] or "unlabeled").replace(" ", "_")}">{html.escape(c.get("raw_label") or "unlabeled")}</span>'
        f'<span class="badge tier">{html.escape(c.get("verification_tier", ""))}</span>'
        f' — {c["start"]:.1f}s–{c["end"]:.1f}s, score {c.get("combined_score", 0):.3f}'
        + (f' <a href="{html.escape(c["export_path"])}">[clip]</a>' if c.get("export_path") else "")
        + f'<br><small>{html.escape(c.get("evidence", ""))}</small></li>'
        for c in enriched
    )
    page = (
        "<html><meta charset='utf-8'><style>"
        "body{font-family:sans-serif;max-width:900px;margin:24px auto}"
        ".badge{font-size:11px;padding:2px 6px;border-radius:8px;background:#eee;margin-left:6px}"
        ".RAW{background:#cdeacd}.likely_edited{background:#f6d5d5}.mixed{background:#fdf3cd}"
        ".tier{background:#e3e0ff;color:#333}"
        ".src_youtube{background:#ffe0e0}.src_twitch{background:#e0e0ff}.src_kick{background:#e0ffe0}"
        ".transparency{background:#f8f8f8;border:1px solid #ddd;padding:12px;border-radius:6px;margin:16px 0}"
        "</style><body>"
        f"<h1>{html.escape(query)}</h1>"
        f"{transparency_html}"
        f"{platform_html}"
        f"<ol>{cards}</ol></body></html>"
    )
    html_path.write_text(page, encoding="utf-8")

    fieldnames = [
        "rank", "raw_label", "verification_tier", "source", "video_id",
        "clip_url", "webpage_url",
        "start", "end", "duration",
        "combined_score", "action_coverage", "entity_coverage",
        "semantic_score", "lexical_score", "rerank_score",
        "llm_moment_score", "llm_moment_reason",
        "vision_score", "vlm_score", "vlm_happening",
        "anchor_sim", "rawness_score", "visual_raw", "overlay", "continuity",
        "editorial_score", "keyword_located",
        "title_match_score", "metadata_score", "match_tier",
        "export_path", "evidence",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for clip in enriched:
            writer.writerow(clip)

    return data_path, html_path, csv_path
