"""Web discovery lane: find the moment on ANY yt-dlp-supported platform.

The verbatim YouTube search misses moments whose uploads live on X/Twitter
mirrors, native Twitch clips, or differently-titled uploads.  This lane runs
the query's search hooks through open-web search (DDG HTML endpoint via
urllib; headed-Chrome Google fallback), keeps hits on supported platforms,
resolves each with yt-dlp metadata, LLM-ranks them against the sentence, and
returns:

  * whole-clip candidates (<= CLIP_MAX_SECONDS) shaped exactly like
    twitch_lane candidates, so they flow through finalize/judge/exporter
    unchanged;
  * long YouTube hits as pool metas for youtube_lane's analysed pool;
  * a rejected record list for the run report.

Login-walled or unresolvable posts are skipped and logged — never fatal.
"""
import hashlib
import logging
import re
import time
import urllib.parse
import urllib.request

import config

logger = logging.getLogger(__name__)

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

_RESULT_HREF_RE = re.compile(r'href="(//|https?://)(?:duckduckgo\.com/l/\?uddg=)?[^"]+"')
_UDDG_RE = re.compile(r"[?&]uddg=([^&\"]+)")
_HREF_EXTRACT_RE = re.compile(r'href="([^"]+)"')

_ALLOWED = (
    ("youtube.com/watch", "youtube"),
    ("youtu.be/", "youtube"),
    ("youtube.com/shorts", "youtube"),
    ("x.com/", "twitter"),
    ("twitter.com/", "twitter"),
    ("clips.twitch.tv/", "twitch"),
    ("twitch.tv/", "twitch"),
    ("kick.com/", "kick"),
)


def _hooks(sentence: str, plan: dict) -> list[str]:
    """Verbatim sentence first, then planner hooks; capped and de-duplicated."""
    hooks = [sentence.strip()]
    for key in ("search_queries", "action_search_queries"):
        for q in (plan.get(key) or []):
            q = str(q).strip()
            if q and q.lower() not in {h.lower() for h in hooks}:
                hooks.append(q)
    hooks = hooks[: config.DISCOVERY_MAX_HOOKS]
    logger.info("Discovery hooks for '%s': %s", sentence, hooks)
    return hooks


def _unwrap(url: str) -> str:
    """Resolve DDG redirect wrappers (/l/?uddg=...) to the real target."""
    if "uddg=" in url:
        match = _UDDG_RE.search(url)
        if match:
            return urllib.parse.unquote(match.group(1))
    if url.startswith("//"):
        return "https:" + url
    return url


# --- Headed-Chrome Google SERP fallback ---------------------------------------
_serp_session = None
_serp_tried = False


def _chrome_serp():
    """Lazily start (once per run) a headed Chrome for Google web search."""
    global _serp_session, _serp_tried
    if _serp_tried:
        return _serp_session
    _serp_tried = True
    try:
        from modules.chrome_session import ChromeSession
        _serp_session = ChromeSession().start()
    except Exception as exc:
        logger.warning("Discovery Chrome session failed to start: %s", exc)
        _serp_session = None
    return _serp_session


def close_serp_session() -> None:
    global _serp_session, _serp_tried
    if _serp_session is not None:
        try:
            _serp_session.close()
        except Exception:
            pass
    _serp_session = None
    _serp_tried = False


def _google_serp_search(query: str, max_results: int) -> list[str]:
    """Google web results rendered in real headed Chrome (beats bot walls)."""
    session = _chrome_serp()
    if session is None:
        return []
    page = session.context.new_page()
    try:
        page.goto("https://www.google.com/search?" +
                  urllib.parse.urlencode({"q": query, "num": "20"}),
                  timeout=45000)
        page.wait_for_timeout(2500)
        urls: list[str] = []
        for a in page.query_selector_all("a[href]"):
            href = a.get_attribute("href") or ""
            if not href.startswith("http"):
                continue
            lowered = href.lower()
            if any(d in lowered for d in ("google.", "gstatic.", "youtube.com/search",
                                          "support.google", "policies.google")):
                continue
            # A result anchor carries a visible headline; ads/menu links do not.
            if a.query_selector("h3") is None:
                continue
            if href not in urls:
                urls.append(href)
            if len(urls) >= max_results:
                break
        return urls
    except Exception as exc:
        logger.info("Discovery Google SERP %r failed: %s", query, exc)
        return []
    finally:
        try:
            page.close()
        except Exception:
            pass


def _web_search(query: str, max_results: int) -> list[str]:
    """DDG HTML first (cheap); headed-Chrome Google when DDG is throttled."""
    urls = _ddg_search(query, max_results)
    if len(urls) >= 3:
        return urls
    logger.info("Discovery: DDG thin (%d) for %r; using Chrome Google", len(urls), query)
    chrome_urls = _google_serp_search(query, max_results)
    merged = list(urls)
    for u in chrome_urls:
        if u not in merged:
            merged.append(u)
    return merged


def _ddg_search(query: str, max_results: int) -> list[str]:
    """DDG HTML endpoint results (no JS needed); falls back to the lite page."""
    endpoints = (
        ("https://html.duckduckgo.com/html/", True),
        ("https://lite.duckduckgo.com/lite/", False),
    )
    for endpoint, is_post in endpoints:
        try:
            body = urllib.parse.urlencode({"q": query}).encode()
            req = urllib.request.Request(
                endpoint, data=body if is_post else None,
                headers={"User-Agent": _UA,
                         "Content-Type": "application/x-www-form-urlencoded"
                         if is_post else "text/html"})
            with urllib.request.urlopen(req, timeout=config.DISCOVERY_TIMEOUT) as resp:
                html = resp.read().decode("utf-8", "replace")
            urls: list[str] = []
            for raw in _HREF_EXTRACT_RE.findall(html):
                target = _unwrap(raw)
                if target.startswith("http") and "duckduckgo.com" not in target:
                    if target not in urls:
                        urls.append(target)
                    if len(urls) >= max_results:
                        return urls
            if urls:
                return urls
        except Exception as exc:
            logger.info("Discovery search %r via %s failed: %s", query, endpoint, exc)
    return []


def _platform(url: str) -> str | None:
    lowered = url.lower()
    for marker, platform in _ALLOWED:
        if marker in lowered:
            return platform
    return None


def _video_id(url: str, platform: str) -> str:
    if platform == "youtube":
        match = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{6,})", url)
        if match:
            return match.group(1)
    if platform == "twitter":
        match = re.search(r"/status/(\d+)", url)
        if match:
            return "x_" + match.group(1)
    if platform == "twitch":
        slug = url.rstrip("/").split("/")[-1]
        if slug and slug not in ("clips", "twitch"):
            return "tw_" + slug[:40]
    return "dsc_" + hashlib.sha1(url.encode()).hexdigest()[:10]


def _lexical_score(sentence: str, title: str) -> float:
    """Fallback relevance when no LLM is available: token overlap."""
    stop = {"a", "an", "the", "in", "on", "at", "of", "and", "or", "to",
            "is", "was", "were", "his", "her", "their", "it", "he", "she"}
    s_tokens = {t for t in re.findall(r"[a-z0-9']+", sentence.lower()) if t not in stop}
    t_tokens = {t for t in re.findall(r"[a-z0-9']+", (title or "").lower()) if t not in stop}
    if not s_tokens:
        return 0.0
    return len(s_tokens & t_tokens) / len(s_tokens)


def _llm_rank(sentence: str, items: list[dict]) -> list[float]:
    """LLM relevance 0..1 per item; [] when no LLM (caller falls back)."""
    from modules import llm_client
    if not llm_client.available():
        return []
    lines = "\n".join(
        f"{i}: [{it['platform']}] {it['channel'] or '?'} | "
        f"{(it['title'] or '')[:110]} | {it['duration']:.0f}s"
        for i, it in enumerate(items))
    prompt = (
        f"Query describing a specific livestream moment:\n\"{sentence}\"\n\n"
        f"Candidate web results:\n{lines}\n\n"
        "For EACH candidate return a relevance score 0.00-1.00 for how likely "
        "it contains THE exact moment described (raw footage of the people and "
        "event named; reactions/compilations/news articles score low). "
        'Reply as JSON: {"scores": {"0": 0.0, "1": 0.0, ...}}')
    reply = llm_client.chat_json(prompt, system="You are a precise media research scorer.")
    scores = reply.get("scores")
    if not isinstance(scores, dict):
        return []
    out: list[float] = []
    for i in range(len(items)):
        try:
            out.append(max(0.0, min(1.0, float(scores.get(str(i), 0.0)))))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def run(sentence: str, plan: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (whole_clip_candidates, youtube_pool_metas, rejected_records)."""
    if not config.DISCOVERY_ENABLED:
        return [], [], []
    hooks = _hooks(sentence, plan)

    # --- 1. Search the open web ------------------------------------------
    # Space DDG requests out (it throttles rapid successive hits) and stop
    # early once enough candidate URLs are collected.
    urls: list[str] = []
    seen_urls: set[str] = set()
    try:
        for index, hook in enumerate(hooks):
            if index:
                time.sleep(2.0)
            for url in _web_search(hook, 12):
                normalized = url.split("#")[0].split("?")[0] if "watch?v=" not in url else url
                if normalized in seen_urls:
                    continue
                platform = _platform(url)
                if not platform:
                    continue
                seen_urls.add(normalized)
                urls.append(url)
                if len(urls) >= config.DISCOVERY_MAX_URLS:
                    break
            if len(urls) >= config.DISCOVERY_MAX_URLS:
                break
    finally:
        close_serp_session()
    if not urls:
        logger.info("Discovery lane '%s': no supported-platform hits", sentence)
        return [], [], []
    logger.info("Discovery lane '%s': %d candidate URL(s)", sentence, len(urls))

    # --- 2. Resolve metadata ------------------------------------------------
    from modules import downloader
    resolved: list[dict] = []
    rejected: list[dict] = []
    for url in urls:
        platform = _platform(url)
        try:
            # Flat extract first: fast, no format selection (Shorts-safe).
            info = downloader.get_video_metadata(url, flat=True)
            if not info or not info.get("duration"):
                info = downloader.get_video_metadata(url)
        except Exception as exc:
            info = None
            logger.info("Discovery resolve failed %s: %s", url, exc)
        if not info:
            rejected.append({"url": url, "reason": "unresolvable",
                             "title": ""})
            continue
        if info.get("is_live") or info.get("live_status") == "is_live":
            rejected.append({"url": url, "reason": "live", "title": info.get("title")})
            continue
        duration = float(info.get("duration") or 0)
        resolved.append({
            "url": url, "platform": platform, "info": info,
            "title": info.get("title") or "",
            "channel": info.get("uploader") or info.get("channel") or "",
            "duration": duration,
        })

    # --- 3. Split: whole clips vs long YouTube pool adds --------------------
    whole_items = [r for r in resolved
                   if config.CLIP_MIN_SECONDS * 0.3 <= r["duration"] <= config.CLIP_MAX_SECONDS]
    yt_pool_items = [r for r in resolved
                     if r["platform"] == "youtube" and r["duration"] > config.CLIP_MAX_SECONDS]
    for r in resolved:
        if r not in whole_items and r not in yt_pool_items:
            rejected.append({"url": r["url"], "reason": f"duration {r['duration']:.0f}s",
                             "title": r["title"]})

    # --- 4. Rank -------------------------------------------------------------
    scores = _llm_rank(sentence, whole_items)
    clips: list[dict] = []
    for index, item in enumerate(whole_items):
        score = scores[index] if scores else _lexical_score(sentence, item["title"])
        if score < config.DISCOVERY_MIN_LLM_SCORE:
            rejected.append({"url": item["url"], "reason": f"relevance {score:.2f}",
                             "title": item["title"]})
            continue
        vid = _video_id(item["url"], item["platform"])
        clips.append({
            "id": vid,
            "video_id": vid,
            "title": item["title"],
            "description": item["info"].get("description") or "",
            "channel": item["channel"],
            "thumbnail": item["info"].get("thumbnail") or "",
            "webpage_url": item["url"],
            "duration": item["duration"],
            "start": 0.0,
            "view_count": float(item["info"].get("view_count") or 0),
            "timestamp": item["info"].get("upload_date") or "",
            "source": item["platform"],
            "query": sentence,
            "whole": True,
            "discovery_pinned": True,
            "discovery_llm_score": round(score, 3),
        })
    clips.sort(key=lambda c: c["discovery_llm_score"], reverse=True)

    yt_metas: list[dict] = []
    for item in yt_pool_items[: config.DISCOVERY_YT_POOL_MAX]:
        vid = _video_id(item["url"], "youtube")
        yt_metas.append({
            "id": vid,
            "video_id": vid,
            "title": item["title"],
            "description": item["info"].get("description") or "",
            "channel": item["channel"],
            "webpage_url": item["url"],
            "duration": item["duration"],
            "view_count": float(item["info"].get("view_count") or 0),
            "upload_date": item["info"].get("upload_date") or "",
            "source": "youtube",
            "discovery_pinned": True,
            "search_score": 0.8,
        })

    logger.info("Discovery lane '%s': %d whole clip(s), %d long YouTube add(s), "
                "%d rejected", sentence, len(clips), len(yt_metas), len(rejected))
    return clips, yt_metas, rejected
