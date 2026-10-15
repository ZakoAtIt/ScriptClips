"""Remote stream helpers: caption retrieval and frame sampling."""
import html
import json
import logging
import re
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

import config
from modules import transcript_cache

logger = logging.getLogger(__name__)

_CAPTION_LANGS = ("en", "en-US", "en-GB")


# ---------------------------------------------------------------------------
# Cookie injection for ffmpeg (ANDROID_VR URLs require session cookies)
# ---------------------------------------------------------------------------

def _load_cookie_header() -> str:
    """Parse cookies.txt (Netscape format) and build a Cookie header string."""
    cookie_file = getattr(config, "YT_COOKIES_FILE", None)
    if not cookie_file or not Path(cookie_file).is_file():
        return ""
    _AUTH_COOKIES = {
        "SID", "HSID", "SSID", "APISID", "SAPISID",
        "__Secure-1PSID", "__Secure-3PSID",
        "__Secure-1PSIDTS", "__Secure-3PSIDTS",
        "__Secure-1PSIDCC", "__Secure-3PSIDCC",
        "LOGIN_INFO", "SIDCC",
    }
    try:
        pairs: list[str] = []
        for line in Path(cookie_file).read_text(
                encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            domain = parts[0]
            name = parts[5]
            value = parts[6]
            if domain.startswith(".") and "youtube.com" in domain:
                if name in _AUTH_COOKIES:
                    pairs.append(f"{name}={value}")
        return "; ".join(pairs)
    except Exception as exc:
        logger.debug("Failed to parse cookies.txt: %s", exc)
        return ""


_cached_cookie_header: str | None = None
_remote_seeks_broken: bool = False


def stream_headers(info: dict) -> list[str]:
    """Build ffmpeg -headers argument list from yt-dlp's http_headers.

    YouTube puts the User-Agent on the format-level dict (not the top level)
    for signed stream URLs; fall back to the frame-stream format's headers so
    remote deep-seeks send a UA instead of 403ing.

    Injects auth cookies from cookies.txt and ensures a Referer header is
    present so WEB client stream URLs don't 403 at the CDN.
    """
    headers = (info.get("http_headers") or {}).copy()
    if not headers:
        stream = get_frame_stream_format(info)
        if stream:
            headers = (stream.get("http_headers") or {}).copy()
    headers.pop("Range", None)  # ffmpeg sets its own ranges
    if not headers:
        return []
    value = "".join(f"{key}: {val}\r\n" for key, val in headers.items())
    if "Referer" not in headers:
        value += "Referer: https://www.youtube.com/\r\n"
    global _cached_cookie_header
    if _cached_cookie_header is None:
        _cached_cookie_header = _load_cookie_header()
    if _cached_cookie_header:
        value += f"Cookie: {_cached_cookie_header}\r\n"
    return ["-headers", value]


def get_best_stream(info: dict) -> str | None:
    video = get_best_stream_format(info)
    return video["url"] if video else None


def get_best_stream_format(info: dict) -> dict | None:
    """Highest-quality video format (DASH fMP4 preferred; deep-seek unfriendly)."""
    formats = info.get("formats") or []
    video = [f for f in formats if f.get("url") and f.get("vcodec") not in (None, "none")]
    if not video:
        return None
    preferred = [f for f in video if (f.get("height") or 0) >= 360]
    pool = preferred if preferred else video
    pool.sort(key=lambda f: ((f.get("height") or 0), (f.get("tbr") or 0)), reverse=True)
    return pool[0]


def get_frame_stream(info: dict) -> str | None:
    video = get_frame_stream_format(info)
    return video["url"] if video else None


def get_frame_stream_format(info: dict) -> dict | None:
    """Pick a stream ffmpeg can deep-seek remotely, fast and timeout-free.

    Progressive combined MP4s (format 18, 22, ...) carry their moov index at the
    front, so ffmpeg input-seeks them via HTTP byte-range in ~0.5s.  YouTube's
    DASH fMP4 and WebM/VP9 streams need fragment/index seeking over HTTP, which
    ffmpeg cannot do here — an input ``-ss`` makes it pull the whole VOD from
    byte 0 and every frame probe times out.  Falls back to any video stream
    (worst case: seeks hang and ``sample_frames_at`` early-aborts).
    """
    formats = info.get("formats") or []
    combined = [f for f in formats if f.get("url")
                and (f.get("vcodec") or "") not in ("", "none")
                and (f.get("acodec") or "") not in ("", "none")]
    progressive = [f for f in combined if "mp4" in (f.get("ext") or "")]
    mid = [f for f in progressive if 240 <= (f.get("height") or 0) <= 480]
    pool = mid or progressive or combined
    if not pool:
        pool = [f for f in formats if f.get("url")
                and (f.get("vcodec") or "") not in ("", "none")]
    pool.sort(key=lambda f: (f.get("height") or 0))
    return pool[0] if pool else None


def get_best_audio_stream(info: dict) -> str | None:
    formats = info.get("formats") or []
    audio = [f for f in formats if f.get("url") and f.get("acodec") not in (None, "none")
             and f.get("vcodec") in (None, "none")]
    if not audio:
        return None
    return max(audio, key=lambda f: f.get("abr") or 0)["url"]


def _caption_track(info: dict):
    """Return (kind, lang, track) for the best available caption track, or None."""
    for kind, pool in (("manual", info.get("subtitles") or {}), ("auto", info.get("automatic_captions") or {})):
        for lang in _CAPTION_LANGS:
            tracks = pool.get(lang)
            if not tracks:
                continue
            track = next((item for item in tracks if item.get("ext") in ("vtt", "json3")), tracks[0])
            return kind, lang, track
    return None


def get_caption_segments(info: dict) -> list[dict]:
    """Fetch available human/automatic VTT captions, cached by (video_id, lang, kind)."""
    found = _caption_track(info)
    if not found:
        return []
    kind, lang, track = found
    video_id = str(info.get("id") or "")
    if video_id:
        cached = transcript_cache.load(video_id, lang, kind)
        if cached is not None:
            logger.info("Caption cache hit for %s (%s)", video_id, kind)
            return cached
    try:
        request = Request(track["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.info("Captions unavailable: %s", exc)
        return []
    segments = _parse_vtt(payload) if track.get("ext") == "vtt" else _parse_json3(payload)
    if video_id:
        transcript_cache.save(video_id, lang, kind, segments)
    return segments


def _to_seconds(value: str) -> float:
    parts = [float(p.replace(",", ".")) for p in value.split(":")]
    return parts[-1] + (parts[-2] * 60 if len(parts) > 1 else 0) + (parts[-3] * 3600 if len(parts) > 2 else 0)


def _parse_vtt(payload: str) -> list[dict]:
    segments, seen = [], set()
    for block in re.split(r"\r?\n\r?\n", payload):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing = next((line for line in lines if " --> " in line), None)
        if not timing:
            continue
        start, end = [part.strip().split(" ")[0] for part in timing.split(" --> ", 1)]
        text = " ".join(line for line in lines if line != timing and not line.startswith("WEBVTT"))
        text = re.sub(r"<[^>]+>", "", html.unescape(text)).strip()
        key = (start, end, text)
        if text and key not in seen:
            seen.add(key)
            segments.append({"start": _to_seconds(start), "end": _to_seconds(end), "text": text})
    return segments


def _parse_json3(payload: str) -> list[dict]:
    try:
        events = json.loads(payload).get("events", [])
    except json.JSONDecodeError:
        return []
    segments = []
    for event in events:
        snippets = event.get("segs") or []
        text = "".join(part.get("utf8", "") for part in snippets).replace("\n", " ").strip()
        if not text or "tStartMs" not in event:
            continue
        start = float(event["tStartMs"]) / 1000
        end = start + float(event.get("dDurationMs", 0)) / 1000
        segments.append({"start": start, "end": end, "text": text})
    return segments


def sample_frames(stream_url: str, duration: float, out_dir: Path, every_seconds: int,
                  maximum: int, extra_args: list[str] | None = None) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    natural_count = max(1, int(duration) // max(1, every_seconds)) + 1
    count = min(natural_count, maximum)
    timestamps = sorted({round(index * max(0, duration - 1) / max(1, count - 1)) for index in range(count)})
    frames = []
    for timestamp in timestamps:
        path = out_dir / f"{timestamp:06d}.jpg"
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *(extra_args or [])]
        cmd += ["-ss", str(timestamp), "-i", stream_url, "-frames:v", "1",
                "-vf", "scale=320:-2,format=yuvj420p", "-q:v", "4", str(path)]
        try:
            subprocess.run(cmd, check=True, timeout=config.FRAME_SEEK_TIMEOUT)
            frames.append({"path": str(path), "timestamp": float(timestamp)})
        except (subprocess.SubprocessError, OSError) as exc:
            logger.debug("Frame at %.1fs skipped: %s", timestamp, exc)
    return frames


def sample_frames_at(stream_url: str, timestamps: list[float], out_dir: Path,
                     extra_args: list[str] | None = None,
                     skip_limit: int = 2,
                     video_id: str | None = None) -> list[dict]:
    """Fetch only the frames needed to verify shortlisted clip moments.

    Stops early once ``skip_limit`` consecutive seeks fail: a stream whose
    frames keep timing out will not recover mid-scan, so callers that sample
    hundreds of frames (vod_scan) abort instead of burning 8s per frame.

    When the remote stream 403s/hangs (YouTube's signature rejects anonymous
    ffmpeg seeks) and ``video_id`` is known, the video is downloaded once via
    in-process yt-dlp and the whole timestamp list is re-scanned from the local
    file, so the moment still gets verified.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    global _remote_seeks_broken
    frames: list[dict] = []
    if not _remote_seeks_broken and stream_url.startswith("http"):
        frames = _sample_list(stream_url, timestamps, out_dir, extra_args,
                              skip_limit)
        if timestamps and not frames:
            # Remote produced nothing at all — seeks are broken for this
            # session (expired/403ing googlevideo URLs).  Flip the session
            # flag so later scans skip straight to local instead of paying
            # the same failed remote round-trip per candidate.
            _remote_seeks_broken = True
            logger.info("Remote seeks produced 0/%d frames; switching "
                        "session to local-download scans", len(timestamps))
    if len(frames) < len(timestamps) and video_id:
        if not frames:
            pass  # remote skipped or all failed — go straight to local
        else:
            _remote_seeks_broken = True
            logger.info("Remote seeks broken for this session; "
                        "switching to local download for remaining videos")
        local = _download_for_scan(video_id, out_dir)
        if local:
            logger.info("Re-scanning %s from local %s", video_id, local)
            frames = _sample_list(local, timestamps, out_dir, None, skip_limit)
    return frames


def _sample_list(stream_url: str, timestamps: list[float], out_dir: Path,
                 extra_args: list[str] | None, skip_limit: int) -> list[dict]:
    frames = []
    skipped = 0
    for index, timestamp in enumerate(timestamps):
        path = out_dir / f"verify_{index:03d}_{timestamp:.1f}.jpg"
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *(extra_args or [])]
        cmd += ["-ss", str(timestamp), "-i", stream_url, "-frames:v", "1",
                "-vf", f"scale={config.VISUAL_FRAME_WIDTH}:-2,format=yuvj420p",
                "-q:v", "3", str(path)]
        try:
            subprocess.run(cmd, check=True, timeout=config.FRAME_SEEK_TIMEOUT)
            frames.append({"path": str(path), "timestamp": float(timestamp)})
            skipped = 0
        except (subprocess.SubprocessError, OSError) as exc:
            logger.info("Verification frame at %.1fs skipped: %s", timestamp, exc)
            skipped += 1
            if skipped >= skip_limit:
                logger.info("Frame scan aborted for %s: %d consecutive "
                            "seek failures", out_dir.name, skipped)
                break
    return frames


def _download_for_scan(video_id: str, out_dir: Path) -> str | None:
    """Download the source video once so frame seeks can run on a local file.

    In-process yt-dlp (the CLI is broken by the injected --enable-ejs config)
    plus the shared cookies file; format 18 progressive MP4 is preferred for
    fast ffmpeg input-seeks.  Best-effort: returns None on any failure.

    Checks a shared download cache and all known scan directories to avoid
    re-downloading the same video across coarse/fine/ultra passes.
    """
    try:
        import yt_dlp
        from modules import downloader
        target = out_dir / f"{video_id}_local.mp4"
        if target.exists() and target.stat().st_size > 1024:
            return str(target)
        # Check shared download cache first (one download per video_id)
        shared = config.FRAMES_DIR / "_shared_downloads" / f"{video_id}.mp4"
        if shared.exists() and shared.stat().st_size > 1024 * 1024:
            logger.info("Reusing shared download cache %s for %s", shared, video_id)
            return str(shared)
        # Check all known scan directories for an existing local copy
        scan_prefixes = ("vodscan_", "sentence_", "vmoment_", "twvod_")
        for sibling_dir in config.FRAMES_DIR.iterdir():
            if not sibling_dir.is_dir():
                continue
            name = sibling_dir.name
            if not any(name.startswith(p) and video_id in name for p in scan_prefixes):
                continue
            if sibling_dir == out_dir:
                continue
            candidate = sibling_dir / f"{video_id}_local.mp4"
            if candidate.exists() and candidate.stat().st_size > 1024 * 1024:
                logger.info("Reusing existing local scan copy %s for %s",
                            candidate, video_id)
                return str(candidate)
        opts = downloader.with_cookies({
            "quiet": True, "no_warnings": True, "nopart": True,
            "socket_timeout": 60,
            "outtmpl": str(target),
            "format": "18/bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
        })
        url = f"https://www.youtube.com/watch?v={video_id}"
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        if target.exists() and target.stat().st_size > 1024:
            # Copy to shared cache so other passes can reuse
            try:
                import shutil
                shared.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(target), str(shared))
            except Exception:
                pass
            return str(target)
    except Exception as exc:
        logger.info("Local fallback download failed for %s: %s", video_id, exc)
    return None
