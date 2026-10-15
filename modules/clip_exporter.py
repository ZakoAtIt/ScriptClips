"""Download only final, selected source ranges as standalone MP4 clips."""
import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import requests

try:
    import yt_dlp
except Exception:
    yt_dlp = None

import config
from modules import content_verifier, event_vision
from modules.downloader import with_cookies

logger = logging.getLogger(__name__)


def _safe_stem(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_")[:80] or "query"


def _user_forced(key: str) -> str | None:
    """User-forced moment reason (``video_id:start`` key) or None when not forced."""
    video_id, _, secs = key.partition(":")
    try:
        start = float(secs)
    except (TypeError, ValueError):
        return None
    try:
        payload = json.loads(config.FORCED_MOMENTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    for fkey, reason in payload.items():
        if not isinstance(fkey, str) or ":" not in fkey:
            continue
        vid, _, fsecs = fkey.partition(":")
        if vid != video_id:
            continue
        try:
            if abs(float(fsecs) - start) <= 1.0:
                return str(reason) if reason else None
        except (TypeError, ValueError):
            continue
    return None


def _probe_duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True)
        out = result.stdout.strip()
        return float(out) if out else None
    except Exception:
        return None


def _has_audio(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "default=noprint_wrappers=1", str(path)],
            capture_output=True, text=True, check=True)
        return bool(result.stdout.strip())
    except Exception:
        return False


def _probe_stream_start(path: Path, stream_type: str) -> float | None:
    """First-frame PTS of a stream in a file (actual media time)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", stream_type,
             "-show_entries", "stream=start_time",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True)
        out = result.stdout.strip()
        return float(out) if out else None
    except Exception:
        return None


def _decodes_a_frame(path: Path) -> bool:
    """True if ffmpeg can actually decode a video frame (stubs with only a
    moov box / undecodable mdat pass ffprobe but carry no usable video)."""
    return _decodes_a_frame_at(path, 0.2)


def _decodes_a_frame_at(path: Path, timestamp: float) -> bool:
    """True if ffmpeg decodes a video frame at the given timestamp.

    Stub files (moov + init bytes, no real mdat) decode only at the very
    start, so validating a mid-clip seek catches broken downloads that would
    otherwise fail the later face/event gates.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-ss", str(timestamp), "-i", str(path), "-frames:v", "1",
             "-f", "null", "-"],
            check=True, timeout=45,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _decodes_frame_to_image(path: Path, timestamp: float) -> bool:
    """True if ffmpeg writes a real downscaled JPEG at the given timestamp.

    Demux-to-null can still pass on fragment-torn files that the visual gate
    later fails to read ("Broken file, trak/mdat not at top-level"), so this
    uses the exact command shape the gate's sampler runs and requires the
    frame to actually land on disk.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="scacc_") as tmpdir:
            out = Path(tmpdir) / "frame.jpg"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-ss", str(timestamp), "-i", str(path), "-frames:v", "1",
                 "-vf", f"scale={config.VISUAL_FRAME_WIDTH}:-2,format=yuvj420p",
                 "-q:v", "3", str(out)],
                check=True, timeout=45,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return out.exists() and out.stat().st_size > 0
    except Exception:
        return False


def _acceptable(path: Path, duration: float) -> bool:
    actual = _probe_duration(path)
    if actual is None or not _has_audio(path):
        return False
    tolerance = max(1.0, 0.05 * duration)  # whole clips vary slightly vs advertised
    if abs(actual - duration) > tolerance:
        return False
    if not _decodes_a_frame(path):
        return False
    mid = min(max(0.5, duration * 0.4), actual)
    if not _decodes_frame_to_image(path, mid):
        return False
    return True


def _run_ffmpeg(cmd: list[str], timeout: int | None = None) -> None:
    timeout = timeout or config.FFMPEG_TRIM_TIMEOUT
    subprocess.run(cmd, check=True, timeout=timeout)


def _scaled_timeout(duration: float) -> int:
    """ffmpeg re-encodes of long clips need timeouts that grow with the clip."""
    return max(config.FFMPEG_TRIM_TIMEOUT, 60 + int(duration))


_FATAL_MARKERS = (
    "requested format is not available",
    "drm protected",
    "drm",
    "sign in to confirm you're not a bot",
    "video unavailable",
    "only available",
    "removed",
)


def _is_fatal(exc: Exception) -> bool:
    """Errors that retrying (or another player client) will never fix."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _FATAL_MARKERS)


def _download_with_retries(ydl_opts: dict, webpage_url: str,
                           attempts: int = 2, fatal: tuple = ()) -> None:
    """yt-dlp download with at most one re-attempt (fresh signed URLs on 403)."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([webpage_url])
            return
        except Exception as exc:
            if isinstance(exc, fatal) or _is_fatal(exc):
                raise
            last = exc
            if i < attempts - 1:
                logger.info("yt-dlp attempt %d/%d failed: %s; retrying",
                            i + 1, attempts, exc)
                time.sleep(8)
    assert last is not None
    raise last


_MAX_SECTION_BYTES = 600 * 1024 * 1024      # ceiling for range-extracted windows
_MAX_WHOLE_CLIP_BYTES = 1024 * 1024 * 1024  # ceiling for whole 15-240s clips

# Player clients to probe for a stream that honours HTTP Range. yt-dlp's
# default often yields non-range-seekable URLs for long VODs; alternate
# clients frequently expose range-supporting streams.
_CLIENTS = ("default", "tv", "mweb", "android", "ios")


def _trim_with_ytdlp(webpage_url: str, start: float, duration: float, out_path: Path,
                     max_section_bytes: int = _MAX_SECTION_BYTES) -> None:
    """Download the video with yt-dlp, then trim exactly with ffmpeg.

    Downloads the whole file (bounded by ``max_section_bytes``), then seeks
    with ffmpeg input ``-ss`` so the extracted window is exact. Probes several
    player clients and aborts early if the download would exceed the cap."""
    if yt_dlp is None:
        raise RuntimeError("yt-dlp not available")

    class _SectionTooBig(Exception):
        pass

    last_error: Exception | None = None
    for client in _CLIENTS:
        tmp = out_path.with_suffix(".tmp.mp4")

        def _abort_hook(d: dict) -> None:
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total > max_section_bytes:
                raise _SectionTooBig(
                    f"section download would fetch {total / 1024 / 1024:.0f} MiB "
                    "(stream not range-seekable)")

        ydl_opts = {
            "quiet": True, "no_warnings": True,
            "socket_timeout": 60,
            "nopart": True,
            "outtmpl": str(tmp),
            "format": ("18/bestvideo[ext=webm]+bestaudio[ext=webm]/"
                       "bestvideo+bestaudio/best"),
            "merge_output_format": "mp4",
            "progress_hooks": [_abort_hook],
        }
        if client != "default":
            ydl_opts["extractor_args"] = {"youtube": {"player_client": [client]}}
        with_cookies(ydl_opts)
        try:
            try:
                _download_with_retries(ydl_opts, webpage_url, fatal=(_SectionTooBig,))
            except _SectionTooBig as exc:
                raise RuntimeError(str(exc))
            if not tmp.exists():
                raise RuntimeError("yt-dlp produced no file")
            size = tmp.stat().st_size
            if size > max_section_bytes:
                raise RuntimeError(
                    f"section download fetched {size / 1024 / 1024:.0f} MiB for a "
                    f"{duration:.0f}s clip (stream not range-seekable)")
            _run_ffmpeg(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                         "-ss", str(start), "-i", str(tmp), "-t", str(duration),
                         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                         "-c:a", "aac", "-b:a", "128k",
                         "-movflags", "+faststart", str(out_path)])
            return
        except Exception as exc:
            last_error = exc
            logger.info("section download with player client %r failed: %s", client, exc)
            if _is_fatal(exc):
                break
        finally:
            for frag in tmp.parent.glob(f"{tmp.stem}.*"):
                try:
                    frag.unlink(missing_ok=True)
                except OSError:
                    pass
    assert last_error is not None
    raise last_error


def _fresh_streams(clip: dict, start: float) -> dict:
    """Re-resolve signed stream URLs right before a direct trim.

    URLs captured at analysis time routinely stall by export time; a fresh
    extract gives YouTube a new signed URL. Falls back to the stale ones.

    Windows deep inside a VOD (``start > 0``) must use a progressive combined
    MP4 (format 18 style): ffmpeg input-seeks it via HTTP byte-range in <1s,
    whereas a DASH fMP4 input-``-ss`` hangs pulling the whole VOD from byte 0
    (the export bottleneck before this fix). A progressive stream carries its
    own audio, so ``audio_url`` is cleared to keep the single-stream path.
    """
    if yt_dlp is None:
        return clip
    webpage = clip.get("webpage_url")
    if not webpage:
        return clip
    try:
        from modules import youtube_search
        with yt_dlp.YoutubeDL(with_cookies({"quiet": True, "no_warnings": True,
                                            "skip_download": True})) as ydl:
            info = ydl.extract_info(webpage, download=False)
        fresh = dict(clip)
        video = (youtube_search.get_frame_stream_format(info)
                 if start > 0.0 else None) or youtube_search.get_best_stream_format(info)
        if video:
            fresh["stream_url"] = video["url"]
            fresh["stream_headers_args"] = youtube_search.stream_headers(video)
            has_audio = (video.get("acodec") or "") not in ("", "none")
            fresh["audio_url"] = (None if has_audio
                                  else youtube_search.get_best_audio_stream(info))
        return fresh
    except Exception as exc:
        logger.info("Fresh URL re-resolution failed: %s", exc)
        return clip


def _trim_direct_streams(clip: dict, start: float, duration: float, out_path: Path) -> None:
    """Fallback: ffmpeg trims straight from freshly resolved video/audio stream URLs."""
    clip = _fresh_streams(clip, start)
    video_url = clip.get("stream_url")
    audio_url = clip.get("audio_url")
    primary = video_url or clip.get("webpage_url")
    if not primary:
        raise RuntimeError("No source URL available for clip export")

    if video_url and audio_url and audio_url != video_url:
        combined = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    *(clip.get("stream_headers_args") or []),
                    "-ss", str(start), "-i", video_url,
                    "-ss", str(start), "-i", audio_url, "-t", str(duration),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart", str(out_path)]
        try:
            _run_ffmpeg(combined, timeout=_scaled_timeout(duration))
            if _acceptable(out_path, duration):
                return
        except Exception as exc:
            logger.info("combined ffmpeg trim failed: %s", exc)

    single = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
              *(clip.get("stream_headers_args") or []),
              "-ss", str(start), "-i", primary, "-t", str(duration),
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
              "-c:a", "aac", "-b:a", "128k",
              "-movflags", "+faststart", str(out_path)]
    try:
        _run_ffmpeg(single, timeout=_scaled_timeout(duration))
        if _acceptable(out_path, duration):
            return
    except Exception as exc:
        logger.info("single ffmpeg trim failed: %s", exc)
    raise RuntimeError("ffmpeg stream trims produced invalid output")


def _http_total(url: str) -> int | None:
    """Total stream size from a tiny HTTP Range probe (Content-Range)."""
    for attempt in range(4):
        try:
            resp = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60,
                                stream=True)
            resp.close()
            cr = resp.headers.get("Content-Range") or ""
            match = re.search(r"/(\d+)$", cr.strip())
            return int(match.group(1)) if match else None
        except Exception as exc:
            if attempt < 3:
                logger.info("size probe attempt %d/4 failed: %s; retrying",
                            attempt + 1, str(exc)[:120])
                time.sleep(5 * (2 ** attempt))
    return None


def _http_get_range(url: str, b0: int, b1: int, dest: Path) -> None:
    for attempt in range(4):
        try:
            resp = requests.get(url, headers={"Range": f"bytes={b0}-{b1}"},
                                timeout=180, stream=True)
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                resp.close()
                raise RuntimeError(f"HTTP {resp.status_code} for range {b0}-{b1}")
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
            return
        except Exception as exc:
            if attempt < 3:
                logger.info("range fetch %d-%d attempt %d/4 failed: %s; retrying",
                            b0, b1, attempt + 1, str(exc)[:120])
                time.sleep(5 * (2 ** attempt))
            else:
                raise


def _range_section_download(clip: dict, start: float, duration: float, out_path: Path) -> None:
    """Fetch only the bytes covering the clip window of a long DASH VOD.

    googlevideo honours HTTP Range (206). A mid-file slice of a fragmented
    MP4 stream is only decodable when prefixed by the stream's init segment
    (ftyp+moov) and aligned to a moof (fragment) boundary. Byte positions are
    NOT proportional to media time in VBR streams (deep seeks drift by
    hundreds of seconds), so the correct slice offset is found by probing the
    real first-frame timestamp of small byte-range samples and iterating the
    estimate. The fetched window is then verified by its actual timestamps and
    extended if needed -- the full video is never downloaded.
    """
    if yt_dlp is None:
        raise RuntimeError("yt-dlp not available")
    webpage = clip.get("webpage_url")
    if not webpage:
        raise RuntimeError("No webpage URL for range export")
    with yt_dlp.YoutubeDL(with_cookies({"quiet": True, "no_warnings": True,
                                        "skip_download": True})) as ydl:
        info = ydl.extract_info(webpage, download=False)
    total_duration = info.get("duration")
    if not total_duration or total_duration <= 0:
        raise RuntimeError("VOD duration unavailable for range export")

    def _pick_formats(info: dict) -> tuple[dict | None, dict | None]:
        formats = info.get("formats") or []
        vpool = [f for f in formats if f.get("url")
                 and (f.get("vcodec") or "") not in ("", "none")
                 and (f.get("acodec") or "") in ("", "none")]
        apool = [f for f in formats if f.get("url")
                 and (f.get("acodec") or "") not in ("", "none")
                 and (f.get("vcodec") or "") in ("", "none")]
        v = max(vpool, key=lambda f: (f.get("height") or 0, f.get("tbr") or 0),
                default=None)
        a = max(apool, key=lambda f: (f.get("abr") or 0), default=None)
        return v, a

    def _refresh_formats() -> tuple[dict | None, dict | None]:
        with yt_dlp.YoutubeDL(with_cookies({"quiet": True, "no_warnings": True,
                                            "skip_download": True})) as ydl:
            return _pick_formats(ydl.extract_info(webpage, download=False))

    vfmt, afmt = _pick_formats(info)
    if not vfmt:
        raise RuntimeError("No separate video-only stream for range export")

    def _first_pts(url: str, b0: int, total: int, prefix: bytes, tmp: Path,
                   label: str) -> float | None:
        for size in (2 << 20, 4 << 20, 8 << 20):
            b1 = min(total - 1, b0 + size)
            if b1 <= b0:
                break
            probe_path = tmp / f"{label}_probe.bin"
            _http_get_range(url, b0, b1, probe_path)
            body = probe_path.read_bytes()
            m = body.find(b"moof")
            if m < 0:
                continue
            merged = tmp / f"{label}_probe_merged.bin"
            merged.write_bytes(prefix + body[m:])
            return _probe_stream_start(merged, f"{label}:0")
        return None

    def _locate_offset(url: str, total: int, rate: float, target: float,
                       prefix: bytes, tmp: Path, label: str) -> tuple[int, float]:
        b = min(total - 1, max(0, int(rate * target)))
        found: tuple[int, float] | None = None
        for _ in range(8):
            t = _first_pts(url, b, total, prefix, tmp, label)
            if t is None or t <= 0:
                break
            found = (b, t)
            if abs(t - target) <= 6.0:
                break
            b = min(total - 1, max(0, int(b * target / t)))
        return found if found is not None else (b, target)

    def _fetch_window(url: str, label: str, tmp: Path) -> tuple[Path, float]:
        total = _http_total(url)
        if not total or total < 1024:
            raise RuntimeError(f"No range-addressable size for {label} stream")
        rate = total / total_duration
        init = tmp / f"{label}_init.bin"
        if not init.exists() or init.stat().st_size < config.RANGE_INIT_BYTES // 4:
            _http_get_range(url, 0, min(config.RANGE_INIT_BYTES - 1, total - 1), init)
        data = init.read_bytes()
        moof = data.find(b"moof")
        if moof < 0:
            raise RuntimeError(f"No init fragment in {label} stream")
        prefix = data[:moof]
        target = start - config.RANGE_MARGIN_SECONDS
        b0, t0 = _locate_offset(url, total, rate, target, prefix, tmp, label)
        local_rate = (b0 / t0) if t0 > 0 else rate
        need = int((duration + 2 * config.RANGE_MARGIN_SECONDS)
                   * local_rate * 1.3)
        need = max(need, 4 * 1024 * 1024)
        if need > config.RANGE_MAX_BYTES:
            raise RuntimeError(
                f"{label} window too large ({need / 1024 / 1024:.0f} MiB)")
        b1 = min(total - 1, b0 + need)
        slice_path = tmp / f"{label}_slice.bin"
        _http_get_range(url, b0, b1, slice_path)
        body = slice_path.read_bytes()
        m = body.find(b"moof")
        if m < 0:
            raise RuntimeError(f"No fragment in {label} slice")
        merged = tmp / f"{label}_merged.bin"
        merged.write_bytes(prefix + body[m:])
        start_actual = _probe_stream_start(merged, f"{label}:0")
        if not start_actual:
            start_actual = t0
        if abs(start_actual - start) > config.RANGE_MARGIN_SECONDS + 10:
            raise RuntimeError(
                f"{label} slice start {start_actual:.0f}s too far from clip "
                f"start {start:.0f}s (fragment locate failed)")
        for _ in range(3):
            dur = _probe_duration(merged)
            if dur is None:
                break
            if start_actual + dur >= start + duration + 1:
                break
            grow = (start + duration + 1 - (start_actual + dur)) * local_rate * 2
            b1 = min(total - 1, int(b1 + grow))
            _http_get_range(url, b0, b1, slice_path)
            body = slice_path.read_bytes()
            m = body.find(b"moof")
            merged.write_bytes(prefix + body[m:])
        return merged, start_actual

    with tempfile.TemporaryDirectory(prefix="scrange_") as tmpdir:
        tmp = Path(tmpdir)
        try:
            vm, t0v = _fetch_window(vfmt["url"], "v", tmp)
        except Exception as exc:
            nv, na = _refresh_formats()
            if not nv:
                raise
            vfmt, afmt = nv, na or afmt
            logger.info("refreshed stream URLs after range failure: %s", str(exc)[:120])
            vm, t0v = _fetch_window(vfmt["url"], "v", tmp)
        if afmt:
            try:
                am, t0a = _fetch_window(afmt["url"], "a", tmp)
            except Exception as exc:
                nv, na = _refresh_formats()
                if na:
                    afmt = na
                    logger.info("refreshed audio URL after range failure: %s",
                                str(exc)[:120])
                am, t0a = _fetch_window(afmt["url"], "a", tmp)
            logger.info("range slice starts v=%.3f a=%.3f (delta=%.3fs)",
                        t0v, t0a, t0a - t0v)
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                   "-err_detect", "ignore_err",
                   "-ss", f"{start - t0v:.3f}", "-i", str(vm),
                   "-ss", f"{start - t0a:.3f}", "-i", str(am),
                   "-t", str(duration),
                   "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                   "-c:a", "aac", "-b:a", "128k",
                   "-movflags", "+faststart", str(out_path)]
        else:
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                   "-err_detect", "ignore_err",
                   "-ss", f"{start - t0v:.3f}", "-i", str(vm),
                   "-t", str(duration), "-an",
                   "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                   "-movflags", "+faststart", str(out_path)]
        _run_ffmpeg(cmd, timeout=_scaled_timeout(duration))
        if not _acceptable(out_path, duration):
            raise RuntimeError("range-exported clip failed validation")


def _trim_kick_hls(clip: dict, start: float, duration: float, out_path: Path) -> None:
    """Export a Kick window by input-seeking the HLS master (no full download).

    ffmpeg's pre-input ``-ss`` on an HLS stream only fetches the segments that
    cover the window, so a 10h VOD is never downloaded.  Kick HLS is not
    byte-range-addressable, so the yt-dlp section / byte-range paths cannot
    apply here.
    """
    hls_url = clip.get("hls_url")
    if not hls_url:
        raise RuntimeError("No HLS URL for kick export")
    _run_ffmpeg(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", str(start), "-i", hls_url, "-t", str(duration),
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-c:a", "aac", "-b:a", "128k",
                 "-movflags", "+faststart", str(out_path)],
                timeout=_scaled_timeout(duration))
    if not _acceptable(out_path, duration):
        raise RuntimeError("kick HLS trim produced invalid output")


def _trim_twitch_vod(clip: dict, start: float, duration: float, out_path: Path) -> None:
    """Export a Twitch VOD window by input-seeking the signed HLS master.

    yt-dlp's Twitch extractor intermittently fails at export time with
    "[twitch:stream] videos: videos does not exist"; the signed HLS streams
    captured at analysis time still work, and can be re-resolved fresh here.
    ffmpeg's pre-input ``-ss`` on HLS only fetches the segments covering the
    window, so the (possibly 10h+) VOD is never downloaded.
    """
    from modules import twitch_source
    from modules.youtube_search import get_best_stream
    from modules.youtube_search import stream_headers as _stream_headers

    info = None
    try:
        info = twitch_source.get_vod_info(clip)
    except Exception as exc:
        logger.info("Twitch VOD fresh re-resolve failed for export: %s", exc)

    video_url = clip.get("export_stream_url") or clip.get("stream_url")
    audio_url = clip.get("audio_url")
    headers = list(clip.get("export_headers") or [])
    if info:
        try:
            fresh_video = get_best_stream(info)
            if fresh_video:
                video_url = fresh_video
        except Exception:
            pass
        try:
            fresh_audio = twitch_source.get_audio_stream(info)
            if fresh_audio:
                audio_url = fresh_audio
        except Exception:
            pass
        headers = _stream_headers(info) or headers
    if not video_url:
        raise RuntimeError("No Twitch HLS stream available for export")

    base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *(headers or [])]
    if video_url and audio_url and audio_url != video_url:
        combined = base + ["-ss", str(start), "-i", video_url,
                           "-ss", str(start), "-i", audio_url, "-t", str(duration),
                           "-map", "0:v:0", "-map", "1:a:0",
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                           "-c:a", "aac", "-b:a", "128k",
                           "-movflags", "+faststart", str(out_path)]
        try:
            _run_ffmpeg(combined, timeout=_scaled_timeout(duration))
            if _acceptable(out_path, duration):
                return
        except Exception as exc:
            logger.info("Twitch VOD combined HLS trim failed: %s", exc)
    single = base + ["-ss", str(start), "-i", video_url, "-t", str(duration),
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                     "-c:a", "aac", "-b:a", "128k",
                     "-movflags", "+faststart", str(out_path)]
    try:
        _run_ffmpeg(single, timeout=_scaled_timeout(duration))
        if _acceptable(out_path, duration):
            return
    except Exception as exc:
        logger.info("Twitch VOD single HLS trim failed: %s", exc)
    raise RuntimeError("Twitch VOD HLS trims produced invalid output")


def _find_local_copy(video_id: str) -> Path | None:
    """Find an existing local download of a video across all scan directories."""
    # Check shared download cache first (fastest path)
    shared = config.FRAMES_DIR / "_shared_downloads" / f"{video_id}.mp4"
    if shared.exists() and shared.stat().st_size > 1024 * 1024:
        return shared
    # Check all known scan directories
    scan_prefixes = ("vodscan_", "sentence_", "vmoment_", "twvod_")
    for d in config.FRAMES_DIR.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if any(name.startswith(p) and video_id in name for p in scan_prefixes):
            candidate = d / f"{video_id}_local.mp4"
            if candidate.exists() and candidate.stat().st_size > 1024 * 1024:
                return candidate
    return None


def _trim_local_fallback(clip: dict, start: float, duration: float, out_path: Path) -> None:
    """Trim the window from an existing on-disk copy of the source VOD.

    YouTube 403s anonymous deep-seeks and DASH section downloads, so when a
    vod/sentence scan already materialized a local MP4 (downloaded once via
    in-process yt-dlp), the window is trimmed from that file instead.
    """
    video_id = clip.get("video_id") or clip.get("id") or ""
    if not video_id:
        raise RuntimeError("No video_id for local fallback trim")
    local = _find_local_copy(video_id)
    if not local:
        raise RuntimeError("No local scan copy for %s" % video_id)
    logger.info("Trimming %s@%.0fs from local %s", video_id, start, local)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-ss", str(start), "-i", str(local), "-t", str(duration),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-c:a", "aac", "-b:a", "128k",
           "-movflags", "+faststart", str(out_path)]
    _run_ffmpeg(cmd, timeout=_scaled_timeout(duration))
    if not _acceptable(out_path, duration):
        raise RuntimeError("local trim output invalid")


def _download_segment(clip: dict, start: float, duration: float, out_path: Path) -> None:
    """Export a clip without ever fetching the whole VOD.

    Windows (start > 0) use range-seekable yt-dlp section extraction first,
    then a direct byte-range download (init segment + fragment-aligned slice)
    for VODs whose streams are not section-downloadable, so only the bytes
    covering the window are ever fetched.  Whole-clip sources (start == 0,
    short videos) may fall back to a direct stream trim, since the entire
    short file is the clip itself.
    """
    webpage = clip.get("webpage_url")
    last_error: Exception | None = None
    if clip.get("source") == "kick":
        try:
            _trim_kick_hls(clip, start, duration, out_path)
            if _acceptable(out_path, duration):
                return
            last_error = RuntimeError("kick HLS output invalid")
        except Exception as exc:
            last_error = exc
            logger.info("kick HLS export failed: %s", exc)
        raise last_error
    is_twitch_vod = (clip.get("source") == "twitch"
                     and webpage
                     and "/videos/" in webpage)
    if is_twitch_vod:
        try:
            _trim_twitch_vod(clip, start, duration, out_path)
            if _acceptable(out_path, duration):
                return
            last_error = RuntimeError("Twitch VOD HLS output invalid")
        except Exception as exc:
            last_error = exc
            logger.info("Twitch VOD HLS export failed: %s", exc)
        raise last_error
    # When a local copy already exists from a prior scan (vodscan, vmoment,
    # etc.), try trimming from it FIRST — it's fast (local ffmpeg seek) and
    # avoids the 403 cascade on remote YouTube streams entirely.
    if start > 0.0 and clip.get("source") == "youtube":
        video_id = clip.get("video_id") or ""
        if video_id and _find_local_copy(video_id):
            try:
                _trim_local_fallback(clip, start, duration, out_path)
                if _acceptable(out_path, duration):
                    return
            except Exception as exc:
                last_error = exc
                logger.info("early local trim failed: %s", exc)
    if webpage:
        if start > 0.0:
            # Windows: trim straight from a fresh progressive MP4 source FIRST.
            # ffmpeg input-seeks the moov-at-front stream via HTTP byte-range in
            # <1s and fetches only the window's bytes, so most VOD clips export
            # in seconds instead of the 45s DASH hangs / multi-GB section fetch
            # attempts that used to run first. Falls through on any failure.
            try:
                _trim_direct_streams(clip, start, duration, out_path)
                if _acceptable(out_path, duration):
                    return
                last_error = RuntimeError("direct progressive stream output invalid")
            except Exception as exc:
                last_error = exc
                logger.info("direct progressive stream trim failed: %s", exc)
        try:
            max_bytes = _MAX_WHOLE_CLIP_BYTES if start <= 0.0 else _MAX_SECTION_BYTES
            _trim_with_ytdlp(webpage, start, duration, out_path, max_bytes)
            if _acceptable(out_path, duration):
                return
            last_error = RuntimeError("yt-dlp section output invalid")
        except Exception as exc:
            last_error = exc
            logger.info("yt-dlp section download failed: %s", exc)
    if start > 0.0 and config.RANGE_SECTION_DOWNLOAD:
        for attempt in (1, 2):
            try:
                _range_section_download(clip, start, duration, out_path)
                if _acceptable(out_path, duration):
                    return
                logger.info("range section download failed validation (attempt %d)",
                            attempt)
            except Exception as exc:
                last_error = exc
                logger.info("range section download failed (attempt %d): %s",
                            attempt, exc)
            out_path.unlink(missing_ok=True)
            if attempt == 1 and config.RANGE_SECTION_DOWNLOAD:
                time.sleep(1.0)
    if start <= 0.0:
        try:
            _trim_direct_streams(clip, start, duration, out_path)
            if _acceptable(out_path, duration):
                return
        except Exception as exc:
            last_error = exc
            logger.info("direct stream trim failed: %s", exc)
    if start > 0.0:
        try:
            _trim_local_fallback(clip, start, duration, out_path)
            if _acceptable(out_path, duration):
                return
        except Exception as exc:
            last_error = exc
            logger.info("local file trim fallback failed: %s", exc)
    if last_error is not None:
        raise last_error
    raise RuntimeError("No exportable source for clip")


def _load_registry() -> set[str]:
    path = config.EXPORTED_REGISTRY
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {item for item in data if isinstance(item, str)}
    except Exception:
        pass
    return set()


def _save_registry(entries: set[str]) -> None:
    config.EXPORTED_REGISTRY.write_text(json.dumps(sorted(entries), indent=2), encoding="utf-8")


def _export_dir(query: str, output_root: Path) -> Path:
    output_dir = output_root / _safe_stem(query)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _face_grade(clip: dict, plan: dict, path: Path) -> None:
    """Sample frames of a downloaded clip and grade subject presence on screen.

    Subject presence is the FIRST-PRIORITY selection signal, so it is graded
    (never gated): every downloaded clip gets `face_all` (every script subject
    on screen) and `face_coverage` (distinct subjects seen / total).  All-subject
    clips always outrank partial-subject clips; the score feeds the final pick.
    """
    clip.setdefault("face_all", False)
    clip.setdefault("face_any", False)
    clip.setdefault("face_coverage", 0.0)
    if not config.FACE_VERIFY_ENABLED:
        clip["face_all"] = True
        clip["face_coverage"] = 1.0
        return
    subjects = [s for s in (plan.get("core_entities") or [])]
    if not subjects:
        clip["face_all"] = True
        clip["face_coverage"] = 1.0
        return
    try:
        from modules import face_verify
        refs, _missing = face_verify.refs_for(subjects)
    except Exception as exc:
        logger.info("Face refs unavailable for %s: %s", path, exc)
        return
    if not refs:
        return  # nothing verifiable: grade neutral rather than reject
    frames = face_verify.sample_frames(path, config.FACE_SAMPLE_FRAMES)
    if not frames:
        clip["face_scores"] = {}
        clip["face_unverifiable"] = True
        logger.info("Face grade for %s: UNDECODABLE (no frames)", path)
        return
    scores = face_verify.verify_frames(frames, refs)
    seen = face_verify.subjects_on_screen(scores)
    clip["face_scores"] = {k: round(v, 3) for k, v in scores.items()}
    clip["face_all"] = all(name in seen for name in refs)
    clip["face_any"] = any(name in seen for name in refs)
    clip["face_coverage"] = sum(1 for name in refs if name in seen) / len(refs)
    logger.info("Face grade for %s: all=%s coverage=%.2f (scores=%s)", path,
                clip["face_all"], clip["face_coverage"], clip["face_scores"])


def _pick_key(clip: dict, plan: dict | None = None) -> tuple:
    """Selection rank: aligned with the finalize reorder.

    For concrete_visual plans (visual_event_kind / object_terms):
      vlm_happening -> vlm_score -> keyword_located -> event_visual -> vision
      -> anchor -> llm_found -> llm_moment -> face -> combined

    For generic plans:
      action_coverage -> entity_coverage -> semantic -> proposal
      -> keyword_located -> llm_found -> llm_moment -> anchor -> face -> vlm
      -> combined
    """
    concrete_visual = bool(plan and (plan.get("visual_event_kind") or plan.get("object_terms")))
    if concrete_visual:
        return (
            bool(clip.get("vlm_happening")),
            float(clip.get("vlm_score") or 0.0),
            bool(clip.get("keyword_located")),
            float(clip.get("event_visual_score") or 0.0),
            float(clip.get("vision_score") or 0.0),
            float(clip.get("anchor_sim") or 0.0),
            bool(clip.get("llm_found")),
            float(clip.get("llm_moment_score") or 0.0),
            float(clip.get("face_all", False)),
            float(clip.get("combined_score") or 0.0),
        )
    return (
        float(clip.get("action_coverage") or 0.0),
        float(clip.get("entity_coverage") or 0.0),
        float(clip.get("semantic_score") or 0.0),
        float(clip.get("proposal_score") or 0.0),
        bool(clip.get("keyword_located")),
        bool(clip.get("llm_found")),
        float(clip.get("llm_moment_score") or 0.0),
        float(clip.get("anchor_sim") or 0.0),
        float(clip.get("face_all", False)),
        float(clip.get("vlm_score") or 0.0),
        float(clip.get("combined_score") or 0.0),
    )


def _anchor_ok(clip: dict, path: Path, anchors) -> bool:
    """Sample frames of the DOWNLOADED clip; confirm the event's pixels are there.

    The event's Google-image anchors prove what the moment looks like.  A clip
    whose frames do not CLIP-match them is a wrong event even when its audio
    scored well, so it is rejected.  Returns True when the clip passes (or no
    enforceable anchor set exists); failed clips are flagged ``anchor_unverified``
    for the report (they are never exported as fallback anymore).
    """
    if anchors is None or not getattr(anchors, "strong", False):
        return True
    if not getattr(anchors, "enforceable", False):
        logger.info("Anchor gate for %s: anchors not event-specific; skipping",
                    path)
        clip["anchor_unverified"] = False
        return True
    duration = float(clip.get("duration") or 0)
    count = config.ANCHOR_FRAMES_PER_WINDOW
    if duration <= 0 or count <= 0:
        clip["anchor_unverified"] = True
        return True
    timestamps = [duration * (i + 1) / (count + 1) for i in range(count)]
    frames_dir = config.FRAMES_DIR / ("export_%s_%.0fs" % (clip.get("video_id", "x"),
                                                           float(clip.get("start", 0))))
    try:
        from modules.youtube_search import sample_frames_at
        frames = sample_frames_at(str(path), timestamps, frames_dir)
    except Exception as exc:
        logger.info("Anchor frame sampling failed for %s: %s", path, exc)
        clip["anchor_unverified"] = True
        return True
    if not frames:
        logger.info("Anchor gate for %s: no frames sampled; unverified", path)
        clip["anchor_unverified"] = True
        return True
    sims = anchors.frame_sims([frame["path"] for frame in frames])
    if not sims:
        logger.info("Anchor gate for %s: no embeddings; unverified", path)
        clip["anchor_unverified"] = True
        return True
    anchor_sim = max(sims)
    fraction = sum(1 for s in sims if s + 1e-6 >= config.ANCHOR_SIM_FLOOR) / len(sims)
    clip["anchor_sim"] = anchor_sim
    clip["anchor_fraction"] = round(fraction, 3)
    passes = (fraction >= config.ANCHOR_FRACTION_PASS
              and anchor_sim + 1e-6 >= config.ANCHOR_SIM_FLOOR)
    clip["anchor_unverified"] = not passes
    logger.info("Anchor gate for %s: anchor_sim=%.3f fraction=%.2f (floor=%.2f, "
                "frac_floor=%.2f) -> %s", path, anchor_sim, fraction,
                config.ANCHOR_SIM_FLOOR, config.ANCHOR_FRACTION_PASS,
                "MATCH" if passes else "no-match")
    return passes


def _anchors_depict_event(anchors, probe) -> bool:
    """True when the anchor images themselves CLIP-match the event's scenes.

    Visual-event queries (helicopter, police car, gel blaster, fight) need
    anchors that show the OBJECT, not just the famous person.  Generic media
    images of a person match any clip showing that person and would wrongly
    override a failed scene probe; only anchors that clear the event's scene
    floor on most frames may do so.
    """
    if anchors is None or not getattr(anchors, "strong", False):
        return False
    paths = [e.get("path") for e in (anchors.entries or [])
             if e.get("path") and Path(e["path"]).exists()]
    if not paths:
        return False
    scores = probe.scene_scores(paths)
    if not scores:
        return False
    passed = sum(1 for v in scores.values() if v >= config.EVENT_VISUAL_FLOOR)
    return passed / len(scores) >= config.EVENT_VISUAL_FRACTION


def _probe_audio_corroborates(clip: dict, probe) -> bool:
    """True when the clip's transcript names the visual event in words."""
    try:
        from modules.content_verifier import normalize
    except Exception:
        return False
    text = normalize(clip.get("snippet") or "")
    if not text or not probe.audio:
        return False
    for term in probe.audio:
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", text):
            return True
    return False


def _scene_audio_corroborates(clip: dict, plan: dict) -> bool:
    """True when the clip's audio names the query's object or action words."""
    try:
        from modules.content_verifier import normalize
    except Exception:
        return False
    text = normalize(clip.get("snippet") or "")
    if not text:
        return False
    terms: list[str] = []
    for t in (plan.get("object_terms") or []):
        if str(t).strip():
            terms.append(str(t).lower().strip())
    for t in (plan.get("action_terms") or []):
        if str(t).strip():
            terms.append(str(t).lower().strip())
    for term in terms:
        if re.search(r"\b" + re.escape(term) + r"\b", text):
            return True
    return False


def _visual_verify(clip: dict, path: Path, anchors, plan: dict) -> tuple[str, bool]:
    """Unified visual verification of a DOWNLOADED clip; returns (tier, passed).

    Tiers, strongest first:
      event_probe          the event-kind scene probe matched on the clip pixels
                           (or its audio / event-anchor escape fired).
      general_scene        the query's scene_prompts matched on the pixels.
      general_scene_audio  the general probe missed, but the clip's audio names
                           the object/action and a frame partially shows it.
      anchor               soft Google-image anchor match (non-visual queries).

    The visual check runs for EVERY query: event-kind probe when a visual-event
    kind is recognized, otherwise the query's own general scene prompts (subject
    + action + objects).  A clip is only accepted on its pixels, or its audio
    corroborating the action with a partial visual; there is no face-only or
    unverified fallback tier, so a failed check hard-rejects the clip.  Frames
    are sampled ONCE per clip and shared by every consumer.
    """
    duration = float(clip.get("duration") or 0)
    hard = True  # the visual check decides for EVERY query, not just event kinds
    if duration <= 0:
        return ("general_scene", False)
    count = max(config.EVENT_VISUAL_FRAMES, config.GENERAL_VISUAL_FRAMES)
    timestamps = [duration * (i + 1) / (count + 1) for i in range(count)]

    frames: list[dict] = []
    try:
        frames = event_vision._sample_local(str(path), timestamps)
    except Exception as exc:
        logger.info("Visual-verify sampling failed on %s: %s", path, exc)
    paths = [f["path"] for f in frames]

    kind = event_vision.probe_for(plan)
    if kind.active() and paths:
        clip["event_visual_frames"] = paths
        scores = kind.scene_scores(paths)
        summary = kind.summarize(scores) if scores else {}
        clip["event_visual_best"] = summary.get("best", 0.0)
        clip["event_visual_mean"] = summary.get("mean", 0.0)
        clip["event_visual_fraction"] = summary.get("fraction", 0.0)
        clip["event_visual_score"] = (1.0 if summary.get("present")
                                      else summary.get("fraction", 0.0))
        if summary.get("present"):
            clip["event_visual_verified"] = True
            return ("event_probe", True)
        if (summary.get("best", 0.0) >= config.EVENT_VISUAL_FLOOR
                and _probe_audio_corroborates(clip, kind)):
            clip["event_visual_audio"] = True
            clip["event_visual_verified"] = True
            return ("event_probe", True)
        if (anchors is not None and getattr(anchors, "enforceable", False)
                and _anchors_depict_event(anchors, kind)
                and _anchor_ok(clip, path, anchors)):
            clip["event_visual_anchor"] = True
            clip["event_visual_verified"] = True
            return ("event_probe", True)
        clip["event_visual_verified"] = False

    general = event_vision.general_probe(plan)
    if general.active() and paths:
        clip["visual_frames"] = paths
        scores = general.scene_scores(paths)
        summary = general.summarize(scores) if scores else {}
        clip["general_visual_best"] = summary.get("best", 0.0)
        clip["general_visual_fraction"] = summary.get("fraction", 0.0)
        clip["vision_score"] = (1.0 if summary.get("present")
                                else summary.get("fraction", 0.0))
        if summary.get("present"):
            return ("general_scene", True)
        best = summary.get("best", 0.0)
        if best >= config.GENERAL_VISUAL_FLOOR:
            # Audio corroboration is the easy path, but a genuinely visual moment
            # (an archery hit, a leg grab) is often silent.  Accept the pixels
            # alone when the scene is strongly matched (best >= 0.30) or the
            # window was selected by the pixel scan in the first place.
            if _scene_audio_corroborates(clip, plan):
                return ("general_scene_audio", True)
            if best >= 0.30 or clip.get("visual_zone"):
                return ("general_scene", True)
        return ("general_scene", False)
    return ("general_scene", False)


def export_media(clips: list[dict], plan: dict, output_root: Path,
                 need: int | None = None, anchors=None) -> list[str]:
    """Export up to `need` distinct video clips, subject-presence first.

    Selection is grade-then-pick: every candidate in the pool is downloaded and
    its PIXELS are verified (event-kind scene probe -> general scene probe ->
    audio corroboration) against the plan's scene, and the anchor gate confirms
    the event's images.  Both are hard rejects — there is no ``face_only`` tier
    and no ``anchor_unverified`` fallback: a clip is exported only when its
    frames show the query's moment.  Visual-event queries (police car /
    helicopter / gel blaster / fight) hard-gate on their pixels; unverifiable
    clips are dropped rather than shipped.

    Subject presence is then the FIRST-PRIORITY ranking signal: clips that put
    every script subject on screen fill the whole quota first; the face
    requirement relaxes to fewer subjects ONLY when no all-subject clip exists
    at all, preferring higher subject coverage.  Within a tier, visual truth
    (event probe / vision / anchor) then combined score decide.  Global dedup:
    a (video_id, start) pair is exported at most once across all sentences.
    """
    if not clips:
        return []
    need = config.CLIPS_PER_SENTENCE if need is None else need
    registry = _load_registry()
    output_dir = _export_dir(clips[0]["query"] or "query", output_root)
    exported: list[str] = []
    graded: list[tuple[dict, Path]] = []

    def _grade_clip(clip: dict) -> None:
        """Download one candidate; keep it only when its PIXELS confirm the query.

        The visual and anchor gates are hard rejects (no face_only tier, no
        anchor-unverified fallback).  Subject presence is then GRADED
        (face_all / face_coverage) as the first-priority ranking signal.
        """
        video_id = clip.get("video_id") or "video"
        start = float(clip["start"])
        duration = float(clip.get("duration", config.CLIP_DURATION_SECONDS))
        key = f"{video_id}:{start:.1f}"
        if config.TRANSCRIPT_TRUST_ENABLED:
            trusted, _treason = content_verifier.transcript_verified(clip, plan)
            clip["trusted"] = trusted
        if config.SKIP_EXPORTED and key in registry:
            logger.info("Skipping already-exported clip %s", key)
            return
        path = output_dir / f"stage_{video_id}_{start:.1f}s.mp4"
        try:
            _download_segment(clip, start, duration, path)
        except Exception as exc:
            logger.warning("Could not download clip %s: %s", key, exc)
            path.unlink(missing_ok=True)
            return
        visual = bool(plan.get("visual_event"))
        # VLM event gate: the vision LLM judges the DOWNLOADED clip's real
        # frames.  A high-confidence "moment not happening" verdict is a HARD
        # reject — face match in a random context is not the moment.  No verdict
        # (vision model down / sampling failed) is neutral: the CLIP tiers above
        # already passed, so the clip is kept and ranked below vlm-verified ones.
        # When the VLM CONFIRMS the moment, it overrides the CLIP/anchor hard
        # gates below (the VLM is the authoritative "is the moment happening"
        # signal; a scene probe can false-reject a real celebration).
        verdict = None
        forced_reason = _user_forced(key)
        if config.VLM_VERIFY_ENABLED:
            from modules import event_verifier
            verdict = event_verifier.sample_and_verify(
                clip, plan, path, config.FRAMES_DIR / f"vlm_{video_id}_{start:.0f}")
            clip["vlm_happening"] = bool(verdict.get("happening")) if verdict else None
            clip["vlm_confidence"] = verdict.get("confidence") if verdict else None
            clip["vlm_score"] = verdict.get("score") if verdict else 0.0
            clip["vlm_reason"] = verdict.get("reason") if verdict else None
            if event_verifier.reject_verdict(verdict) and not forced_reason \
                    and not clip.get("trusted"):
                logger.info("VLM event gate rejected %s (%s); dropping",
                            key, verdict.get("reason", "not happening"))
                path.unlink(missing_ok=True)
                return
            if forced_reason:
                logger.info("VLM event gate bypassed for forced moment %s (%s)",
                            key, forced_reason)
            elif clip.get("trusted"):
                logger.info("VLM event gate bypassed for transcript-verified "
                            "moment %s", key)
        vlm_confirmed = (bool(verdict and verdict.get("happening"))
                         or bool(forced_reason) or bool(clip.get("trusted")))
        tier, passed = _visual_verify(clip, path, anchors, plan)
        clip["verification_tier"] = tier
        if not vlm_confirmed:
            if not passed:
                logger.info("Visual gate rejected %s (%s); dropping", key, tier)
                path.unlink(missing_ok=True)
                return
            if not visual and not _anchor_ok(clip, path, anchors):
                # Visual-event queries are anchor-gated inside _visual_verify
                # (_anchors_depict_event escape); non-visual queries gate here.
                logger.info("Anchor gate rejected %s; dropping", key)
                path.unlink(missing_ok=True)
                return
        _face_grade(clip, plan, path)
        graded.append((clip, path))

    # Download-and-grade until the quota can be filled with VLM-confirmed clips
    # (the vision LLM must see enough candidates to confirm the right windows).
    # A transcript-perfect moment (action_coverage full) is NEVER starved by the
    # quota break: even when enough clips are already VLM-confirmed, ungraded
    # full-coverage candidates are still downloaded and graded so the exact
    # spoken moment can displace a look-alike that merely has chase pixels.
    # When the VLM is unavailable the whole pool is graded and face-presence
    # decides, exactly as before.
    for clip in clips:
        if (sum(1 for c, _ in graded if c.get("vlm_happening")) >= need
                and not any(
                    float(c.get("action_coverage") or 0.0) >= config.TRANSCRIPT_MOMENT_COVERAGE
                    and not c.get("_export_attempted") for c in clips)):
            break
        clip["_export_attempted"] = True
        _grade_clip(clip)

    # VLM-confirmed clips (the vision LLM saw the moment actually happen) fill
    # the quota first; only when none exist does the pool fall back to subject
    # presence on the pixels.  Within a tier, the new _pick_key still lets the
    # VLM verdict outrank a plain face match.
    confirmed = [g for g in graded if g[0].get("vlm_happening")]
    trusted_pool = [g for g in graded if g[0].get("trusted")]
    if confirmed:
        ranked = sorted(confirmed, key=lambda g: _pick_key(g[0], plan), reverse=True)
    else:
        ranked = sorted([g for g in graded if not g[0].get("trusted")],
                        key=lambda g: _pick_key(g[0], plan), reverse=True)
    # Face gate for NAMED-subject queries: a clip that only reached the weak
    # general_scene tier, whose frames were actually measured and matched NO
    # script subject, and that the VLM did not confirm is a "random vid" (e.g.
    # an off-moment crowd shot).  Never export that as a subject clip — export
    # a face-present / VLM-confirmed / event-probe alternative instead.
    named_subjects = bool(plan.get("core_entities")) and not plan.get("subjects_unknown")
    if named_subjects and config.FACE_VERIFY_ENABLED:
        def _subject_present(g) -> bool:
            clip, _ = g
            return (bool(clip.get("face_all"))
                    or float(clip.get("face_coverage") or 0.0) > 0.0
                    or bool(clip.get("vlm_happening"))
                    or bool(clip.get("trusted"))
                    or clip.get("verification_tier") == "event_probe")
        have_present = any(_subject_present(g) for g in ranked)
        if have_present:
            kept = [g for g in ranked
                    if (_subject_present(g)
                        or (g[0].get("verification_tier") != "general_scene")
                        or not g[0].get("face_scores"))]
            if len(kept) != len(ranked):
                logger.info("Face gate dropped %d face-less general_scene "
                            "clip(s); face-present alternative exists",
                            len(ranked) - len(kept))
                ranked = kept
    nsubj = max(1, len(plan.get("core_entities") or []))
    trust_quota = (max(0, int(config.TRANSCRIPT_TRUST_QUOTA or 0))
                   if config.TRANSCRIPT_TRUST_ENABLED else 0)
    cap = need + trust_quota
    exported_keys: set[str] = set()

    def _export_one(g: tuple) -> None:
        clip, path = g
        video_id = clip.get("video_id") or "video"
        start = float(clip["start"])
        key = f"{video_id}:{start:.1f}"
        if key in exported_keys:
            return
        final_path = output_dir / f"{len(exported) + 1:02d}_{video_id}_{start:.1f}s.mp4"
        try:
            if path != final_path:
                shutil.move(str(path), str(final_path))
                path = final_path
        except Exception:
            pass
        clip["export_path"] = str(path)
        exported.append(str(path))
        exported_keys.add(key)
        registry.add(key)
        if clip.get("face_all"):
            presence = "all subjects"
        else:
            presence = ("%.0f/%d subjects" %
                        (float(clip.get("face_coverage") or 0.0) * nsubj, nsubj))
        logger.info("Exported #%d (%s, %s): %s", len(exported), presence,
                    clip.get("verification_tier", "pending"), path)

    # Pass 1: pixel/face-confirmed clips fill the base quota first, so a
    # transcript-trusted clip can never crowd out a visual match.  Pass 2 gives
    # transcript-trusted clips their OWN extra slots (TRANSCRIPT_TRUST_QUOTA).
    for g in ranked:
        if len(exported) >= need:
            break
        _export_one(g)
    for g in sorted(trusted_pool, key=lambda g: _pick_key(g[0], plan), reverse=True):
        if len(exported) >= cap:
            break
        _export_one(g)
    for clip, path in graded:
        if not any(p == path for p in exported):
            path.unlink(missing_ok=True)
    _save_registry(registry)
    return exported


def export_images(images: list[dict], plan: dict, output_root: Path,
                  need: int | None = None) -> list[str]:
    """Copy up to `need` verified Google images, globally deduped by content.

    Near-duplicate photos (the same shot at slightly different crops) are also
    skipped so a query's image set is three DISTINCT moments, not one photo
    three times under different cache names.
    """
    if not images:
        return []
    need = config.IMAGES_PER_SENTENCE if need is None else need
    registry = _load_registry()
    output_dir = _export_dir(plan.get("sentence") or "query", output_root)
    exported: list[str] = []
    accepted_vecs: list = []

    def _vector(path) -> list | None:
        try:
            from modules import vision_ranker
            vectors = vision_ranker.image_features([str(path)])
            return vectors[0] if vectors else None
        except Exception:
            return None

    def _near_dup(vector) -> bool:
        if not accepted_vecs:
            return False
        try:
            import numpy as np
            return max(float(np.dot(vector, v)) for v in accepted_vecs) \
                >= config.IMAGE_NEAR_DUP_COSINE
        except Exception:
            return False

    for image in images:
        if len(exported) >= need:
            break
        try:
            src = Path(image["image_path"])
            if not src.exists():
                continue
            digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
        except Exception as exc:
            logger.info("Could not hash image %s: %s", image.get("image_path"), exc)
            continue
        key = f"img:{digest}"
        if config.SKIP_EXPORTED and key in registry:
            logger.info("Skipping already-exported image %s", key)
            continue
        vector = _vector(src)
        if vector is not None and _near_dup(vector):
            logger.info("Skipping near-duplicate image %s", src)
            continue
        path = output_dir / f"{len(exported) + 1:02d}_{digest[:10]}.jpg"
        try:
            shutil.copy2(src, path)
            image["export_path"] = str(path)
            exported.append(str(path))
            registry.add(key)
            if vector is not None:
                accepted_vecs.append(vector)
            logger.info("Exported image #%d: %s", len(exported), path)
        except Exception as exc:
            logger.warning("Could not export image #%d: %s", len(exported) + 1, exc)
    _save_registry(registry)
    return exported
