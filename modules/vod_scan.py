"""Visual localization of a visual event inside a long VOD.

Captions rarely say "a helicopter is landing" or "here comes the police car",
so the transcript-based window proposers miss exactly the moments a visual-event
query needs.  When a video's captions produced no candidates, this samples the
video on a coarse time grid, CLIP-scores every frame against the event's scene
prompts, and returns the strongest contiguous zones as candidate windows.
"""
import logging

import config

logger = logging.getLogger(__name__)


def localize(stream_url: str, extra_args: list[str], duration: float,
             probe, video_id: str, window_seconds: int,
             max_windows: int, max_frames: int = 220,
             min_stride: int = 15, floor: float | None = None) -> list[dict]:
    """Return candidate window dicts for the strongest scene zones.

    Two-pass scan: a coarse grid over the whole video locates hit regions, then
    each region is re-sampled at a finer stride so a brief moment inside a large
    hit region is pinpointed instead of smeared over the coarse stride.

    Each window has start/duration/best/mean/fraction/evidence.  Empty list when
    nothing visual is found or sampling fails (callers degrade to audio-only).
    """
    floor = config.EVENT_VISUAL_FLOOR if floor is None else floor
    if not probe or not probe.active() or duration <= 0:
        return []
    out_dir = config.FRAMES_DIR / ("vodscan_%s" % video_id)

    def _sample(timestamps: list[int]) -> list[dict]:
        try:
            from modules.youtube_search import sample_frames_at
            return sample_frames_at(stream_url, timestamps, out_dir, extra_args,
                                    video_id=video_id)
        except Exception as exc:
            logger.info("vod_scan sample failed for %s: %s", video_id, exc)
            return []

    def _score_rows(frames: list[dict]) -> list[tuple[float, float]]:
        if not frames:
            return []
        scores = probe.scene_scores([f["path"] for f in frames])
        rows = []
        for frame in frames:
            score = scores.get(frame["path"])
            if score is not None:
                rows.append((float(frame["timestamp"]), float(score)))
        rows.sort(key=lambda r: r[0])
        return rows

    def _zones(rows: list[tuple[float, float]], gap: float) -> list[list[tuple[float, float]]]:
        zones: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        for ts, score in rows:
            if score >= floor:
                if current and ts - current[-1][0] > gap:
                    zones.append(current)
                    current = []
                current.append((ts, score))
        if current:
            zones.append(current)
        return zones

    stride = max(min_stride, int(duration // max_frames))
    if stride < 1:
        stride = 1
    gap = max(window_seconds * 2.0, stride * 2.0)
    rows = _score_rows(_sample(list(range(0, int(duration), stride))))
    if not rows:
        return []
    coarse_zones = _zones(rows, gap)

    # Fine pass: zoom each coarse zone at (up to) the minimum stride, with a
    # bounded per-zone frame budget so remote seeks stay cheap overall.
    if coarse_zones:
        per_zone = max(8, max_frames // max(1, len(coarse_zones)))
        fine_rows: list[tuple[float, float]] = []
        fine_stride = max(2, min_stride)
        for zone in coarse_zones:
            lo = max(0, int(zone[0][0]) - window_seconds)
            hi = min(int(duration), int(zone[-1][0]) + window_seconds)
            fine_ts = list(range(lo, hi + 1, fine_stride))
            if len(fine_ts) > per_zone:
                step = len(fine_ts) / per_zone
                fine_ts = [fine_ts[int(i * step)] for i in range(per_zone)]
            fine_rows.extend(_score_rows(_sample(fine_ts)))
        seen_ts: set[int] = set()
        merged_rows: list[tuple[float, float]] = []
        for r in sorted(rows + fine_rows, key=lambda r: r[0]):
            if int(r[0]) in seen_ts:
                continue
            seen_ts.add(int(r[0]))
            merged_rows.append(r)
        rows = merged_rows

    zones = _zones(rows, gap)
    windows: list[dict] = []
    for zone in zones:
        ts0 = zone[0][0]
        ts1 = zone[-1][0]
        start = max(0.0, ts0 - window_seconds * 0.4)
        if ts1 - start < window_seconds * 0.5:
            start = max(0.0, ts1 - window_seconds * 0.8)
        best = max(score for _, score in zone)
        mean = sum(score for _, score in zone) / len(zone)
        span = max(1, int((ts1 - ts0) / stride) + 1)
        fraction = min(1.0, len(zone) / span)
        windows.append({
            "start": start,
            "duration": float(window_seconds),
            "best": round(best, 3),
            "mean": round(mean, 3),
            "fraction": round(fraction, 3),
            "evidence": (f"[visual scan] {probe.kind or 'scene'} scene present "
                         f"(best={best:.2f} mean={mean:.2f})"),
        })
    windows.sort(key=lambda w: w["mean"], reverse=True)
    logger.info("vod_scan %s: %d zone(s) -> %d window(s)", video_id,
                len(zones), len(windows))
    return windows[:max_windows]
