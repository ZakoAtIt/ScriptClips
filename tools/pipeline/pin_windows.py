import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from _util import PIPELINE, load_json, log, probe, query_for_dir, save_json


def envelope(path, hop_s=0.05):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
         "-f", "s16le", "-"], capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(16000 * hop_s)
    n = len(x) // hop
    if n == 0:
        return np.zeros(0), hop_s
    e = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    return e, hop_s


def bursts(e, hop_s, thresh_db=-35.0):
    ref = float(e.max()) if len(e) else 0.0
    th = ref * (10 ** (thresh_db / 20))
    on = e > th
    segs = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if (j - i) * hop_s >= 0.25:
                peak_t = (i + int(np.argmax(e[i:j]))) * hop_s
                segs.append({
                    "start": round(i * hop_s, 2),
                    "end": round(j * hop_s, 2),
                    "peak": round(peak_t, 2),
                    "peak_rms": round(float(e[i:j].max()), 3),
                })
            i = j
        else:
            i += 1
    return segs


def pad_clamp(start, end, dur, min_win, max_win):
    start = max(0.0, start)
    end = min(dur, end)
    if end - start < min_win:
        need = min_win - (end - start)
        start = max(0.0, start - need / 2)
        end = min(dur, end + need / 2)
        if end - start < min_win:
            end = min(dur, start + min_win)
    if end - start > max_win:
        end = start + max_win
    return round(start, 2), round(end, 2)


def overlaps(w, s, e):
    return not (w["end"] <= s or w["start"] >= e)


def propose_windows(segments, burst_list, kws, dur, min_win, max_win):
    wins = []
    for seg in segments:
        text = seg.get("text", "").lower()
        hit = next((k for k in kws if k.lower() in text), None)
        if not hit:
            continue
        s, e = pad_clamp(seg["start"] - 2, seg["end"] + 2, dur, min_win, max_win)
        snippet = seg["text"].strip()[:60]
        wins.append({"start": s, "end": e, "why": f"keyword '{hit}': \"{snippet}\""})
    if not wins and segments:
        best = max(segments, key=lambda s: s["end"] - s["start"])
        s, e = pad_clamp(best["start"] - 2, best["end"] + 2, dur, min_win, max_win)
        wins.append({"start": s, "end": e,
                     "why": f"longest speech segment \"{best['text'].strip()[:50]}\""})
    merged = []
    for w in sorted(wins, key=lambda x: x["start"]):
        if merged and w["start"] <= merged[-1]["end"]:
            if w["end"] > merged[-1]["end"]:
                merged[-1]["end"] = w["end"]
            merged[-1]["why"] += f" | {w['why']}"
        else:
            merged.append(dict(w))
    for b in sorted(burst_list, key=lambda x: -x["peak_rms"])[:3]:
        if any(overlaps(m, b["start"], b["end"]) for m in merged):
            continue
        s, e = pad_clamp(b["peak"] - min_win * 0.4, b["end"] + 1.0, dur, min_win, max_win)
        merged.append({"start": s, "end": e,
                       "why": f"rms burst peak@{b['peak']}s rms={b['peak_rms']}"})
    return merged[:6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qdir")
    ap.add_argument("--keywords", nargs="*", default=[])
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--min-win", type=float, default=6.0)
    ap.add_argument("--max-win", type=float, default=30.0)
    a = ap.parse_args()

    p = Path(a.qdir)
    qdir = p if p.is_absolute() else PIPELINE / p.name
    media = qdir / "media"
    cands = sorted(media.glob("candidate*.mp4"))
    if not cands:
        print(f"no candidates in {media}")
        raise SystemExit(1)

    meta = query_for_dir(qdir)
    kws = a.keywords or [w for w in meta["query"].lower().split() if len(w) > 3]
    manifest = load_json(qdir / "media_manifest.json", {"downloaded": {}})
    by_file = {v.get("file"): v for v in manifest.get("downloaded", {}).values()}

    from faster_whisper import WhisperModel
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    tmp = Path(tempfile.gettempdir())

    transcripts = {}
    cand_entries = []
    for f in cands:
        name = f.stem
        info = probe(f)
        dur = info["duration"]
        print(f"\n=== {name} ({dur:.1f}s {info['video']}/{info['audio']}) ===")
        if not info["exists"] or dur < 1:
            print("[SKIP] unusable file")
            continue

        wav = tmp / f"pin_{name}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(f),
                        "-ac", "1", "-ar", "16000", str(wav)], check=True)
        segments_out = []
        seg_iter, _ = model.transcribe(str(wav), vad_filter=True)
        for s in seg_iter:
            txt = s.text.strip()
            if txt:
                segments_out.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": txt})

        e, hs = envelope(str(f))
        burst_list = bursts(e, hs)

        transcripts[name] = {"file": str(f), "duration": round(dur, 2),
                             "segments": segments_out, "bursts": burst_list}

        for seg in segments_out:
            print(f"  [{seg['start']:7.2f}-{seg['end']:7.2f}] {seg['text']}")
        print(f"  ({len(burst_list)} rms bursts)")

        wins = propose_windows(segments_out, burst_list, kws, dur, a.min_win, a.max_win)
        kw_hit = any(k.lower() in " ".join(s["text"] for s in segments_out).lower() for k in kws)
        conf = 0.55
        if kw_hit:
            conf += 0.25
        if wins:
            conf += 0.10
        conf = min(0.9, conf)
        m = by_file.get(str(f), {})
        cand_entries.append({
            "file": str(f),
            "id": m.get("id", ""),
            "title": m.get("title", ""),
            "channel": m.get("channel", ""),
            "confidence": round(conf, 2),
            "moment_present": bool(wins),
            "windows": wins,
            "notes": "PROVISIONAL (script heuristic) - agent must read transcript and finalize",
        })
        for w in wins:
            print(f"  -> window {w['start']}-{w['end']} | {w['why']}")

        if a.frames > 0:
            fdir = qdir / "frames" / name
            fdir.mkdir(parents=True, exist_ok=True)
            fps = a.frames / max(dur, 0.1)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(f),
                            "-vf", f"fps={fps:.5f}", str(fdir / "f%03d.jpg")], check=True)
            print(f"  frames -> {fdir}")

    save_json(qdir / "transcripts.json", transcripts)

    final_path = qdir / "analysis.json"
    out_path = analysis_target(final_path)
    payload = {
        "query": meta["query"],
        "kws": kws,
        "provisional": True,
        "candidates": cand_entries,
        "missing_matches": [],
    }
    save_json(out_path, payload)
    print(f"\nwrote {out_path.name}" + (" (analysis.json already existed - agent version preserved)" if out_path != final_path else ""))
    log(f"pin {qdir.name}: {len(cand_entries)} candidates transcribed, {sum(len(c['windows']) for c in cand_entries)} provisional windows")


def analysis_target(final_path):
    if final_path.exists():
        try:
            import json as _json
            data = _json.loads(final_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and not data.get("provisional"):
                return final_path.with_name("analysis_provisional.json")
        except Exception:
            pass
    return final_path.with_name("analysis_provisional.json")


if __name__ == "__main__":
    main()
