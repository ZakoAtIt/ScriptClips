import argparse
import json
import subprocess
from pathlib import Path

from _util import CLIPS, PIPELINE, REPO, log, probe


def resolve_src(src):
    p = Path(src)
    if p.is_absolute():
        return p if p.exists() else None
    for base in (REPO, PIPELINE, Path.cwd()):
        cand = base / src
        if cand.exists():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", help="path to plan JSON: array of cuts or {\"cuts\": [...]}")
    a = ap.parse_args()

    data = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    cuts = data["cuts"] if isinstance(data, dict) else data

    CLIPS.mkdir(parents=True, exist_ok=True)
    results = []
    for i, c in enumerate(cuts, 1):
        out = Path(c["out"])
        out_path = out if out.is_absolute() else CLIPS / out
        src = resolve_src(c["src"])
        start = float(c["start"])
        dur = float(c["dur"])
        conf = c.get("conf")
        why = c.get("why", "")
        tag = f"[{i}/{len(cuts)}] {out_path.name}"

        if src is None:
            print(f"FAIL {tag} <- missing source: {c['src']}")
            results.append((out_path.name, False, 0.0))
            continue

        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", str(start), "-i", str(src), "-t", str(dur),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
             "-movflags", "+faststart", str(out_path)],
            capture_output=True, text=True)

        info = probe(out_path)
        ok = (r.returncode == 0 and info["video"] == "h264"
              and info["audio"] == "aac"
              and abs(info["duration"] - dur) <= 1.0 + 0.05 * dur)
        results.append((out_path.name, ok, info["duration"]))
        line = (f"{'OK ' if ok else 'BAD'} {tag} {info['duration']:.1f}s "
                f"{info['video']}/{info['audio']} conf={conf} | {why[:60]}")
        print(line)
        if not ok and r.stderr:
            print(f"     ffmpeg: {r.stderr.strip()[:200]}")

    good = [n for n, ok, _ in results if ok]
    bad = [n for n, ok, _ in results if not ok]
    log(f"export run {Path(a.plan).name}: {len(good)}/{len(results)} verified h264+aac"
        + (f" | BAD: {', '.join(bad)}" if bad else ""))
    for n, ok, d in results:
        log(f"exported {n} ({d:.1f}s)" + ("" if ok else " [VERIFY FAILED]"))

    print(f"\n{len(good)}/{len(results)} clips exported & ffprobe-verified h264+aac -> {CLIPS}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
