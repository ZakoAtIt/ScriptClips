import argparse
from pathlib import Path

from _util import PIPELINE, load_json, log, probe, save_json, ytdlp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qdir")
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--max-mb", type=int, default=300)
    a = ap.parse_args()

    p = Path(a.qdir)
    qdir = p if p.is_absolute() else PIPELINE / p.name
    media = qdir / "media"
    media.mkdir(parents=True, exist_ok=True)

    manifest = load_json(qdir / "media_manifest.json", {"downloaded": {}})
    dl = manifest.setdefault("downloaded", {})

    ids = list(a.ids)
    if a.top:
        search = load_json(qdir / "search_results.json", {"results": []})
        have = set(dl.keys())
        for it in search.get("results", []):
            if len(ids) >= a.top:
                break
            if it["id"] in have or it["id"] in ids:
                continue
            ids.append(it["id"])

    existing = sorted(media.glob("candidate*.mp4"))
    n = len(existing)
    ok_count = 0
    for vid in ids:
        if vid in dl and probe(dl[vid]["file"]).get("duration", 0) >= 5:
            print(f"[SKIP] {vid} already downloaded -> {dl[vid]['file']}")
            continue
        n += 1
        out = media / f"candidate{n:02d}.mp4"
        r = ytdlp([
            "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
            "--merge-output-format", "mp4",
            "--no-playlist",
            "-o", str(out),
            f"https://www.youtube.com/watch?v={vid}",
        ], timeout=900)
        info = probe(out)
        good = r.returncode == 0 and info["exists"] and info["duration"] >= 5 and info["video"] and info["audio"]
        size_ok = info.get("size_mb", 0) <= a.max_mb
        meta = {}
        if r.returncode == 0:
            m = load_json(qdir / "search_results.json", {"results": []})
            meta = next((x for x in m.get("results", []) if x["id"] == vid), {})
        dl[vid] = {
            "file": str(out),
            "title": meta.get("title", ""),
            "channel": meta.get("channel", ""),
            "channel_id": meta.get("channel_id", ""),
            "source_url": f"https://www.youtube.com/watch?v={vid}",
            "duration": info["duration"],
            "size_mb": info.get("size_mb", 0),
            "ok": bool(good and size_ok),
        }
        status = "OK " if (good and size_ok) else ("BIG" if good else "FAIL")
        print(f"[{status}] candidate{n:02d} <- {vid} [{(meta.get('channel') or '?')[:22]}] "
              f"{info['duration']:.1f}s {info.get('size_mb', 0)}MB {info['video']}/{info['audio']} | {(meta.get('title') or '')[:60]}")
        if good and not size_ok:
            out.unlink(missing_ok=True)
            print(f"       deleted: exceeds --max-mb {a.max_mb}")
            dl[vid]["ok"] = False
        if good and size_ok:
            ok_count += 1

    save_json(qdir / "media_manifest.json", manifest)
    total_ok = sum(1 for v in dl.values() if v.get("ok"))
    print(f"\nusable candidates now on disk: {total_ok}")
    log(f"download {qdir.name}: +{ok_count} new usable ({total_ok} total)")


if __name__ == "__main__":
    main()
