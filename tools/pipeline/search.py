import argparse
import json
import time
from pathlib import Path

from _util import PIPELINE, load_json, log, query_for_dir, save_json, score_item, ytdlp


def parse_flat(lines):
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        vid = e.get("id")
        if not vid:
            continue
        out.append({
            "id": vid,
            "title": e.get("title") or "",
            "channel": e.get("channel") or e.get("uploader") or "",
            "channel_id": e.get("channel_id") or "",
            "duration": round(e["duration"]) if e.get("duration") else -1,
            "views": e.get("view_count") or -1,
            "url": e.get("url") or f"https://www.youtube.com/watch?v={vid}",
            "source": "search",
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qdir")
    ap.add_argument("phrases", nargs="*", default=[])
    ap.add_argument("--kws", nargs="*", default=[])
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--channels", nargs="*", default=[])
    a = ap.parse_args()

    p = Path(a.qdir)
    qdir = p if p.is_absolute() else PIPELINE / p.name

    seen = {}
    chan_ids = {}

    for phrase in a.phrases:
        tag = f"ytsearch{a.per}:{phrase}"
        r = ytdlp(["--flat-playlist", "--dump-json", "--no-warnings", tag], timeout=150)
        items = parse_flat((r.stdout or "").splitlines()) if r and r.returncode == 0 else []
        for it in items:
            if it["channel_id"]:
                chan_ids[it["channel"]] = it["channel_id"]
            prev = seen.get(it["id"])
            if not prev:
                it["phrase_hits"] = [phrase]
                seen[it["id"]] = it
            else:
                prev.setdefault("phrase_hits", []).append(phrase)
        print(f"[{'OK ' if items else 'ERR'}] '{phrase}' -> {len(items)} results")
        time.sleep(1.5)

    for curl in a.channels:
        base = curl.rstrip("/")
        for tab in ("videos", "shorts", "streams"):
            r = ytdlp(["--flat-playlist", "--dump-json", "--no-warnings", f"{base}/{tab}"], timeout=240)
            items = parse_flat((r.stdout or "").splitlines()) if r and r.returncode == 0 else []
            for it in items:
                it["source"] = f"channel:{tab}"
                prev = seen.get(it["id"])
                if not prev:
                    it["phrase_hits"] = [base]
                    seen[it["id"]] = it
                elif f"channel:{tab}" not in prev["source"]:
                    prev["source"] += f"|channel:{tab}"
            print(f"[{'OK ' if items else 'ERR'}] {base}/{tab} -> {len(items)} entries")
            time.sleep(1.5)

    results = list(seen.values())
    kws = a.kws or [w for w in (query_for_dir(qdir)["query"].lower().split()) if len(w) > 3]
    for it in results:
        it["score"] = score_item(it, kws)
    results.sort(key=lambda x: (-x["score"], -(x["views"] if x["views"] and x["views"] > 0 else 0)))

    payload = {
        "qdir": qdir.name,
        "query": query_for_dir(qdir)["query"],
        "kws": kws,
        "phrases": a.phrases,
        "channels_crawled": a.channels,
        "channels": chan_ids,
        "results": results,
    }
    save_json(qdir / "search_results.json", payload)

    print(f"\n--- {len(results)} unique videos | top 25 by score ---")
    for it in results[:25]:
        t = it["title"][:70]
        print(f"{it['score']:>3} | {it['id']} | {it['duration']:>6}s | {(it['channel'] or '?')[:22]:<22} | {t}")

    manifest = load_json(qdir / "media_manifest.json", {"downloaded": {}})
    have = set(manifest.get("downloaded", {}).keys())
    fresh = [r for r in results if r["id"] not in have]
    if fresh:
        print(f"\nnot yet downloaded: {len(fresh)} (top: {fresh[0]['id']} '{fresh[0]['title'][:50]}')")
    log(f"search {qdir.name}: {len(results)} unique from {len(a.phrases)} phrases + {len(a.channels)} channel crawls")


if __name__ == "__main__":
    main()
