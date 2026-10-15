import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLIPS = REPO / "clips"
PIPELINE = REPO / "output" / "pipeline"
LOG = REPO / "pipeline.log"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CLIENTS = [None, "android", "web_embedded", "mweb"]

JUNK = ["react", "compilation", "best of", "fails", "funny moments", "tiktok compilation"]


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {msg}\n")


def slug(text, maxlen=40):
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:maxlen].rstrip("-")


def ytdlp(args, timeout=180):
    last = None
    for client in CLIENTS:
        cmd = ["yt-dlp"] + list(args)
        if client:
            cmd += ["--extractor-args", f"youtube:player_client={client}"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            last = subprocess.CompletedProcess(cmd, 1, "", f"timeout after {timeout}s")
            continue
        if r.returncode == 0:
            return r
        last = r
    return last


def probe(path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {"exists": False, "duration": 0.0, "video": "", "audio": ""}

    def q(extra):
        r = subprocess.run(["ffprobe", "-v", "error"] + extra + [str(p)],
                           capture_output=True, text=True)
        return r.stdout.strip()

    dur = q(["-show_entries", "format=duration", "-of", "csv=p=0"])
    v = q(["-select_streams", "v", "-show_entries", "stream=codec_name", "-of", "csv=p=0"])
    a = q(["-select_streams", "a", "-show_entries", "stream=codec_name", "-of", "csv=p=0"])
    try:
        d = float(dur.splitlines()[0])
    except Exception:
        d = 0.0
    return {
        "exists": True,
        "size_mb": round(p.stat().st_size / 1e6, 1),
        "duration": d,
        "video": v.splitlines()[0] if v else "",
        "audio": a.splitlines()[0] if a else "",
    }


def queries(filter_sub=None):
    lines = [l.strip() for l in (REPO / "queries.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for i, qtext in enumerate(lines, 1):
        if filter_sub and filter_sub.lower() not in qtext.lower():
            continue
        out.append({"index": i, "query": qtext, "slug": slug(qtext), "dir": PIPELINE / f"q{i}-{slug(qtext)}"})
    return out


def query_for_dir(qdir):
    m = re.match(r"^q(\d+)-", Path(qdir).name)
    if m:
        idx = int(m.group(1))
        allq = queries()
        for q in allq:
            if q["index"] == idx:
                return q
    return {"index": None, "query": Path(qdir).name, "slug": slug(Path(qdir).name), "dir": Path(qdir)}


def score_item(item, kws):
    title = (item.get("title") or "").lower()
    chan = (item.get("channel") or "").lower()
    s = sum(2 for k in kws if k.lower() in title)
    dur = item.get("duration") or -1
    if 5 <= dur <= 1800:
        s += 1
    if dur > 3600:
        s -= 2
    blob = title + " " + chan
    s -= 3 * sum(1 for j in JUNK if j in blob)
    return s


def load_json(path, default=None):
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default


def save_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
