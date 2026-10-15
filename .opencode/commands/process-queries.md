---
description: Direct yt-dlp/whisper clip pipeline over queries.txt - agent judgment drives everything, tools do mechanical work.
agent: build
---

Process `queries.txt` into exported clips. `$ARGUMENTS` is an optional substring filter; when empty, process EVERY non-empty line. Work from the repo root. Do NOT modify repo code (`config.py`, `modules/`, `run.py`, `tools/`).

## Prime directive

YOU are the brain; the scripts are hands. Scripts exist only to automate failure-prone mechanics (yt-dlp client rotation, encoding-safe python, proven ffmpeg flags). Every decision is yours, made from real evidence you read yourself (search tables, transcripts, RMS bursts): which phrasings to try, whether attribution is correct, which candidates matter, where a moment starts and ends, what deserves export. Script scores/confidences are provisional sorting aids — never let them override your judgment.

## Environment facts

- Windows PowerShell 5.1; python/yt-dlp/ffmpeg/ffprobe on PATH.
- `cookies.txt` is EMPTY — NEVER pass `--cookies` to yt-dlp.
- YouTube HTTP 403 / sign-in blocks: `_util.ytdlp` already rotates `youtube:player_client=` default→android→web_embedded→mweb. If calling yt-dlp directly, rotate manually in that order.
- Tools (run from repo root): `python tools\pipeline\search.py`, `download.py`, `pin_windows.py`, `export.py`.
- Throwaway python goes to `%TEMP%\opencode\*.py`; first line `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`. Fix scripts with the edit tool — never regex-in-shell. Never combine `2>$null` with `$?` checks on native commands.

## Phase 0 — setup

1. Read `queries.txt`; filter by `$ARGUMENTS` if given. Per query index N: slug = lowercase, non-alnum→`-`, ≤40 chars; create `output/pipeline/qN-<slug>/{media,frames}`.
2. Append `pipeline.log`: `[ts] BATCH START | K queries: <slugs>`.

## Phase 1 — search

1. **Verify attribution FIRST** via web search across ≥2 independent sources before trusting query wording — queries are frequently misattributed. Log corrections in `pipeline.log` and aim searches at the TRUE creator/event names.
2. Run `search.py` per query with 2–4 phrasings YOU devise (literal wording + alternate nicknames + how uploaders actually title such clips) plus `--kws`:
   `python tools\pipeline\search.py qN-<slug> "phrase one" "phrase two" --kws word1 word2`
   Add `--channels <url_or_id>` to crawl a channel's /videos /shorts /streams tabs when hunting raw uploads.
3. Read the ranked table yourself. Thin results (<2 plausible originals) → invent new phrasings, crawl channels of known participants.
4. Sourcing hierarchy: original upload > contemporaneous mirror/reupload > news report embedding the moment (LAST RESORT). Reaction/compilation channels only as carrier of an otherwise-unavailable exact moment.

## Phase 2 — download

Hand-pick IDs after reading the table (preferred over blind --top):
`python tools\pipeline\download.py qN-<slug> --ids id1 id2 ...`
Files land in `media/candidateNN.mp4`, ffprobe-verified (>5 s, video+audio), tracked in `media_manifest.json`. Re-runs skip verified files.

## Phase 3 — pin the moments

`python tools\pipeline\pin_windows.py qN-<slug> [--keywords w1 w2] [--frames 8]`
→ writes `transcripts.json` (full whisper segments + RMS bursts) and a PROVISIONAL analysis file.

Then YOU finalize by reading the transcripts:
- Locate payoff lines/actions; set windows 10–30 s starting just BEFORE the payoff; ≤3 windows/query; diversify across different uploads; drop near-duplicate sibling windows.
- Set real confidence per candidate (transcript corroboration + source quality).
- Record `missing_matches` (ideal-but-unavailable sources + why) and attribution notes.
- Write your finalized version to `analysis.json` (script output stays in `analysis_provisional.json`).

## Phase 4 — export

Write `%TEMP%\opencode\plan_qN.json`: array of
`{"out": "qN_<slug>_NN_<label>.mp4", "src": "<path or candidateNN.mp4>", "start": 12.3, "dur": 15.2, "conf": 0.95, "why": "..."}`
(`src` may be absolute, repo-relative, or a bare filename inside the query dir). Then:
`python tools\pipeline\export.py %TEMP%\opencode\plan_qN.json`
It cuts with proven settings (libx264 crf20 veryfast, aac 128k, faststart) into `clips/`, ffprobe-verifies h264+aac, and logs each clip. Fix any BAD entries before moving on.

## Phase 5 — raw pass

For every query whose best exported source was news/reaction-carried: hunt the ORIGINAL (extra phrasings, X/Twitter mirror searches, channel tab crawls, wayback machine for dead hosts like vine.co). Export extras named `qN_raw_NN_<label>.mp4`. If truly unavailable, append a RAW line to `pipeline.log` citing concrete checked evidence (tabs crawled, posts deleted, VOD absent) — never a vague "not found".

## Escalation — subagents (rare)

- @api-scraper ONLY if a query still has <2 usable candidates after Phases 1–2: launch once, prompt demands it writes `output/pipeline/qN-<slug>/scrape_result.json` FIRST; check disk on return; resume at most once; then abandon silently and continue directly.
- @vision-engineer: skip by default (no reliable image input). Transcript/RMS pinning IS the verification standard.

## Logging contract

Per query one RESULTS line: `<slug> | N cuts | top conf X (method: whisper-pinned/rms-pinned/etc.) | missing: ... | attribution notes`. End with `PIPELINE COMPLETE: X queries | Y cuts exported & ffprobe-verified h264+aac`. One query failing must never block the others.
