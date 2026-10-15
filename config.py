"""Settings for ScriptClips — queries.txt -> 3 verified clips + 3 images per query.

Each line of queries.txt is a well-detailed query naming the subjects, object
and action.  Primary lane: the named streamer's own Twitch channel clips, each
*content verified* by transcribing the short clip audio (no title-only
guessing).  Fallback lane: YouTube raw footage (official-channel VODs / raw
stream uploads), scanned by caption where possible, sectioned to 10s windows.
Edited / reaction / re-upload content is hard-rejected.  Face-gate keeps as
many script subjects on screen as possible (all first, then any).
"""
from os import environ
from pathlib import Path

# Cap native thread pools BEFORE torch / onnxruntime / ctranslate2 are imported.
# With 24 logical CPUs the model libraries each default to 24-thread pools; with
# ~16 GB RAM that oversubscription caused silent native crashes ~30 min into
# runs.  Capping to a few threads avoids the memory blowup at a modest speed cost.
CPU_THREADS = environ.get("SCRIPTCLIPS_CPU_THREADS", "4")
environ.setdefault("OMP_NUM_THREADS", CPU_THREADS)
environ.setdefault("MKL_NUM_THREADS", CPU_THREADS)
environ.setdefault("OPENBLAS_NUM_THREADS", CPU_THREADS)
environ.setdefault("NUMEXPR_NUM_THREADS", CPU_THREADS)
environ.setdefault("VECLIB_MAXIMUM_THREADS", CPU_THREADS)

# Free-RAM floor (MB) below which the runtime watchdog forces a GC before heavy steps.
MEMORY_GC_THRESHOLD_MB = int(environ.get("SCRIPTCLIPS_MEMORY_GC_MB", "1536"))

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
FRAMES_DIR = OUTPUT_DIR / "frames"
CLIPS_DIR = BASE_DIR / "clips"
CACHE_DIR = BASE_DIR / "cache"
TRANSCRIPT_CACHE_DIR = CACHE_DIR / "transcripts"
MOMENT_FINDER_CACHE_DIR = CACHE_DIR / "moments"
OCR_FRAME_CACHE = CACHE_DIR / "ocr"
TWITCH_CLIP_CACHE = CACHE_DIR / "twitch_clips"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

# --- Input -----------------------------------------------------------------
QUERIES_FILE = BASE_DIR / "queries.txt"          # one detailed query per line
SENTENCES_CACHE = BASE_DIR / "data" / "sentences.json"   # planner output (edit + re-run)
STREAMERS_FILE = BASE_DIR / "data" / "streamers.json"    # alias -> twitch/youtube identities
PEOPLE_FILE = BASE_DIR / "data" / "people.json"          # notable non-KB people for subject extraction
TOPICS_FILE = BASE_DIR / "data" / "topics.json"          # non-person subjects (movies, franchises, events)
EVENTS_FILE = BASE_DIR / "data" / "events.json"          # event phrase -> participant handles (per-streamer expansion)
EVENT_VIDEOS_FILE = BASE_DIR / "data" / "event_videos.json"  # curated original-event broadcast/POV registry

# --- Auto-registry (per-batch discovered identities/events/VODs/lexicon) ----
# The auto-registry stage (modules/auto_registry.py) generates these before
# planning.  Manual entries always WIN over auto entries: *_manual merge order
# is defaults <- auto <- manual.  Review reports land under AUTO_REPORT_DIR.
AUTO_STREAMERS_FILE = DATA_DIR / "auto_streamers.json"
AUTO_PEOPLE_FILE = DATA_DIR / "auto_people.json"
AUTO_EVENTS_FILE = DATA_DIR / "auto_events.json"
AUTO_EVENT_VIDEOS_FILE = DATA_DIR / "auto_event_videos.json"
AUTO_MOMENT_LEXICON_FILE = DATA_DIR / "auto_moment_lexicon.json"
# video_id:start.0 -> reason; moments here are never judge-rejected (user-forced
# exports, e.g. a moment the LLM judge kept mis-flagging as a reaction).
FORCED_MOMENTS_FILE = DATA_DIR / "forced_moments.json"
AUTO_REGISTRY_CACHE = DATA_DIR / "auto_registry.json"
AUTO_REPORT_DIR = OUTPUT_DIR / "reports"

# --- Output volume ----------------------------------------------------------
CLIPS_PER_SENTENCE = 3          # video clips exported per sentence
IMAGES_PER_SENTENCE = 3         # google images exported per sentence
EXPORT_TRY_LIMIT = 16           # ranked candidates the exporter walks to fill the quotas
CLIP_DURATION_SECONDS = 10      # window length used for YouTube VOD sections
MIN_CLIP_GAP_SECONDS = 12
MAX_CLIPS_PER_VIDEO = 3         # up to three windows from one VOD (range-downloaded, never the full video)

# --- Section (window) downloads ----------------------------------------------
RANGE_SECTION_DOWNLOAD = True   # fetch only the bytes covering each window via HTTP Range
RANGE_MARGIN_SECONDS = 12.0     # buffer around a window so fragment/keyframe alignment is safe
RANGE_INIT_BYTES = 1 * 1024 * 1024
RANGE_MAX_BYTES = 150 * 1024 * 1024

# --- Whole-clip mode (Twitch clips / YouTube short clips) ------------------
CLIP_MIN_SECONDS = 15
CLIP_MAX_SECONDS = 240
CLIP_MIN_VIEWS = 100
CLIP_MIN_TITLE_MATCH = 0.20
CLIP_EDIT_OVERLAY = 0.30        # avg OCR text-coverage above this = edited recap

# --- Search -----------------------------------------------------------------
YOUTUBE_MAX_RESULTS = 20
YOUTUBE_MAX_RESULTS_EVENT = 30   # wider result net for event queries (raw VODs are hard to rank by title)
YOUTUBE_QUERY_LIMIT_EVENT = 10   # search hooks consumed for event queries (entity-pair + verb variants)
VIDEOS_TO_ANALYSE = 10
MIN_VIDEO_LENGTH = 30
MAX_VIDEO_LENGTH = 60 * 60 * 8
RAWNESS_TOP_SLOTS = 2
LONG_VIDEO_SECONDS = 2 * 60 * 60

# Netscape-format cookies.txt exported for youtube.com (e.g. via the
# "Get cookies.txt LOCALLY" extension). Supplied to every yt-dlp call so
# YouTube's "Sign in to confirm you're not a bot" gate does not block
# metadata lookups or clip downloads. Omit / leave missing to run anonymous.
YT_COOKIES_FILE = BASE_DIR / "cookies.txt"

# --- Exact-query "human search" pass ----------------------------------------
# The verbatim query is searched as-is and its top results are deep-analyzed
# (LLM transcript scan + visual localization) like a human would, instead of
# being re-ranked by the metadata heuristic that can bury the platform's #1.
# YouTube is the FIRST-PRIORITY lane: the first EXACT_TOP_RESULTS 16:9 videos
# (regular watch?v= uploads, NOT 9:16 Shorts) in YouTube's own ranking are the
# most reliable source of the best clip, so they are searched, analyzed for
# EXACT_WINDOW_SECONDS windows, and ranked ahead of Twitch/Kick candidates.
EXACT_TOP_RESULTS = 15           # first 15 platform-ordered 16:9 results checked
EXACT_SHORTS_EXCLUDE = True      # skip 9:16 Shorts ("/shorts/" URLs); 16:9 only
MIN_LONG_FORM_SECONDS = 120      # legacy long-form floor (no longer enforced in the exact pass)
EXACT_WINDOW_SECONDS = 30        # window length exported for this pass
EXACT_SEARCH_FETCH = 10          # raw results fetched before 16:9/live filtering

# --- Relevance gates (lenient: only clearly unrelated evidence is dropped) --
RELEVANCE_GATE_COMBINED = 0.20
RELEVANCE_GATE_SEMANTIC = 0.12
RELEVANCE_GATE_RERANK = 0.05
RELEVANCE_GATE_ENTITY = 1
# Event sentences require action or object evidence (or visual proof) in addition
# to entity presence — entity alone is never sufficient for events.  For narration
# sentences the existing entity-only path is unchanged.
RELEVANCE_GATE_ACTION_OR_OBJECT = 1
RAWNESS_SLOT_MIN_TITLE_MATCH = 0.20

# --- Transcripts ------------------------------------------------------------
MAX_TRANSCRIPTION_SECONDS = 600  # Whisper cap for caption-less sources
CAPTION_CACHE_TTL_DAYS = 30

# --- Candidate moments ------------------------------------------------------
WINDOW_STRIDE_SECONDS = 2
MAX_CANDIDATES_PER_VIDEO = 4
SEMANTIC_MIN_SCORE = 0.45
VIDEO_SEMANTIC_SKIP_FLOOR = 0.25   # skip full analysis for videos below this

# --- Action-aware selection (event/action moments) ---------------------------
ACTION_WINDOW_SECONDS = 30       # YouTube window size for event/action lines (narration keeps 10s)
ACTION_MIN_HITS = 1              # distinct action terms a Twitch clip transcript must contain to pass
TWITCH_VERIFY_PER_LANE_EVENT = 18  # Twitch clips audio-verified per event/action query
VIDEOS_TO_ANALYSE_EVENT = 18       # YouTube videos analyzed per event/action query
# Synonym expansion for action terms extracted from a query line.  Keys are the
# stemmed action word; values are spoken forms a transcript may contain.  Only
# DISTINCTIVE spoken forms are listed: weak filler words (right/answer/world/
# money/open/...) pollute action matching and moment grounding, so they live in
# GENERIC_ACTION_TERMS instead and never count as event signal.
ACTION_SYNONYMS = {
    "call": {"call", "calls", "called", "calling", "phone", "phones", "dial", "dialed",
             "facetime", "face time", "video call", "ring", "ringing", "ringed"},
    "answer": {"answer", "answers", "answered", "answering", "hello", "hey", "pick",
               "picked", "picked up", "pick up", "on the phone"},
    "pick": {"pick", "picks", "picked", "picking", "answer", "answers", "answered", "hello"},
    "guess": {"guess", "guesses", "guessed", "guessing", "nailed", "nails", "correct",
              "got it", "good job"},
    "nail": {"nail", "nails", "nailed", "correct", "got it"},
    "burger": {"burger", "burgers", "cheeseburger", "cheeseburgers", "food", "sandwich",
               "mcdonald's", "mcdonalds", "mcflurry", "fries", "ketchup"},
    "give": {"give", "gives", "gave", "given", "giving", "gift", "hand", "hands",
             "handed", "handed over", "hand over", "present", "presented"},
    "gift": {"gift", "gifts", "gifted", "gifting", "hand", "hands", "handed",
             "present", "presented", "give away"},
    "boots": {"boots", "boot", "shoe", "shoes", "sneaker", "sneakers", "cleat", "cleats",
              "game worn", "game-worn", "pair"},
    "meet": {"meet", "meets", "met", "meeting", "arrive", "arrives", "arrived",
             "introduce", "introduces", "introduced", "face to face", "in person",
             "ran into", "bumped into", "bump into", "he's here", "she's here",
             "showed up", "show up", "came in", "come in", "walked in", "walk in",
             "finally here", "there he is", "there she is"},
    "donate": {"donate", "donates", "donated", "donating", "donation", "donations",
               "my bad", "oops", "paypal", "bits", "gift sub", "sub"},
    "spend": {"spend", "spends", "spent", "spending", "paid", "wasted", "give away",
              "gifting", "donated", "donation"},
    "leak": {"leak", "leaks", "leaked", "leaking", "dm", "dms", "private messages",
             "screenshots", "texts", "texting", "dms open", "accidentally",
             "on screen", "on stream"},
    "chase": {"chase", "chases", "chased", "chasing", "run", "runs", "ran", "running",
              "running after", "follow", "followed", "follows"},
    "receive": {"receive", "receives", "received", "receiving", "got", "gets",
                "hands", "handed", "presents"},
    "viral": {"viral", "trend", "trending", "famous", "popular", "blew up", "blown up"},
    "react": {"react", "reacts", "reacted", "reacting", "response", "responds", "responded"},
    "shove": {"shove", "shoves", "shoved", "shoving", "push", "pushes", "pushed",
              "pushing", "shoving match", "tumble", "into the pool", "in the pool",
              "into the water", "in the water"},
    "push": {"push", "pushes", "pushed", "pushing", "shove", "shoves", "shoved",
             "tackle", "tackled", "tackling", "into the pool", "in the pool",
             "into the water", "in the water"},
    "fight": {"fight", "fights", "fought", "fighting", "brawl", "brawling",
              "scuffle", "wrestle", "wrestling", "grapple", "grappling",
              "punch", "punched", "swing", "swung", "smack", "smacked"},
    "punch": {"punch", "punches", "punched", "punching", "hit", "hits", "knock",
              "knocked", "knocked out", "jab", "jabbed", "slug", "slugged",
              "socked", "chin check", "chin checked", "rocked"},
    "slap": {"slap", "slaps", "slapped", "slapping", "smack", "smacked",
             "hit", "hits"},
    "attack": {"attack", "attacks", "attacked", "attacking", "lunge", "lunged",
               "lunging", "swing", "swung", "rushed", "rushing"},
    "throw": {"throw", "throws", "threw", "throwing", "toss", "tossed", "tossing",
              "fling", "flung", "heaved"},
    "confront": {"confront", "confronts", "confronted", "confrontation",
                 "confronting", "call out", "calls out", "called out",
                 "calling out", "stop him", "stop her", "come here", "hold on",
                 "apologize", "apologizes", "apologized", "apologizing",
                 "you need to apologize", "that's rude", "you're wrong"},
    "spit": {"spit", "spits", "spat", "spitting", "splatter", "splattered",
             "spat on", "spit on", "spitting on", "on my leg", "on her leg",
             "apologize to my leg", "apologize to her leg", "the ground"},
    "expel": {"expel", "expels", "expelled", "expelling", "expulsion",
              "kicked out", "kicks out", "kick out", "you're expelled",
              "you are expelled", "detention", "arrest", "arrested",
              "arresting", "get out", "get him out", "off the campus",
              "out of here"},
    "escort": {"escort", "escorts", "escorted", "escorting", "driving us",
               "taking us", "took us", "taking everyone", "transporting",
               "pick us up", "picked us up", "get in", "get on"},
    "fly": {"fly", "flies", "flew", "flying", "chopper", "helicopter",
            "airborne", "land", "landed", "landing", "took off", "take off",
            "in the sky", "overhead"},
    "drive": {"drive", "drives", "drove", "driving", "pull up", "pulled up",
              "pull up", "rolling up", "drove up", "driving us"},
    "ride": {"ride", "rides", "rode", "riding", "hop on", "hopping on",
             "get on", "jump on", "jumping on"},
    "bust": {"bust", "busts", "busted", "busting", "raided", "raid", "stormed",
             "storm", "showed up to", "rolled up on"},
    "arrest": {"arrest", "arrests", "arrested", "arresting", "in handcuffs",
               "handcuffs", "detain", "detained", "booked", "jail"},
    "injure": {"injure", "injures", "injured", "injuring", "hurt", "wound",
               "wounded", "bleeding", "blood", "got hit", "hit me",
               "my eye", "my face", "ouch", "ow", "pain"},
    "shoot": {"shoot", "shoots", "shot", "shooting", "fired", "fire",
              "shots fired", "pulled the trigger", "gunshot", "gunshots"},
}

# Words that appear in queries but carry NO event signal on their own.  They are
# excluded from LLM-moment cue building and grounding so that a stray "world" /
# "right" / "answer" in the chat can never mark a wrong window as "grounded".
GENERIC_ACTION_TERMS = (
    "full", "clip", "clips", "video", "footage", "reaction", "react", "reacts",
    "reacting", "live", "livestream", "stream", "streams", "streaming",
    "watching", "watch", "watches", "view", "views",
    "right", "wrong", "answer", "answers", "answered", "answering",
    "world", "cup", "game", "worn", "give", "gives", "given", "giving",
    "him", "her", "their", "huge", "ever", "luckiest", "randomly",
    "impossible", "private", "mistake", "misclick", "misclicked", "sends",
    "sent", "messages", "message", "open", "opened", "opening", "showed",
    "showing", "money", "cash", "dollars", "bucks", "thousand", "thousands",
    "million", "like", "know", "think", "get", "got", "going", "gonna",
    "today", "yeah", "okay", "really", "stuff", "thing", "things", "way",
    "good", "best", "hit", "big", "small", "new",
    "she", "he", "during", "trying", "try", "out", "university", "water", "pool",
)

# --- Visual verification ----------------------------------------------------
FRAMES_PER_CANDIDATE = 3
OCR_ENABLED = True
VISUAL_FRAME_WIDTH = 384
FRAME_SEEK_TIMEOUT = 12
FFMPEG_TRIM_TIMEOUT = 45

# --- Visual-anchor matching --------------------------------------------------
# Google Images anchors are fetched up-front for every query.  A clip candidate
# survives only when a sampled frame CLIP-matches the anchors (image->image
# cosine), because audio + metadata alone let a wrong event pass when its
# transcript merely mentions the same words.  The gate is skipped when no
# relevant anchor image is found for a query.
ANCHOR_ENABLED = True
ANCHOR_TOP_K = 5               # strongest unique anchors kept per query
ANCHOR_NEAR_DUP_COSINE = 0.92  # above this cosine two anchors are the same image
ANCHOR_SIM_FLOOR = 0.55        # min max-cosine(frame, anchors) for a candidate to pass
ANCHOR_FRAMES_PER_WINDOW = 6   # frames sampled per candidate window
ANCHOR_GATE_STRICT = True      # True: reject below-floor candidates; False: keep but flag anchor_failed
# The floor above is only trusted when the anchors are *event-specific*: multiple
# photos of the SAME incident score mutually similar (e.g. Kai set mean ~0.58).
# Generic media imagery of a famous person is a diverse grab-bag (e.g. Rampage set
# mean ~0.44) and the floor would wrongly reject real footage, so below this mean
# pairwise cosine the anchor gate is disabled and lanes fall back to audio/metadata
# verification only.
ANCHOR_GATE_MIN_COHERENCE = 0.50
# Visual-event gate: the fraction of sampled frames that must clear
# ANCHOR_SIM_FLOOR for a visual-event clip (police car / helicopter / gel
# blaster queries) to pass.  Requiring most frames to match proves the object is
# actually on screen, not that a single frame happened to resemble the anchors.
ANCHOR_FRACTION_PASS = 0.70
ANCHOR_CACHE = CACHE_DIR / "anchors"
ANCHOR_TERM_FLOOR = 0.28            # CLIP score for a term to be "present" in an anchor image
ANCHOR_MIN_COVERAGE = 0.40          # min fraction of entity+object terms that must hit
ANCHOR_STRICT_MIN_GROUPS = 2        # min anchor groups (entity, object) that must match a frame

# --- Event-visual probing (semantic scene verification) ----------------------
# Visual-event queries (police-car escort / school bus / helicopter / gel
# blaster / fight) are probed with CLIP text scenes on LOCAL frames BEFORE the
# expensive anchor gate runs, so a candidate with the object/action actually on
# screen out-ranks audio-only lookalikes and fewer wrong clips reach download.
EVENT_VISUAL_ENABLED = True
EVENT_VISUAL_FRAMES = 6             # frames probed per candidate/window
EVENT_VISUAL_FLOOR = 0.20           # min CLIP cosine (scene vs frame) to count
EVENT_VISUAL_FRACTION = 0.5         # fraction of frames that must clear the floor
EVENT_VISUAL_IMAGE_K = 4            # top-K visual frames kept per candidate
EVENT_PROBE_AUDIO_BOOST = 0.04      # combine-score lift when scene + audio agree

# --- General visual scene probing (every query, not just 6 event kinds) -------
# Every plan gets its own CLIP scene prompts (script_planner), so ALL candidates
# get a real pixel signal for ranking and the export gate.  Previously the only
# pixel checks were the 6 hardcoded event kinds, leaving e.g. lamborghini/jet/
# yacht queries with no visual grounding at all.
GENERAL_VISUAL_ENABLED = True
GENERAL_VISUAL_FRAMES = 4           # frames sampled per candidate for general scenes
GENERAL_VISUAL_FLOOR = 0.20         # min CLIP cosine (scene vs frame) to count
GENERAL_VISUAL_FRACTION = 0.5       # fraction of frames that must clear the floor
# Frame budget for the ALWAYS-RUN visual scan on remote (YouTube) streams.
# Remote ffmpeg seeks are slow on this setup, so the always-run scan is bounded
# (a ~64-frame coarse pass plus a bounded fine zoom around hit regions; it still
# early-aborts on consecutive timeouts); Twitch/Kick HLS scans keep the denser
# default budget.
YOUTUBE_VISUAL_SCAN_FRAMES = 64
YOUTUBE_VISUAL_MIN_STRIDE = 30
# A single composite scene image (frames tiled in one JPEG) sent to the LLM
# judge so it can *see* the event pixels instead of guessing from audio alone.
CONTACT_SHEET_ENABLED = True
CONTACT_SHEET_ROWS = 4
CONTACT_SHEET_COLS = 6
CONTACT_SHEET_THUMB = 160          # thumbnail width (px) per sheet cell
EVENT_VISION_CACHE = CACHE_DIR / "event_vision"

# --- Visual-first discovery (thumbnail CLIP + stream frame-seek) ------------
# Layer 1: YouTube thumbnail CLIP scoring (pre-metadata, cheap).
# Tries maxresdefault -> hqdefault -> mqdefault for each video ID.
VISUAL_THUMBNAIL_TOP_N = 10         # top N visually-ranked candidates that get full metadata
VISUAL_THUMBNAIL_TIMEOUT = 5        # seconds per thumbnail download
VISUAL_THUMBNAIL_WORKERS = 8        # concurrent thumbnail download threads
# Layer 2: Stream frame-seek moment-finding (post-metadata, moderate cost).
# Fast 2-pass coarse-to-fine: 40 frames find the right 2-minute region,
# then 20 frames pinpoint the exact moment.  Only runs on top-3 candidates.
VISUAL_MOMENT_COARSE_FRAMES = 40    # first pass: spread across entire video
VISUAL_MOMENT_FINE_FRAMES = 20      # second pass: dense within best region
VISUAL_MOMENT_REGION_SECONDS = 120  # region size for fine scan (seconds)
VISUAL_MOMENT_COARSE_FLOOR = 0.25   # min mean score in coarse scan to proceed to fine
VISUAL_MOMENT_FLOOR = 0.35          # min anchor cosine for a visual moment to be trusted
VISUAL_MOMENT_PAD = 8               # seconds of padding around the best visual timestamp
VISUAL_MOMENT_ULTRA_FRAMES = 0      # disabled: ultra-fine pass costs too much per video
VISUAL_MOMENT_ULTRA_RANGE = 15      # ±seconds around best timestamp for ultra-fine scan

# --- Models -----------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
WHISPER_MODEL_SIZE = "small"     # content verification of short clips
WHISPER_BIG_MODEL_SIZE = "tiny"  # long caption-less VODs (fast CPU fallback)
WHISPER_DEVICE = "auto"          # 'auto'|'cuda'|'cpu' (auto checks torch.cuda)
WHISPER_COMPUTE = "auto"         # 'auto'|'float16'|'int8'
# Per-language Whisper model overrides.  `small` is weak on Japanese, so
# Japanese-language VODs (Ozemi/Nanatty-style streams) transcribe with the
# large-v3-turbo model instead (slower, but usable) when no captions exist.
WHISPER_LANG_MODELS = {
    "ja": "large-v3-turbo",
}

# --- LLM providers ----------------------------------------------------------
# Primary: any OpenAI-compatible HTTP API (set LLM_API_BASE + LLM_API_KEY,
# e.g. OpenAI, Groq, OpenRouter, LM Studio, vLLM).  When unset — or when the
# API fails at call time — every LLM call falls back to local Ollama below.
#   export SCRIPTCLIPS_LLM_API_BASE="https://api.openai.com/v1"
#   export SCRIPTCLIPS_LLM_API_KEY="sk-..."
#   export SCRIPTCLIPS_LLM_API_MODEL="gpt-4o-mini"
LLM_API_BASE = environ.get("SCRIPTCLIPS_LLM_API_BASE", "").rstrip("/")
LLM_API_KEY = environ.get("SCRIPTCLIPS_LLM_API_KEY", "")
LLM_API_MODEL = environ.get("SCRIPTCLIPS_LLM_API_MODEL", "gpt-4o-mini")
LLM_API_VISION_MODEL = environ.get("SCRIPTCLIPS_LLM_API_VISION_MODEL", "")
LLM_API_TIMEOUT = int(environ.get("SCRIPTCLIPS_LLM_API_TIMEOUT", "120"))

# --- Local LLM (ollama) -----------------------------------------------------
LLM_ENABLED = True
OLLAMA_HOST = "http://127.0.0.1:11434"
LLM_MODEL = "qwen2.5:7b"          # installed and stable (qwen3:8b was erroring with HTTP 500)
LLM_MODEL_FALLBACK = "llama2:latest" # used if primary is missing/unreachable
LLM_THINK = False                 # disable qwen3 chain-of-thought (fast, direct JSON)
LLM_TIMEOUT = 240
LLM_MAX_TOKENS = 1400
LLM_TEMPERATURE = 0.1
LLM_CONTEXT = 16384             # per-request num_ctx; vision sheets need >> 4096
LLM_JUDGE_POOL = 8                # top-N candidates shown to the LLM judge

# Vision-capable model used ONLY by the LLM judge so it can actually SEE the
# candidate frames (a text-only model could never judge footage).  The rest of
# the pipeline keeps LLM_MODEL (fast, stable).  When LLM_VISION_MODEL is not
# installed, the judge falls back to text judgment with a loud warning.
LLM_VISION_ENABLED = True
LLM_VISION_MODEL = "qwen2.5vl:7b"
LLM_VISION_FALLBACK = "llava-llama3"
LLM_VISION_MODELS = (
    "llava", "llama3.2-vision", "llama3.2v", "qwen2-vl", "qwen2.5vl",
    "qwen2.5-vl", "gemma3", "minicpm-v", "moondream", "bakllava",
)

# --- VLM event verifier (is the moment actually happening) --------------------
# qwen2.5vl answers "does this window show / contain the described moment" on a
# 6-frame contact sheet of the window.  The verdict is a HARD gate at export
# time (high-confidence "not happening" clips are dropped) and the primary
# ranking signal, so a wrong event with the right faces can no longer ship.
VLM_VERIFY_ENABLED = True
VLM_VERIFY_FRAMES = 6             # frames tiled into one contact sheet per window
VLM_VERIFY_MAX_SHEETS = 8         # cap sheets per LLM call (context budget)
VLM_REJECT_CONFIDENCE = 0.65       # confidence floor for a hard "not happening" reject
VLM_CACHE = CACHE_DIR / "vlm"

# --- Transcript-trust (the transcript is the primary signal) ------------------
# A clip whose captions LITERALLY speak the moment (keyword-located + action +
# an entity name heard, on an official/raw channel) exports even when the vision
# stack false-rejects it.  Strictly additive: visual matches are unaffected and
# the trusted clips get their OWN quota so they never crowd out pixel-confirmed
# exports.
TRANSCRIPT_TRUST_ENABLED = True
TRANSCRIPT_TRUST_MIN_ACTION_HITS = 1
TRANSCRIPT_TRUST_MIN_ENTITY_COVERAGE = 0.5
TRANSCRIPT_TRUST_REQUIRE_KEYWORD = True
TRANSCRIPT_TRUST_ACCEPT_CHANNELS = ("official", "raw")
TRANSCRIPT_TRUST_QUOTA = 2          # extra export slots beyond CLIPS_PER_SENTENCE
TRANSCRIPT_MOMENT_COVERAGE = 1.0    # full plan action coverage = the exact spoken moment;
                                    # such candidates are always export-graded (never starved)

# --- LLM moment finder (smart transcript scan) --------------------------------
# The heuristic window proposers only surface transcript regions whose words
# overlap the query.  Real moments often carry NO query words (a phone call is
# "Hello. Hello. Hello.", a meeting is crowd noise + introductions, a donation is
# "I just donated four thousand dollars").  The moment finder asks the local LLM
# to read the transcript and return exact [start,end] timestamps for the event.
MOMENT_FINDER_TOP_VIDEOS = 7       # pool videos (by search_score) given an LLM transcript scan
MOMENT_FINDER_MIN_CONFIDENCE = 0.5  # lowest confidence that still becomes a candidate window
LLM_MOMENT_MIN_CONFIDENCE = 0.6     # confidence floor that bypasses the verify/relevance gates
LLM_MOMENT_UNCUED_CONFIDENCE = 0.9  # floor for moments whose audio speaks no event cue (model inference only)
MOMENT_FINDER_TOKEN_BUDGET = 2400   # whole-transcript read below this token estimate
MOMENT_FINDER_ZONE_NEIGHBORHOOD = 45  # seconds kept around each keyword-zoom hit in long VODs
MOMENT_FINDER_ZONE_MAX = 6           # max zoom zones scanned per video
MOMENT_FINDER_LINE_LIMIT = 120       # max timestamped lines per LLM prompt
MOMENT_FINDER_COARSE_SCAN = 3        # coarse chunk scans when a long VOD has no cue words

# --- Keyword moment lexicon (exact-moment localization by spoken phrase) -----
# Per known event (key matches data/event_videos.json): each entry lists the
# spoken phrases that MARK the moment in the captions.  When a segment speaks a
# phrase, a window is pinned there (keyword_located), which is a STRONG ranking
# + export signal independent of the fuzzy semantic/LLM windows.  Patterns are
# matched case-insensitively against normalized caption text with re.search.
# Only the specs whose ``query_terms`` appear in the sentence are active, so an
# apple-cut query never gets 'champions' windows and vice versa.  Patterns must
# be DISTINCTIVE spoken forms — never generic celebration words like "we won".
EVENT_MOMENT_LEXICON = {
    "streamer games": {
        "apple cut": {
            "query_terms": ("apple", "cut", "cutting", "precision", "gram"),
            "patterns": (r"cutting apples", r"cutting the apples", r"cut the apple",
                         r"cut the apples", r"cut an apple", r"cutting an apple",
                         r"apple is cut", r"apples are cut", r"apple cut",
                         r"2 gram", r"two gram", r"2grams"),
            "pad": 12,
            "priority": 5,
        },
        "crate stacking": {
            "query_terms": ("crate", "crates", "stack", "stacking"),
            "patterns": (r"crates", r"crate", r"stacking", r"18 crates"),
            "pad": 12,
            "priority": 5,
        },
        "crate stacking maya": {
            "query_terms": ("maya", "crate", "crates", "stack", "stacking"),
            "patterns": (r"maya with \d+ crates", r"crates high for maya",
                         r"this is her last crate", r"last crate",
                         r"maya.{0,40}crate", r"maya.{0,40}crates",
                         r"maya.{0,40}stack"),
            "pad": 12,
            "priority": 7,
        },
        "simon says": {
            "query_terms": ("simon", "lightning", "round"),
            "patterns": (r"simon says", r"lightning round", r"simon"),
            "pad": 10,
            "priority": 5,
        },
        "archery": {
            "query_terms": ("bow", "arrow", "arrows", "archery", "archer",
                            "invasion"),
            "patterns": (r"bow and arrow", r"bow & arrow", r"bow and arrows",
                         r"archery", r"arrows", r"archers", r"arrow",
                         r"seven people run", r"live suicide", r"live suicide attempt",
                         r"people run before", r"wait for security"),
            "pad": 14,
            "priority": 5,
        },
        "champions": {
            "query_terms": ("won", "win", "wins", "winner", "champion",
                            "champions", "victory", "worldwide", "grand final",
                            "final"),
            "patterns": (r"crown our champions", r"crowned", r"they have dominated",
                         r"we came, we dominated", r"team worldwide wins",
                         r"worldwide wins", r"champions of the streamer games",
                         r"are the champions", r"the champions"),
            "pad": 8,
            "priority": 6,
        },
    },
}

# --- Signature mining from short clips (autonomous event phrasing) ----------
# When the exact-query pass rejects short clips on title/channel policy, their
# transcripts are the ONE place the event's signature dialogue is guaranteed to
# exist verbatim ("I know my route gang, I'm watching bro" for the Kai Cenat
# police-chase).  Mining transcribes the top rejected shorts, asks the LLM for
# the exact spoken phrases that MARK the moment, and feeds them back as
# keyword-lexicon patterns so long VODs in the pool get pinned at the precise
# second (keyword_located) — the same knowledge a human gets from watching the
# short clip first.
SIGNATURE_MINING_ENABLED = True
SIGNATURE_MINING_MAX_CLIPS = 4      # rejected shorts transcribed per sentence
SIGNATURE_MINING_MAX_SECONDS = 300  # short-clip cap (long VODs are in the pool anyway)
SIGNATURE_MINING_MIN_SIM = 0.15     # title+transcript vs sentence relevance floor
SIGNATURE_MINING_MAX_PHRASES = 5    # lexicon patterns kept per sentence

# --- Cross-source corroboration ---------------------------------------------
# A mined signature phrase heard in BOTH a short clip and a long-VOD window is
# independent confirmation that the window is the same moment ("two witnesses").
CORROBORATION_ENABLED = True
CORROBORATION_LIFT = 0.10           # proposal-score lift for corroborated windows

# --- Source rejection --------------------------------------------------------
REJECT_REACTION = True            # 'reaction/REACTS TO' titles are hard-rejected
REJECT_REUPLOAD = True            # third-party channels re-uploading a streamer are rejected
# YouTube channels that legitimately carry a streamer's RAW footage (VODs/streams).
# A third-party channel qualifies only if it also names the subject streamer and
# carries none of the reactor/aggregator markers below.
STREAMER_RAW_CHANNEL_MARKERS = ("stream", "streams", "live", "vod", "vods", "raw",
                                "irl", "unfiltered", "unedited")
# Channel names that mark a REACTOR or edited clip shop (never a raw host).
REACTOR_CHANNEL_MARKERS = ("reaction", "reacting", "reacts", "react to", "reacts to",
                           "watch party", "watching", "watches", "watch along",
                           "watchalong")
# Channel names that mark an aggregator / compilation shop ("Daily Dose of
# LivestreamFail") or a third-party recap channel.
AGGREGATOR_CHANNEL_MARKERS = ("daily dose", "dose", "livestreamfail", "streamfail",
                              "best twitch", "best of", "compilation", "compilations",
                              "montage", "highlight", "highlights", "moments",
                              "recap", "shorts", "fail")

# --- Twitch clip content verification ----------------------------------------
TWITCH_CLIP_PAGES = 50           # max pages to fetch (100 clips/page; 50 = up to 5000 clips)
TWITCH_VERIFY_PER_LANE = 25      # how many title-ranked clips get audio-verified
TWITCH_VERIFY_PER_LANE_EVENT = 40  # event queries: more clips verified
TWITCH_FOLLOWERS_SORT = True      # rank ambiguous streamer names by followers

# --- Content verification -----------------------------------------------------
VERIFY_SEMANTIC_MIN = 0.35        # transcript-verified clips must beat this MiniLM bar
VERIFY_SEMANTIC_MIN_NARRATION = 0.25  # narration/context sentences: topic rarely spoken verbatim
VERIFY_RERANK_MIN = 0.001        # informational only: ms-marco CrossEncoder scores real
                                 # conversational captions ~0.00, so it never gates verification;
                                 # semantic (MiniLM) + source-lane authenticity guard quality
VERIFY_MIN_ENTITY_HITS = 0       # soft signal: source lanes already guarantee the streamer; requiring the name
                                 # to be spoken in each window is over-strict for narration/context
VERIFY_REQUIRE_TOPIC = True       # clip transcript must mention the sentence topic

# --- Face identity verification (YuNet detect + ArcFace embed) ---------------
FACE_VERIFY_ENABLED = True
FACE_MATCH_THRESHOLD = 0.42     # cosine similarity floor for a subject match
FACE_SAMPLE_FRAMES = 12         # frames sampled per downloaded clip for verification
FACE_REF_FALLBACK = True        # export top-semantic images when <IMAGES_PER_SENTENCE pass the face gate
FACE_REQUIRE_ALL = True         # images: require EVERY subject on screen for face_verified rank (clips use a two-tier all-then-any gate)
MODELS_DIR = BASE_DIR / "cache" / "models"
FACE_CACHE = BASE_DIR / "cache" / "faces"

# --- Image fallback lane (used when Twitch/YouTube find nothing) --------------
IMAGE_LANE_ENABLED = True
IMAGE_LANE_SOURCES = ("google", "bing")   # try Google via Chrome session first, then Bing
IMAGE_MAX_RESULTS = 20            # images fetched per query
IMAGE_QUERY_MAX = 3               # queries built from the sentence + streamer
IMAGE_SEMANTIC_MIN = 0.26         # CLIP relevance floor vs the sentence (unrelated images score ~0.2)
IMAGE_MIN_SIDE = 200              # ignore tiny icons/logos (px)
IMAGE_MIN_BYTES = 3 * 1024
IMAGE_MAX_BYTES = 10 * 1024 * 1024
IMAGE_NET_TIMEOUT = 20
IMAGE_NEAR_DUP_COSINE = 0.90     # exported image closer to an accepted one than this = near-dup
IMAGES_DIR = BASE_DIR / "images"

# --- Scoring fusion ----------------------------------------------------------
SCORE_WEIGHTS = {
    "semantic": 0.15,
    "rerank": 0.10,
    "llm": 0.10,        # LLM judge (0-10, /10)
    "lexical": 0.05,
    "llm_moment": 0.10, # LLM moment finder exactness (0-1)
    "vision": 0.15,     # general CLIP scene match on sampled frames (always computed now)
    "rawness": 0.05,
    "visual_raw": 0.02,
    "editorial": 0.03,
    "title_match": 0.05,
    "metadata": 0.02,
    "anchor": 0.15,     # visual-anchor match (image->image CLIP; present when anchors exist)
    "event_visual": 0.10,  # EventProbe scene match on local frames (visual-event queries)
    "vlm": 0.15,        # VLM verdict that the moment is actually happening (0-1)
}

# --- Keyword models ----------------------------------------------------------
EDITED_VIDEO_KEYWORDS = [
    "highlight", "highlights", "recap", "review", "reaction", "commentary",
    "edited", "compilation", "montage", "best of", "story time", "storytime",
    "narration", "voiceover", "voice-over", "supercut", "analysis", "breakdown",
    "summary", "retrospective", "clips", "clip", "trailer", "rewind", "meme",
    "snippet", "teaser", "best moments", "top 10",
    "tips", "tip", "lesson", "guide", "tutorial", "explainer", "course",
    "documentary", "official", "music video", "lecture",
    "watch party", "reacting", "watching live", "day 1", "day 178",
]
RAW_VIDEO_KEYWORDS = [
    "stream", "live", "raw", "unedited", "unfiltered", "twitch", "irl", "full stream",
    "full video", "live stream", "recording", "direct", "real", "original", "vod",
]
EDITORIAL_EVIDENCE_KEYWORDS = [
    "fans", "noticed", "highlights", "highlight", "reaction", "edited",
    "clip", "watch", "subscribe", "views", "trending", "best of",
    "reuploaded", "vlog", "montage", "recap", "review", "analysis",
    "story time", "storytime", "supercut", "reaction video", "meme",
    "watch party", "reacting to", "day 178",
]

# --- Twitch source ----------------------------------------------------------
TWITCH_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
TWITCH_GQL_ENDPOINT = "https://gql.twitch.tv/gql"
TWITCH_SEARCH_RESULTS = 15
TWITCH_VODS_TO_ANALYSE = 3
TWITCH_MAX_VOD_SECONDS = 8 * 3600
TWITCH_QUICK_CLIP_SECONDS = 3600
TWITCH_FRAME_RENDITION_INDEX = -1
TWITCH_TRANSCRIPT_CHUNK_SECONDS = 1800
TWITCH_CHUNK_EXTRACT_TIMEOUT = 600
TWITCH_TRANSCRIPT_CACHE_DIR = CACHE_DIR / "twitch_transcripts"
TWITCH_EDITORIAL_MARKERS = ("watch party", "reaction", "reacting", "day 178", "recap", "review")
# Skip the full-VOD transcription lane entirely (multi-hour VODs are too slow on
# CPU); only the short-clips section (twitch_lane) is searched for Twitch.
TWITCH_VOD_LANE_ENABLED = False

# --- Kick VOD lane --------------------------------------------------------
# Kick has no captions and no public search API; discovery is the subject
# streamer's own channel VOD list.  VODs are processed via HLS window seeks
# (never fully downloaded) and transcribed in resumable chunks.
KICK_API_BASE = "https://kick.com/api"
KICK_CHANNEL_VIDEOS_LIMIT = 30
KICK_VODS_TO_ANALYSE = 1
KICK_MAX_VOD_SECONDS = 24 * 3600
KICK_QUICK_CLIP_SECONDS = 3600
KICK_TRANSCRIPT_CHUNK_SECONDS = 1800
KICK_CHUNK_EXTRACT_TIMEOUT = 600
KICK_TRANSCRIPT_CACHE_DIR = CACHE_DIR / "kick_transcripts"
# The Kick VOD lane pulled wrong/off-topic candidates for event queries and Kick
# has no captions/search; off unless a streamer's own Kick POV is explicitly
# wanted (then the HLS window lane still works).
KICK_VOD_LANE_ENABLED = False

# --- Web discovery lane -------------------------------------------------------
# Open-web discovery: per query, search DDG (urllib HTML POST; headed-Chrome
# Google fallback) for the moment on ANY yt-dlp-supported platform (X/Twitter
# mirrors, native Twitch clips, Kick, YouTube uploads the verbatim search
# missed).  Short hits become whole-clip candidates; long YouTube hits join
# the YouTube lane pool.  Login-walled posts are skipped and logged.
DISCOVERY_ENABLED = True
DISCOVERY_MAX_HOOKS = 6           # search hooks per query (sentence + planner hints)
DISCOVERY_MAX_URLS = 24           # result URLs resolved per query
DISCOVERY_MIN_LLM_SCORE = 0.35    # LLM relevance floor (title/channel vs sentence)
DISCOVERY_TIMEOUT = 15            # seconds per HTTP fetch
DISCOVERY_YT_POOL_MAX = 6         # long YouTube hits handed to the YouTube lane

# --- Run report (structured confidence + missing matches) ---------------------
# After each sentence, run_report appends JSON lines to this file: one per
# final candidate (all scoring signals + exported flag) and a summary line
# with quota/shortfall and WHICH gate rejected what (missing_matches).
RUN_REPORT_FILE = BASE_DIR / "pipeline.log"

# --- Export ----------------------------------------------------------------
SKIP_EXPORTED = True
EXPORTED_REGISTRY = BASE_DIR / "exported_registry.json"

# --- Short builder (make_short.py) -------------------------------------------
# Turns a folder of clips/images + a voiceover MP3 into a 9:16 YouTube Short:
# the voiceover is split into scenes (sentences), the best 3-6s window per scene
# is trimmed from the clips (motion/sharpness/shot/face scoring) and stitched
# exactly in sync with the voiceover.  Images fill scenes with no matching window.
SHORTS_DIR = OUTPUT_DIR / "shorts"
SHORT_SCAN_DIR = INPUT_DIR              # default media folder for make_short.py
SHORT_CACHE_DIR = CACHE_DIR / "short_builder"
SHORT_SCAN_CACHE_DIR = SHORT_CACHE_DIR / "scans"
SHORT_WORK_DIR = SHORT_CACHE_DIR / "work"
SHORT_SCENES_CACHE = SHORT_CACHE_DIR / "scenes.json"
SHORT_ASSIGNMENT_REPORT = SHORT_CACHE_DIR / "assignment.json"
SHORT_ASSIGNMENT_REPORT_TXT = SHORT_CACHE_DIR / "assignment.txt"

SHORT_SCAN_VERSION = 3            # bump to invalidate cached per-clip scans
SHORT_PLAN_VERSION = 6            # bump to invalidate the cached scene/beat plan
SHORT_SCAN_FPS = 2.0              # frame extraction rate for window scoring
SHORT_FRAME_WIDTH = 640           # scan frame width (keeps motion/sharpen fast)
SHORT_WINDOW_STRIDE = 1.0         # candidate start spacing (seconds)
SHORT_LENGTH_BUCKETS = (2.0, 2.5)
SHORT_MAX_WINDOW = 2.5            # longest scored window (= beat cap)
SHORT_NMS_GAP = 2.0               # min seconds between windows kept from one clip
SHORT_MAX_CANDIDATES_PER_CLIP = 8
SHORT_BEAT_TARGET_SECONDS = 2.0   # fixed beat length -> uniform cuts every ~2s
SHORT_BEAT_MIN_SECONDS = 2.0      # beats rendering shorter than this merge into neighbours
SHORT_BEAT_MAX_SECONDS = 2.5      # every on-screen visual stays <= this
SHORT_SENTENCE_MAX_SECONDS = 6.0  # sentence-forming safety cap (beats split further)
SHORT_SPLIT_MIN_SECONDS = 3.5     # sentences must exceed this before splitting
SHORT_SENTENCE_PAUSE = 0.45       # word gap that forces a scene break
SHORT_MIN_SCORE = 0.0             # 0 = auto: global percentile of candidate scores
SHORT_MIN_SCORE_PERCENTILE = 60   # auto threshold keeps windows above this percentile
SHORT_MIN_SCORE_FLOOR = 0.30      # absolute floor beneath the auto threshold
SHORT_SCORE_WEIGHTS = {
    "motion": 0.40,
    "sharpness": 0.15,
    "stability": 0.10,
    "shot": 0.20,
    "face": 0.15,
}
SHORT_FACE_SAMPLE_INTERVAL = 1.5  # seconds between face-detection samples
SHORT_SEMANTIC_ENABLED = True     # CLIP re-rank of the per-scene shortlist
SHORT_SEMANTIC_SHORTLIST = 6      # candidates shown to CLIP per scene
SHORT_SEMANTIC_WEIGHT = 0.30      # CLIP share in the final candidate ranking
SHORT_STIMULUS_BONUS = 0.06       # extra weight given to high-motion windows (tie-break)
SHORT_IMAGE_FALLBACK_SEMANTIC = True  # CLIP picks the best-matching unused image
SHORT_OUT_WIDTH = 1080
SHORT_OUT_HEIGHT = 1920
SHORT_OUT_FPS = 30
SHORT_OUT_PRESET = "veryfast"
SHORT_OUT_CRF = 23
SHORT_OUT_AUDIO_BITRATE = "128k"
SHORT_KENBURNS_ZOOM = 0.10        # total zoom amount for still-image scenes
SHORT_WORKER_JOBS = 2             # parallel ffmpeg scene renders
SHORT_LLM_LABELS = True           # qwen2.5 condenses scene text into CLIP-friendly labels
SHORT_STACK_ENABLED = True        # render shots as top/bottom full-bleed duplicates
SHORT_STACK_TOP_ZOOM = (1.00, 1.05)   # (start, end) zoom of the top copy over the shot
SHORT_STACK_TOP_PAN = (10, 0)         # (dx, dy) px camera drift of the top copy over the shot
SHORT_STACK_BOT_ZOOM = (1.15, 1.25)   # (start, end) zoom of the bottom copy (more than top)
SHORT_STACK_BOT_PAN = (-20, 12)       # (dx, dy) px camera drift of the bottom copy
SHORT_FOLDER_MATCH = True         # assign each beat to its best topic folder first
SHORT_FOLDER_MIN_SCORE = 0.20     # floor for a beat's matched topic-folder pool
SHORT_PERSON_MIN_SCORE = 0.15     # floor for person-pinned pools (identity > quality)
SHORT_PERSON_CARRY_BEATS = 6      # keep a named person across the following beats
SHORT_PERSON_CARRY_TOPIC_STRENGTH = 1.8  # a stronger unrelated topic breaks carry
SHORT_PERSON_ALIASES = {
    # Whisper commonly mis-hears streamer names; map a mis-transcription to the
    # person id derived from the topic folders so the story still pins correctly.
    "secuna": "sykkuno",
}
SHORT_PERSON_CARRY_TRANSITIONS = ("finally", "in conclusion", "up next", "lastly")

# --- Light synonym map -------------------------------------------------------
SYNONYMS = {
    "canada": {"canadian"},
    "rule": {"rules", "banned", "ban", "restricted"},
    "banned": {"ban", "rule", "rules"},
    "ban": {"banned", "rule", "rules"},
    "lifelong": {"life", "forever", "permanent", "lifetime"},
    "forever": {"life", "lifelong", "permanent"},
    "never": {"cant", "cannot", "won't", "wont", "won`t"},
    "confront": {"argument", "argue", "yell", "fight", "defend", "harass", "attack"},
    "harass": {"bother", "yell", "touch", "security", "fight", "attack"},
    "fight": {"argument", "argue", "yell", "confront", "attack"},
    "stream": {"live", "vod", "recording"},
    "viral": {"trend", "trending", "famous", "popular"},
    "million": {"mil", "1.4m", "1,400,000", "1.4 million"},
    "money": {"cash", "dollars", "bucks"},
    "spend": {"spent", "spending", "paid", "wasted"},
}

for directory in (INPUT_DIR, OUTPUT_DIR, REPORTS_DIR, FRAMES_DIR, CLIPS_DIR,
                  CACHE_DIR, TRANSCRIPT_CACHE_DIR, MOMENT_FINDER_CACHE_DIR,
                  OCR_FRAME_CACHE, LOGS_DIR, TWITCH_TRANSCRIPT_CACHE_DIR,
                  TWITCH_CLIP_CACHE, KICK_TRANSCRIPT_CACHE_DIR, DATA_DIR,
                  IMAGES_DIR, MODELS_DIR, FACE_CACHE, ANCHOR_CACHE,
                  EVENT_VISION_CACHE, VLM_CACHE,
                  SHORTS_DIR, SHORT_CACHE_DIR, SHORT_SCAN_CACHE_DIR,
                  SHORT_WORK_DIR):
    directory.mkdir(parents=True, exist_ok=True)
