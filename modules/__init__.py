"""
VidGetter V2 Modules Package

Each module exposes a small set of focused functions and is orchestrated from
run.py.  Modules do not call each other except for the documented scoring and
visual helpers.
"""

__all__ = [
    "downloader",
    "youtube_search",
    "transcript_cache",
    "audio_transcriber",
    "semantic_embedder",
    "text_match",
    "scoring",
    "moment_proposer",
    "vision_ranker",
    "overlay_detector",
    "llm_reranker",
    "clip_exporter",
    "report_writer",
]
