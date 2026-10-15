"""Short builder — voiceover-synced 9:16 shorts from a folder of clips/images.

Pipeline: voiceover_plan (Whisper scenes) -> footage_index (input scan) ->
window_scanner (best 3-6s windows per clip) -> semantic_refine (CLIP re-rank)
-> assigner (scene -> window / image fallback) -> compositor (ffmpeg render).
"""
from modules.short_builder import (assigner, compositor, footage_index,
                                   semantic_refine, voiceover_plan,
                                   window_scanner)

__all__ = [
    "voiceover_plan", "footage_index", "window_scanner",
    "semantic_refine", "assigner", "compositor",
]
