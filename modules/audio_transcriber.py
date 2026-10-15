"""Whisper-based transcription fallback for videos without captions."""
import subprocess
import tempfile
import logging

import config

logger = logging.getLogger(__name__)

_model_cache = {}


def model_for(languages: list[str] | None = None) -> str:
    """Whisper model size for the query's spoken languages.

    Defaults to WHISPER_MODEL_SIZE; languages listed in WHISPER_LANG_MODELS get
    their dedicated (usually larger, more accurate) model.
    """
    langs = {str(l).lower().split("-")[0] for l in (languages or [])}
    for lang in langs:
        override = config.WHISPER_LANG_MODELS.get(lang)
        if override:
            return override
    return config.WHISPER_MODEL_SIZE


def _device_and_compute() -> tuple[str, str]:
    device = config.WHISPER_DEVICE
    compute = config.WHISPER_COMPUTE
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def _get_model(model_size: str):
    from faster_whisper import WhisperModel
    if model_size not in _model_cache:
        device, compute = _device_and_compute()
        logger.info("Loading Whisper model '%s' on %s (%s)...", model_size, device, compute)
        try:
            _model_cache[model_size] = WhisperModel(model_size, device=device, compute_type=compute)
        except Exception as exc:
            logger.warning("Whisper on %s failed (%s); falling back to CPU int8", device, exc)
            _model_cache[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _model_cache[model_size]


def _extract_audio(stream_url, out_path, extra_args=None, max_seconds=None,
                   start_seconds=None, timeout=300):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *(extra_args or [])]
    if start_seconds:
        cmd += ["-ss", str(start_seconds)]
    cmd += ["-i", stream_url]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", out_path]
    subprocess.run(cmd, check=True, timeout=timeout)


def transcribe_file(audio_path, model_size="tiny"):
    """Transcribe an already-extracted local WAV; returns segment dicts or []."""
    try:
        model = _get_model(model_size)
    except Exception as exc:
        logger.warning("Could not load Whisper model: %s", exc)
        return []
    try:
        segments, _ = model.transcribe(audio_path, beam_size=1, vad_filter=True)
        return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    except Exception as exc:
        logger.warning("Whisper transcription failed: %s", exc)
        return []


def transcribe(stream_url, extra_args=None, model_size="tiny", timeout=400, max_seconds=None):
    """Extract audio (small relative to video) and transcribe with Whisper.

    Returns [{"start": float, "end": float, "text": str}, ...] or [] on failure.
    """
    try:
        model = _get_model(model_size)
    except Exception as exc:
        logger.warning("Could not load Whisper model: %s", exc)
        return []

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = tmp + "\\audio.wav"
        try:
            _extract_audio(stream_url, audio_path, extra_args=extra_args,
                           max_seconds=max_seconds, timeout=timeout)
        except Exception as exc:
            logger.warning("Audio extraction failed: %s", exc)
            return []
        try:
            segments, _ = model.transcribe(audio_path, beam_size=1, vad_filter=True)
            return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
        except Exception as exc:
            logger.warning("Whisper transcription failed: %s", exc)
            return []