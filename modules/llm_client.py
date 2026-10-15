"""LLM client: OpenAI-compatible HTTP API first, local Ollama fallback.

All HTTP goes through urllib (matching the rest of the codebase).  When
config.LLM_API_BASE + LLM_API_KEY are set, every call goes to the
OpenAI-compatible chat-completions endpoint (any provider: OpenAI, Groq,
OpenRouter, LM Studio, vLLM); on failure it falls back to Ollama, and any
remaining failure returns None so callers degrade to rule-based behaviour.
"""
import base64
import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import config

logger = logging.getLogger(__name__)

_models = None
_model_attempted = False
_api_ok: bool | None = None

_CONTROL_CHARS = dict.fromkeys(
    list(range(0x00, 0x08)) + list(range(0x0B, 0x1F)) + list(range(0x7F, 0xA0)))


def _sanitize(text: str | None) -> str:
    """Strip control characters and lone surrogates so json.dumps never fails.

    Ollama payloads built with ``json.dumps(...).encode("utf-8")`` crash on
    control chars and on lone surrogates (common in YouTube auto-captions), and
    ollama returns HTTP 400 for some of them.  Clean before sending."""
    if not text:
        return ""
    text = text.translate(_CONTROL_CHARS)
    return "".join(ch for ch in text
                   if not (0xD800 <= ord(ch) <= 0xDFFF))


def _api_configured() -> bool:
    return bool(config.LLM_API_BASE and config.LLM_API_KEY)


def available() -> bool:
    global _models, _model_attempted, _api_ok
    if _api_ok is None and _api_configured():
        # A cheap models-list probe; most compatible providers implement it.
        try:
            req = Request(config.LLM_API_BASE + "/models",
                          headers={"Authorization": "Bearer " + config.LLM_API_KEY})
            with urlopen(req, timeout=15) as resp:
                resp.read()
            _api_ok = True
            logger.info("LLM API active: %s (model %s)",
                        config.LLM_API_BASE, config.LLM_API_MODEL)
        except Exception as exc:
            _api_ok = False
            logger.info("LLM API %s unreachable (%s); using Ollama",
                        config.LLM_API_BASE, exc)
    if _api_ok:
        return True
    if _model_attempted:
        return bool(_models)
    _model_attempted = True
    try:
        with urlopen(config.OLLAMA_HOST + "/api/tags", timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
        _models = {m.get("name", "").split(":")[0]: m.get("name") for m in payload.get("models", [])}
        logger.info("Ollama models: %s", list(_models.values()))
    except Exception as exc:
        logger.info("Ollama unavailable (%s); falling back to rule-based decisions", exc)
        _models = {}
    return bool(_models)


def _resolve_model(require_vision: bool = False) -> str | None:
    if not available():
        return None
    if require_vision and not config.LLM_VISION_ENABLED:
        return None
    if require_vision:
        for name in (config.LLM_VISION_MODEL, config.LLM_VISION_FALLBACK):
            key = name.split(":")[0]
            if key in _models or name in _models.values():
                return _models.get(key, name)
        # Any installed vision-capable model will do.
        for key, full in _models.items():
            if key in config.LLM_VISION_MODELS:
                return full
        return None
    primary = config.LLM_MODEL
    fallback = config.LLM_MODEL_FALLBACK
    for name in (primary, fallback):
        key = name.split(":")[0]
        if key in _models or name in _models.values():
            return _models.get(key, name)
    if _models:
        return next(iter(_models.values()))
    return None


def vision_capable() -> bool:
    """True when a vision model is installed AND enabled."""
    if not config.LLM_ENABLED or not config.LLM_VISION_ENABLED or not available():
        return False
    return _resolve_model(require_vision=True) is not None


def _api_image_data_uri(b64: str) -> str:
    """Wrap a raw base64 image in a data URI (mime sniffed from the magic bytes)."""
    mime = "image/png" if b64.startswith("iVBOR") else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _api_chat(prompt: str, system: str | None, json_mode: bool,
              images: list[str] | None) -> str | None:
    """One OpenAI-compatible chat completion; None on any failure."""
    model = config.LLM_API_MODEL
    if images and config.LLM_API_VISION_MODEL:
        model = config.LLM_API_VISION_MODEL
    content: object = _sanitize(prompt)
    if images:
        parts = [{"type": "text", "text": _sanitize(prompt)}]
        for img in images:
            if img:
                parts.append({"type": "image_url",
                              "image_url": {"url": _api_image_data_uri(img)}})
        content = parts
    messages = []
    if system:
        messages.append({"role": "system", "content": _sanitize(system)})
    messages.append({"role": "user", "content": content})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + config.LLM_API_KEY}
    try:
        req = Request(config.LLM_API_BASE + "/chat/completions", data=data,
                      headers=headers)
        with urlopen(req, timeout=config.LLM_API_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
        choice = (body.get("choices") or [{}])[0]
        return (choice.get("message") or {}).get("content") or None
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:600]
        except Exception:
            pass
        # Some providers reject response_format; retry once without it.
        if json_mode and exc.code == 400:
            payload.pop("response_format", None)
            try:
                req = Request(config.LLM_API_BASE + "/chat/completions",
                              data=json.dumps(payload).encode(), headers=headers)
                with urlopen(req, timeout=config.LLM_API_TIMEOUT) as resp:
                    body = json.loads(resp.read().decode("utf-8", "replace"))
                choice = (body.get("choices") or [{}])[0]
                return (choice.get("message") or {}).get("content") or None
            except Exception as exc2:
                logger.warning("LLM API call failed (%s): %s", model, exc2)
                return None
        logger.warning("LLM API call failed (%s): HTTP %s %s", model, exc.code, detail)
        return None
    except (URLError, OSError, ValueError) as exc:
        logger.warning("LLM API call failed (%s): %s", model, exc)
        return None


def chat(prompt: str, system: str | None = None, json_mode: bool = False,
         images: list[str] | None = None) -> str | None:
    """Single-turn completion; API provider first, Ollama fallback.

    Pass ``images`` (base64-encoded JPEG/PNG) to send a vision-model query; the
    vision model is resolved automatically and the text fallback model is never
    used for image calls.
    """
    if not config.LLM_ENABLED or not available():
        return None
    use_vision = bool(images)
    if _api_ok:
        reply = _api_chat(prompt, system, json_mode, images)
        if reply is not None:
            return reply
        logger.info("LLM API failed; falling back to Ollama")
    model = _resolve_model(require_vision=use_vision)
    if not model:
        if use_vision:
            logger.warning("No vision-capable model available; image query dropped")
        return None
    messages = []
    if system:
        messages.append({"role": "system", "content": _sanitize(system)})
    messages.append({"role": "user", "content": _sanitize(prompt)})
    if use_vision:
        messages[-1]["images"] = [img for img in images if img]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": bool(config.LLM_THINK),
        "options": {
            "temperature": config.LLM_TEMPERATURE,
            "num_predict": config.LLM_MAX_TOKENS,
            # Ollama's default num_ctx is 4096; a vision call with several
            # contact sheets (each ~470 image tokens) blows past it and ollama
            # answers HTTP 400 "exceeds the available context size".  Raise the
            # per-request context so the judge can actually SEE the frames.
            "num_ctx": config.LLM_CONTEXT,
        },
    }
    if json_mode:
        payload["format"] = "json"
    data = json.dumps(payload).encode()
    request = Request(config.OLLAMA_HOST + "/api/chat", data=data,
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=config.LLM_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
        return (body.get("message") or {}).get("content") or None
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:600]
        except Exception:
            pass
        logger.warning("Ollama call failed (%s): HTTP %s %s", model, exc.code, detail)
        return None
    except (URLError, OSError, ValueError) as exc:
        logger.warning("Ollama call failed (%s): %s", model, exc)
        return None


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of an LLM reply that may include markdown fences."""
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def chat_json(prompt: str, system: str | None = None,
              images: list[str] | None = None) -> dict:
    """Ask the LLM for a JSON object; returns {} on any failure."""
    reply = chat(prompt, system=system, json_mode=True, images=images)
    if not reply:
        return {}
    parsed = _extract_json(reply)
    if not parsed:
        logger.warning("LLM returned non-JSON: %.200s", reply)
    return parsed


def force_reload() -> None:
    global _models, _model_attempted, _api_ok
    _models = None
    _model_attempted = False
    _api_ok = None
