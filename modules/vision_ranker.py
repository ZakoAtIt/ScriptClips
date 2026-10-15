"""Optional local visual semantic scoring, loaded only when frames are needed."""
import logging

import numpy as np

import config
from modules.scoring import visual_prompt

logger = logging.getLogger(__name__)

_model = _preprocess = _tokenizer = _device = None
_load_attempted = False


def _load() -> bool:
    global _model, _preprocess, _tokenizer, _device, _load_attempted
    if _load_attempted:
        return _model is not None
    _load_attempted = True
    try:
        import open_clip
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED
        )
        _model.to(_device).eval()
        _tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)
        logger.info("VisionRanker: CLIP model loaded on %s", _device)
        return True
    except Exception as exc:
        logger.warning("Visual model unavailable; continuing with captions/audio: %s", exc)
        return False


def image_features(paths: list[str]):
    """Return normalized CLIP image vectors for the given frame paths (or None)."""
    if not paths or not _load():
        return None
    try:
        import torch
        from PIL import Image
        with torch.no_grad():
            vectors = []
            for path in paths:
                with Image.open(path) as image:
                    tensor = _preprocess(image.convert("RGB")).unsqueeze(0).to(_device)
                vec = _model.encode_image(tensor)
                vec = vec / vec.norm(dim=-1, keepdim=True)
                vectors.append(vec.cpu().numpy()[0])
        return vectors
    except Exception as exc:
        logger.warning("Image feature extraction failed: %s", exc)
        return None


def score_frames(frames: list[dict], query: str) -> list[dict]:
    """Return [{path, timestamp, similarity}] entries sorted best first."""
    if not frames or not _load():
        return []
    prompt = visual_prompt(query)
    try:
        import torch
        from PIL import Image
        text_features = _model.encode_text(_tokenizer([prompt]).to(_device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            scored = []
            for frame in frames:
                with Image.open(frame["path"]) as image:
                    tensor = _preprocess(image.convert("RGB")).unsqueeze(0).to(_device)
                image_features = _model.encode_image(tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = float((text_features @ image_features.T).item())
                scored.append({
                    "path": frame["path"],
                    "timestamp": frame["timestamp"],
                    "similarity": max(0.0, similarity),
                })
        return sorted(scored, key=lambda item: item["similarity"], reverse=True)
    except Exception as exc:
        logger.warning("Visual scoring failed: %s", exc)
        return []


def text_features(texts: list[str]):
    """Normalized CLIP text vectors for the given prompts (or None)."""
    if not texts or not _load():
        return None
    try:
        import torch
        with torch.no_grad():
            feats = _model.encode_text(_tokenizer(texts).to(_device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy()
    except Exception as exc:
        logger.warning("Text feature extraction failed: %s", exc)
        return None


def text_image_scores(texts: list[str], paths: list[str]):
    """CLIP cosine matrix (len(texts) x len(paths)), or None on failure.

    Rows are the text prompts, columns the image paths.  Lets the event probe
    score many scene descriptions against many frames in one batched pass.
    """
    text_vectors = text_features(texts)
    if text_vectors is None or not paths:
        return None
    image_vectors = image_features(paths)
    if image_vectors is None:
        return None
    try:
        mat = np.asarray(image_vectors) @ np.asarray(text_vectors).T
        return mat.T
    except Exception as exc:
        logger.warning("Text-image scoring failed: %s", exc)
        return None


def image_query_scores(paths: list[str], query: str) -> list[float]:
    """CLIP cosine similarity of each image to the raw query text (0..1).

    Used by the image fallback lane; unlike score_frames this scores against
    the exact query text (no visual_prompt decoration).
    """
    if not paths or not _load():
        return []
    try:
        import torch
        from PIL import Image
        with torch.no_grad():
            text_features = _model.encode_text(_tokenizer([query]).to(_device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            scores = []
            for path in paths:
                with Image.open(path) as image:
                    tensor = _preprocess(image.convert("RGB")).unsqueeze(0).to(_device)
                vec = _model.encode_image(tensor)
                vec = vec / vec.norm(dim=-1, keepdim=True)
                scores.append(max(0.0, float((text_features @ vec.T).item())))
        return scores
    except Exception as exc:
        logger.warning("Image text scoring failed: %s", exc)
        return []


def pairwise_similarity(paths: list[str]):
    """Average cosine similarity between consecutive frames (0..1), or None."""
    vectors = image_features(paths)
    if not vectors:
        return None
    sims = [float(np.dot(a, b)) for a, b in zip(vectors, vectors[1:])]
    return float(np.mean(sims)) if sims else 0.5
