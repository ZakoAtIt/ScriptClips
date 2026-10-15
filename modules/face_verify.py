"""Face identity verification: YuNet detect + ArcFace embed.

Confirms which known subjects (from script core_entities) are actually ON
SCREEN in an image, a Twitch clip, or a downloaded YouTube clip — the check
that audio transcripts can never provide.

References: a KB streamer's Twitch avatar (authoritative), otherwise the
Wikipedia lead image for the display name, cached under cache/faces. A subject
with no usable reference falls back to name-in-metadata/audio verification.
Models (YuNet + ArcFace/MobileFaceNet ONNX) are downloaded once to cache/models.
"""
import hashlib
import json
import logging
import math
import re
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image

import config
from modules import streamers, twitch_source

logger = logging.getLogger(__name__)

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

_ONNX = None
_YUNET_SESSION = None
_ARCFACE_SESSION = None
_THRESHOLD = config.FACE_MATCH_THRESHOLD

_YUNET_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx")
_YUNET_HF = ("opencv/face_detection_yunet", "face_detection_yunet_2023mar.onnx")
_ARC_SOURCES = (
    ("DavidHoa/buffalo_l", "w600k_r50.onnx"),
    ("richarrrddd/w600k_r50_v1", "w600k_r50.onnx"),
)
_ARC_GITHUB_BUNDLES = (
    ("https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
     "w600k_mbf.onnx"),
    ("https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
     "w600k_r50.onnx"),
)

_ARC_DST = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def _models_ready() -> bool:
    return (_YUNET_SESSION is not None and _ARCFACE_SESSION is not None)


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(256 * 1024):
                    fh.write(chunk)
        tmp.replace(dest)
        return True
    except Exception as exc:
        logger.warning("Model download failed for %s: %s", url, exc)
        tmp.unlink(missing_ok=True)
        return False


def _extract_from_zip(zip_url: str, wanted: str, dest_dir: Path) -> Path | None:
    tmp = dest_dir / "bundle.zip"
    if not _download(zip_url, tmp):
        return None
    try:
        with zipfile.ZipFile(tmp) as zf:
            names = zf.namelist()
            match = next((n for n in names if wanted in n.lower() and n.lower().endswith(".onnx")), None)
            if not match:
                match = next((n for n in names if n.lower().endswith(".onnx") and "mbf" in n.lower()), None)
            if not match:
                logger.warning("No embedding onnx found in bundle %s (files: %s)",
                               zip_url, names)
                return None
            out = dest_dir / Path(match).name
            with zf.open(match) as src, open(out, "wb") as fh:
                fh.write(src.read())
            return out
    except Exception as exc:
        logger.warning("Bundle extract failed for %s: %s", zip_url, exc)
        return None
    finally:
        tmp.unlink(missing_ok=True)


def _hf_model(repo_id: str, filename: str) -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id, filename))
    except Exception as exc:
        logger.warning("HF model download failed for %s/%s: %s", repo_id, filename, exc)
        return None


def _ensure_models() -> bool:
    global _YUNET_SESSION, _ARCFACE_SESSION
    if _models_ready():
        return True
    if not config.FACE_VERIFY_ENABLED:
        return False
    try:
        import onnxruntime as ort
    except Exception as exc:
        logger.warning("onnxruntime unavailable: %s", exc)
        return False
    _ONNX = ort

    yunet_path = config.MODELS_DIR / "face_detection_yunet.onnx"
    if not yunet_path.exists() or yunet_path.stat().st_size < 100_000:
        yunet_path.unlink(missing_ok=True)
        found = _hf_model(*_YUNET_HF)
        if found:
            if found != yunet_path:
                try:
                    yunet_path.write_bytes(found.read_bytes())
                except Exception:
                    yunet_path = found
            else:
                yunet_path = found
        if not yunet_path.exists() or yunet_path.stat().st_size < 100_000:
            yunet_path.unlink(missing_ok=True)
            _download(_YUNET_URL, yunet_path)

    arc_path = None
    for repo_id, filename in _ARC_SOURCES:
        dest = config.MODELS_DIR / filename
        if not dest.exists():
            found = _hf_model(repo_id, filename)
            if found and found != dest:
                try:
                    dest.write_bytes(found.read_bytes())
                except Exception:
                    dest = found
            elif found:
                dest = found
        if dest.exists():
            arc_path = dest
            break
    if arc_path is None:
        for url, wanted in _ARC_GITHUB_BUNDLES:
            found = _extract_from_zip(url, wanted, config.MODELS_DIR)
            if found:
                arc_path = found
                break

    if not yunet_path.exists() or not arc_path or not arc_path.exists():
        logger.warning("Face models unavailable (yunet=%s arc=%s)",
                       yunet_path.exists(), bool(arc_path))
        return False

    try:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        _YUNET_SESSION = ort.InferenceSession(str(yunet_path), sess_options=opts)
        _ARCFACE_SESSION = ort.InferenceSession(str(arc_path), sess_options=opts)
        logger.info("Face models loaded: %s / %s", yunet_path.name, arc_path.name)
        return True
    except Exception as exc:
        logger.warning("Face model session init failed: %s", exc)
        return False


# --------------------------------------------------------------------------
# Detection + embedding
# --------------------------------------------------------------------------
def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = src.astype(np.float32)
    dst = dst.astype(np.float32)
    n = src.shape[0]
    src_mean = src.mean(axis=0, keepdims=True)
    dst_mean = dst.mean(axis=0, keepdims=True)
    sc, dc = src - src_mean, dst - dst_mean
    src_var = np.mean(np.sum(sc * sc, axis=1))
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / src_var
    t = dst_mean.T - scale * R @ src_mean.T
    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = scale * R
    M[:, 2] = t.ravel()
    return M


def _warp_crop(img: Image.Image, landmark: np.ndarray, size: int = 112) -> Image.Image:
    M = _similarity_transform(landmark, _ARC_DST)
    a, b, c = M[0, 0], M[0, 1], M[0, 2]
    d, e, f = M[1, 0], M[1, 1], M[1, 2]
    det = a * e - b * d
    if abs(det) < 1e-9:
        raise ValueError("degenerate landmark transform")
    ia, ib, id_, ie = e / det, -b / det, -d / det, a / det
    ic = -(ia * c + ib * f)
    if_ = -(id_ * c + ie * f)
    return img.transform((size, size), Image.AFFINE,
                         (ia, ib, ic, id_, ie, if_), resample=Image.BILINEAR)


def _run_yunet_raw(arr: np.ndarray) -> list[dict]:
    """Decode YuNet detections for a single RGB float32 HxWx3 array.

    Boxes and landmarks are returned in the input array's coordinate space.
    """
    in_w, in_h = 640, 640
    h, w = arr.shape[:2]
    ratio = min(in_w / w, in_h / h)
    nw, nh = max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))
    resized = np.asarray(Image.fromarray(arr.astype(np.uint8)).resize(
        (nw, nh), Image.BILINEAR), dtype=np.float32)
    canvas = np.zeros((in_h, in_w, 3), dtype=np.float32)
    canvas[:nh, :nw] = resized
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None]
    outs = _YUNET_SESSION.run(None, {"input": x})
    faces: list[dict] = []
    strides = (8, 16, 32)
    for si, s in enumerate(strides):
        fw = in_w // s
        cls = np.asarray(outs[si]).reshape(-1)
        obj = np.asarray(outs[3 + si]).reshape(-1)
        bb = np.asarray(outs[6 + si]).reshape(-1, 4)
        kp = np.asarray(outs[9 + si]).reshape(-1, 10)
        score = cls * obj
        for i in np.where(score > 0.30)[0]:
            px = float(i % fw) * s
            py = float(i // fw) * s
            cx = float(bb[i, 0]) * s + px
            cy = float(bb[i, 1]) * s + py
            fw_px = math.exp(float(bb[i, 2])) * s
            fh_px = math.exp(float(bb[i, 3])) * s
            lms = np.asarray(
                [[float(kp[i, 2 * j]) * s + px, float(kp[i, 2 * j + 1]) * s + py]
                 for j in range(5)], dtype=np.float32)
            x1, y1 = cx - fw_px / 2, cy - fh_px / 2
            faces.append({"box": (x1 / ratio, y1 / ratio,
                                  (x1 + fw_px) / ratio, (y1 + fh_px) / ratio),
                          "landmark": lms / ratio, "score": float(score[i])})
    return faces


def _detect_faces(img: Image.Image) -> list[dict]:
    """Run YuNet on a 640x640 letterbox and decode the raw head outputs.

    Small images are upscaled first so faces at thumbnail resolution are not
    missed by YuNet's 8-64px feature grid.  Large frames (>=1080p) are also
    scanned on 2x2 half-frame tiles so small on-screen faces (PiP, stadium
    screens, etc.) are detected at a useful scale and embed cleanly.
    """
    if _YUNET_SESSION is None:
        if not _ensure_models():
            return []
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = base.shape[:2]
    scale = 1.0
    if min(h, w) < 360:
        scale = 360.0 / min(h, w)
        base = np.asarray(img.resize((int(round(w * scale)),
                                      int(round(h * scale))), Image.BILINEAR),
                          dtype=np.float32)
    h, w = base.shape[:2]
    raw = _run_yunet_raw(base)
    if min(h, w) >= 1080:
        th, tw = (h + 1) // 2, (w + 1) // 2
        for oy in (0, h // 2):
            for ox in (0, w // 2):
                tile = base[oy:oy + th, ox:ox + tw]
                if tile.size == 0:
                    continue
                for f in _run_yunet_raw(tile):
                    bx1, by1, bx2, by2 = f["box"]
                    f["box"] = (bx1 + ox, by1 + oy, bx2 + ox, by2 + oy)
                    f["landmark"] = f["landmark"] + np.asarray(
                        [ox, oy], dtype=np.float32)
                    raw.append(f)
    faces: list[dict] = []
    for f in raw:
        box = tuple(v / scale for v in f["box"])
        faces.append({"box": box,
                      "landmark": f["landmark"] / scale,
                      "score": f["score"],
                      "area": (box[2] - box[0]) * (box[3] - box[1])})
    faces.sort(key=lambda f: -f["score"])
    kept: list[dict] = []
    for f in faces:
        bx1, by1, bx2, by2 = f["box"]
        overlap = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k["box"]
            ix = max(0.0, min(bx2, kx2) - max(bx1, kx1))
            iy = max(0.0, min(by2, ky2) - max(by1, ky1))
            inter = ix * iy
            uni = (bx2 - bx1) * (by2 - by1) + (kx2 - kx1) * (ky2 - ky1) - inter
            if uni > 0 and inter / uni > 0.3:
                overlap = True
                break
        if not overlap:
            kept.append(f)
    return kept


def _embed_face(img: Image.Image, landmark: np.ndarray) -> np.ndarray:
    if _ARCFACE_SESSION is None:
        return np.zeros(512, dtype=np.float32)
    crop = _warp_crop(img, landmark)
    arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
    arr = (arr - 127.5) / 127.5
    x = arr.transpose(2, 0, 1)[None]
    in_name = _ARCFACE_SESSION.get_inputs()[0].name
    out_name = _ARCFACE_SESSION.get_outputs()[0].name
    emb = np.asarray(_ARCFACE_SESSION.run([out_name], {in_name: x})[0][0],
                     dtype=np.float32)
    norm = float(np.linalg.norm(emb))
    return emb / norm if norm > 0 else emb


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# --------------------------------------------------------------------------
# References (per subject, cached)
# --------------------------------------------------------------------------
def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:64]


def _twitch_avatar(handle: str) -> str | None:
    try:
        payload = {"query": 'query { user(login: "%s") { profileImageURL(width: 300) } }' % handle}
        result = twitch_source._gql(payload)
        return ((result.get("data") or {}).get("user") or {}).get("profileImageURL")
    except Exception as exc:
        logger.info("Twitch avatar fetch failed for %s: %s", handle, exc)
        return None


def _wikipedia_image(name: str) -> str | None:
    from urllib.parse import quote
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(name.replace(' ', '_'))}"
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": _UA})
        resp.raise_for_status()
        data = resp.json()
        return (data.get("originalimage") or {}).get("source") or None
    except Exception as exc:
        logger.info("Wikipedia image lookup failed for %r: %s", name, exc)
        return None


def _reference_image_url(name: str) -> tuple[str | None, str]:
    resolved = streamers.lookup(name)
    if resolved:
        manual = resolved.get("reference_image")
        if manual:
            return str(manual), "kb"
        handle = streamers.twitch_handle(resolved)
        if handle:
            avatar = _twitch_avatar(handle)
            if avatar:
                return avatar, "twitch"
    wiki = _wikipedia_image(name)
    if wiki:
        return wiki, "wikipedia"
    return None, "none"


def reference_for(name: str, force: bool = False) -> dict | None:
    """Return {name, embedding, source} for a subject, or None if unavailable."""
    slug = _slug(name)
    meta_path = config.FACE_CACHE / f"{slug}.json"
    emb_path = config.FACE_CACHE / f"{slug}.npy"
    if not force and meta_path.exists() and emb_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return {"name": meta["name"], "embedding": np.load(emb_path),
                    "source": meta.get("source", "")}
        except Exception:
            pass
    if not _models_ready():
        if not _ensure_models():
            return None
    url, kind = _reference_image_url(name)
    if not url:
        logger.info("No reference image available for %r", name)
        return None
    try:
        resp = requests.get(url, timeout=60, headers={"User-Agent": _UA})
        resp.raise_for_status()
        import io
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as exc:
        logger.info("Reference image download failed for %r: %s", name, exc)
        return None
    faces = _detect_faces(img)
    if not faces:
        logger.info("No face detected in reference image for %r", name)
        return None
    best = max(faces, key=lambda f: (f["score"], f["area"]))
    try:
        emb = _embed_face(img, best["landmark"])
    except Exception as exc:
        logger.info("Reference embedding failed for %r: %s", name, exc)
        return None
    config.FACE_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    ref_img = config.FACE_CACHE / f"{slug}.jpg"
    try:
        _warp_crop(img, best["landmark"]).save(ref_img)
    except Exception:
        pass
    meta_path.write_text(json.dumps({"name": name, "source": kind}, ensure_ascii=False),
                         encoding="utf-8")
    logger.info("Reference face ready for %r (%s)", name, kind)
    return {"name": name, "embedding": emb, "source": kind}


def refs_for(subjects: list[str]) -> tuple[dict[str, np.ndarray], set[str]]:
    """Return (embeddings by name, names with no usable reference)."""
    refs: dict[str, np.ndarray] = {}
    missing: set[str] = set()
    for name in subjects:
        ref = reference_for(name)
        if ref and ref.get("embedding") is not None:
            refs[name] = ref["embedding"]
        else:
            missing.add(name)
    return refs, missing


# --------------------------------------------------------------------------
# Frame extraction + verification
# --------------------------------------------------------------------------
def _probe_duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True)
        out = result.stdout.strip()
        return float(out) if out else None
    except Exception:
        return None


def _extract_frame(video_path: Path, t: float, out: Path) -> Image.Image | None:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-ss", str(t), "-i", str(video_path), "-frames:v", "1",
             "-vf", "scale=1280:-2,format=yuvj420p", "-q:v", "3",
             str(out)], check=True, timeout=60)
        if out.exists():
            return Image.open(out).convert("RGB")
    except Exception as exc:
        logger.info("Frame extraction at %.1fs failed: %s", t, exc)
    return None


def sample_frames(video_path: Path, n: int = 4) -> list[Image.Image]:
    """Extract n frames spread across the video via ffmpeg (no decode in Python)."""
    duration = _probe_duration(video_path)
    if not duration or duration <= 0:
        return []
    frames: list[Image.Image] = []
    with tempfile.TemporaryDirectory(prefix="scface_") as tmpdir:
        for i in range(max(1, n)):
            t = (i + 1) * duration / (n + 1)
            img = _extract_frame(video_path, t, Path(tmpdir) / f"f{i}.jpg")
            if img is not None:
                frames.append(img)
        if not frames:
            # Range-assembled downloads are sometimes seek-broken past the
            # first fragment; early frames are the most likely to decode.
            for i in range(max(1, n)):
                img = _extract_frame(video_path, 0.5 + i * 1.0,
                                     Path(tmpdir) / f"fb{i}.jpg")
                if img is not None:
                    frames.append(img)
    return frames


def verify_frames(frames: list[Image.Image], refs: dict[str, np.ndarray]) -> dict[str, float]:
    """Return best cosine per subject across all frames."""
    if not refs or not frames:
        return {}
    best: dict[str, float] = {name: 0.0 for name in refs}
    for img in frames:
        for face in _detect_faces(img):
            try:
                emb = _embed_face(img, face["landmark"])
            except Exception:
                continue
            for name, ref in refs.items():
                score = _cosine(emb, ref)
                if score > best[name]:
                    best[name] = score
    return best


def verify_image(path, refs: dict[str, np.ndarray]) -> dict[str, float]:
    if not refs:
        return {}
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return {}
    return verify_frames([img], refs)


def subjects_on_screen(scores: dict[str, float]) -> list[str]:
    return [name for name, score in scores.items() if score >= _THRESHOLD]
