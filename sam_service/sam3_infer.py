"""SAM 3.1 image-mode segmentation, wrapped for serving.

The model itself is loaded by `teacher.load_teacher` — the SAME loader the
distillation pipeline uses, so the served model and the teacher that trained
the on-device student can never silently diverge. This module adds only what
serving needs on top of that:

  - load ONCE, keep resident (the worker survives across jobs; a fresh
    `set_image` per frame is unavoidable — that's the backbone encode — but
    the 3.5 GB weights load exactly once per worker lifetime)
  - reuse the encoded text-prompt stage across images in a request
  - turn raw (masks, boxes, scores) into a compact, transport-friendly result:
    polygons or RLE instead of H×W bool arrays, optional masked crops

Nothing here is CUDA-specific; on this Mac it runs on MPS (see the SAM 3.1 MPS
port notes), which is exactly how `test_local.py` validates the handler with no
pod and no spend.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import teacher
from teacher import detect, load_teacher

_HF_REPO = "facebook/sam3.1"
_HF_FILE = "sam3.1_multiplex.pt"


def _find_on_volume(root: str = "/runpod-volume", depth: int = 5) -> str | None:
    """A checkpoint already staged on an attached network volume (any path,
    e.g. a previous pod's HF cache) beats a fresh download: no token, no
    billed transfer. Bounded walk so a big volume can't stall the cold start."""
    if not os.path.isdir(root):
        return None
    base = root.rstrip("/").count("/")
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.count("/") - base >= depth:
            dirnames[:] = []
        if _HF_FILE in filenames:
            return os.path.join(dirpath, _HF_FILE)
    return None


def _ensure_checkpoint() -> str:
    """Resolve the SAM 3.1 checkpoint and point `teacher.CKPT` at it.

    Order: an explicit SAM3_CKPT env, then the path teacher already knows
    (repo checkout / baked image), then a Hugging Face fetch when
    SAM3_FROM_HF=1 (RunPod cached-models pre-stages this into HF_HOME, so the
    "fetch" is a cache hit that costs seconds and is unbilled). Setting the
    module global rather than editing teacher.py keeps one loader shared with
    the distillation pipeline.
    """
    env = os.environ.get("SAM3_CKPT")
    if env and os.path.exists(env):
        teacher.CKPT = env
        return env
    if os.path.exists(teacher.CKPT):
        return teacher.CKPT
    staged = _find_on_volume()
    if staged:
        teacher.CKPT = staged
        return staged
    if os.environ.get("SAM3_FROM_HF") == "1":
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(_HF_REPO, _HF_FILE,
                               token=os.environ.get("HF_TOKEN"))
        teacher.CKPT = path
        return path
    raise FileNotFoundError(
        f"SAM 3.1 checkpoint not found (teacher.CKPT={teacher.CKPT}); set "
        f"SAM3_CKPT to a local file or SAM3_FROM_HF=1 with an HF token")


@dataclass
class Instance:
    box: list[float]                 # [x1,y1,x2,y2] in ORIGINAL image pixels
    score: float
    area_frac: float                 # mask area / frame area
    polygon: list[list[float]] | None = None   # normalized 0-1, largest contour
    rle: dict | None = None          # COCO-style {size,counts} if requested
    crop_b64: str | None = None      # masked bbox crop, JPEG b64, if requested


@dataclass
class FrameResult:
    width: int
    height: int
    instances: list[Instance] = field(default_factory=list)
    n: int = 0
    error: str | None = None


def _largest_polygon(mask: np.ndarray, eps_frac: float = 0.005
                     ) -> list[list[float]] | None:
    """Normalized outer polygon of the biggest contour (mirrors label_frames)."""
    h, w = mask.shape
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(c) < 3:
        return None
    return np.column_stack([c[:, 0] / w, c[:, 1] / h]).clip(0, 1).tolist()


def _rle(mask: np.ndarray) -> dict:
    """COCO column-major RLE. Compact and lossless where a polygon isn't."""
    m = np.asfortranarray(mask.astype(np.uint8))
    flat = m.ravel(order="F")
    counts, prev, run = [], 0, 0
    # RLE always starts with a run of 0s (COCO convention)
    for v in flat:
        if v == prev:
            run += 1
        else:
            counts.append(run)
            prev, run = v, 1
    counts.append(run)
    return {"size": [int(m.shape[0]), int(m.shape[1])], "counts": counts}


def _masked_crop_b64(bgr: np.ndarray, mask: np.ndarray, box: np.ndarray,
                     quality: int = 85) -> str | None:
    import base64
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(bgr.shape[1], x2), min(bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = bgr[y1:y2, x1:x2].copy()
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("ascii") if ok else None


class SAM3Segmenter:
    """Warm, reusable SAM 3.1 image segmenter. Construct once per process."""

    def __init__(self, device: str | None = None):
        t0 = time.time()
        _ensure_checkpoint()
        self.model, self.processor, self.device = load_teacher(device=device)
        self.model_load_s = time.time() - t0

    def segment(self, bgr: np.ndarray, prompt: str = "card",
                min_score: float = 0.5, min_area_frac: float = 0.001,
                max_dim: int = 1008, masks: str = "polygon",
                crops: bool = False, max_instances: int = 64) -> FrameResult:
        """One frame. `bgr` is a full-res OpenCV image; boxes come back in its
        pixel space regardless of the internal `max_dim` downscale."""
        H, W = bgr.shape[:2]
        s = min(1.0, max_dim / max(H, W))
        small = cv2.resize(bgr, (round(W * s), round(H * s))) if s < 1.0 else bgr

        _, m_masks, m_boxes, m_scores = detect(self.processor, small, prompt)

        order = np.argsort(-m_scores)
        out = FrameResult(width=W, height=H)
        inv = 1.0 / s
        frame_area = float(small.shape[0] * small.shape[1])
        for i in order:
            if m_scores[i] < min_score:
                continue
            mask = m_masks[i]
            area = float(mask.sum())
            af = area / frame_area if frame_area else 0.0
            if af < min_area_frac:
                continue
            box_orig = (m_boxes[i] * inv).tolist()   # scale back to original px
            inst = Instance(box=[round(v, 1) for v in box_orig],
                            score=round(float(m_scores[i]), 4),
                            area_frac=round(af, 5))
            if masks == "polygon":
                inst.polygon = _largest_polygon(mask)
            elif masks == "rle":
                inst.rle = _rle(mask)
            if crops:
                # crop from the original-res frame using the scaled-back box
                inst.crop_b64 = _masked_crop_b64(
                    bgr, cv2.resize(mask.astype(np.uint8), (W, H)) > 0,
                    np.array(box_orig))
            out.instances.append(inst)
            if len(out.instances) >= max_instances:
                break
        out.n = len(out.instances)
        return out
