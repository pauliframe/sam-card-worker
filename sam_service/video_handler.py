"""RunPod handler for SAM 3.1 VIDEO mode — detect + track cards across a clip.

Separate from the image `handler.py` because the shape is different: the video
tracker is session-based (it wants a directory of consecutive frames), so this
handler samples a clip into frames first, then tracks.

  input = {
    "video":   "<url>" | {"b64": "..."},   # the clip (required)
    "prompt":  "card",
    "t0": 0.0, "t1": null,        # window in seconds (null t1 = to end)
    "every_s": 0.3,               # sample spacing; the tracker wants CONSECUTIVE
                                  # frames, so keep this small (0.2-0.4s)
    "max_frames": 60,             # hard cap (cost + memory guard)
    "max_dim": 1008,
    "masks": "polygon" | "rle" | "none"
  }

  output = {
    "model": "sam3.1-video", "device": "cuda", "prompt": "card",
    "n_frames", "unique_objects", "object_ids", "per_frame_counts",
    "frames": [ {frame_index, n, instances:[{obj_id, box_xywh, prob,
                 polygon|rle}]} ],
    "timings"
  }

`unique_objects` is the headline the image path can't give: distinct persistent
track ids = distinct cards, deduped by the model instead of downstream.

Like `handler.py`, the tracker is a module-level singleton so the 3.5 GB model
loads once per worker. NOTE video mode wants a bigger GPU than image mode for
real-time (H100-class); on cheaper cards it still runs, just slower.
"""
from __future__ import annotations

import base64
import os
import tempfile
import time
from dataclasses import asdict

import cv2

from video_infer import VideoSegmenter

SEGMENTER = VideoSegmenter(device=os.environ.get("SAM3_DEVICE"))

MAX_FRAMES_CAP = int(os.environ.get("SAM_MAX_VIDEO_FRAMES", "120"))
MAX_BYTES = int(os.environ.get("SAM_MAX_VIDEO_BYTES", str(500 * 1024 * 1024)))


def _fetch_video(item, dst: str) -> None:
    if isinstance(item, dict) and "b64" in item:
        with open(dst, "wb") as f:
            f.write(base64.b64decode(item["b64"]))
        return
    if isinstance(item, str) and item.startswith(("http://", "https://")):
        # reuse the image handler's SSRF gate for the URL check
        from handler import _check_url
        import urllib.request
        _check_url(item)
        with urllib.request.urlopen(item, timeout=60) as r:   # noqa: S310
            data = r.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError(f"video exceeds {MAX_BYTES} bytes")
        with open(dst, "wb") as f:
            f.write(data)
        return
    raise ValueError("expected input.video = url | {b64}")


def _sample(video_path: str, frames_dir: str, t0: float, t1, every_s: float,
            max_frames: int, max_dim: int) -> int:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = int((t1 if t1 is not None else n_total / fps) * fps)
    end = min(end, n_total)
    step = max(1, int(fps * every_s))
    fno, idx = int(t0 * fps), 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    while fno < end and idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if (fno - int(t0 * fps)) % step == 0:
            h, w = frame.shape[:2]
            s = min(1.0, max_dim / max(h, w))
            if s < 1.0:
                frame = cv2.resize(frame, (round(w * s), round(h * s)))
            cv2.imwrite(os.path.join(frames_dir, f"{idx}.jpg"), frame)
            idx += 1
        fno += 1
    cap.release()
    return idx


def handler(event: dict) -> dict:
    t0_wall = time.time()
    inp = event.get("input") or {}
    if not isinstance(inp, dict) or not inp.get("video"):
        return {"error": "expected input.video = url | {b64}"}

    every_s = float(inp.get("every_s", 0.3))
    max_frames = min(int(inp.get("max_frames", 60)), MAX_FRAMES_CAP)
    max_dim = int(inp.get("max_dim", 1008))
    prompt = str(inp.get("prompt", "card"))[:128]
    masks = inp.get("masks", "polygon")
    if masks not in ("polygon", "rle", "none"):
        return {"error": "masks must be polygon|rle|none"}

    with tempfile.TemporaryDirectory() as td:
        vp = os.path.join(td, "clip.mp4")
        fd = os.path.join(td, "frames")
        os.makedirs(fd)
        try:
            _fetch_video(inp["video"], vp)
        except (ValueError, OSError) as e:
            return {"error": f"video: {e}"}
        n = _sample(vp, fd, float(inp.get("t0", 0.0)), inp.get("t1"),
                    every_s, max_frames, max_dim)
        if n < 2:
            return {"error": "need >=2 sampled frames; check t0/t1/every_s"}
        res = SEGMENTER.segment_dir(fd, prompt=prompt, masks=masks)

    d = asdict(res)
    d["model"] = "sam3.1-video"
    d["sampled_frames"] = n
    d["wall_s"] = round(time.time() - t0_wall, 2)
    # strip None mask fields to keep the payload lean
    for fr in d["frames"]:
        fr["instances"] = [{k: v for k, v in i.items() if v is not None}
                           for i in fr["instances"]]
    return d


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
