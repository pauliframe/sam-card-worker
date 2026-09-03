"""SAM 3.1 VIDEO mode — detect + track cards across frames with persistent IDs.

The difference from `sam3_infer` (image mode) is the whole point of video: the
detector runs every frame AND a memory tracker propagates "masklets" so the
same physical card keeps the same object id across the clip. That id IS the
cross-frame dedup — 34 raw per-frame detections collapse to "9 distinct cards"
for free, instead of being deduped downstream by embedding.

The call sequence is the sam3 predictor's session API, proven in
`backend/sam3_video.py`:
    build_sam3_predictor -> start_session(frames_dir)
    -> add_prompt(frame 0, "card") -> stream propagate_in_video

Per-frame the model returns: out_obj_ids, out_probs, out_boxes_xywh (normalized
xywh), out_binary_masks (N,H,W bool), frame_stats. We reshape that into the same
transport-friendly instance shape the image service uses, plus the persistent
`obj_id` and a `tracks` summary.

Device notes: on a 24 GB+ CUDA GPU this runs stock fp32 — no tricks. The fp16
vision-trunk + chunked-SDPA dance is ONLY needed to fit the 18 GB Mac (MPS), and
is applied únicamente when device == "mps" (see the SAM 3.1 MPS port notes).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from sam3_infer import _ensure_checkpoint, _largest_polygon, _rle  # reuse encoders


@dataclass
class TrackInstance:
    obj_id: int
    box_xywh: list[float]            # normalized [x, y, w, h]
    prob: float | None = None
    polygon: list[list[float]] | None = None
    rle: dict | None = None


@dataclass
class VideoFrameResult:
    frame_index: int
    instances: list[TrackInstance] = field(default_factory=list)
    n: int = 0


@dataclass
class VideoResult:
    prompt: str
    device: str
    n_frames: int
    unique_objects: int              # distinct persistent ids = distinct cards
    object_ids: list[int]
    per_frame_counts: dict[int, int]
    frames: list[VideoFrameResult] = field(default_factory=list)
    timings: dict = field(default_factory=dict)


def _np(x):
    import torch
    if torch.is_tensor(x):
        x = x.detach().to("cpu")
    return np.asarray(x)


class VideoSegmenter:
    """Warm, reusable SAM 3.1 video tracker. Construct once per process."""

    def __init__(self, device: str | None = None):
        import torch
        t0 = time.time()
        self.device = device or os.environ.get("SAM3_DEVICE") or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu")
        ckpt = _ensure_checkpoint()

        if self.device == "mps":
            import sam3_mps_sdpa
            sam3_mps_sdpa.install()
            sam3_mps_sdpa.install_cuda_redirect(self.device)

        from sam3.model_builder import build_sam3_predictor
        self.pred = build_sam3_predictor(checkpoint_path=ckpt, version="sam3.1",
                                         use_fa3=False, warm_up=False)
        m = self.pred.model
        # Upstream bug (present at 5dd401d and on main 2026-08): the predictor's
        # start_session forwards `offload_state_to_cpu` to model.init_state, but
        # the multiplex tracker's init_state has no such parameter -> TypeError
        # on the very first session. Filter kwargs to the real signature here,
        # at runtime, instead of patching the library inside the image.
        import inspect
        _orig_init = m.init_state
        _valid = set(inspect.signature(_orig_init).parameters)
        def _init_state(*a, **kw):
            return _orig_init(*a, **{k: v for k, v in kw.items() if k in _valid})
        m.init_state = _init_state
        m.float()
        if self.device == "mps":
            # fp16 ONLY the memory-heavy vision trunk; decoder/tracker stay fp32
            # (MPS asserts on some fp16 matmuls). CUDA needs none of this.
            for mod in m.modules():
                if hasattr(mod, "patch_embed") and hasattr(mod, "blocks"):
                    mod.half()
        for mod in m.modules():
            if hasattr(mod, "cache") and isinstance(getattr(mod, "cache"), dict):
                mod.cache = {k: (v.to(self.device) if hasattr(v, "to") else v)
                             for k, v in mod.cache.items()}
        m.to(self.device)
        if self.device == "mps":
            # MPS-only: the tracker's hardcoded .cuda() reloads are redirected
            # and need a default device. On CUDA this must NOT be set: it makes
            # every fresh tensor CUDA-resident, which breaks the image path's
            # pinned-memory post-processing ("cannot pin torch.cuda.FloatTensor")
            # for the rest of the worker's life.
            torch.set_default_device(self.device)
        self.model_load_s = time.time() - t0

    def segment_dir(self, frames_dir: str, prompt: str = "card",
                    masks: str = "polygon", min_prob: float = 0.0
                    ) -> VideoResult:
        """Track `prompt` across the JPG frames in `frames_dir` (named 0.jpg,
        1.jpg, …). Returns per-frame tracked instances + a unique-object count.
        """
        t0 = time.time()
        r = self.pred.handle_request(dict(type="start_session",
                                          resource_path=os.path.abspath(frames_dir)))
        sid = r["session_id"]
        self.pred.handle_request(dict(type="add_prompt", session_id=sid,
                                      frame_index=0, text=prompt))

        outs: dict[int, dict] = {}
        for resp in self.pred.handle_stream_request(
                dict(type="propagate_in_video", session_id=sid)):
            outs[resp["frame_index"]] = resp["outputs"]

        ids: set[int] = set()
        per_frame: dict[int, int] = {}
        frames: list[VideoFrameResult] = []
        for fidx, fo in sorted(outs.items()):
            if not isinstance(fo, dict):
                continue
            obj_ids = _np(fo.get("out_obj_ids")).astype(int).ravel()
            boxes = _np(fo.get("out_boxes_xywh")).reshape(-1, 4)
            probs = fo.get("out_probs")
            probs = _np(probs).ravel() if probs is not None else None
            bmasks = _np(fo.get("out_binary_masks"))
            fr = VideoFrameResult(frame_index=int(fidx))
            for j, oid in enumerate(obj_ids):
                p = float(probs[j]) if probs is not None and j < len(probs) else None
                if p is not None and p < min_prob:
                    continue
                inst = TrackInstance(obj_id=int(oid),
                                     box_xywh=[round(float(v), 4) for v in boxes[j]],
                                     prob=None if p is None else round(p, 4))
                if masks != "none" and j < len(bmasks):
                    mk = bmasks[j].astype(bool)
                    if masks == "polygon":
                        inst.polygon = _largest_polygon(mk)
                    elif masks == "rle":
                        inst.rle = _rle(mk)
                fr.instances.append(inst)
                ids.add(int(oid))
            fr.n = len(fr.instances)
            per_frame[int(fidx)] = fr.n
            frames.append(fr)

        dt = time.time() - t0
        return VideoResult(
            prompt=prompt, device=self.device, n_frames=len(frames),
            unique_objects=len(ids), object_ids=sorted(ids),
            per_frame_counts=per_frame, frames=frames,
            timings={"model_load_s": round(self.model_load_s, 2),
                     "propagate_s": round(dt, 2),
                     "per_frame_s": round(dt / max(1, len(frames)), 2)})
