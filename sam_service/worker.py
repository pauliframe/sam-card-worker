"""One RunPod worker, both SAM 3.1 modes, one endpoint.

    input.mode = "image"  -> handler.py        (per-frame open-vocab detection)
    input.mode = "video"  -> video_handler.py  (detect every frame + track ids)
    input.mode = "ping"   -> no inference; reports GPU + what is loaded

`mode` is inferred when omitted: a `video` key means video, otherwise image.

Why one worker instead of two images: the evaluation question is "what does
the RunPod API give us for photos AND clips", and a single endpoint answers it
with one cold start. Each mode's model is a lazily-built singleton, so a worker
only pays for the mode it is actually asked for; SAM_PRELOAD (image|video|both|
none, default image) decides what is warmed at container start, i.e. inside
the cold start rather than inside the first job.

Every response gains a `worker` block: which GPU ran it, how long the process
has been up, per-mode load seconds, and the handler wall time — the numbers
the cost model needs, straight from the job that produced them.
"""
from __future__ import annotations

import os
import time

T_BOOT = time.time()
LOADS: dict[str, float] = {}
_mods: dict[str, object] = {}
_gpu: str | None = None


def _gpu_name() -> str:
    global _gpu
    if _gpu is None:
        try:
            import torch
            _gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        except Exception as e:  # pragma: no cover
            _gpu = f"unknown ({e})"
    return _gpu


def _volume_listing(root: str = "/runpod-volume") -> dict:
    """What an attached network volume holds (top level + any SAM checkpoints),
    so a ping tells us whether weights are already staged before we spend a
    download on them."""
    if not os.path.isdir(root):
        return {"mounted": False}
    from sam3_infer import _find_on_volume
    try:
        top = sorted(os.listdir(root))[:40]
    except OSError as e:
        top = [f"<{e}>"]
    return {"mounted": True, "top": top, "checkpoint": _find_on_volume(root)}


def _mod(mode: str):
    """Import (and thereby construct the model singleton of) a mode once."""
    if mode not in _mods:
        t = time.time()
        if mode == "image":
            import handler as m
        else:
            import video_handler as m
        _mods[mode] = m
        LOADS[mode] = round(time.time() - t, 2)
    return _mods[mode]


def _worker_block(mode: str, t0: float) -> dict:
    return {
        "gpu": _gpu_name(),
        "mode": mode,
        "worker_id": os.environ.get("RUNPOD_POD_ID"),
        "uptime_s": round(time.time() - T_BOOT, 1),
        "loads_s": dict(LOADS),
        "handler_s": round(time.time() - t0, 2),
    }


def handler(event: dict) -> dict:
    t0 = time.time()
    inp = event.get("input") or {}
    if not isinstance(inp, dict):
        return {"error": "input must be an object"}
    mode = inp.get("mode") or ("video" if "video" in inp else "image")
    if mode == "ping":
        return {"ok": True, "volume": _volume_listing(),
                "worker": _worker_block("ping", t0)}
    if mode not in ("image", "video"):
        return {"error": "mode must be image|video|ping"}
    try:
        out = _mod(mode).handler(event)
    except Exception as e:  # surface the failure as data, keep the worker alive
        out = {"error": f"{type(e).__name__}: {e}"}
    if isinstance(out, dict):
        out["worker"] = _worker_block(mode, t0)
    return out


if __name__ == "__main__":
    pre = os.environ.get("SAM_PRELOAD", "image")
    for m in ("image", "video"):
        if pre in (m, "both"):
            _mod(m)
    print(f"[worker] gpu={_gpu_name()} preloaded={LOADS} boot={time.time()-T_BOOT:.1f}s",
          flush=True)
    import runpod
    runpod.serverless.start({"handler": handler})
