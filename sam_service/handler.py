"""RunPod serverless handler for SAM 3.1 card segmentation.

Request/response contract (the authority is `schema.py`):

  input = {
    "images":  [ "<url>" | {"b64": "<jpeg/png base64>"}, ... ],   # 1+ required
    "prompt":  "card",           # concept text prompt
    "min_score":     0.5,        # confidence gate
    "min_area_frac": 0.001,      # drop specks (fraction of frame area)
    "max_dim":       1008,       # longest-edge resize before inference
    "masks":  "polygon" | "rle" | "none",   # mask encoding (default polygon)
    "crops":  false,             # also return masked bbox crops (JPEG b64)
    "max_instances": 64
  }

  output = {
    "model": "sam3.1", "device": "cuda", "prompt": "card",
    "images": [ { "width", "height", "n", "instances": [ {box,score,
                  area_frac, polygon|rle, crop?} ], "error" }, ... ],
    "timings": { "model_load_s", "total_s", "per_image_s" },
    "n_images": N
  }

The single most important line in this file is that `SEGMENTER` is built at
IMPORT time, not inside `handler`. RunPod keeps a warm worker alive across
jobs, so the 3.5 GB model loads once per worker and every subsequent request
pays only for the backbone encode. Building it inside the handler would reload
the weights on every request — the entire cost story (see EVALUATION.md)
depends on not doing that.
"""
from __future__ import annotations

import base64
import ipaddress
import os
import socket
import time
import urllib.parse
import urllib.request
from dataclasses import asdict
from typing import Any

import cv2
import numpy as np

from sam3_infer import SAM3Segmenter

# --- warm singleton: loaded ONCE per worker, at import time --------------------
SEGMENTER = SAM3Segmenter(device=os.environ.get("SAM3_DEVICE"))

# Guardrails. This worker takes untrusted input (URLs and base64 blobs) even
# though the endpoint is API-key gated — defense in depth, and a cap keeps one
# request from evicting the warm model or hanging the queue.
MAX_IMAGES = int(os.environ.get("SAM_MAX_IMAGES", "32"))
MAX_BYTES = int(os.environ.get("SAM_MAX_BYTES", str(20 * 1024 * 1024)))  # 20 MB
MAX_PIXELS = int(os.environ.get("SAM_MAX_PIXELS", str(50_000_000)))      # 50 MP
_ALLOW_PRIVATE = os.environ.get("SAM_ALLOW_PRIVATE_URLS") == "1"         # tests


class ImageError(ValueError):
    """A per-image problem reported back in that image's slot, not a 500."""


def _clamp(v: Any, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _check_url(url: str) -> None:
    """Reject anything that isn't a plain public http(s) host — the SSRF gate.

    A serverless worker sits next to cloud metadata (169.254.169.254) and
    internal services; an attacker who can hand us a URL must not be able to
    make us fetch those. We resolve the host and refuse if ANY resolved
    address is private, loopback, link-local, or otherwise not global.
    """
    p = urllib.parse.urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ImageError("url must be http(s) with a host")
    if _ALLOW_PRIVATE:
        return
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or
                                   (443 if p.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ImageError(f"cannot resolve host: {e}") from e
    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if not ip.is_global or ip.is_multicast:
            raise ImageError(f"refusing non-public address {ip}")


def _read_capped(resp) -> bytes:
    """Read at most MAX_BYTES, honoring Content-Length when present."""
    clen = resp.headers.get("Content-Length")
    if clen and int(clen) > MAX_BYTES:
        raise ImageError(f"image too large ({clen} bytes > {MAX_BYTES})")
    data = resp.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ImageError(f"image exceeds {MAX_BYTES} bytes")
    return data


def _decode_image(item: Any) -> np.ndarray:
    """A request image is a URL string or {"b64": ...}. Raises ImageError."""
    if isinstance(item, dict) and "b64" in item:
        raw = base64.b64decode(item["b64"], validate=False)
    elif isinstance(item, str) and item.startswith(("http://", "https://")):
        _check_url(item)
        # No redirect-follow to a private host: the default opener would chase
        # a 302 into the metadata service. Disable redirects entirely.
        opener = urllib.request.build_opener(_NoRedirect())
        with opener.open(item, timeout=15) as r:   # noqa: S310
            raw = _read_capped(r)
    elif isinstance(item, str):
        raw = base64.b64decode(item, validate=False)   # bare b64 string
    else:
        raise ImageError("expected a URL string or {\"b64\": ...}")
    if len(raw) > MAX_BYTES:
        raise ImageError(f"image exceeds {MAX_BYTES} bytes")
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageError("could not decode image")
    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        raise ImageError(f"image over {MAX_PIXELS} px (decompression guard)")
    return img


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):   # noqa: D401 - refuse all redirects
        return None


def handler(event: dict) -> dict:
    t0 = time.time()
    inp = event.get("input") or {}
    if not isinstance(inp, dict):
        return {"error": "input must be an object"}
    images = inp.get("images")
    if not images:
        return {"error": "no images: expected input.images = [url | {b64}, ...]"}
    if isinstance(images, (str, dict)):
        images = [images]
    if not isinstance(images, list):
        return {"error": "images must be a list"}
    if len(images) > MAX_IMAGES:
        return {"error": f"too many images ({len(images)} > {MAX_IMAGES})"}

    try:
        masks = inp.get("masks", "polygon")
        if masks not in ("polygon", "rle", "none"):
            return {"error": "masks must be polygon|rle|none"}
        opts = dict(
            prompt=str(inp.get("prompt", "card"))[:128],
            min_score=_clamp(inp.get("min_score", 0.5), 0.0, 1.0),
            min_area_frac=_clamp(inp.get("min_area_frac", 0.001), 0.0, 1.0),
            max_dim=int(_clamp(inp.get("max_dim", 1008), 256, 2048)),
            masks=masks,
            crops=bool(inp.get("crops", False)),
            max_instances=int(_clamp(inp.get("max_instances", 64), 1, 256)),
        )
    except (TypeError, ValueError):
        return {"error": "invalid option types in input"}

    results = []
    for item in images:
        try:
            bgr = _decode_image(item)
        except ImageError as e:
            results.append({"error": str(e), "instances": [],
                            "n": 0, "width": 0, "height": 0})
            continue
        fr = SEGMENTER.segment(bgr, **opts)
        d = asdict(fr)
        # asdict keeps None fields; drop them so the payload stays lean
        d["instances"] = [{k: v for k, v in i.items() if v is not None}
                          for i in d["instances"]]
        results.append(d)

    total = time.time() - t0
    return {
        "model": "sam3.1",
        "device": SEGMENTER.device,
        "prompt": opts["prompt"],
        "images": results,
        "n_images": len(results),
        "timings": {
            "model_load_s": round(SEGMENTER.model_load_s, 2),
            "total_s": round(total, 3),
            "per_image_s": round(total / max(1, len(results)), 3),
        },
    }


if __name__ == "__main__":
    # `python handler.py` starts the RunPod worker loop. Importing runpod is
    # deferred to here so the module can be imported for local testing (and by
    # test_local.py) on a machine without the runpod SDK.
    import runpod
    runpod.serverless.start({"handler": handler})
