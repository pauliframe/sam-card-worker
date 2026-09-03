"""The wire contract, as data — importable by client and handler, and the
one place the request/response shape is defined. Kept dependency-free
(dataclasses only) so it imports anywhere.

This is deliberately NOT wired into the pipeline yet: it is the seam a future
`detect.py` backend would call instead of the local YOLO/classical detector,
but that integration is out of scope here.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SegmentRequest:
    images: list                      # [url:str | {"b64": str}]
    prompt: str = "card"
    min_score: float = 0.5
    min_area_frac: float = 0.001
    max_dim: int = 1008
    masks: str = "polygon"            # "polygon" | "rle" | "none"
    crops: bool = False
    max_instances: int = 64

    def to_input(self) -> dict:
        """The dict that goes under RunPod's `input` key."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class InstanceOut:
    box: list                         # [x1,y1,x2,y2] original-image pixels
    score: float
    area_frac: float
    polygon: list | None = None       # normalized 0-1
    rle: dict | None = None
    crop_b64: str | None = None


@dataclass
class FrameOut:
    width: int
    height: int
    n: int = 0
    instances: list = field(default_factory=list)
    error: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "FrameOut":
        insts = [InstanceOut(**{k: i.get(k) for k in
                                ("box", "score", "area_frac",
                                 "polygon", "rle", "crop_b64")})
                 for i in d.get("instances", [])]
        return cls(width=d.get("width", 0), height=d.get("height", 0),
                   n=d.get("n", len(insts)), instances=insts,
                   error=d.get("error"))
