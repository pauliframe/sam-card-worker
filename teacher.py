"""Shared SAM 3.1 teacher (image mode, MPS) for the distillation pipeline."""
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch import
BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import cv2
import numpy as np
import torch
from PIL import Image

VIDEO = os.path.join(BACKEND, "..", "IMG_3321.mov")
CKPT = os.path.join(BACKEND, "sam3.1_multiplex.pt")


def load_teacher(device=None):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = device or os.environ.get("SAM3_DEVICE") or (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda":
        # stock sam3 is CUDA-native: no patches, fused perflib kernels active
        model = build_sam3_image_model(device="cuda", load_from_HF=False,
                                       checkpoint_path=CKPT, eval_mode=True)
        return model, Sam3Processor(model, device=device), device

    if device == "mps":  # see sam31-mps-port notes: chunked SDPA + .cuda redirect
        import sam3_mps_sdpa
        sam3_mps_sdpa.install()
        sam3_mps_sdpa.install_cuda_redirect(device)

    model = build_sam3_image_model(device="cpu", load_from_HF=False,
                                   checkpoint_path=CKPT, eval_mode=True)
    model = model.float().to(device)
    for mod in model.modules():  # plain-dict caches .to() misses
        if hasattr(mod, "cache") and isinstance(getattr(mod, "cache"), dict):
            mod.cache = {k: (v.to(device) if hasattr(v, "to") else v)
                         for k, v in mod.cache.items()}
    if device != "cpu":
        torch.set_default_device(device)
    return model, Sam3Processor(model, device=device), device


def video_duration(video=VIDEO):
    cap = cv2.VideoCapture(video)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1.0, cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return dur


def grab_frame(seconds, max_dim=1008, video=VIDEO):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = bgr.shape[:2]
    s = max_dim / max(h, w)
    return cv2.resize(bgr, (int(w * s), int(h * s)))


def detect(processor, bgr, prompt="card", state=None):
    """Run teacher on a BGR frame. Reuse `state` to skip the backbone re-encode."""
    import contextlib
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)  # stock sam3 CUDA convention
           if str(processor.device).startswith("cuda") else contextlib.nullcontext())
    with ctx:
        if state is None:
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            state = processor.set_image(pil)
        state = processor.set_text_prompt(prompt=prompt, state=state)
    masks = state["masks"].detach().cpu().numpy()[:, 0]              # (N, H, W) bool
    boxes = state["boxes"].detach().float().cpu().numpy()            # (N, 4) xyxy px
    scores = state["scores"].detach().float().cpu().numpy()          # (N,)
    return state, masks, boxes, scores


@torch.inference_mode()
def raw_query_probs(model, processor, state):
    """All 200 DETR query scores (unthresholded) + presence, for threshold ablation."""
    out = model.forward_grounding(
        backbone_out=state["backbone_out"],
        find_input=processor.find_stage,
        geometric_prompt=state.get("geometric_prompt", model._get_dummy_prompt()),
        find_target=None)
    probs = out["pred_logits"].sigmoid()
    presence = out["presence_logit_dec"].sigmoid()
    final = (probs * presence.unsqueeze(1)).squeeze(-1)[0]
    return final.detach().cpu().float().numpy(), presence.reshape(-1)[0].item()


def match_masks(ref_masks, test_masks, thr=0.5):
    """Greedy IoU matching. Returns (pairs [(i, j, iou)], n_ref_unmatched, n_test_unmatched)."""
    if len(ref_masks) == 0 or len(test_masks) == 0:
        return [], len(ref_masks), len(test_masks)
    iou = np.zeros((len(ref_masks), len(test_masks)), dtype=np.float32)
    for i, r in enumerate(ref_masks):
        rs = r.sum()
        for j, t in enumerate(test_masks):
            inter = np.logical_and(r, t).sum()
            union = rs + t.sum() - inter
            iou[i, j] = inter / union if union else 0.0
    pairs = []
    m = iou.copy()
    while m.max() > thr:
        i, j = np.unravel_index(m.argmax(), m.shape)
        pairs.append((int(i), int(j), float(m[i, j])))
        m[i, :] = 0
        m[:, j] = 0
    return pairs, len(ref_masks) - len(pairs), len(test_masks) - len(pairs)
