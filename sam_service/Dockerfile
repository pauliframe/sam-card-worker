# SAM 3.1 card-segmentation worker for RunPod serverless (image + video modes).
#
# Build CONTEXT is backend/ (so the loader that the distillation pipeline
# already uses can be copied in verbatim — one source of truth):
#
#   cd backend
#   docker build -f sam_service/Dockerfile -t <registry>/sam-card:0.2 .
#
# In practice the image is built by GitHub Actions (see build.yml) because a
# 10 GB CUDA image is not something a laptop should push. Base is the official
# slim PyTorch runtime — not the RunPod pod image, which carries Jupyter/SSH the
# worker never uses; a smaller image is a faster cold start.
#
# CHECKPOINT — never in the image (gated weights). Two ways at deploy time:
#   (A) HF fetch at cold start: env SAM3_FROM_HF=1 + HF_TOKEN. With a network
#       volume mounted and HF_HOME on it, the 3.5 GB download happens ONCE and
#       later cold starts are a cache hit — same effect as RunPod cached-models.
#   (B) SAM3_CKPT=/runpod-volume/<path>.pt if the file is already staged.
#
# The stock sam3 package is CUDA-native and needs NO patches here — the MPS
# patches (sam3_mps_sdpa.py) are only for running on the Mac.

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface-cache \
    SAM3_DEVICE=cuda \
    SAM3_FROM_HF=1 \
    SAM_PRELOAD=image \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY sam_service/requirements.txt .
RUN pip install -r requirements.txt

# The teacher loader is shared with backend/distill — copy it in as-is.
COPY teacher.py sam3_mps_sdpa.py /app/
COPY sam_service/sam3_infer.py sam_service/handler.py sam_service/schema.py \
     sam_service/video_infer.py sam_service/video_handler.py \
     sam_service/worker.py /app/

CMD ["python", "-u", "worker.py"]
