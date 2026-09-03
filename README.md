# sam-card-worker

RunPod serverless worker for SAM 3.1 card segmentation — image mode (per-frame
open-vocabulary detection) and video mode (detect + track with persistent ids)
behind one endpoint. Built by GitHub Actions to `ghcr.io/<owner>/sam-card-worker`.

This is a build mirror of `backend/sam_service` in the CardScanner repo; edit
there, then run `publish_worker.sh`. The model checkpoint is NOT in the image:
the worker fetches `facebook/sam3.1` from Hugging Face at start (`HF_TOKEN`).
