"""Deploy the whole app (frontend + model server) to one public HTTPS URL on Modal.

    pip install modal
    modal setup                      # one-time login
    modal deploy deploy/modal_app.py # from the repo root

The URL Modal prints is the model server: a fallback UI at / and the API (/ws, /health, /tiers)
that the main React UI connects to.

Cost control (both read at deploy time):
  DARKROOM_GPU   T4 (default, cheapest; fits every tier incl. the FP32 baseline) | A10G (faster)
  DARKROOM_WARM  0 (default: nothing runs while idle, first request cold-starts in ~2-4 min)
                 1 (keep one GPU running: use only during the judging window)

    modal deploy deploy/modal_app.py                                   # cheap, idle = $0
    DARKROOM_WARM=1 modal deploy deploy/modal_app.py                   # judging: always warm
"""
import os

import modal

GPU = os.environ.get("DARKROOM_GPU", "T4")
WARM = int(os.environ.get("DARKROOM_WARM", "0"))

app = modal.App("darkroom")
cache = modal.Volume.from_name("darkroom-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("server/requirements.txt")
    .env({
        "HF_HOME": "/cache/hf",
        "DARKROOM_BASELINE": "1",
        "DARKROOM_FRONTEND": "/root/static/index.html",
        "DARKROOM_RESULTS": "/root/eval/results/results.json",
    })
    .add_local_dir("server/darkroom", "/root/darkroom")
    .add_local_dir("server/static", "/root/static")
    .add_local_dir("eval/results", "/root/eval/results", ignore=["images/**"])
)


@app.function(image=image, gpu=GPU, volumes={"/cache": cache}, min_containers=WARM,
              scaledown_window=600, timeout=3600)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import sys
    sys.path.insert(0, "/root")
    from darkroom.server import app as fastapi_app
    return fastapi_app
