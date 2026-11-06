"""Deploy the whole app (frontend + model server) to one public HTTPS URL on Modal.

    pip install modal
    modal setup                      # one-time login
    modal deploy deploy/modal_app.py # from the repo root

The URL Modal prints is your live application URL. It serves a lightweight fallback UI at /
and the API (/ws, /health, /tiers) that the main React UI connects to. min_containers=1 keeps one GPU warm so judges
never wait for a cold start; set it to 0 after judging to stop spending credits.
"""
import modal

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


@app.function(image=image, gpu="A10G", volumes={"/cache": cache}, min_containers=1,
              scaledown_window=1200, timeout=3600)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import sys
    sys.path.insert(0, "/root")
    from darkroom.server import app as fastapi_app
    return fastapi_app
