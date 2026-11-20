"""Record exactly what produced a set of numbers: library versions, GPU, and the Hugging Face
commit of every model used. Saved next to each results.json, so a rerun can pin the same
versions (pip install pkg==version, from_pretrained(..., revision=sha))."""
import platform
import subprocess
import sys
from importlib import metadata

PACKAGES = ["torch", "diffusers", "transformers", "peft", "optimum-quanto", "accelerate", "torchmetrics",
            "opencv-python-headless", "numpy", "pillow", "fastapi", "huggingface-hub"]


def model_commits(repos):
    out = {}
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for r in sorted({x for x in repos if x}):
            try:
                out[r] = api.model_info(r).sha
            except Exception as e:  # noqa: BLE001 - offline or gated: record why
                out[r] = f"unknown ({type(e).__name__})"
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
    return out


def collect(repos=()):
    import torch
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
                                timeout=60).stdout.splitlines()
    except Exception:  # noqa: BLE001
        freeze = []
    return {"python": platform.python_version(), "platform": platform.platform(), "gpu": gpu,
            "cuda": torch.version.cuda, "packages": versions, "models": model_commits(repos), "pip_freeze": freeze}


def all_model_repos():
    from . import config
    repos = [config.BASE_MODEL, config.TAESD, config.CLIP_MODEL, config.DETECTOR, config.SAFETY_MODEL]
    repos += [a["repo"] for a in config.ADAPTERS.values()]
    repos += [b["repo"] for b in config.BASES.values()] + [b["vae"] for b in config.BASES.values()]
    repos += ["openai/clip-vit-large-patch14", "google/owlv2-base-patch16-ensemble", "yuvalkirstain/PickScore_v1"]
    return repos
