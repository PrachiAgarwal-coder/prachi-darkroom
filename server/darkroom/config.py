"""Tier definitions shared by the server and the evaluation script.

Every tier is a point on the quality-vs-cost curve. The optimisation strategies
used (the problem statement asks for at least two):

  1. Reduced sampling steps      - LCM-LoRA (latent consistency distillation): 50 -> 8/4/3 steps
  2. Knowledge distillation      - LCM-LoRA is a distilled student of SD 1.5; TAESD is a distilled VAE decoder
  3. Reduced precision           - FP32 -> FP16, and INT8 weight-only quantisation of the UNet (optimum-quanto)
  4. Latent-space optimisation   - TAESD decodes latents with ~1M params instead of the ~50M-param VAE decoder
  5. Resolution scaling          - Lite generates at 384 px and is upscaled with Lanczos + edge-aware sharpening
"""
import os

BASE_MODEL = os.getenv("DARKROOM_BASE_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
LCM_LORA = os.getenv("DARKROOM_LCM_LORA", "latent-consistency/lcm-lora-sdv1-5")
TAESD = os.getenv("DARKROOM_TAESD", "madebyollin/taesd")
CLIP_MODEL = os.getenv("DARKROOM_CLIP", "openai/clip-vit-base-patch32")
DETECTOR = os.getenv("DARKROOM_DETECTOR", "google/owlvit-base-patch32")
# Few-step adapter for the fast tiers. "lcm" is what the submitted results use; "tcd" and "hyper"
# are drop-in alternatives compared in eval/improve.py (same base model, same step counts).
ADAPTER = os.getenv("DARKROOM_ADAPTER", "lcm")
ADAPTERS = {
    "lcm":   dict(name="LCM-LoRA", repo=LCM_LORA, weight_name=None, scheduler="lcm", call={}),
    "tcd":   dict(name="TCD-LoRA", repo="h1t/TCD-SD15-LoRA", weight_name=None, scheduler="tcd", call={"eta": 0.3}),
    "hyper": dict(name="Hyper-SD 4-step LoRA", repo="ByteDance/Hyper-SD",
                  weight_name="Hyper-SD15-4steps-lora.safetensors", scheduler="ddim_trailing", call={}),
}
# Base checkpoints with the same SD 1.5 architecture (same size, speed and memory). Community
# fine-tunes are trained further on curated images; the few-step LoRAs still apply to them.
# The baseline tier always stays the original SD 1.5 so the efficiency comparison is unchanged.
BASE = os.getenv("DARKROOM_BASE", "sd15")
BASES = {
    "sd15": dict(name="Stable Diffusion 1.5", repo=BASE_MODEL, vae=None),
    "dreamshaper8": dict(name="DreamShaper 8", repo="Lykon/dreamshaper-8", vae=None),
    "rv51": dict(name="Realistic Vision 5.1", repo="SG161222/Realistic_Vision_V5.1_noVAE",
                 vae="stabilityai/sd-vae-ft-mse"),
}
# Stable Diffusion's NSFW safety checker (a CLIP-based classifier). Off by default to save memory and
# time; DARKROOM_SAFETY=1 turns it on for any public deployment. A flagged image is blurred, not shown.
SAFETY = os.getenv("DARKROOM_SAFETY", "0") == "1"
SAFETY_MODEL = "CompVis/stable-diffusion-safety-checker"

# Best-of-N: generate up to this many candidates in one batch and keep the one the verifier prefers.
MAX_CANDIDATES = 4
# Optional: your own LoRA trained with eval/train_lora.py ("meaningful adaptation")
CUSTOM_LORA = os.getenv("DARKROOM_CUSTOM_LORA", "")

TIERS = {
    # The unoptimised reference. Loaded only if DARKROOM_BASELINE=1 (needs ~7 GB VRAM).
    "baseline": dict(name="Baseline", unet="SD 1.5 UNet (860M)", steps=50, res=512, precision="FP32",
                     decoder="Full VAE", pipe="baseline", guidance=7.5),
    "quality":  dict(name="Quality", unet="SD 1.5 + LCM-LoRA", steps=8, res=512, precision="FP16",
                     decoder="Full VAE", pipe="fast", guidance=1.0),
    "balanced": dict(name="Balanced", unet="SD 1.5 + LCM-LoRA", steps=4, res=512, precision="FP16",
                     decoder="TAESD", pipe="fast_taesd", guidance=1.0),
    "lite":     dict(name="Lite", unet="SD 1.5 + LCM-LoRA, INT8", steps=3, res=384, precision="INT8",
                     decoder="TAESD", pipe="lite", guidance=1.0),
}
# best first: used for out-of-memory fallback
FALLBACK_ORDER = ["quality", "balanced", "lite"]
