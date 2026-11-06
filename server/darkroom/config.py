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
