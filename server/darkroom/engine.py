"""Generation engine: loads every tier once, runs a request, and reports
per-stage timings, peak memory and model size alongside the image."""
from pathlib import Path
import base64
import copy
import io
import logging
import os
import threading
import time

import numpy as np
import torch
from PIL import Image

from . import config, postprocess, prompts

log = logging.getLogger("darkroom")


def pick_device():
    forced = os.getenv("DARKROOM_DEVICE")          # cuda | mps | cpu
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def module_bytes(module):
    """Size of a module's weights in bytes. Handles quanto's INT8 tensors, whose
    raw storage lives in ._data (int8) plus a per-channel ._scale."""
    total = 0
    for _, p in module.state_dict().items():
        try:
            if hasattr(p, "_data"):
                total += p._data.numel() * p._data.element_size()
                if hasattr(p, "_scale"):
                    total += p._scale.numel() * p._scale.element_size()
            else:
                total += p.numel() * p.element_size()
        except Exception:
            total += p.numel() * 2
    return total


def to_data_url(img: np.ndarray, fmt="JPEG", quality=92):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format=fmt, quality=quality)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


class Engine:
    def __init__(self, load_baseline=None, pipes=None, taesd=None, only=None):
        """only: optional list of tier keys to load (e.g. ['lite']) - used to measure one tier's memory in isolation."""
        self.device = pick_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.lock = threading.Lock()
        self.pipes, self.taesd = {}, taesd
        self.precision_actual = {}
        self._clip = self._det = None
        if load_baseline is None:
            load_baseline = os.getenv("DARKROOM_BASELINE", "1" if self.device == "cuda" else "0") == "1"
        if pipes is not None:          # injected (tests)
            self.pipes = pipes
        else:
            self._load(load_baseline, only)
        self.model_mb = {k: self._pipe_mb(p) for k, p in self.pipes.items()}
        self.available = [k for k, t in config.TIERS.items() if t["pipe"] in self.pipes]
        log.info("Tiers ready: %s on %s", self.available, self.device)

    # ------------------------------------------------------------------ loading
    def _load(self, load_baseline, only=None):
        import gc

        from diffusers import AutoencoderTiny, LCMScheduler, StableDiffusionPipeline

        wanted = None if not only else {config.TIERS[t]["pipe"] for t in only}

        def need(pipe_key):
            return wanted is None or pipe_key in wanted

        common = dict(safety_checker=None, requires_safety_checker=False)
        self.taesd = AutoencoderTiny.from_pretrained(config.TAESD, torch_dtype=self.dtype).to(self.device)

        if need("fast") or need("fast_taesd") or need("lite"):
            log.info("Loading %s (%s)", config.BASE_MODEL, self.dtype)
            fast = StableDiffusionPipeline.from_pretrained(config.BASE_MODEL, torch_dtype=self.dtype, **common)
            fast.scheduler = LCMScheduler.from_config(fast.scheduler.config)
            fast.load_lora_weights(config.LCM_LORA, adapter_name="lcm")
            if config.CUSTOM_LORA:
                lora_dir = Path(config.CUSTOM_LORA)
                lora_file = lora_dir / "pytorch_lora_weights.safetensors"
                if not lora_file.is_file():
                    raise FileNotFoundError(f"DARKROOM_CUSTOM_LORA is set but {lora_file} does not exist. "
                                            "Train it first with eval/train_lora.py and check it printed 'Saved LoRA'.")
                fast.load_lora_weights(str(lora_dir), weight_name=lora_file.name, adapter_name="custom")
                fast.set_adapters(["lcm", "custom"], adapter_weights=[1.0, 0.8])
            fast.fuse_lora()
            fast.unload_lora_weights()
            fast.to(self.device)

            def variant(unet, vae):
                return StableDiffusionPipeline(vae=vae, text_encoder=fast.text_encoder, tokenizer=fast.tokenizer,
                                               unet=unet, scheduler=fast.scheduler, safety_checker=None,
                                               feature_extractor=None, requires_safety_checker=False)

            if need("fast"):
                self.pipes["fast"] = fast
            if need("fast_taesd"):
                self.pipes["fast_taesd"] = variant(fast.unet, self.taesd)
            if need("lite"):
                # share nothing with the FP16 UNet when other tiers use it; quantise in place when Lite is alone
                alone = not (need("fast") or need("fast_taesd"))
                unet_lite = fast.unet if alone else copy.deepcopy(fast.unet)
                try:
                    if os.getenv("DARKROOM_INT8", "1") != "1":
                        raise RuntimeError("disabled with DARKROOM_INT8=0")
                    from optimum.quanto import freeze, qint8, quantize
                    quantize(unet_lite, weights=qint8)
                    freeze(unet_lite)
                    self.precision_actual["lite"] = "INT8"
                except Exception as e:  # quanto missing or unsupported device: keep running, but say so
                    log.warning("INT8 quantisation unavailable (%s); Lite runs in %s", e, self.dtype)
                    self.precision_actual["lite"] = "FP16" if self.dtype == torch.float16 else "FP32"
                self.pipes["lite"] = variant(unet_lite, self.taesd)
            del fast                      # unused parts (e.g. the full VAE for TAESD-only tiers) are freed here
            gc.collect()

        if need("baseline") and (load_baseline or wanted is not None):
            log.info("Loading FP32 baseline")
            base = StableDiffusionPipeline.from_pretrained(config.BASE_MODEL, torch_dtype=torch.float32, **common)
            self.pipes["baseline"] = base.to(self.device)

        if self.device == "cuda":
            torch.cuda.empty_cache()
        for p in self.pipes.values():
            p.set_progress_bar_config(disable=True)

    def _pipe_mb(self, pipe):
        return sum(module_bytes(m) for m in (pipe.unet, pipe.text_encoder, pipe.vae)) / 1e6

    # ------------------------------------------------------------------ helpers
    def _sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def _preview(self, latents, size=256):
        x = self.taesd.decode(latents.to(self.taesd.dtype), return_dict=False)[0]
        x = ((x[0].float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        if x.shape[0] != size:
            x = np.array(Image.fromarray(x).resize((size, size), Image.BILINEAR))
        return to_data_url(x, quality=70)

    def tier_info(self, key):
        t = dict(config.TIERS[key])
        if key in self.precision_actual:
            t["precision"] = self.precision_actual[key]
        t["model_mb"] = round(self.model_mb.get(t["pipe"], 0), 1)
        return t

    # ------------------------------------------------------------------ generate
    @torch.no_grad()
    def generate(self, prompt, seed=0, tier="balanced", on_event=None, preview=True, post=None, verify=True,
                 steps=None, res=None, score=True):
        """Runs one request. on_event(dict) receives stage/step messages as they happen.
        steps/res override the tier (used by the ablation study)."""
        emit = on_event or (lambda m: None)
        t = dict(config.TIERS[tier])
        if steps: t["steps"] = steps
        if res: t["res"] = res
        pipe = self.pipes[t["pipe"]]
        cfg = t["guidance"] > 1.0
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            alloc0 = torch.cuda.memory_allocated()

        # 1. prompt encoding
        emit({"type": "stage", "stage": "encode", "state": "run"})
        self._sync(); t0 = time.perf_counter()
        pe, ne = pipe.encode_prompt(prompt, self.device, 1, cfg, negative_prompt=None)
        self._sync(); enc_ms = (time.perf_counter() - t0) * 1000
        emit({"type": "stage", "stage": "encode", "state": "ok", "ms": enc_ms})

        # 2. denoising in latent space
        emit({"type": "stage", "stage": "denoise", "state": "run"})
        every = 1 if t["steps"] <= 10 else 5

        def cb(p, i, timestep, kw):
            step = i + 1
            msg = {"type": "step", "step": step, "total": t["steps"]}
            if preview and self.taesd is not None and (step % every == 0 or step == t["steps"]):
                msg["preview"] = self._preview(kw["latents"])
            emit(msg)
            return kw

        self._sync(); t1 = time.perf_counter()
        latents = pipe(prompt_embeds=pe, negative_prompt_embeds=ne if cfg else None,
                       num_inference_steps=t["steps"], guidance_scale=t["guidance"],
                       height=t["res"], width=t["res"], generator=gen, output_type="latent",
                       callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"]).images
        self._sync(); den_ms = (time.perf_counter() - t1) * 1000
        emit({"type": "stage", "stage": "denoise", "state": "ok", "ms": den_ms})

        # 3. decode latents -> pixels (full VAE or TAESD, both use the same formula)
        emit({"type": "stage", "stage": "decode", "state": "run"})
        self._sync(); t2 = time.perf_counter()
        img = pipe.vae.decode(latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor, return_dict=False)[0]
        img = pipe.image_processor.postprocess(img, output_type="np")[0]
        self._sync(); dec_ms = (time.perf_counter() - t2) * 1000
        emit({"type": "stage", "stage": "decode", "state": "ok", "ms": dec_ms})
        # peak_process_mb: everything on the GPU (all loaded tiers + scorer models) - only meaningful in isolation.
        # peak_mb: this tier's footprint = its own weights + the activation peak of this request.
        peak_process_mb = torch.cuda.max_memory_allocated() / 1e6 if self.device == "cuda" else None
        peak_mb = (self.model_mb[t["pipe"]] + (torch.cuda.max_memory_allocated() - alloc0) / 1e6) \
            if self.device == "cuda" else None

        raw = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # 4-5. classical post-processing + quality check
        post = post or {}
        processed, qa, post_ms, qa_ms = postprocess.run(
            raw, contrast=post.get("contrast", True), wb=post.get("wb", True),
            sharp=post.get("sharp", True), up=post.get("upscale", True))
        emit({"type": "stage", "stage": "post", "state": "ok", "ms": post_ms})
        emit({"type": "stage", "stage": "qa", "state": "ok", "ms": qa_ms})

        # 6. semantic verification
        detections, checks, ver_ms = [], [], 0.0
        if verify:
            t3 = time.perf_counter()
            detections, checks = self.verify(prompt, processed)
            ver_ms = (time.perf_counter() - t3) * 1000
            emit({"type": "stage", "stage": "verify", "state": "ok", "ms": ver_ms})
        clip = self.clip_score(prompt, processed) if score else None

        return {
            "raw": raw, "processed": processed, "detections": detections, "checks": checks,
            "metrics": {"clip": clip, "peak_mb": peak_mb, "peak_process_mb": peak_process_mb,
                        "model_mb": self.model_mb[t["pipe"]],
                        "encode_ms": enc_ms, "denoise_ms": den_ms, "decode_ms": dec_ms,
                        "post_ms": post_ms, "qa_ms": qa_ms, "verify_ms": ver_ms,
                        "total_ms": enc_ms + den_ms + dec_ms + post_ms + qa_ms + ver_ms},
            "qa": qa, "tier_used": tier,
        }

    # ------------------------------------------------------------------ scoring
    def _load_clip(self):
        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor
            m = CLIPModel.from_pretrained(config.CLIP_MODEL).to(self.device).eval()
            self._clip = (m, CLIPProcessor.from_pretrained(config.CLIP_MODEL))
        return self._clip

    @torch.no_grad()
    def clip_score(self, prompt, img):
        """Cosine similarity between CLIP image and text embeddings (0..1 scale, typically 0.25-0.35)."""
        m, proc = self._load_clip()
        inp = proc(text=[prompt], images=Image.fromarray(img), return_tensors="pt", padding=True, truncation=True).to(self.device)
        out = m(**inp)
        i = torch.nn.functional.normalize(out.image_embeds, dim=-1)
        t = torch.nn.functional.normalize(out.text_embeds, dim=-1)
        return float((i * t).sum())

    def _load_det(self):
        if self._det is None:
            from transformers import OwlViTForObjectDetection, OwlViTProcessor
            m = OwlViTForObjectDetection.from_pretrained(config.DETECTOR).to(self.device).eval()
            self._det = (m, OwlViTProcessor.from_pretrained(config.DETECTOR))
        return self._det

    @torch.no_grad()
    def verify(self, prompt, img, threshold=float(os.getenv("DARKROOM_DET_THRESHOLD", "0.10"))):
        parsed = prompts.parse(prompt)
        objs = parsed["objects"]
        if not objs:
            return [], [{"text": "No countable objects named, so only CLIP similarity is scored", "ok": True, "muted": True}]
        m, proc = self._load_det()
        queries = [f"a photo of a {prompts.phrase(o)}" for o in objs]
        inp = proc(text=[queries], images=Image.fromarray(img), return_tensors="pt").to(self.device)
        out = m(**inp)
        scores = out.logits[0].sigmoid()            # (patches, queries)
        boxes = out.pred_boxes[0]                   # (patches, 4) cx, cy, w, h in 0..1
        used, detections, checks = set(), [], []
        for q, o in enumerate(objs):
            order = torch.argsort(scores[:, q], descending=True)
            best = next((int(k) for k in order[:50] if int(k) not in used), int(order[0]))
            used.add(best)
            cx, cy, w, h = boxes[best].tolist()
            box = [max(0, cx - w / 2), max(0, cy - h / 2), min(1, cx + w / 2), min(1, cy + h / 2)]
            s = float(scores[best, q])
            ok = s >= threshold
            o["box"], o["found"] = box, ok
            detections.append({"label": prompts.phrase(o), "box": box, "score": s, "ok": ok})
            checks.append({"text": f"Found a {prompts.phrase(o)}", "ok": ok})
        for r in parsed["relations"]:
            a, b = objs[r["a"]], objs[r["b"]]
            ok = a["found"] and b["found"] and prompts.check_relation(r["rel"], a["box"], b["box"])
            checks.append({"text": f"{prompts.phrase(a)} is {prompts.RELATION_WORDS[r['rel']]} the {prompts.phrase(b)}", "ok": ok})
        return detections, checks
