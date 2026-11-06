"""A stand-in engine with the same interface as Engine, for running the UI and
tests on a machine without a GPU (DARKROOM_MOCK=1). It does NOT generate real
images and must never be used for reported metrics."""
import time

import numpy as np
from PIL import Image, ImageDraw

from . import config, postprocess, prompts
from .engine import to_data_url

PALETTE = {"red": (216, 67, 58), "blue": (61, 111, 214), "green": (63, 158, 90), "yellow": (242, 194, 48),
           "orange": (238, 138, 42), "purple": (139, 85, 201), "pink": (238, 134, 180), "white": (237, 232, 223),
           "black": (38, 33, 31), "brown": (138, 90, 59), "silver": (191, 196, 201)}


class MockEngine:
    device = "mock"

    def __init__(self):
        self.available = list(config.TIERS)
        self.model_mb = {"baseline": 4100, "fast": 2050, "fast_taesd": 1720, "lite": 860}
        self.taesd = None

    def tier_info(self, key):
        t = dict(config.TIERS[key])
        t["model_mb"] = self.model_mb[t["pipe"]]
        return t

    def _scene(self, prompt, res):
        img = Image.new("RGB", (res, res))
        d = ImageDraw.Draw(img)
        for y in range(res):
            c = int(40 + 120 * y / res)
            d.line([(0, y), (res, y)], fill=(c, c // 2 + 30, 90 - c // 4))
        parsed = prompts.parse(prompt)
        n = max(1, len(parsed["objects"]))
        for i, o in enumerate(parsed["objects"]):
            col = next((PALETTE[a] for a in o["attrs"] if a in PALETTE), (200, 200, 200))
            cx, r = int(res * (i + 1) / (n + 1)), res // 9
            d.ellipse([cx - r, int(res * .62) - r, cx + r, int(res * .62) + r], fill=col)
            o["box"] = [(cx - r) / res, (res * .62 - r) / res, (cx + r) / res, (res * .62 + r) / res]
        return np.array(img), parsed

    def generate(self, prompt, seed=0, tier="balanced", on_event=None, preview=True, post=None, verify=True):
        emit = on_event or (lambda m: None)
        t = config.TIERS[tier]
        rng = np.random.default_rng(seed)
        target, parsed = self._scene(prompt, t["res"])
        noise = rng.integers(0, 255, target.shape, dtype=np.uint8)
        emit({"type": "stage", "stage": "encode", "state": "ok", "ms": 20})
        emit({"type": "stage", "stage": "denoise", "state": "run"})
        t0 = time.perf_counter()
        for s in range(1, t["steps"] + 1):
            time.sleep(0.03 if tier != "baseline" else 0.06)
            a = (1 - s / t["steps"]) ** 1.3
            frame = (target * (1 - a) + noise * a).astype(np.uint8)
            emit({"type": "step", "step": s, "total": t["steps"],
                  "preview": to_data_url(frame, quality=60) if preview else None})
        den = (time.perf_counter() - t0) * 1000
        emit({"type": "stage", "stage": "denoise", "state": "ok", "ms": den})
        emit({"type": "stage", "stage": "decode", "state": "ok", "ms": 10})
        processed, qa, post_ms, qa_ms = postprocess.run(target, **{
            "contrast": (post or {}).get("contrast", True), "wb": (post or {}).get("wb", True),
            "sharp": (post or {}).get("sharp", True), "up": (post or {}).get("upscale", True)})
        emit({"type": "stage", "stage": "post", "state": "ok", "ms": post_ms})
        emit({"type": "stage", "stage": "qa", "state": "ok", "ms": qa_ms})
        dets = [{"label": prompts.phrase(o), "box": o["box"], "score": 0.5, "ok": True} for o in parsed["objects"]]
        checks = [{"text": f"Found a {d['label']}", "ok": True} for d in dets]
        emit({"type": "stage", "stage": "verify", "state": "ok", "ms": 5})
        return {"raw": target, "processed": processed, "detections": dets, "checks": checks, "qa": qa,
                "tier_used": tier,
                "metrics": {"clip": None, "peak_mb": None, "model_mb": self.model_mb[t["pipe"]],
                            "encode_ms": 20, "denoise_ms": den, "decode_ms": 10, "post_ms": post_ms,
                            "qa_ms": qa_ms, "verify_ms": 5, "total_ms": 35 + den + post_ms + qa_ms}}
