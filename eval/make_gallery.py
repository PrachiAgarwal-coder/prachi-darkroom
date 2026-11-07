"""Generate a small gallery of real outputs: the same prompts and seeds on every tier.

    cd server && python ../eval/make_gallery.py
    -> eval/gallery/<n>_<tier>.jpg  (384 px, JPEG) and eval/gallery/gallery.json

Used by the UI (Benchmarks comparison, Studio empty state) and the README, so real model
output is visible even when no live server is attached.
"""
import json
import sys
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "server"))
from darkroom.engine import Engine  # noqa: E402

PROMPTS = [
    ("a cozy cabin in a snowy pine forest at dusk, warm light in the windows", 11),
    ("a yellow auto rickshaw on a busy street in Mumbai, golden hour", 21),
    ("a red apple and a green pear on a wooden table, studio lighting", 31),
    ("a lighthouse on a cliff above a stormy sea, dramatic sky", 41),
]
TIERS = ["baseline", "balanced", "lite"]


def main():
    out = HERE / "gallery"
    out.mkdir(exist_ok=True)
    eng = Engine(load_baseline=True)
    rows = []
    for i, (prompt, seed) in enumerate(PROMPTS):
        for tier in TIERS:
            t0 = time.perf_counter()
            r = eng.generate(prompt, seed=seed, tier=tier, preview=False, verify=False)
            m = r["metrics"]
            name = f"{i}_{tier}.jpg"
            Image.fromarray(r["processed"]).resize((384, 384), Image.LANCZOS).save(out / name, quality=86)
            rows.append({"prompt": prompt, "seed": seed, "tier": tier, "file": name,
                         "gen_ms": m["encode_ms"] + m["denoise_ms"] + m["decode_ms"], "clip": m["clip"]})
            print(f"{tier:9s} {rows[-1]['gen_ms']:8.0f} ms  clip {m['clip']:.3f}  {prompt[:50]}", flush=True)
    (out / "gallery.json").write_text(json.dumps(rows, indent=1))
    print(f"Saved {len(rows)} images to {out}")


if __name__ == "__main__":
    main()
