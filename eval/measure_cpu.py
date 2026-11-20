"""How fast is Darkroom with NO GPU at all? An honest proxy for a device without a usable GPU.

Runs a tier on the CPU only (FP32 for the FP16 tiers, since CPUs lack fast FP16; Lite keeps its
INT8 UNet weights) and reports seconds per image and the process's peak RAM. This is not a
phone measurement: a phone's NPU would be faster than a server CPU on this workload, and its
memory limits are stricter. It answers "does it run at all without a GPU, and how slowly?".

    cd server
    python ../eval/measure_cpu.py --tiers lite,balanced --n 3

Writes eval/results_cpu/results.json and prints a table.
"""
import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "server"))
os.environ["DARKROOM_DEVICE"] = "cpu"        # must be set before the engine is imported

PROMPTS = ["a yellow auto rickshaw on a busy street in Mumbai, golden hour",
           "a red bicycle leaning against a blue wall",
           "a cozy cabin in a snowy pine forest at dusk"]


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if platform.system() != "Darwin" else r / 1e6   # Linux reports KiB, macOS bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="lite,balanced")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = library default)")
    ap.add_argument("--out", default=str(HERE / "results_cpu"))
    a = ap.parse_args()

    import torch
    from darkroom.engine import Engine
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"cpu": platform.processor() or platform.machine(), "threads": torch.get_num_threads(),
           "created": time.strftime("%Y-%m-%d %H:%M"), "tiers": {}}
    for tier in [t.strip() for t in a.tiers.split(",") if t.strip()]:
        print(f"=== {tier} on CPU ({torch.get_num_threads()} threads)", flush=True)
        t0 = time.time()
        eng = Engine(load_baseline=False, only=[tier])
        load_s = time.time() - t0
        rss_loaded = peak_rss_mb()
        eng.generate(PROMPTS[0], seed=999, tier=tier, preview=False, verify=False, score=False)   # warm-up
        gens = []
        for i in range(a.n):
            r = eng.generate(PROMPTS[i % len(PROMPTS)], seed=i, tier=tier, preview=False, verify=False, score=False)
            m = r["metrics"]
            gens.append((m["encode_ms"] + m["denoise_ms"] + m["decode_ms"]) / 1000)
            print(f"    image {i + 1}: {gens[-1]:.1f} s", flush=True)
        res["tiers"][tier] = {"seconds_per_image": sum(gens) / len(gens), "min_s": min(gens), "max_s": max(gens),
                              "n": a.n, "load_s": load_s, "peak_rss_mb_after_load": rss_loaded,
                              "peak_rss_mb": peak_rss_mb(), "precision": eng.tier_info(tier)["precision"]}
        del eng
    (out / "results.json").write_text(json.dumps(res, indent=1))
    from darkroom import provenance
    (out / "environment.json").write_text(json.dumps(provenance.collect(provenance.all_model_repos()), indent=1))
    print("\n| Tier | Seconds per image (CPU) | Peak RAM (process) |\n|---|---|---|")
    for k, v in res["tiers"].items():
        print(f"| {k} | {v['seconds_per_image']:.1f} s ({v['min_s']:.1f}-{v['max_s']:.1f}) | {v['peak_rss_mb']:.0f} MB |")
    print("\nPeak RAM is the whole Python process, so later tiers include earlier ones; run one tier per call "
          "for a clean number.")


if __name__ == "__main__":
    main()
