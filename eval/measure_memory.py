"""Measure each tier's peak GPU memory IN ISOLATION and update the evaluation report.

Why: evaluate.py loads every tier plus the CLIP scorer and the OWL-ViT verifier into one
process, so torch.cuda.max_memory_allocated() there includes all of them. Here each tier
runs in its own fresh process that loads only that tier's weights, and nothing else, so
the peak is exactly what a device running that tier would need.

    cd server
    python ../eval/measure_memory.py --team T03 --usns "USN1,USN2,..."

Reads and updates eval/results/results.json, then regenerates eval_P3_<TEAM>.md and the plot.
About 10 minutes on a T4 (mostly loading weights four times).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "server"))

PROMPTS = [l.strip() for l in (HERE / "prompts_compositional.txt").read_text().splitlines() if l.strip()]


def child(tier, n):
    """Runs inside a fresh process: load one tier, generate n images, print one JSON line."""
    import torch
    from darkroom.engine import Engine

    eng = Engine(load_baseline=(tier == "baseline"), only=[tier])
    torch.cuda.synchronize()
    weights_mb = torch.cuda.memory_allocated() / 1e6
    eng.generate(PROMPTS[0], seed=999, tier=tier, preview=False, verify=False, score=False)  # warm-up
    peaks, gens = [], []
    for i in range(n):
        r = eng.generate(PROMPTS[i % len(PROMPTS)], seed=i, tier=tier, preview=False, verify=False, score=False)
        m = r["metrics"]
        peaks.append(m["peak_process_mb"])
        gens.append(m["encode_ms"] + m["denoise_ms"] + m["decode_ms"])
    print("RESULT " + json.dumps({"tier": tier, "peak_mb": max(peaks), "resident_weights_mb": weights_mb,
                                  "gen_ms_isolated": sum(gens) / len(gens), "n": n}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team")
    ap.add_argument("--usns")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        child(a.child, a.n)
        return
    if not (a.team and a.usns):
        ap.error("--team and --usns are required")

    out = Path(a.out)
    R = json.loads((out / "results.json").read_text())
    for tier in ["baseline", "quality", "balanced", "lite"]:
        if tier not in R["tiers"]:
            continue
        print(f"=== {tier}: isolated process ===", flush=True)
        p = subprocess.run([sys.executable, __file__, "--child", tier, "--n", str(a.n)],
                           capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            print(p.stdout[-2000:], p.stderr[-4000:])
            sys.exit(f"{tier} failed - see the error above")
        res = json.loads(line[7:])
        R["tiers"][tier]["peakMB"] = res["peak_mb"]
        R["tiers"][tier]["resident_weights_mb"] = res["resident_weights_mb"]
        R["tiers"][tier]["gen_ms_isolated"] = res["gen_ms_isolated"]
        print(f"    peak {res['peak_mb']:.0f} MB  (weights resident {res['resident_weights_mb']:.0f} MB)", flush=True)
    R["memory_method"] = "isolated: one process per tier, only that tier's weights loaded, no scorer models"
    (out / "results.json").write_text(json.dumps(R, indent=1, default=float))

    import evaluate
    evaluate.plot(R, out / "quality_vs_cost.png")
    evaluate.write_markdown(R, argparse.Namespace(team=a.team, usns=a.usns), out)
    print(f"\nUpdated {out}/eval_P3_{a.team}.md")


if __name__ == "__main__":
    main()
