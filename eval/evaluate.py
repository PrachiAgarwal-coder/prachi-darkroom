"""Full evaluation: baseline vs optimised tiers, ablation study, failure cases.

    cd server
    python ../eval/evaluate.py --team T01 --usns "1RV23CS001,1RV23CS002" --coco ../data/coco

Outputs (in eval/results/):
  results.json                 machine-readable numbers (the server's /tiers endpoint reads this)
  eval_P3_<TEAM>.md            the submission file (first line = USNs)
  quality_vs_cost.png          Pareto plot
  images/<run>/*.jpg           every generated image
  failures/*.jpg + failures.md failure cases from the verifier

Notes on honesty:
  * FID needs thousands of images to be stable. With a few hundred it is biased upward;
    we also report KID, which is unbiased at small sample sizes.
  * Latency is measured without live previews, after one warm-up call, with CUDA synchronised.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "server"))
from darkroom import config, postprocess  # noqa: E402
from darkroom.engine import Engine  # noqa: E402

TIER_ORDER = ["baseline", "quality", "balanced", "lite"]


# --------------------------------------------------------------------------- helpers
def pct(v, q):
    return float(np.percentile(v, q)) if v else None


def run_set(eng, prompts, tier, out_dir, steps=None, res=None, post=None, verify=False, seed0=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    eng.generate(prompts[0], seed=999, tier=tier, preview=False, verify=False, steps=steps, res=res, score=False)  # warm-up
    for i, p in enumerate(prompts):
        r = eng.generate(p, seed=seed0 + i, tier=tier, preview=False, post=post, verify=verify, steps=steps, res=res)
        Image.fromarray(r["processed"]).save(out_dir / f"{i:04d}.jpg", quality=95)
        rows.append({"prompt": p, "img": r["processed"], **r["metrics"], "qa": r["qa"], "checks": r["checks"],
                     "detections": r["detections"], "file": str(out_dir / f"{i:04d}.jpg")})
    return rows


def summarise(rows):
    gen = [r["encode_ms"] + r["denoise_ms"] + r["decode_ms"] for r in rows]
    tot = [r["total_ms"] for r in rows]
    peaks = [r["peak_mb"] for r in rows if r["peak_mb"]]
    return {
        "n": len(rows),
        "clip": float(np.mean([r["clip"] for r in rows])),
        "clip_std": float(np.std([r["clip"] for r in rows])),
        "gen_ms_mean": float(np.mean(gen)), "gen_ms_p50": pct(gen, 50), "gen_ms_p95": pct(gen, 95),
        "total_ms_mean": float(np.mean(tot)),
        "encMs": float(np.mean([r["encode_ms"] for r in rows])),
        "denoise_ms": float(np.mean([r["denoise_ms"] for r in rows])),
        "decMs": float(np.mean([r["decode_ms"] for r in rows])),
        "post_ms": float(np.mean([r["post_ms"] for r in rows])),
        "peakMB": max(peaks) if peaks else None,
        "sharpness": float(np.mean([r["qa"]["sharpness"] for r in rows])),
        "clipped_pct": float(np.mean([r["qa"]["clipped_pct"] for r in rows])),
    }


def distribution_metrics(real_dir, rows, device):
    """FID and KID between generated images and real COCO images."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    def batch(imgs):
        arr = np.stack([np.array(Image.fromarray(i).resize((299, 299), Image.BICUBIC)) for i in imgs])
        return torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)

    reals = [np.array(Image.open(p).convert("RGB")) for p in sorted(Path(real_dir).glob("*.jpg"))[:len(rows)]]
    fakes = [r["img"] for r in rows]
    n = min(len(reals), len(fakes))
    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(subset_size=min(50, n)).to(device)
    for i in range(0, n, 32):
        fid.update(batch(reals[i:i + 32]), real=True); kid.update(batch(reals[i:i + 32]), real=True)
        fid.update(batch(fakes[i:i + 32]), real=False); kid.update(batch(fakes[i:i + 32]), real=False)
    k_mean, k_std = kid.compute()
    return float(fid.compute()), float(k_mean), float(k_std)


def verify_rate(rows):
    checks = [c for r in rows for c in r["checks"] if not c.get("muted")]
    objs = [c for c in checks if c["text"].startswith("Found")]
    rels = [c for c in checks if not c["text"].startswith("Found")]
    full = [all(c["ok"] for c in r["checks"]) for r in rows]
    rate = lambda xs: float(np.mean([c["ok"] for c in xs])) if xs else None
    return {"object_recall": rate(objs), "relation_acc": rate(rels), "prompt_pass": float(np.mean(full))}


def fmt(v, d=3, unit=""):
    return "—" if v is None else f"{v:.{d}f}{unit}"


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="Team ID, e.g. T01")
    ap.add_argument("--usns", required=True, help="Comma-separated USNs of every member")
    ap.add_argument("--coco", default=str(HERE.parent / "data" / "coco"))
    ap.add_argument("--n", type=int, default=200, help="COCO captions per tier (FID/KID/CLIP)")
    ap.add_argument("--n-baseline", type=int, default=100, help="baseline is slow; use fewer captions")
    ap.add_argument("--n-ablation", type=int, default=40)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--tiers", default="baseline,quality,balanced,lite", help="comma-separated subset to evaluate")
    ap.add_argument("--no-root-copy", action="store_true", help="do not copy the report to the repo root")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    caps = json.loads((Path(a.coco) / "captions.json").read_text())
    coco_prompts = [c["caption"] for c in caps]
    comp_prompts = [l.strip() for l in (HERE / "prompts_compositional.txt").read_text().splitlines() if l.strip()]

    chosen = [t.strip() for t in a.tiers.split(",") if t.strip()]
    eng = Engine(load_baseline="baseline" in chosen)
    dev = eng.device
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else dev
    results = {"device": gpu, "tiers": {}, "ablation": [], "created": time.strftime("%Y-%m-%d %H:%M")}
    all_comp = {}

    for k in [t for t in TIER_ORDER if t in eng.available and t in chosen]:
        n = a.n_baseline if k == "baseline" else a.n
        print(f"\n=== {k}: {n} COCO prompts ===")
        rows = run_set(eng, coco_prompts[:n], k, out / "images" / k)
        s = summarise(rows)
        s["fid"], s["kid"], s["kid_std"] = distribution_metrics(Path(a.coco) / "images", rows, dev)
        print(f"=== {k}: compositional prompts ===")
        comp = run_set(eng, comp_prompts, k, out / "images" / f"{k}_compositional", verify=True, seed0=1000)
        all_comp[k] = comp
        s.update(verify_rate(comp))
        info = eng.tier_info(k)
        s.update({"name": info["name"], "steps": info["steps"], "res": info["res"], "precision": info["precision"],
                  "decoder": info["decoder"], "unet": info["unet"], "sizeMB": info["model_mb"],
                  "stepMs": s["denoise_ms"] / info["steps"]})
        results["tiers"][k] = s
        print(json.dumps({x: s[x] for x in ("clip", "fid", "kid", "gen_ms_mean", "peakMB", "sizeMB", "prompt_pass")}, indent=1))

    # ----------------------------------------------------------------- ablation
    if not a.skip_ablation:
        P = coco_prompts[:a.n_ablation]
        exps = [
            ("Sampling steps", "LCM 1 step", "fast", dict(tier="quality", steps=1)),
            ("Sampling steps", "LCM 2 steps", "fast", dict(tier="quality", steps=2)),
            ("Sampling steps", "LCM 4 steps", "fast", dict(tier="quality", steps=4)),
            ("Sampling steps", "LCM 8 steps", "fast", dict(tier="quality", steps=8)),
            ("Decoder", "Full VAE, 4 steps", "fast", dict(tier="quality", steps=4)),
            ("Decoder", "TAESD, 4 steps", "fast_taesd", dict(tier="balanced", steps=4)),
            ("Precision", "FP16 UNet (TAESD, 4 steps, 512)", "fast_taesd", dict(tier="balanced", steps=4, res=512)),
            ("Precision", "INT8 UNet (TAESD, 4 steps, 512)", "lite", dict(tier="lite", steps=4, res=512)),
            ("Resolution", "Lite at 384 px + upscale", "lite", dict(tier="lite", res=384)),
            ("Resolution", "Lite at 512 px", "lite", dict(tier="lite", res=512)),
            ("Post-processing", "Balanced, post-processing off", "fast_taesd",
             dict(tier="balanced", post=dict(contrast=False, wb=False, sharp=False, upscale=True))),
            ("Post-processing", "Balanced, post-processing on", "fast_taesd", dict(tier="balanced")),
        ]
        for group, label, pipe, kw in [e for e in exps if e[3]["tier"] in eng.available]:
            print(f"--- ablation: {label}")
            slug = label.lower().replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
            rows = run_set(eng, P, out_dir=out / "images" / f"abl_{slug}", **kw)
            s = summarise(rows)
            results["ablation"].append({"group": group, "label": label, "size_mb": eng.model_mb[pipe], **s})

    # ----------------------------------------------------------------- failures
    fail_dir = out / "failures"; fail_dir.mkdir(exist_ok=True)
    failures = []
    for k, rows in all_comp.items():
        for r in rows:
            bad = [c["text"] for c in r["checks"] if not c["ok"]]
            if bad:
                kind = "missing object" if any(b.startswith("Found") for b in bad) else "wrong spatial relation"
                failures.append({"tier": k, "prompt": r["prompt"], "failed": bad, "kind": kind, "file": r["file"]})
    for i, f in enumerate(failures[:12]):
        dst = fail_dir / f"{i:02d}_{f['tier']}.jpg"
        Image.open(f["file"]).save(dst); f["file"] = str(dst.relative_to(out))
    results["failures"] = failures

    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    plot(results, out / "quality_vs_cost.png")
    write_markdown(results, a, out, root_copy=not a.no_root_copy)
    print(f"\nDone. See {out}/eval_P3_{a.team}.md")


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for k, s in results["tiers"].items():
        ax.scatter(s["gen_ms_mean"] / 1000, s["clip"], s=40 + (s["peakMB"] or 500) / 25,
                   color="#7a7a7a" if k == "baseline" else "#1f9e89", alpha=.85)
        ax.annotate(s["name"], (s["gen_ms_mean"] / 1000, s["clip"]), xytext=(6, 4), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("Generation latency per image (s, log)"); ax.set_ylabel("CLIP similarity")
    ax.set_title("Quality vs cost (marker size = peak memory)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path)


def write_markdown(R, a, out, root_copy=True):
    T = R["tiers"]
    b = T.get("baseline")
    L = [", ".join(u.strip() for u in a.usns.split(",")), "",
         f"# Evaluation — P3 Resource-Efficient Text-to-Image Generation — Team {a.team}", "",
         f"Hardware: {R['device']}. Generated {R['created']}. Seeds are fixed (0..N-1); "
         f"COCO val2017 captions, deterministic subset.", ""]

    def row(k, cells):
        return f"| {T[k]['name']} | " + " | ".join(cells) + " |"

    L += ["## 1. CLIP similarity (text-image alignment)", "",
          "| Tier | CLIP ↑ | Std | N |", "|---|---|---|---|"]
    L += [row(k, [fmt(s["clip"]), fmt(s["clip_std"]), str(s["n"])]) for k, s in T.items()]
    L += ["", "## 2. FID and KID (distribution-level, vs real COCO images)", "",
          "| Tier | FID ↓ | KID ×1000 ↓ |", "|---|---|---|"]
    L += [row(k, [fmt(s["fid"], 1), f"{s['kid'] * 1000:.2f} ± {s['kid_std'] * 1000:.2f}"]) for k, s in T.items()]
    ns = {s["n"] for s in T.values()}
    L += ["", "_FID with a few hundred samples is biased upward; compare tiers against each other, not against "
          "published numbers. KID is unbiased at this sample size._"]
    if len(ns) > 1:
        L += ["", "_**Tiers were evaluated on different numbers of images** (" +
              ", ".join(f"{s['name']} {s['n']}" for s in T.values()) +
              "). FID is strongly biased by sample count, so FID values are **not comparable across these rows**; "
              "use KID for cross-tier comparison._"]
    L += ["",
          "## 3. Image generation latency", "",
          "| Tier | Steps | Encode | Denoise | Decode | Generation mean | p50 | p95 | Speed-up |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, s in T.items():
        sp = f"{b['gen_ms_mean'] / s['gen_ms_mean']:.1f}×" if b else "—"
        L.append(row(k, [str(s["steps"]), fmt(s["encMs"], 0, " ms"), fmt(s["denoise_ms"], 0, " ms"),
                         fmt(s["decMs"], 0, " ms"), fmt(s["gen_ms_mean"], 0, " ms"), fmt(s["gen_ms_p50"], 0, " ms"),
                         fmt(s["gen_ms_p95"], 0, " ms"), sp]))
    isolated = "isolated" in R.get("memory_method", "")
    L += ["", "## 4. Peak memory consumption", "",
          ("Measured in isolation (`eval/measure_memory.py`): one process per tier, loading only that tier's "
           "weights and no scorer models, so each number is what a device running that tier needs."
           if isolated else
           "**Not yet isolated:** these peaks include every tier and the scorer models loaded in the same process. "
           "Run `eval/measure_memory.py` to replace them with per-tier measurements."), "",
          "| Tier | Peak GPU memory | Reduction |", "|---|---|---|"]
    for k, s in T.items():
        red = f"{b['peakMB'] / s['peakMB']:.1f}×" if b and s["peakMB"] and b["peakMB"] else "—"
        L.append(row(k, [fmt(s["peakMB"], 0, " MB"), red]))
    L += ["", "## 5. Model size", "", "| Tier | Weights (UNet + text encoder + decoder) | Precision | Decoder |",
          "|---|---|---|---|"]
    L += [row(k, [fmt(s["sizeMB"], 0, " MB"), s["precision"], s["decoder"]]) for k, s in T.items()]
    L += ["", "## 6. Quality versus computational cost", "", "![quality vs cost](quality_vs_cost.png)", "",
          "| Tier | CLIP | FID | Latency | Peak memory | Size |", "|---|---|---|---|---|---|"]
    L += [row(k, [fmt(s["clip"]), fmt(s["fid"], 1), fmt(s["gen_ms_mean"], 0, " ms"), fmt(s["peakMB"], 0, " MB"),
                  fmt(s["sizeMB"], 0, " MB")]) for k, s in T.items()]
    L += ["", "## 7. Semantic consistency on compositional prompts", "",
          f"{sum(1 for l in (HERE / 'prompts_compositional.txt').read_text().splitlines() if l.strip())} prompts with multiple objects, "
          "colour attributes and spatial relations, checked by an open-vocabulary detector (OWL-ViT).", "",
          "| Tier | Object recall | Relation accuracy | Prompts fully satisfied |", "|---|---|---|---|"]
    L += [row(k, [fmt(s["object_recall"], 2), fmt(s["relation_acc"], 2), fmt(s["prompt_pass"], 2)]) for k, s in T.items()]
    if R["ablation"]:
        L += ["", "## 8. Ablation study", "",
              "Each row changes one factor; everything else is held fixed.", "",
              "| Factor | Configuration | CLIP | Generation latency | Sharpness | Clipped px |",
              "|---|---|---|---|---|---|"]
        L += [f"| {x['group']} | {x['label']} | {fmt(x['clip'])} | {fmt(x['gen_ms_mean'], 0, ' ms')} | "
              f"{fmt(x['sharpness'], 0)} | {fmt(x['clipped_pct'], 2, ' %')} |"
              for x in R["ablation"]]
        L += ["", "_Memory per configuration is reported in §4 (measured per tier in isolation)._"]
    L += ["", "## 9. Failure cases", ""]
    if R["failures"]:
        L += ["| Tier | Prompt | What failed | Type | Image |", "|---|---|---|---|---|"]
        L += [f"| {f['tier']} | {f['prompt']} | {'; '.join(f['failed'])} | {f['kind']} | "
              f"{'![](' + f['file'] + ')' if f['file'].startswith('failures') else ''} |" for f in R["failures"][:12]]
    else:
        L.append("No verifier failures on this prompt set.")
    L += ["", "_Write the cause of at least three of these failures in docs/DESIGN.md §Failure analysis._", ""]
    text = "\n".join(L)
    (out / f"eval_P3_{a.team}.md").write_text(text)
    if root_copy:   # the submission asks for eval_<ProbID>_<TeamID>.md; keep a copy at the repo root as well
        (HERE.parent / f"eval_P3_{a.team}.md").write_text(
            text.replace("](quality_vs_cost.png)", "](eval/results/quality_vs_cost.png)")
                .replace("](failures/", "](eval/results/failures/"))


if __name__ == "__main__":
    main()
