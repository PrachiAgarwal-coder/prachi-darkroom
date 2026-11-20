"""Improvement experiments for round 2, graded by judges that are NOT used to pick images.

Each run is "adapter:tier[@candidates][+compile]", for example:
    lcm:balanced            the submitted Balanced tier (reference for every delta)
    lcm:balanced@4          best-of-4: four seeds in one batch, the verifier keeps one
    tcd:balanced            TCD-LoRA instead of LCM-LoRA, same 4 steps
    hyper:balanced          Hyper-SD 4-step LoRA instead of LCM-LoRA
    lcm:balanced+compile    torch.compile on the UNet
    base:baseline           the FP32 50-step baseline (slow: ~32 s an image)

    cd server
    python ../eval/improve.py --coco ../data/coco --n 300 --baseline-n 100 \\
        --runs lcm:balanced,lcm:balanced@4,tcd:balanced,hyper:balanced,lcm:balanced+compile

Why separate judges: best-of-N picks the candidate that OWL-ViT and CLIP ViT-B/32 like most, so
scoring it with the same two models would be marking our own homework. Here every image of every
run is graded by CLIP ViT-L/14, OWLv2 and PickScore (a human-preference model), none of which
take part in selection. The old CLIP B/32 and OWL-ViT numbers are kept only for continuity.

Writes eval/results_improve/results.json and improve_report.md. Images in images/<run>/.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "server"))
sys.path.insert(0, str(HERE))
from darkroom import config, prompts as P  # noqa: E402

OFFICIAL_BASELINE_MS = 31814.8      # eval/results/results.json, used when the baseline is not re-run


# --------------------------------------------------------------------------- run specs
def parse_run(spec):
    """'lcm:balanced@4+compile' -> dict(adapter, tier, candidates, compile, key)."""
    s = spec.strip()
    comp = s.endswith("+compile")
    s = s.removesuffix("+compile")
    adapter, _, rest = s.partition(":")
    tier, _, cand = rest.partition("@")
    adapter, tier = adapter.strip(), tier.strip()
    if tier not in config.TIERS:
        raise SystemExit(f"Unknown tier in {spec!r}")
    if tier == "baseline":
        adapter = "base"
    elif adapter not in config.ADAPTERS:
        raise SystemExit(f"Unknown adapter in {spec!r}; choose from {list(config.ADAPTERS)}")
    return {"key": spec.strip(), "adapter": adapter, "tier": tier, "candidates": int(cand or 1), "compile": comp}


def group_runs(runs):
    """One Engine per (adapter, compile) so each set of weights is loaded once."""
    groups = {}
    for r in runs:
        groups.setdefault((r["adapter"], r["compile"]), []).append(r)
    return groups


# --------------------------------------------------------------------------- generation
def generate_set(eng, run, prompt_list, out_dir, seed0):
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = {"candidates": run["candidates"]} if run["candidates"] > 1 else {}
    eng.generate(prompt_list[0], seed=999, tier=run["tier"], preview=False, verify=False, score=False, **kw)  # warm-up
    rows = []
    for i, p in enumerate(prompt_list):
        r = eng.generate(p, seed=seed0 + i, tier=run["tier"], preview=False, verify=False, score=False, **kw)
        m = r["metrics"]
        f = out_dir / f"{i:04d}.jpg"
        Image.fromarray(r["processed"]).save(f, quality=95)
        rows.append({"prompt": p, "file": str(f), "gen_ms": m["encode_ms"] + m["denoise_ms"] + m["decode_ms"],
                     "select_ms": m.get("select_ms", 0.0), "peak_mb": m["peak_mb"], "chosen": m.get("chosen", 0),
                     "sharpness": r["qa"]["sharpness"], "clipped_pct": r["qa"]["clipped_pct"]})
    return rows


def run_all(runs, coco_prompts, comp_prompts, out, make_engine):
    for (adapter, comp), group in group_runs(runs).items():
        tiers = sorted({r["tier"] for r in group})
        print(f"\n##### loading engine: adapter={adapter} compile={comp} tiers={tiers}", flush=True)
        eng = None
        try:                      # one failed experiment (a download, a compile error) must not end the others
            eng = make_engine(adapter, comp, tiers)
            for r in group:
                print(f"=== {r['key']}: {len(r['coco_prompts'])} COCO + {len(comp_prompts)} compositional", flush=True)
                t0 = time.time()
                r["coco"] = generate_set(eng, r, r["coco_prompts"], out / "images" / slug(r["key"]), 0)
                r["comp"] = generate_set(eng, r, comp_prompts, out / "images" / (slug(r["key"]) + "_comp"), 1000)
                r["model_mb"] = eng.model_mb[config.TIERS[r["tier"]]["pipe"]]
                print(f"    done in {time.time() - t0:.0f} s", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            for r in group:
                if "comp" not in r:
                    r["error"] = f"{type(e).__name__}: {e}"
                    print(f"!!! {r['key']} FAILED: {r['error']}", flush=True)
        del eng
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def slug(key):
    return key.replace(":", "_").replace("@", "_x").replace("+", "_")


# --------------------------------------------------------------------------- judges
class Judges:
    """CLIP ViT-L/14 (prompt match), OWLv2 (objects + positions), PickScore (human preference),
    plus the old CLIP B/32 and OWL-ViT for continuity with round 1."""

    def __init__(self, device, pickscore=True, owl_threshold=0.10):
        from transformers import (AutoModel, AutoProcessor, CLIPModel, CLIPProcessor, Owlv2ForObjectDetection,
                                  Owlv2Processor, OwlViTForObjectDetection, OwlViTProcessor)
        self.device, self.thr = device, owl_threshold
        self.clip_l = (CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval(),
                       CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14"))
        self.clip_b = (CLIPModel.from_pretrained(config.CLIP_MODEL).to(device).eval(),
                       CLIPProcessor.from_pretrained(config.CLIP_MODEL))
        self.owl2 = (Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval(),
                     Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble"))
        self.owl1 = (OwlViTForObjectDetection.from_pretrained(config.DETECTOR).to(device).eval(),
                     OwlViTProcessor.from_pretrained(config.DETECTOR))
        self.pick = None
        if pickscore:
            try:
                self.pick = (AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to(device).eval(),
                             AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K"))
            except Exception as e:  # optional: never let it stop the run
                print(f"PickScore unavailable ({e}); continuing without it")

    @torch.no_grad()
    def clip(self, which, prompt, img):
        m, proc = self.clip_l if which == "L" else self.clip_b
        inp = proc(text=[prompt], images=img, return_tensors="pt", padding=True, truncation=True).to(self.device)
        out = m(**inp)
        i = torch.nn.functional.normalize(out.image_embeds, dim=-1)
        t = torch.nn.functional.normalize(out.text_embeds, dim=-1)
        return float((i * t).sum())

    @torch.no_grad()
    def pickscore(self, prompt, img):
        if self.pick is None:
            return None
        m, proc = self.pick
        im = proc(images=img, return_tensors="pt").to(self.device)
        tx = proc(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        i = torch.nn.functional.normalize(m.get_image_features(**im), dim=-1)
        t = torch.nn.functional.normalize(m.get_text_features(**tx), dim=-1)
        return float(m.logit_scale.exp() * (i * t).sum())

    @torch.no_grad()
    def checks(self, which, prompt, img):
        """Same rules as the app's verifier (prompts.parse + check_relation), different detector."""
        parsed = P.parse(prompt)
        objs = parsed["objects"]
        if not objs:
            return []
        m, proc = self.owl2 if which == "v2" else self.owl1
        queries = [f"a photo of a {P.phrase(o)}" for o in objs]
        out = m(**proc(text=[queries], images=img, return_tensors="pt").to(self.device))
        scores, boxes = out.logits[0].sigmoid(), out.pred_boxes[0]
        used, res = set(), []
        for q, o in enumerate(objs):
            order = torch.argsort(scores[:, q], descending=True)
            best = next((int(k) for k in order[:50] if int(k) not in used), int(order[0]))
            used.add(best)
            cx, cy, w, h = boxes[best].tolist()
            o["box"] = [max(0, cx - w / 2), max(0, cy - h / 2), min(1, cx + w / 2), min(1, cy + h / 2)]
            o["found"] = float(scores[best, q]) >= self.thr
            res.append(("obj", o["found"]))
        for r in parsed["relations"]:
            a, b = objs[r["a"]], objs[r["b"]]
            res.append(("rel", a["found"] and b["found"] and P.check_relation(r["rel"], a["box"], b["box"])))
        return res


def faithfulness(check_lists):
    objs = [ok for cl in check_lists for kind, ok in cl if kind == "obj"]
    rels = [ok for cl in check_lists for kind, ok in cl if kind == "rel"]
    full = [all(ok for _, ok in cl) for cl in check_lists if cl]
    f = lambda xs: float(np.mean(xs)) if xs else None
    return {"object_recall": f(objs), "relation_acc": f(rels), "prompt_pass": f(full), "n_prompts": len(full)}


def score_runs(runs, judges, coco_dir, fid=True, fid_n=100):
    if fid:
        from evaluate import distribution_metrics
    runs = [r for r in runs if "error" not in r]
    for r in runs:
        print(f"--- judging {r['key']}", flush=True)
        imgs = [Image.open(x["file"]).convert("RGB") for x in r["coco"]]
        prom = [x["prompt"] for x in r["coco"]]
        clip_l = [judges.clip("L", p, i) for p, i in zip(prom, imgs)]
        clip_b = [judges.clip("B", p, i) for p, i in zip(prom, imgs)]
        pick = [judges.pickscore(p, i) for p, i in zip(prom, imgs)]
        comp_imgs = [Image.open(x["file"]).convert("RGB") for x in r["comp"]]
        comp_prom = [x["prompt"] for x in r["comp"]]
        v2 = faithfulness([judges.checks("v2", p, i) for p, i in zip(comp_prom, comp_imgs)])
        v1 = faithfulness([judges.checks("v1", p, i) for p, i in zip(comp_prom, comp_imgs)])
        gen = [x["gen_ms"] for x in r["coco"]]
        e2e = [x["gen_ms"] + x["select_ms"] for x in r["coco"]]
        s = {
            "adapter": r["adapter"], "tier": r["tier"], "candidates": r["candidates"], "compile": r["compile"],
            "n": len(imgs), "model_mb": r.get("model_mb"),
            "gen_ms_mean": float(np.mean(gen)), "gen_ms_p95": float(np.percentile(gen, 95)),
            "select_ms_mean": float(np.mean([x["select_ms"] for x in r["coco"]])),
            "e2e_ms_mean": float(np.mean(e2e)), "e2e_ms_p95": float(np.percentile(e2e, 95)),
            "peak_mb": max((x["peak_mb"] for x in r["coco"] if x["peak_mb"]), default=None),
            "clip_l14": float(np.mean(clip_l)), "clip_l14_std": float(np.std(clip_l)),
            "clip_b32": float(np.mean(clip_b)),
            "pickscore": float(np.mean([p for p in pick if p is not None])) if any(p is not None for p in pick) else None,
            "sharpness": float(np.mean([x["sharpness"] for x in r["coco"]])),
            "clipped_pct": float(np.mean([x["clipped_pct"] for x in r["coco"]])),
            "owlv2": v2, "owlvit": v1,
        }
        if fid:
            rows = [{"img": np.array(i)} for i in imgs]
            s["fid"], s["kid"], s["kid_std"] = distribution_metrics(Path(coco_dir) / "images", rows, judges.device)
            n100 = min(fid_n, len(rows))
            s["fid_at_100"], s["kid_at_100"], _ = distribution_metrics(Path(coco_dir) / "images", rows[:n100], judges.device)
        r["summary"] = s
    return {r["key"]: r["summary"] for r in runs}


# --------------------------------------------------------------------------- report
def speedup(base, ms):
    x = base / ms
    return f"{x:.1f}×" if x < 10 else f"{x:.0f}×"


def fmt(v, d=3, scale=1.0):
    return "—" if v is None else f"{v * scale:.{d}f}"


def write_report(results, out, ref_key):
    S = results["runs"]
    ref = S.get(ref_key)
    base_ms = S.get("base:baseline", {}).get("gen_ms_mean") or OFFICIAL_BASELINE_MS
    L = ["# Round 2 improvement experiments", "",
         f"Device: {results['device']} · {results['created']} · reference run: `{ref_key}`", "",
         "Every image is graded by **CLIP ViT-L/14, OWLv2 and PickScore**, none of which choose images. "
         "Best-of-N runs pick with OWL-ViT + CLIP B/32, so those two columns are biased for them and kept only for continuity.", "",
         "## Cost", "",
         "| Run | N | Generate ms (mean / p95) | Select ms | End-to-end ms | Speed-up vs baseline | Peak MB |",
         "|---|---|---|---|---|---|---|"]
    for k, s in S.items():
        L.append(f"| `{k}` | {s['n']} | {s['gen_ms_mean']:.0f} / {s['gen_ms_p95']:.0f} | {s['select_ms_mean']:.0f} | "
                 f"{s['e2e_ms_mean']:.0f} | {speedup(base_ms, s['e2e_ms_mean'])} | {fmt(s['peak_mb'], 0)} |")
    L += ["", "## Quality (independent judges)", "",
          "| Run | CLIP L/14 ↑ | PickScore ↑ | KID ×1000 ↓ (all N) | FID @100 ↓ | KID ×1000 @100 ↓ | Sharpness | Clipped % |",
          "|---|---|---|---|---|---|---|---|"]
    for k, s in S.items():
        L.append(f"| `{k}` | {fmt(s['clip_l14'], 4)} ± {fmt(s['clip_l14_std'], 3)} | {fmt(s['pickscore'], 2)} | "
                 f"{fmt(s.get('kid'), 2, 1000)} ± {fmt(s.get('kid_std'), 2, 1000)} | {fmt(s.get('fid_at_100'), 1)} | "
                 f"{fmt(s.get('kid_at_100'), 2, 1000)} | {s['sharpness']:.0f} | {s['clipped_pct']:.2f} |")
    L += ["", "## Prompt faithfulness, 30 compositional prompts (OWLv2 judge; OWL-ViT in brackets)", "",
          "| Run | Objects found | Positions right | Fully passed |", "|---|---|---|---|"]
    for k, s in S.items():
        a, b = s["owlv2"], s["owlvit"]
        L.append(f"| `{k}` | {fmt(a['object_recall'], 2)} ({fmt(b['object_recall'], 2)}) | "
                 f"{fmt(a['relation_acc'], 2)} ({fmt(b['relation_acc'], 2)}) | "
                 f"{fmt(a['prompt_pass'], 2)} ({fmt(b['prompt_pass'], 2)}) |")
    if ref:
        L += ["", f"## Change against `{ref_key}`", "", "| Run | Δ CLIP L/14 | Δ PickScore | Δ KID ×1000 | Δ positions right (OWLv2) | Δ fully passed (OWLv2) | Time × |",
              "|---|---|---|---|---|---|---|"]
        d = lambda a, b, sc=1.0, dd=3: "—" if a is None or b is None else f"{(a - b) * sc:+.{dd}f}"
        for k, s in S.items():
            if k == ref_key:
                continue
            L.append(f"| `{k}` | {d(s['clip_l14'], ref['clip_l14'], 1, 4)} | {d(s['pickscore'], ref['pickscore'], 1, 2)} | "
                     f"{d(s.get('kid'), ref.get('kid'), 1000, 2)} | {d(s['owlv2']['relation_acc'], ref['owlv2']['relation_acc'], 1, 2)} | "
                     f"{d(s['owlv2']['prompt_pass'], ref['owlv2']['prompt_pass'], 1, 2)} | {s['e2e_ms_mean'] / ref['e2e_ms_mean']:.2f}× |")
    for k, err in results.get("failed", {}).items():
        L.append(f"\n**{k} did not run:** {err}")
    L += ["", "Read differences in CLIP below about 0.005 and in KID within the ± values as noise."]
    (out / "improve_report.md").write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default=str(HERE.parent / "data" / "coco"))
    ap.add_argument("--n", type=int, default=300, help="COCO captions per fast run")
    ap.add_argument("--baseline-n", type=int, default=100, help="COCO captions for base:baseline (32 s each)")
    ap.add_argument("--runs", default="lcm:balanced,lcm:balanced@4,tcd:balanced,hyper:balanced,lcm:balanced+compile")
    ap.add_argument("--reference", default="lcm:balanced")
    ap.add_argument("--out", default=str(HERE / "results_improve"))
    ap.add_argument("--no-fid", action="store_true")
    ap.add_argument("--no-pickscore", action="store_true")
    ap.add_argument("--owl-threshold", type=float, default=0.10)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    caps = json.loads((Path(a.coco) / "captions.json").read_text())
    coco_prompts = [c["caption"] for c in caps]
    comp_prompts = [l.strip() for l in (HERE / "prompts_compositional.txt").read_text().splitlines() if l.strip()]
    runs = [parse_run(s) for s in a.runs.split(",") if s.strip()]
    for r in runs:
        n = a.baseline_n if r["tier"] == "baseline" else a.n
        if n > len(coco_prompts):
            raise SystemExit(f"Only {len(coco_prompts)} COCO captions downloaded; rerun download_coco.py --n {n}")
        r["coco_prompts"] = coco_prompts[:n]

    from darkroom.engine import Engine

    def make_engine(adapter, comp, tiers):
        if adapter == "base":
            return Engine(load_baseline=True, only=["baseline"])
        return Engine(load_baseline=False, only=tiers, adapter=adapter, compile_unet=comp)

    run_all(runs, coco_prompts, comp_prompts, out, make_engine)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    judges = Judges(dev, pickscore=not a.no_pickscore, owl_threshold=a.owl_threshold)
    results = {"device": torch.cuda.get_device_name(0) if dev == "cuda" else dev,
               "created": time.strftime("%Y-%m-%d %H:%M"), "reference": a.reference,
               "runs": score_runs(runs, judges, a.coco, fid=not a.no_fid),
               "failed": {r["key"]: r["error"] for r in runs if "error" in r}}
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    write_report(results, out, a.reference)
    print((out / "improve_report.md").read_text())


if __name__ == "__main__":
    main()
