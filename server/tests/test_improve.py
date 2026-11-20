"""Dry run of eval/improve.py on a tiny random model with fake judges: catches wiring bugs
(run parsing, best-of-N, scoring, report) before any GPU time is spent."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import improve  # noqa: E402
from darkroom.engine import Engine  # noqa: E402
from test_engine_tiny import tiny_pipe  # noqa: E402


class FakeJudges:
    device = "cpu"

    def clip(self, which, prompt, img):
        return float(np.asarray(img).mean()) / 255

    def pickscore(self, prompt, img):
        return 20.0

    def checks(self, which, prompt, img):
        return [("obj", True), ("obj", False), ("rel", False)]


def test_improve_dry_run(tmp_path, monkeypatch):
    import darkroom.config as c
    for k in c.TIERS:
        c.TIERS[k]["res"] = 64
        c.TIERS[k]["steps"] = min(c.TIERS[k]["steps"], 3)
    runs = [improve.parse_run(s) for s in ["lcm:balanced", "lcm:balanced@3", "base:baseline",
                                           "lcm:balanced+nopost+style+cfg1.5"]]
    assert runs[1]["candidates"] == 3 and runs[2]["adapter"] == "base"
    assert runs[3]["post"] is False and runs[3]["style"] and runs[3]["guidance"] == 1.5
    r = improve.parse_run("tcd/dreamshaper8:lite@2+compile")
    assert (r["adapter"], r["base"], r["tier"], r["candidates"], r["compile"]) == ("tcd", "dreamshaper8", "lite", 2, True)
    prompts = ["a red cube left of a blue sphere", "a cat on a chair", "a dog"]
    for r in runs:
        r["coco_prompts"] = prompts

    def make_engine(adapter, base, comp, tiers):
        pipes, taesd = tiny_pipe()
        eng = Engine(pipes=pipes, taesd=taesd)
        monkeypatch.setattr(eng, "clip_score", lambda p, i: 0.3)
        monkeypatch.setattr(eng, "verify", lambda p, i: ([], [{"text": "Found x", "ok": True}]))
        return eng

    improve.run_all(runs, prompts, prompts[:2], tmp_path, make_engine, showcase=prompts[:2])
    improve.showcase_grid(runs, prompts[:2], tmp_path / "showcase.jpg")
    assert (tmp_path / "showcase.jpg").exists()
    res = improve.score_runs(runs, FakeJudges(), tmp_path, fid=False)
    assert res["lcm:balanced@3"]["select_ms_mean"] > 0 and res["lcm:balanced"]["select_ms_mean"] == 0
    assert res["lcm:balanced"]["owlv2"]["object_recall"] == 0.5
    improve.write_report({"device": "cpu", "created": "now", "runs": res}, tmp_path, "lcm:balanced")
    text = (tmp_path / "improve_report.md").read_text()
    assert "lcm:balanced@3" in text and "Change against" in text


def test_pickscore_runs_on_a_real_clip_model():
    """Regression: on transformers 5, get_image_features() returns an object, not a tensor (crashed Kaggle)."""
    import torch
    from transformers import BatchEncoding, CLIPConfig, CLIPModel
    from PIL import Image
    cfg = CLIPConfig(text_config=dict(hidden_size=32, intermediate_size=37, num_attention_heads=4, num_hidden_layers=2,
                                      vocab_size=1000, projection_dim=16, bos_token_id=0, eos_token_id=1, pad_token_id=1),
                     vision_config=dict(hidden_size=32, intermediate_size=37, num_attention_heads=4, num_hidden_layers=2,
                                        image_size=32, patch_size=8, projection_dim=16), projection_dim=16)
    model = CLIPModel(cfg).eval()

    def proc(text=None, images=None, **_):
        return BatchEncoding({"input_ids": torch.randint(2, 1000, (1, 7)), "pixel_values": torch.randn(1, 3, 32, 32)})

    j = improve.Judges.__new__(improve.Judges)
    j.device, j.pick = "cpu", (model, proc)
    s = j.pickscore("a cat", Image.new("RGB", (32, 32)))
    assert isinstance(s, float)
