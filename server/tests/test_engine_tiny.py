"""Runs the real Engine.generate code path on a tiny, randomly initialised
Stable Diffusion (no downloads), to catch API/shape bugs without a GPU.
CLIP scoring and the detector are stubbed because they need pretrained weights.
Run: python -m pytest -q tests  (from backend/)"""
import json, tempfile, os
import numpy as np, torch
from diffusers import AutoencoderKL, AutoencoderTiny, LCMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from darkroom.engine import Engine


def tiny_pipe():
    torch.manual_seed(0)
    unet = UNet2DConditionModel(block_out_channels=(32, 64), layers_per_block=1, sample_size=16, in_channels=4,
                                out_channels=4, down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32, attention_head_dim=4, norm_num_groups=8)
    vae = AutoencoderKL(norm_num_groups=8, block_out_channels=(16, 16, 16, 16), in_channels=3, out_channels=3, latent_channels=4,
                        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4)
    te = CLIPTextModel(CLIPTextConfig(hidden_size=32, intermediate_size=37, num_attention_heads=4, num_hidden_layers=2,
                                      vocab_size=1000, max_position_embeddings=77, bos_token_id=0, eos_token_id=1, pad_token_id=1))
    d = tempfile.mkdtemp()
    json.dump({"<|startoftext|>": 0, "<|endoftext|>": 1}, open(os.path.join(d, "vocab.json"), "w"))
    open(os.path.join(d, "merges.txt"), "w").write("#version: 0.2\n")
    tok = CLIPTokenizer(os.path.join(d, "vocab.json"), os.path.join(d, "merges.txt"), model_max_length=77)
    sched = LCMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    p = StableDiffusionPipeline(vae=vae, text_encoder=te, tokenizer=tok, unet=unet, scheduler=sched,
                                safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    taesd = AutoencoderTiny()
    p2 = StableDiffusionPipeline(vae=taesd, text_encoder=te, tokenizer=tok, unet=unet, scheduler=sched,
                                 safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    for x in (p, p2): x.set_progress_bar_config(disable=True)
    return {"fast": p, "fast_taesd": p2, "lite": p2, "baseline": p}, taesd


def test_generate_all_tiers(monkeypatch):
    pipes, taesd = tiny_pipe()
    eng = Engine(pipes=pipes, taesd=taesd)
    monkeypatch.setattr(eng, "clip_score", lambda p, i: 0.3)
    monkeypatch.setattr(eng, "verify", lambda p, i: ([], []))
    import darkroom.config as c
    for k in c.TIERS:
        c.TIERS[k]["res"] = 64            # keep the tiny model fast
        c.TIERS[k]["steps"] = min(c.TIERS[k]["steps"], 3)
    events = []
    for tier in ["quality", "balanced", "lite", "baseline"]:
        r = eng.generate("a red cube left of a blue sphere", seed=1, tier=tier, on_event=events.append)
        assert r["processed"].shape == (512, 512, 3) and r["processed"].dtype == np.uint8
        assert r["metrics"]["total_ms"] > 0
    steps = [e for e in events if e["type"] == "step"]
    assert steps and all(e.get("preview", "").startswith("data:image/jpeg") for e in steps)
    assert eng.model_mb["fast"] > 0
