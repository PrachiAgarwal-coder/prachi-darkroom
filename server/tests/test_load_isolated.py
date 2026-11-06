"""Exercises Engine._load(only=[tier]) with the hub calls replaced by a tiny random pipeline,
so single-tier loading (used by eval/measure_memory.py) is tested without downloads."""
import pytest
from diffusers import AutoencoderTiny, StableDiffusionPipeline

import test_engine_tiny as tt
from darkroom.engine import Engine


@pytest.fixture
def fake_hub(monkeypatch):
    def fake_sd(*a, **k):
        p, _ = tt.tiny_pipe()
        return p["fast"]
    monkeypatch.setattr(StableDiffusionPipeline, "from_pretrained", classmethod(lambda cls, *a, **k: fake_sd()))
    monkeypatch.setattr(AutoencoderTiny, "from_pretrained", classmethod(lambda cls, *a, **k: AutoencoderTiny()))
    for name in ("load_lora_weights", "fuse_lora", "unload_lora_weights", "set_adapters"):
        monkeypatch.setattr(StableDiffusionPipeline, name, lambda self, *a, **k: None)


@pytest.mark.parametrize("tier,pipe", [("quality", "fast"), ("balanced", "fast_taesd"), ("lite", "lite"), ("baseline", "baseline")])
def test_only_one_tier(fake_hub, tier, pipe):
    eng = Engine(load_baseline=False, only=[tier])
    assert list(eng.pipes) == [pipe]
    assert eng.available == [tier]


def test_lite_alone_is_quantised(fake_hub):
    eng = Engine(load_baseline=False, only=["lite"])
    assert eng.precision_actual["lite"] == "INT8"
    assert eng.model_mb["lite"] > 0


def test_full_load(fake_hub):
    eng = Engine(load_baseline=True)
    assert set(eng.pipes) == {"fast", "fast_taesd", "lite", "baseline"}
