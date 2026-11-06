import numpy as np
from darkroom import prompts, postprocess


def test_parse_relations():
    p = prompts.parse("a red cube to the left of a blue sphere on a wooden table at sunset")
    assert [prompts.phrase(o) for o in p["objects"]] == ["red cube", "blue sphere"]
    assert p["relations"] == [{"a": 0, "rel": "left", "b": 1}]
    p = prompts.parse("two orange cats beside a purple lamp")
    assert len(p["objects"]) == 3 and p["relations"][0]["rel"] == "beside"
    p = prompts.parse("a yellow auto rickshaw in front of a white temple at night")
    assert prompts.phrase(p["objects"][0]) == "yellow auto rickshaw" and p["relations"][0]["rel"] == "front"


def test_check_relation():
    assert prompts.check_relation("left", [0, 0, .2, .2], [.6, 0, .8, .2])
    assert not prompts.check_relation("left", [.6, 0, .8, .2], [0, 0, .2, .2])
    assert prompts.check_relation("above", [0, 0, .2, .2], [0, .6, .2, .8])


def test_postprocess_chain():
    img = (np.random.default_rng(0).random((384, 384, 3)) * 255).astype(np.uint8)
    out, qa, post_ms, qa_ms = postprocess.run(img)
    assert out.shape == (512, 512, 3) and out.dtype == np.uint8
    for k in ("sharpness", "clipped_pct", "colorfulness", "checkerboard", "fixes"):
        assert k in qa
