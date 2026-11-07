"""End-to-end checks of the WebSocket server: out-of-memory fallback and bad input.
The engine is replaced by a fake whose Quality tier always runs out of GPU memory."""
import numpy as np
from fastapi.testclient import TestClient

from darkroom import server


class FakeEngine:
    device = "cuda"
    available = ["quality", "balanced", "lite"]
    model_mb = {}

    def tier_info(self, key):
        from darkroom import config
        return dict(config.TIERS[key])

    def generate(self, prompt, seed, tier, emit, preview, post):
        if tier == "quality":
            raise RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")
        emit({"type": "stage", "stage": "encode", "state": "ok", "ms": 1.0})
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        return {"raw": img, "processed": img, "detections": [], "checks": [], "qa": {}, "tier_used": tier,
                "metrics": {"clip": 0.3, "peak_mb": 100.0, "model_mb": 1.0, "encode_ms": 1, "denoise_ms": 1,
                            "decode_ms": 1, "post_ms": 1, "qa_ms": 1, "verify_ms": 0, "total_ms": 5}}


def client(monkeypatch):
    monkeypatch.setattr(server, "build_engine", lambda: FakeEngine())
    return TestClient(server.app)


def test_out_of_memory_falls_back_to_lighter_tier(monkeypatch):
    with client(monkeypatch) as c, c.websocket_connect("/ws") as ws:
        ws.send_json({"prompt": "a red cube", "seed": 1, "tier": "quality"})
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m["type"] in ("result", "error"):
                break
    assert any(m["type"] == "fallback" and m["from"] == "quality" for m in msgs)
    assert msgs[-1]["type"] == "result"
    assert msgs[-1]["tier_used"] == "balanced"
    assert msgs[-1]["tiers_tried"] == ["quality", "balanced"]


def test_rejects_bad_input(monkeypatch):
    with client(monkeypatch) as c:
        for bad in ({"prompt": "   "}, {"prompt": "x" * 301}, {"prompt": "ok", "tier": "ultra"}):
            with c.websocket_connect("/ws") as ws:
                ws.send_json(bad)
                m = ws.receive_json()
                assert m["type"] == "error" and m["message"]
