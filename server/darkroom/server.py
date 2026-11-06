"""HTTP + WebSocket server. Serves the frontend at / and the generator at /ws,
so the whole app lives behind one URL.

Run:  uvicorn darkroom.server:app --host 0.0.0.0 --port 8000
"""
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import config
from .engine import to_data_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("darkroom")

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = Path(os.getenv("DARKROOM_FRONTEND", ROOT / "server" / "static" / "index.html"))
RESULTS = Path(os.getenv("DARKROOM_RESULTS", ROOT / "eval" / "results" / "results.json"))
MAX_PROMPT = 300

state = {"engine": None, "gpu_lock": None}


def build_engine():
    if os.getenv("DARKROOM_MOCK") == "1":
        from .mock import MockEngine
        log.warning("MOCK MODE: images are placeholders, not model output")
        return MockEngine()
    from .engine import Engine
    eng = Engine()
    if os.getenv("DARKROOM_WARMUP", "1") == "1":
        for k in eng.available:       # first call compiles kernels and loads CLIP/detector
            try:
                eng.generate("a red cube next to a blue sphere", 0, k, preview=False)
                log.info("warmed %s", k)
            except Exception as e:
                log.warning("warmup %s failed: %s", k, e)
    return eng


@asynccontextmanager
async def lifespan(app):
    state["gpu_lock"] = asyncio.Lock()
    state["engine"] = await asyncio.to_thread(build_engine)
    yield


app = FastAPI(title="Darkroom", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def index():
    return FileResponse(FRONTEND)


@app.get("/health")
def health():
    eng = state["engine"]
    return {"ok": eng is not None, "device": getattr(eng, "device", None),
            "tiers": getattr(eng, "available", []), "mock": os.getenv("DARKROOM_MOCK") == "1"}


@app.get("/tiers")
def tiers():
    """Tier table for the UI. Uses numbers measured by eval/evaluate.py when present."""
    eng = state["engine"]
    measured = {}
    if RESULTS.exists():
        try:
            measured = json.loads(RESULTS.read_text()).get("tiers", {})
        except Exception:
            pass
    out = {}
    for k in (eng.available if eng else config.TIERS):
        t = eng.tier_info(k) if eng else dict(config.TIERS[k])
        t.update(measured.get(k, {}))
        out[k] = t
    return JSONResponse(out)


def _validate(msg, available):
    prompt = str(msg.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Write a prompt first.")
    if len(prompt) > MAX_PROMPT:
        raise ValueError(f"Prompts are limited to {MAX_PROMPT} characters.")
    tier = msg.get("tier", "balanced")
    if tier not in config.TIERS:
        raise ValueError(f"Unknown tier '{tier}'.")
    if tier not in available:
        raise ValueError(f"The {config.TIERS[tier]['name']} tier is not loaded on this server.")
    try:
        seed = int(msg.get("seed", 0)) % (2 ** 31)
    except (TypeError, ValueError):
        seed = 0
    post = msg.get("post") if isinstance(msg.get("post"), dict) else {}
    return prompt, tier, seed, post, bool(msg.get("preview", True))


@app.websocket("/ws")
async def ws_generate(ws: WebSocket):
    await ws.accept()
    eng = state["engine"]
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    try:
        msg = json.loads(await ws.receive_text())
        prompt, tier, seed, post, preview = _validate(msg, eng.available)
    except (ValueError, json.JSONDecodeError) as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
        return
    except WebSocketDisconnect:
        return

    def emit(m):
        loop.call_soon_threadsafe(queue.put_nowait, m)

    async def pump():
        while True:
            m = await queue.get()
            if m is None:
                return
            await ws.send_json(m)

    pumper = asyncio.create_task(pump())
    if state["gpu_lock"].locked():
        await ws.send_json({"type": "queued", "message": "Another image is being generated; you are next."})

    result, error = None, None
    tried = []
    async with state["gpu_lock"]:
        order = [tier] + [k for k in config.FALLBACK_ORDER
                          if k in eng.available and config.FALLBACK_ORDER.index(k) > config.FALLBACK_ORDER.index(tier)] \
            if tier in config.FALLBACK_ORDER else [tier, "lite"]
        for k in order:
            tried.append(k)
            try:
                result = await asyncio.to_thread(eng.generate, prompt, seed, k, emit, preview, post)
                break
            except Exception as e:  # torch.cuda.OutOfMemoryError lands here too
                oom = "out of memory" in str(e).lower()
                log.exception("generation failed on %s", k)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                if not oom:
                    error = str(e)
                    break
                emit({"type": "fallback", "from": k, "message": f"{config.TIERS[k]['name']} ran out of memory; retrying on a lighter tier."})
    emit(None)
    await pumper

    try:
        if result:
            await ws.send_json({
                "type": "result",
                "image": to_data_url(result["raw"]),
                "processed": to_data_url(result["processed"], quality=94),
                "metrics": result["metrics"], "qa": result["qa"],
                "detections": result["detections"], "checks": result["checks"],
                "tier_used": result["tier_used"], "tiers_tried": tried,
            })
        else:
            await ws.send_json({"type": "error", "message": error or "Every tier ran out of memory.",
                                "fallback_tier": "lite"})
        await ws.close()
    except (WebSocketDisconnect, RuntimeError):
        pass
