# Darkroom — resource-efficient text-to-image for mobile devices (P3)

Type a scene and watch it form from noise in a few steps. A distilled, quantised Stable Diffusion pipeline picks how much compute to spend based on the phone it is running for. Classical CV cleans up the result, and an open-vocabulary detector checks that the image actually contains what you asked for.

- **Live app:** _<paste your published Lovable URL>_
- **Model server:** _<paste your Modal URL>_
- **Evaluation:** [`eval/results/eval_P3_<TEAM>.md`](eval/results/)
- **Design document:** [`docs/DESIGN.md`](docs/DESIGN.md)
- **Architecture:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

| Tier | Denoiser | Steps | Precision | Decoder | Resolution |
|---|---|---|---|---|---|
| Baseline | SD 1.5 | 50 (+CFG) | FP32 | VAE | 512 |
| Quality | SD 1.5 + LCM-LoRA | 8 | FP16 | VAE | 512 |
| Balanced | SD 1.5 + LCM-LoRA | 4 | FP16 | TAESD | 512 |
| Lite | SD 1.5 + LCM-LoRA | 3 | INT8 UNet | TAESD | 384 → 512 |

## Repository layout

```
src/                     React UI (TanStack Start + Tailwind + Framer Motion), built with Lovable
  routes/index.tsx       the Darkroom app
  lib/darkroom.ts        tier selection, server contract (normalizeTiers, runJob), demo scene renderer
server/                  Python model server (FastAPI + diffusers)
  darkroom/config.py     tier definitions + model IDs
  darkroom/engine.py     loads tiers, generates, times stages, measures memory, CLIP score, verifier
  darkroom/postprocess.py classical CV: resize, white balance, CLAHE, edge-aware sharpening, QA, artifact fix
  darkroom/prompts.py    prompt → objects / attributes / spatial relations
  darkroom/server.py     /ws (generation), /health, /tiers, fallback UI at /
  darkroom/mock.py       GPU-free stand-in for UI work and tests
  static/index.html      lightweight fallback UI served by the model server
  tests/                 pytest (runs the real engine on a tiny random model, CPU only)
eval/                    download_coco.py, evaluate.py, train_lora.py, prompts_compositional.txt
deploy/                  modal_app.py, colab_backup.md
docs/                    DESIGN.md, ARCHITECTURE.md
Dockerfile               one-command GPU server
```

## 1. Run the UI

```bash
bun install        # or: npm install
bun run dev        # http://localhost:5173
```

With no server configured, the UI runs a clearly labelled **demo mode** (procedural scenes, not model output). To use the real model, set the server URL:

```bash
cp .env.example .env    # then edit: VITE_DARKROOM_WS=wss://<your-server>/ws
```

You can also click the connection pill in the header and paste the URL there.

## 2. Run the model server (needs an NVIDIA GPU with 8 GB+; 16 GB to also load the FP32 baseline)

```bash
cd server
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn darkroom.server:app --host 0.0.0.0 --port 8000
```

- The first start downloads about 5 GB of weights and warms up every tier (3–6 min).
- Point the UI at `ws://localhost:8000/ws`.
- No GPU? `DARKROOM_MOCK=1 uvicorn darkroom.server:app --port 8000` runs placeholder images for UI work. Never report numbers from mock mode.
- Low on GPU memory? `DARKROOM_BASELINE=0` skips the FP32 baseline.

**One command with Docker:**
```bash
docker build -t darkroom . && docker run --gpus all -p 8000:8000 -v hf:/root/.cache/huggingface darkroom
```

## 3. Deploy

**Model server on Modal (one HTTPS URL):**
```bash
pip install modal
modal setup                          # log in once
modal deploy deploy/modal_app.py     # from the repo root; prints https://<...>.modal.run
```

- `min_containers=1` keeps one A10G warm so judges never hit a cold start.
- Set it to `0` after the event.
- Backup on a free Colab/Kaggle GPU: [`deploy/colab_backup.md`](deploy/colab_backup.md).

**UI:**
1. In Lovable, set `VITE_DARKROOM_WS=wss://<...>.modal.run/ws`.
2. Publish.

## 4. Reproduce the evaluation

```bash
cd server
python ../eval/download_coco.py --n 300 --out ../data/coco
python ../eval/evaluate.py --team T01 --usns "USN1,USN2,USN3,USN4" --coco ../data/coco
```

This writes the following to `eval/results/`:
- `eval_P3_T01.md` (the first line holds the USNs)
- `results.json`
- `quality_vs_cost.png`
- the ablation table and the `failures/` images

Redeploy the server afterwards; `/tiers` then serves the measured numbers and the UI shows them. A full run takes about 60–90 min on an A10G. For a quick pass, use `--n 100 --n-baseline 50 --n-ablation 20`.

**Optional, meaningful adaptation:** train a LoRA on a COCO split that is disjoint from the eval split:
```bash
python ../eval/download_coco.py --n 1000 --offset 5000 --out ../data/coco_train
python ../eval/train_lora.py --data ../data/coco_train --steps 1500 --out ../lora/coco
DARKROOM_CUSTOM_LORA=../lora/coco python ../eval/evaluate.py --team T01 --usns "..." --out ../eval/results_lora
```

## 5. Tests

```bash
cd server && python -m pytest -q tests     # engine, parser, OpenCV chain
bun run test                               # UI
```

## Limitations and ethics

See [`docs/DESIGN.md`](docs/DESIGN.md) §10–11. In short:
- Inference runs on a cloud GPU; the phone memory budget is simulated from measured numbers.
- FID on a few hundred images is noisy.
- The base model's dataset biases are inherited.
- The safety checker is disabled for latency.

No API keys or secrets are needed or committed. The UI was scaffolded with [Lovable](https://lovable.dev) and edited by hand; the model pipeline does not call any third-party image API.
