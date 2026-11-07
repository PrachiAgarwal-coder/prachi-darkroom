# Demo script and viva preparation

## 3-minute demo

| Time | Who | Do | Say |
|---|---|---|---|
| 0:00 | Lead | Open the judges' link on a phone; point at the green **Live model · cuda** pill | "This is Stable Diffusion running on our own optimised pipeline, not an image API." |
| 0:20 | Model owner | Studio: pick the rickshaw prompt, press **Generate** | "Watch it form: each frame is a real denoising step, decoded by a tiny 1M-parameter decoder. Four steps instead of fifty." |
| 0:45 | CV owner | Drag the raw-vs-processed slider; toggle **Adaptive contrast** | "Classical CV after generation: white balance, CLAHE, edge-only sharpening. It cut clipped pixels from 1.3% to 0.05% without changing CLIP." |
| 1:10 | Systems owner | Device → **Budget phone**, tier → **Quality** | "Quality needs 2.8 GB; a budget phone allows 1.8. The app steps down to Lite and says why. If the GPU still runs out, the server retries on a lighter tier." |
| 1:35 | Verifier owner | Point at **Does it match the prompt?** and the boxes | "An open-vocabulary detector checks each object and position. A failed check triggers one retry with a new seed." |
| 1:55 | Eval owner | **Benchmarks → Start the race** | "Same prompt and seed. Ours: under a second. The FP32 50-step baseline: about 30 seconds." |
| 2:30 | Lead | Scroll to the chart and the ablation table | "79× faster at 98% of the baseline's CLIP score, 4× less memory, 3.8× smaller. Every number is from `eval_P3_T03.md`." |

**If the live server fails:** say so, switch to the screen recording, then show Benchmarks (it uses the measured numbers) and the README results. Don't pretend demo mode is the model.

## Likely questions, with answers

### Model and optimisation
- **Which model, and why not a smaller architecture?** SD 1.5. It has mature distillation tools (LCM-LoRA, TAESD), and a pruned UNet loses more quality per millisecond saved than step distillation does.
- **What makes it lightweight?** Five independent switches: LCM consistency distillation (50 → 3–8 steps, and no classifier-free guidance, so one UNet pass per step instead of two), FP16, INT8 UNet weights, the TAESD decoder, and 384 px generation for Lite.
- **Why exactly 4 steps for Balanced?** Ablation: CLIP 0.277 / 0.293 / 0.297 / 0.293 at 1 / 2 / 4 / 8 steps. It peaks at 4, and 8 steps is 66% slower for nothing.
- **Is INT8 faster?** Not on a T4: 400 → 537 ms, because quanto dequantises weights on the fly. It halves the UNet and cuts memory; on phone NPUs with native INT8 it is also faster. We report what we measured.
- **What is the "meaningful adaptation"?** We adapted the pretrained model: LCM-LoRA fused into the weights (consistency distillation), INT8 weight quantisation, the TAESD decoder swap and per-tier step/resolution schedules, each measured in the ablation. Our own LoRA fine-tuning script with a disjoint COCO train split is implemented (`eval/train_lora.py`) but was not run within the 24 hours. Say this plainly if asked.

### Classical CV
- **What runs after generation?** Lanczos upscaling, gray-world white balance (gains clamped 0.85–1.18), CLAHE on the L channel in LAB space (so colours don't shift), and an unsharp mask applied only where a Canny edge map is on.
- **How do you detect artifacts?** FFT energy at the 8-pixel latent-grid frequency catches decoder checkerboarding; more than 2% clipped pixels triggers tone compression.
- **Does it help?** Clipped pixels 1.31% → 0.05%, sharpness 1333 → 3538, CLIP unchanged.

### Evaluation
- **Why is the baseline's FID worse than the fast tiers'?** It was scored on 50 images and the others on 100; FID is strongly biased by sample size. We compare with KID instead, where the baseline is best (1.7 vs 7.6–12.7 ×10⁻³), as expected.
- **How was latency measured?** Encode + denoise + decode, after a warm-up, with CUDA synchronised and previews off; mean, p50 and p95 over 100 prompts.
- **How was memory measured?** One process per tier loading only that tier's weights, `torch.cuda.max_memory_allocated()`: 5.11 / 2.81 / 2.17 / 1.28 GB.
- **Weakest result?** Spatial relations: 0.31 accuracy even for the baseline. It is a limit of SD 1.5's training data, not of our compression.

### Adaptivity, fallback and serving
- **How does it pick a tier?** The browser reports device memory, which maps to an allowance (1.8 / 2.6 / 4.2 GB). It picks the best tier whose measured peak fits.
- **What if it still runs out?** The server catches CUDA out-of-memory, frees the cache and retries the next lighter tier; the app shows a notice. Covered by `server/tests/test_server_fallback.py`.
- **Does it run on a phone?** The weights and peaks fit a phone budget (Lite: 1.11 GB of weights, 1.28 GB peak), but inference runs on a cloud GPU for the demo. Say this plainly.

### Interface
- **Why stream previews?** It shows the denoising happening, and it makes a sub-second wait feel instant.
- **Does it work without the server?** Yes. It enters a clearly labelled demo mode and still shows the measured benchmarks.
