# Design document — Darkroom (P3: Resource-Efficient Text-to-Image Generation for Mobile Devices)

> If the organisers gave you a design-doc template, keep its headings and move these sections under them. Every heading the template asks for must exist.

**Team:** _<Team ID>_ · **Members:** _<names + USNs + who owns which part>_

## 1. Problem

Generate semantically accurate images from complex natural-language prompts on resource-constrained mobile devices. The baseline (Stable Diffusion 1.5 at 50 steps in FP32) needs about 4 GB of weights and several seconds per image on a datacentre GPU. Both are far outside a phone's budget.

## 2. Approach in one paragraph

We keep the SD 1.5 latent-diffusion architecture and attack the cost in five places. We cut the number of denoising steps through consistency distillation (LCM-LoRA, fused into the weights). We reduce numeric precision (FP16, plus INT8 weight-only quantisation of the UNet). We replace the VAE decoder with a distilled tiny decoder (TAESD). We generate at a lower resolution and recover detail with classical upscaling and edge-aware sharpening. We adapt the cost to the device by choosing among four tiers. A classical CV stage corrects colour and contrast and detects artifacts. An open-vocabulary detector checks that the image actually contains what the prompt asked for.

## 3. Models and meaningful adaptation

| Component | Model | Why |
|---|---|---|
| Text encoder | CLIP ViT-L/14 text tower (from SD 1.5) | Prompt encoding; shared by every tier |
| Denoiser | SD 1.5 UNet + **LCM-LoRA fused** | 4–8 steps instead of 50; no classifier-free guidance pass needed (guidance 1.0), which halves the UNet cost per step |
| Decoder | Full VAE (Quality) / **TAESD** (Balanced, Lite) | TAESD has ~1M decoder params vs ~50M, so decode cost drops sharply |
| Adaptation | **Our LoRA** (`eval/train_lora.py`) trained on a COCO train split that is disjoint from the evaluation split, then fused together with LCM-LoRA | Moves the output distribution toward natural photos; measured as the change in FID/CLIP with vs without it |
| Verifier | OWL-ViT B/32 | Finds each prompt object by text query, which enables attribute and relation checks |
| Scorer | CLIP ViT-B/32 | Text-image cosine similarity (CLIP score) |

**Status:** the training script and the disjoint train split are ready (`eval/train_lora.py`, `download_coco.py --offset 5000`). Within the 24-hour window we did not run the LoRA training, so all reported results use the LCM-LoRA–adapted model without the custom LoRA. Our adaptation therefore consists of the fused LCM-LoRA, INT8 quantisation, the TAESD decoder swap and the tier-specific schedules, each measured in the ablation.

## 4. Optimisation strategies (the brief asks for at least two; we use five)

1. **Reduced sampling steps:** 50 → 8 / 4 / 3 with LCM-LoRA.
2. **Knowledge distillation:** LCM-LoRA is a consistency-distilled student of SD 1.5, and TAESD is a distilled VAE decoder.
3. **Reduced precision:** FP32 → FP16; the Lite tier's UNet weights are INT8 (optimum-quanto). On GPU, INT8 weight-only quantisation mainly reduces **size and memory**, not latency. The dequantisation overhead can even make it slightly slower. On mobile NPUs with native INT8 kernels, it is also faster. We report exactly what we measured.
4. **Latent-space optimisation:** latents are decoded by TAESD, and previews are decoded from latents without the full VAE.
5. **Lightweight resolution strategy:** Lite generates at 384 px (0.56× the latent area of 512 px) and is upscaled with Lanczos interpolation plus edge-aware sharpening.

## 5. Classical computer vision (`server/darkroom/postprocess.py`)

| Requirement | Implementation |
|---|---|
| Resizing / resolution optimisation | Lanczos-4 upscaling from the tier's native resolution to 512 px |
| Colour correction | Gray-world white balance with clamped per-channel gains (0.85–1.18) |
| Enhancement | CLAHE on the L channel in LAB space (clip 2.0, 8×8 tiles), so colour is untouched |
| Edge-aware post-processing | Unsharp mask whose detail layer comes from a bilateral filter, applied only where a dilated Canny edge map is on, so flat regions do not get noise amplified |
| Image quality assessment | Laplacian variance (sharpness), percentage of clipped pixels, Hasler–Süsstrunk colourfulness |
| Artifact detection and correction | FFT energy at the 8-px latent-grid frequency detects decoder checkerboarding and triggers a light Gaussian; more than 2% clipped pixels triggers tone compression |

## 6. Adaptive generation and fallback

- **Client side:** device class → memory allowance (budget 1.8 GB, mid 2.6 GB, flagship 4.2 GB). **Adaptive** picks the highest-quality tier whose *measured* peak memory fits: Lite 1.28 GB on budget phones, Balanced 2.17 GB on mid-range, Quality 2.81 GB on flagships (peaks measured per tier in isolation on a T4). A forced tier that does not fit is downgraded and the UI says why.
- **Server side:** CUDA out-of-memory errors are caught; the server frees the cache and retries on the next lighter tier, then reports `tier_used`.
- **Semantic retry:** if the verifier finds a missing object or a wrong relation, the client regenerates once with a new seed.

## 7. Evaluation protocol

- **Data:** MS-COCO val2017. There is a deterministic eval subset (`download_coco.py --n 300`) and a disjoint training subset (`--offset 5000`).
- **Metrics:** CLIP similarity; FID and KID against the real images (KID because FID is biased at a few hundred samples); generation latency (mean, p50, p95), measured after warm-up with CUDA synchronised and previews off; peak GPU memory (`torch.cuda.max_memory_allocated`); model size (bytes of UNet + text encoder + decoder weights, INT8 counted at 1 byte); quality-vs-cost Pareto plot.
- **Compositional prompts:** 30 prompts with multiple objects, colours and spatial relations (`eval/prompts_compositional.txt`), scored by object recall, relation accuracy and the share of prompts fully satisfied.
- Results: `eval/results/eval_P3_<TEAM>.md`.

## 8. Ablation study

`evaluate.py` changes one factor at a time:
- Steps (1, 2, 4, 8)
- Decoder (full VAE vs TAESD at 4 steps)
- Precision (FP16 vs INT8 at the same steps and resolution)
- Resolution (384 vs 512 for Lite)
- Post-processing (off vs on)

Measured on a T4 over 20 COCO captions per configuration (full table: `eval/results/eval_P3_T03.md` §8).

| Factor | CLIP | Generation latency | Other |
|---|---|---|---|
| LCM 1 / 2 / 4 / 8 steps | 0.277 / 0.293 / 0.297 / 0.293 | 295 / 389 / 575 / 957 ms | sharpness falls as steps rise (5742 → 3393): fewer steps leave more high-frequency noise |
| Full VAE vs TAESD (4 steps) | 0.297 vs 0.296 | 577 vs 400 ms | — |
| FP16 vs INT8 UNet (4 steps, 512 px) | 0.296 vs 0.294 | 400 vs 537 ms | UNet weights about 2× smaller |
| Lite 512 px vs 384 px | 0.299 vs 0.293 | 416 vs 351 ms | sharpness 3747 vs 1510 |
| Post-processing off vs on | 0.297 vs 0.296 | 404 vs 403 ms (generation) | clipped pixels 1.31% → 0.05%; sharpness 1333 → 3538 |

- **Steps.** Quality saturates at 4 steps. One step loses 0.02 CLIP; going from 4 to 8 steps adds 66% latency for no measurable gain. Balanced therefore uses 4 steps, and Quality's 8 steps is kept only for its slightly better texture on flagship devices.
- **Decoder.** TAESD removes 31% of generation time with no CLIP change. It is the cheapest optimisation in the system.
- **Precision.** INT8 weight-only quantisation keeps alignment (−0.002 CLIP) and halves UNet weight size, but is *slower* on a T4. quanto dequantises weights on the fly, and the T4 has no fast path for INT8 weights × FP16 activations. On a phone NPU with native INT8 matmuls this trade-off reverses. We report the GPU result as measured.
- **Resolution.** 384 px saves 16% latency at a small alignment cost; classical upscaling recovers size but not fine detail (sharpness 1510 vs 3747), which is the visible quality cost of Lite.
- **Post-processing.** The classical chain does not change semantic alignment (CLIP is computed on content), but it removes almost all clipped highlights and shadows and restores edge contrast lost by few-step sampling and TAESD.

## 9. Failure analysis

The verifier ran on 30 compositional prompts per tier. Images are in `eval/results/failures/`; the full list is in the report §9. Overall, object recall is 0.60–0.75 and relation accuracy is 0.19–0.31 across tiers.

1. **Object absorption: one subject is duplicated, the other dropped.** *"a black cat next to a white dog"* produced **two black cats and no dog**; the "white" ended up on one cat's chest (`failures/02_baseline.jpg`). *Cause:* in SD 1.5, cross-attention lets the first and most frequent noun dominate the layout, so the second subject is absorbed into it. This happens even at 50 steps, so it is a property of the base model, not of our compression.
2. **Attribute leakage between objects.** *"a blue sphere to the left of a red cube"* produced **a single cube with one red face and one blue face** and no sphere (`failures/00_baseline.jpg`). *"a green tree to the left of a yellow house"* produced **a green house with yellow shutters** (`failures/07_baseline.jpg`). *Cause:* colour words bind to whichever object's tokens they share attention with. The colours appear in the image but on the wrong object, so the verifier correctly fails the object and the relation.
3. **Second object omitted, and counting.** *"two red apples beside a green bottle"* produced **one apple and no bottle** (`failures/03_baseline.jpg`); *"a green cup on top of a brown book"* produced **only the cup** (`failures/01_baseline.jpg`). Latent diffusion at this scale handles neither numbers nor "on top of" stacking reliably.
4. **Verifier false negative, and an ignored scene word.** *"a yellow auto rickshaw in front of a white temple at night"* shows the rickshaw in front of clearly visible white temples, yet the verifier reported "temple not found" (`failures/04_baseline.jpg`). OWL-ViT's confidence for this ornate, non-Western temple fell below the threshold. The reported consistency is therefore a **conservative lower bound**, and the detector itself has a cultural bias. Separately, the image is in daylight: "at night" was ignored, which is a scene-level failure the verifier does not check.

Across tiers, relation accuracy is low for every model (0.31 baseline, 0.19 for the few-step tiers), so spatial failures (2) come mainly from SD 1.5's training data. Fewer steps make it worse because layout is fixed in the first denoising steps.

**Mitigation in the system:** the client regenerates once with a new seed when a check fails. This often fixes cases 1 and 3, which are seed-dependent, but rarely case 2, which comes from the model's training data.

## 10. Limitations

- The deployed system runs on a cloud GPU and *simulates* the phone budget from measured memory. We have not run on-device inference on an actual phone NPU.
- FID on a few hundred images is noisy and biased; compare tiers against each other only.
- Only INT8 weight quantisation is used (activations stay in FP16), so GPU latency gains from INT8 are small.
- The verifier only understands the relations listed in `prompts.py`.

## 11. Ethics and responsible use

- SD 1.5 inherits biases from LAION (gender and ethnicity stereotypes for occupations, a Western default for "a house" or "a wedding"). We did not mitigate this.
- The safety checker is disabled for latency. The public demo should not be left open unattended. State this in the demo.
- Generated images can be mistaken for photographs; the app labels them as generated.
- Model licences: SD 1.5 is under CreativeML Open RAIL-M; LCM-LoRA and TAESD are under their repositories' licences. COCO images are used for evaluation only.
