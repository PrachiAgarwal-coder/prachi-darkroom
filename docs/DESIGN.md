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

_Fill in after training: steps, final train/val loss, and the eval delta with vs without the LoRA._

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

- **Client side:** device class → memory allowance (budget 1.8 GB, mid 2.6 GB, flagship 4.2 GB). **Adaptive** picks the highest-quality tier whose *measured* peak memory fits. A forced tier that does not fit is downgraded and the UI says why.
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

_Paste the table and write 2–3 sentences per factor on what it cost and what it saved._

## 9. Failure analysis

_After running the evaluation, pick at least three images from `eval/results/failures/` and explain each cause. Typical causes to check:_

1. **Attribute leakage** ("a red cube and a blue sphere" → both red). CLIP's text encoder pools colour words weakly per noun. Fewer steps make it worse because there is less time to separate the objects.
2. **Spatial relations ignored** ("left of" swapped). SD 1.5 was trained mostly on captions that rarely state positions, so the model has a weak spatial prior.
3. **Counting** ("three bananas" → two or five). This is a known limitation of latent diffusion at this scale.
4. **Verifier false negatives:** OWL-ViT misses small or stylised objects. The image may be correct while the check fails.

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
