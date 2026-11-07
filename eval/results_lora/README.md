# Custom LoRA adaptation — results

Team T03 · Balanced tier · same 100 COCO val2017 captions and seeds as the main evaluation · Tesla T4 (Kaggle)

**Training:** rank-16 LoRA on the UNet attention projections, 400 optimiser steps (gradient accumulation 4), learning rate 1e-4, on 540 COCO **train2017** images with 60 held out for validation. train2017 never overlaps the val2017 evaluation images. At serving time it is fused together with LCM-LoRA at weight 0.8 (`DARKROOM_CUSTOM_LORA`).

**Validation loss (noise-prediction MSE):** 0.1625 → 0.1625 → 0.1622 → 0.1622 at steps 100 / 200 / 300 / 400.

| Metric | Balanced, without our LoRA | Balanced, with our LoRA |
|---|---|---|
| CLIP score ↑ | 0.3008 | 0.3020 |
| KID ×1000 ↓ | 7.61 | 7.93 |
| FID ↓ (N=100, indicative only) | 212.0 | 212.2 |
| Compositional prompts passed ↑ | 27% (8/30) | 30% (9/30) |
| Mean generation time | 402 ms | 412 ms |
| Peak memory | 2168 MB | 2138 MB |

## Reading

The LoRA made **no measurable difference**. CLIP moved by +0.001, KID by +0.3 ×10⁻³ and the compositional pass rate by one prompt out of 30. All three are within run-to-run noise at N = 100. The flat validation loss says the same: 400 steps on 540 images did not move the denoiser's error on held-out COCO images.

This is the expected outcome for a short run. SD 1.5 was pre-trained on web photos that already resemble COCO, so a small LoRA has little distribution gap to close. A shift of this size would need many more images and steps than a 24-hour window allows.

**Conclusion:** the efficiency adaptations (fused LCM-LoRA, INT8, TAESD, per-tier schedules) carry the result. The fine-tuning pipeline is complete and measured, but it adds no quality, so the deployed tiers do not use it.

The source log is the Kaggle run of `eval/train_lora.py` followed by `eval/evaluate.py --tiers balanced` with `DARKROOM_CUSTOM_LORA` set.
