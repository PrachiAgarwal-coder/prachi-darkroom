"""Meaningful adaptation: fine-tune a LoRA on the SD 1.5 UNet with a training split
that never overlaps the evaluation split, then serve it fused together with LCM-LoRA.

    python eval/download_coco.py --n 1000 --offset 5000 --out data/coco_train
    python eval/train_lora.py --data data/coco_train --steps 1500 --out lora/coco
    # then serve / evaluate with it:
    DARKROOM_CUSTOM_LORA=lora/coco uvicorn darkroom.server:app ...

Takes roughly 40-70 min on a T4 / 15-25 min on an A10G at 512 px, batch 1, grad-accum 4.
The validation loss on a held-out 10% of the training split is printed every 100 steps.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
from darkroom import config  # noqa: E402


def load_split(data_dir, val_frac=0.1, seed=0):
    rows = json.loads((Path(data_dir) / "captions.json").read_text())
    random.Random(seed).shuffle(rows)
    k = max(1, int(len(rows) * val_frac))
    return rows[k:], rows[:k]


def to_tensor(path, res):
    img = Image.open(path).convert("RGB").resize((res, res), Image.BICUBIC)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 127.5 - 1
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="lora/coco")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--accum", type=int, default=4)
    a = ap.parse_args()

    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    dev = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(config.BASE_MODEL, torch_dtype=torch.float32, safety_checker=None)
    vae, te, tok, unet = pipe.vae.to(dev, torch.float16), pipe.text_encoder.to(dev, torch.float16), pipe.tokenizer, pipe.unet.to(dev)
    sched = DDPMScheduler.from_config(pipe.scheduler.config)
    for m in (vae, te, unet):
        m.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=a.rank, lora_alpha=a.rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    unet.enable_gradient_checkpointing()
    params = [p for p in unet.parameters() if p.requires_grad]
    print(f"Trainable LoRA params: {sum(p.numel() for p in params) / 1e6:.2f} M of {sum(p.numel() for p in unet.parameters()) / 1e6:.0f} M")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-2)
    lr_at = lambda s: a.lr * min(1, s / 100) * 0.5 * (1 + math.cos(math.pi * s / a.steps))
    scaler = torch.amp.GradScaler("cuda")

    train, val = load_split(a.data)
    img_dir = Path(a.data) / "images"
    print(f"train {len(train)} / val {len(val)}")

    def loss_on(row, gen=None):
        x = to_tensor(img_dir / row["image"], a.res).unsqueeze(0).to(dev, torch.float16)
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.sample(generator=gen) * vae.config.scaling_factor
            ids = tok(row["caption"], padding="max_length", max_length=tok.model_max_length, truncation=True,
                      return_tensors="pt").input_ids.to(dev)
            emb = te(ids)[0]
        noise = torch.randn(lat.shape, device=dev, generator=gen, dtype=lat.dtype)
        t = torch.randint(0, sched.config.num_train_timesteps, (1,), device=dev, generator=gen)
        noisy = sched.add_noise(lat, noise, t)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = unet(noisy, t, emb).sample
        return F.mse_loss(pred.float(), noise.float())

    step = 0
    unet.train()
    while step < a.steps:
        random.shuffle(train)
        for i, row in enumerate(train):
            loss = loss_on(row) / a.accum
            scaler.scale(loss).backward()
            if (i + 1) % a.accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                unet.eval()
                g = torch.Generator(dev).manual_seed(0)
                with torch.no_grad():
                    vl = np.mean([loss_on(r, g).item() for r in val[:50]])
                unet.train()
                print(f"step {step}/{a.steps}  train {loss.item() * a.accum:.4f}  val {vl:.4f}")
            if step >= a.steps:
                break

    Path(a.out).mkdir(parents=True, exist_ok=True)
    lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(a.out, unet_lora_layers=lora, safe_serialization=True)
    print(f"Saved LoRA to {a.out}")


if __name__ == "__main__":
    main()
