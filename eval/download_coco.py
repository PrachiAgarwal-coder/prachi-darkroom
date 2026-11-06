"""Download a deterministic subset of MS-COCO val2017 (captions + images) for evaluation.

    python eval/download_coco.py --n 300 --out data/coco

For the LoRA training split (disjoint from evaluation):
    python eval/download_coco.py --n 1000 --offset 5000 --out data/coco_train

Writes data/coco/captions.json  -> [{"image": "000000xxxxx.jpg", "caption": "..."}]
and   data/coco/images/*.jpg   (center-cropped to 512x512 for FID).
"""
import argparse
import io
import json
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ANN = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def center_512(img):
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((512, 512), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/coco")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many images first; use it to build a training split that never overlaps the eval split")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    cap_file = out / "captions_val2017.json"
    if not cap_file.exists():
        print("Downloading COCO annotations (~240 MB, once)...")
        data = urllib.request.urlopen(ANN).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            cap_file.write_bytes(z.read("annotations/captions_val2017.json"))
    ann = json.loads(cap_file.read_text())
    images = {im["id"]: im for im in ann["images"]}
    first_caption = {}
    for c in ann["annotations"]:
        first_caption.setdefault(c["image_id"], c["caption"].strip())

    ids = sorted(first_caption)
    random.Random(a.seed).shuffle(ids)
    ids = ids[a.offset:a.offset + a.n]

    def fetch(i):
        im = images[i]
        dest = out / "images" / im["file_name"]
        if not dest.exists():
            raw = urllib.request.urlopen(im["coco_url"], timeout=30).read()
            center_512(Image.open(io.BytesIO(raw)).convert("RGB")).save(dest, quality=95)
        return {"image": im["file_name"], "caption": first_caption[i]}

    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(fetch, ids))
    (out / "captions.json").write_text(json.dumps(rows, indent=1))
    print(f"Saved {len(rows)} image-caption pairs to {out}")


if __name__ == "__main__":
    main()
