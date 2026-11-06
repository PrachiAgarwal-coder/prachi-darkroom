"""Classical computer vision: resizing, colour correction, edge-aware enhancement,
image quality assessment, and artifact detection + correction.

All functions take/return uint8 RGB numpy arrays (H, W, 3).
"""
import time
import cv2
import numpy as np


# ---------- resizing ----------
def upscale(img, size=512):
    if img.shape[0] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)


# ---------- colour ----------
def gray_world_wb(img, max_gain=1.18, min_gain=0.85):
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(0) + 1e-6
    gains = np.clip(means.mean() / means, min_gain, max_gain)
    return np.clip(f * gains, 0, 255).astype(np.uint8)


def clahe_luminance(img, clip=2.0, tiles=8):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


# ---------- edge-aware sharpening ----------
def edge_aware_sharpen(img, amount=0.6, edge_thresh=40):
    """Unsharp mask whose strength is gated by an edge map, so flat regions
    (sky, skin) do not get noise amplified. The base layer comes from a
    bilateral filter, which is itself edge-preserving."""
    base = cv2.bilateralFilter(img, d=7, sigmaColor=40, sigmaSpace=7)
    detail = img.astype(np.float32) - base.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, edge_thresh, edge_thresh * 2.5)
    mask = cv2.GaussianBlur(cv2.dilate(edges, None), (5, 5), 0).astype(np.float32)[..., None] / 255.0
    out = img.astype(np.float32) + amount * detail * mask
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- quality assessment ----------
def sharpness(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def clipped_pct(img):
    hi = np.all(img >= 254, axis=2)
    lo = np.all(img <= 1, axis=2)
    return float((hi | lo).mean() * 100)


def colorfulness(img):
    r, g, b = [img[..., i].astype(np.float32) for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def checkerboard_score(img):
    """Energy at the Nyquist frequency of the 8x latent grid. Decoders (especially
    tiny ones) can leave a periodic 8-px pattern; this spikes when that happens."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean())))
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    fy, fx = h // 8, w // 8
    peaks = [spec[cy + dy, cx + dx] for dy, dx in [(fy, 0), (-fy, 0), (0, fx), (0, -fx)]]
    ring = spec[cy - fy - 3:cy + fy + 4, cx - fx - 3:cx + fx + 4].mean() + 1e-6
    return float(np.mean(peaks) / ring)


def assess(img):
    return {"sharpness": sharpness(img), "clipped_pct": clipped_pct(img),
            "colorfulness": colorfulness(img), "checkerboard": checkerboard_score(img)}


# ---------- artifact correction ----------
def correct_artifacts(img, qa):
    fixes = []
    if qa["checkerboard"] > 6.0:          # periodic decoder pattern -> notch it out with a small blur
        img = cv2.GaussianBlur(img, (3, 3), 0.8)
        fixes.append("checkerboard smoothed")
    if qa["clipped_pct"] > 2.0:           # blown highlights / crushed shadows -> gentle tone compression
        f = img.astype(np.float32) / 255.0
        img = np.clip((0.04 + 0.92 * f) * 255, 0, 255).astype(np.uint8)
        fixes.append("tone range compressed")
    return img, fixes


# ---------- full chain ----------
def run(img, contrast=True, wb=True, sharp=True, up=True, size=512):
    t0 = time.perf_counter()
    if up:
        img = upscale(img, size)
    if wb:
        img = gray_world_wb(img)
    if contrast:
        img = clahe_luminance(img)
    if sharp:
        img = edge_aware_sharpen(img)
    post_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    qa = assess(img)
    img, fixes = correct_artifacts(img, qa)
    qa_ms = (time.perf_counter() - t1) * 1000
    qa["fixes"] = fixes
    return img, qa, post_ms, qa_ms
