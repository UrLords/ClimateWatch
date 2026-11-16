"""Synthetic CCTV scene with depth-dependent haze (Koschmieder: I = J*t + A*(1-t), t = exp(-beta*d)).
Gives known sky ground truth. CAVEAT: the haze index shares this physical model, so passing here proves
internal consistency and robustness to camera factors, NOT real-world accuracy (needs real data + station calibration)."""

from __future__ import annotations
import cv2
import numpy as np

CLEAR_SKY = np.array([0.92, 0.72, 0.45], np.float32)
OVERCAST = np.array([0.72, 0.72, 0.72], np.float32)
GRAY_A = np.array([0.80, 0.80, 0.80], np.float32)
SMOKE_A = np.array([0.45, 0.68, 0.85], np.float32)


def make_scene(
    beta=2e-4,
    A=GRAY_A,
    sky0=CLEAR_SKY,
    sky_frac=0.35,
    seed=0,
    size=(480, 640),
    sky_dist=4000.0,
):
    H, W = size
    rng = np.random.default_rng(seed)
    hz = int(H * sky_frac)
    J = np.zeros((H, W, 3), np.float32)
    d = np.full((H, W), np.inf, np.float32)
    sky = np.zeros((H, W), bool)
    sky[:hz] = True
    ys = np.arange(hz, H)
    d[hz:] = (15 + 1985 * (1 - (ys - hz) / max(1, H - hz - 1)) ** 2)[:, None]
    tex = cv2.resize(
        rng.uniform(0.08, 0.6, (H // 8, W // 8, 3)).astype(np.float32),
        (W, H),
        interpolation=cv2.INTER_NEAREST,
    )
    J[hz:] = tex[hz:]
    for _ in range(14):
        bd = float(rng.uniform(150, 1500))
        x = int(rng.integers(0, W - 40))
        bw = int(rng.integers(40, 120))
        top = int(rng.integers(max(0, hz - 140), hz - 5))
        bot = hz + int(rng.integers(5, 40))
        J[top:bot, x : x + bw] = rng.uniform(0.15, 0.5, 3).astype(np.float32)
        d[top:bot, x : x + bw] = bd
        sky[top:bot, x : x + bw] = False
    t = np.exp(-beta * np.where(np.isinf(d), sky_dist, d))[..., None]
    J[sky] = sky0
    img = J * t + A * (1 - t)
    return np.clip(img * 255, 0, 255).astype(np.uint8), sky


def jpeg(img, q):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)


def gain(img, g):
    return np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)


def wb_shift(img, b=1.0, g=1.0, r=1.0):
    return np.clip(
        img.astype(np.float32) * np.array([b, g, r], np.float32), 0, 255
    ).astype(np.uint8)


def noise(img, sigma, seed=0):
    n = np.random.default_rng(seed).normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0
