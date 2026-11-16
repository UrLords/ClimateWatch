"""Haze / visibility index from a fixed CCTV frame.

NOT a PM2.5 estimator. It is an uncalibrated 0-100 index until scripts/calibrate.py
has fit it against a nearby reference station (OpenAQ / ISPU / AirGradient).

Method: segment sky first, then
  * Dark Channel Prior transmission on NON-sky pixels (DCP breaks on bright sky),
  * RMS contrast of non-sky pixels (contrast loss with distance/haze),
  * sky colour is reported separately (`sky_yellowness`, flag `yellow_sky_cue`), never mixed into the index.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import cv2
import numpy as np

OMEGA = 0.95
PATCH = 15


@dataclass
class HazeResult:
    haze_index: float
    confidence: float
    transmission_mean: float
    contrast: float
    sky_yellowness: float | None
    sky_fraction: float
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in asdict(self).items()
        }


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Accept gray / BGRA / float / uint16 and return uint8 BGR."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (
            np.clip(
                img
                / (
                    65535.0
                    if img.dtype == np.uint16
                    else (1.0 if img.max() <= 1.0 else 255.0)
                ),
                0,
                1,
            )
            * 255
        ).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def dark_channel(img: np.ndarray, patch: int = PATCH) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(np.min(img, axis=2), k)


def estimate_atmosphere(
    img: np.ndarray, dark: np.ndarray, top: float = 0.001
) -> np.ndarray:
    n = max(1, int(dark.size * top))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    a = img.reshape(-1, 3)[idx].mean(axis=0)
    return np.clip(a, 0.1, 1.0)


def compute_haze(
    bgr: np.ndarray, sky_mask: np.ndarray | None = None, width: int = 640
) -> HazeResult:
    bgr = ensure_bgr(bgr)
    h0, w0 = bgr.shape[:2]
    scale = width / w0
    small = cv2.resize(
        bgr, (width, max(1, int(h0 * scale))), interpolation=cv2.INTER_AREA
    )

    img = cv2.GaussianBlur(small, (5, 5), 1.2).astype(np.float32) / 255.0
    H, W = img.shape[:2]

    sky = None
    if sky_mask is not None:
        sky = cv2.resize(
            sky_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    sky_frac = float(sky.mean()) if sky is not None else 0.0
    ground = ~sky if sky is not None else np.ones((H, W), bool)

    flags: list[str] = []
    conf = 1.0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_dn = (
        cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(
            np.float32
        )
        / 255.0
    )

    if ground.mean() < 0.10:
        flags.append("little_non_sky_area")
        conf -= 0.3
        ground = np.ones((H, W), bool)

    dark = dark_channel(img)
    A = estimate_atmosphere(img, dark)
    t = 1.0 - OMEGA * dark_channel(img / A)
    t_mean = float(np.clip(t[ground].mean(), 0.0, 1.0))
    dc_score = 1.0 - t_mean

    contrast = float(gray_dn[ground].std() / (gray_dn[ground].mean() + 1e-3))
    contrast_score = float(np.clip(1.0 - contrast / 0.9, 0.0, 1.0))

    yellow = None
    if sky is not None and sky_frac > 0.05:
        b_mean = (
            float(cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 2][sky].mean()) - 128.0
        )
        yellow = float(np.clip(b_mean / 30.0, 0.0, 1.0))
    else:
        flags.append("no_usable_sky")
        conf -= 0.2
    idx = 0.55 * dc_score + 0.45 * contrast_score

    b, g_, r = (small[..., i].astype(np.float32) for i in range(3))
    if float(np.abs(b - g_).mean() + np.abs(g_ - r).mean()) < 1.5:
        flags.append("ir_mono_mode")
        conf -= 0.5
    mean_v = float(gray.mean())
    if mean_v < 0.15:
        flags.append("night_or_dark")
        conf -= 0.6
    elif mean_v > 0.90:
        flags.append("overexposed")
        conf -= 0.4
    if float((gray >= 250 / 255.0).mean()) > 0.15:
        flags.append("saturated_clipping")
        conf -= 0.3
    if cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F).var() < 20:
        flags.append("blurry_or_lens_fog")
        conf -= 0.3

    if yellow is not None and yellow >= 0.6:
        flags.append("yellow_sky_cue")
    return HazeResult(
        haze_index=float(np.clip(idx, 0, 1) * 100),
        confidence=float(np.clip(conf, 0.0, 1.0)),
        transmission_mean=t_mean,
        contrast=contrast,
        sky_yellowness=yellow,
        sky_fraction=sky_frac,
        flags=flags,
    )


def relative_haze(
    raw: float, baseline: float | None, span: float = 35.0
) -> float | None:
    """Haze above this camera's own clear-day baseline, 0-100. `span` = raw points that count as full haze.
    Removes camera-specific offsets (scene content, WB, lens). Replace by calibrate.py fit when station data exist."""
    if baseline is None:
        return None
    return float(np.clip((raw - baseline) / span, 0.0, 1.0) * 100)
