"""Sky mask. Tries YOLO26-sem (Cityscapes, sky = class 10); falls back to a heuristic.

VERIFY: the exact result attribute that holds the semantic map in ultralytics>=8.4
was not confirmed. `_parse` tries several access patterns; check against the docs
(https://docs.ultralytics.com/tasks/semantic) and simplify once confirmed.
"""

from __future__ import annotations
import os
import cv2
import numpy as np

SKY_CLASS_ID = 10


def heuristic_sky(bgr: np.ndarray) -> np.ndarray:
    """Colour-agnostic (works for blue, grey overcast, yellow smoke): sky = smooth, bright-ish region
    connected to the top edge. Assumes a fixed, upright camera."""
    h, w = bgr.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (7, 7), 0).astype(
        np.float32
    )
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    grad = cv2.blur(np.hypot(gx, gy), (9, 9))
    cand = (grad < 6.0) & (g > 0.25 * 255)
    cand[int(h * 0.7) :] = False
    cand = cv2.morphologyEx(
        cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
    )
    n, lab = cv2.connectedComponents(cand)
    keep = np.zeros_like(cand)
    for i in range(1, n):
        comp = lab == i
        if comp[0].any() and comp.sum() > 0.01 * h * w:
            keep |= comp.astype(np.uint8)
    return keep.astype(bool)


class SkySegmenter:
    def __init__(self, weights: str | None = None):
        self.model = None
        weights = weights or os.getenv("SKY_WEIGHTS", "yolo26n-sem.pt")
        if os.getenv("DISABLE_SEM") == "1":
            return
        try:
            from ultralytics import YOLO

            self.model = YOLO(weights)
        except Exception as e:
            print(f"[sky] model unavailable ({e}); using heuristic")

    def _parse(self, res, shape) -> np.ndarray | None:
        h, w = shape
        for name in ("sem_mask", "semantic", "seg_map", "mask_map"):
            m = getattr(res, name, None)
            if m is not None:
                m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
                return cv2.resize(
                    (m == SKY_CLASS_ID).astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
        if (
            getattr(res, "masks", None) is not None
            and getattr(res, "boxes", None) is not None
        ):
            out = np.zeros((h, w), bool)
            for m, c in zip(res.masks.data.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                if int(c) == SKY_CLASS_ID:
                    out |= cv2.resize(
                        m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
            return out
        return None

    def mask(self, bgr: np.ndarray) -> tuple[np.ndarray, str]:
        if self.model is not None:
            try:
                res = self.model.predict(
                    bgr, imgsz=int(os.getenv("SKY_IMGSZ", "512")), verbose=False
                )[0]
                m = self._parse(res, bgr.shape[:2])
                if m is not None:
                    return m, "yolo26-sem"
            except Exception as e:
                print(f"[sky] inference failed ({e}); using heuristic")
        return heuristic_sky(bgr), "heuristic"
