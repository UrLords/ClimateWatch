import cv2
import numpy as np


def blur_boxes(
    bgr: np.ndarray, boxes: list[list[float]], pad: float = 0.15
) -> np.ndarray:
    out = bgr.copy()
    H, W = out.shape[:2]
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        x1, x2 = int(max(0, x1 - pad * bw)), int(min(W, x2 + pad * bw))
        y1, y2 = int(max(0, y1 - pad * bh)), int(min(H, y2 + pad * bh))
        if x2 > x1 and y2 > y1:
            k = max(15, ((x2 - x1) // 3) | 1)
            out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (k, k), 0)
    return out
