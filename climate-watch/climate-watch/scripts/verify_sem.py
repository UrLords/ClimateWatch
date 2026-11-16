"""Inspect YOLO26 semantic segmentation output for a supplied camera frame."""

import sys
import cv2
import numpy as np
from ultralytics import YOLO

img = cv2.imread(sys.argv[1])
m = YOLO("yolo26n-sem.pt")
print("model.names:", getattr(m, "names", None))
r = m.predict(img, imgsz=512, verbose=False)[0]
print("result type:", type(r).__name__)
for k, v in vars(r).items():
    shp = getattr(v, "shape", None) or (len(v) if hasattr(v, "__len__") else "")
    print(f"  attr {k}: {type(v).__name__} {shp}")
print("public attrs:", [a for a in dir(r) if not a.startswith("_")])

sys.path.insert(0, ".")
from climatewatch.sky import SkySegmenter, heuristic_sky, SKY_CLASS_ID

seg = SkySegmenter()
mask, src = seg.mask(img)
print(
    f"\nsky source: {src}  sky fraction: {mask.mean():.2f}  (heuristic: {heuristic_sky(img).mean():.2f})"
)
if src != "yolo26-sem":
    print(
        "!! _parse could not read the result: use the attribute list above to fix SkySegmenter._parse"
    )
ov = img.copy()
ov[mask] = (0.5 * ov[mask] + 0.5 * np.array([255, 120, 0])).astype(np.uint8)
cv2.imwrite("sem_check.jpg", ov)
print("wrote sem_check.jpg (blue = predicted sky) -> eyeball it")
