"""2-day feasibility spike. Run on 3-5 frames per candidate camera (day/night/rain).
  python scripts/spike.py data/spike_frames out/spike
Outputs per image: sky-mask overlay + CSV of haze metrics. Eyeball masks first, THEN trust numbers."""

import csv, glob, os, sys
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from climatewatch.haze import compute_haze
from climatewatch.sky import SkySegmenter

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
seg, rows = SkySegmenter(), []
for p in sorted(glob.glob(os.path.join(src, "*.*"))):
    img = cv2.imread(p)
    if img is None:
        continue
    m, how = seg.mask(img)
    r = compute_haze(img, m)
    ov = img.copy()
    ov[m] = (0.5 * ov[m] + 0.5 * np.array([255, 120, 0])).astype(np.uint8)
    cv2.imwrite(os.path.join(dst, "mask_" + os.path.basename(p)), ov)
    rows.append({"file": os.path.basename(p), "sky_src": how, **r.to_dict()})
    print(rows[-1])
if rows:
    with open(os.path.join(dst, "spike.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
