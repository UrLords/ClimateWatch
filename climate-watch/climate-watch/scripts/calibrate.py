"""Fit haze_index against a reference station. CSV columns: timestamp,haze_index,pm25
  python scripts/calibrate.py data/camX_vs_station.csv out/camX_calibration.json
Report Spearman (+bootstrap 95% CI) honestly. Need >= ~1 week of daytime data per camera."""

import csv, json, sys
import numpy as np


def rank(x):
    o = np.argsort(x)
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


rows = list(csv.DictReader(open(sys.argv[1])))
h = np.array([float(r["haze_index"]) for r in rows])
p = np.array([float(r["pm25"]) for r in rows])
rho = spearman(h, p)
rng = np.random.default_rng(0)
boots = [
    spearman(h[i], p[i]) for i in (rng.integers(0, len(h), len(h)) for _ in range(1000))
]
a, b = np.polyfit(h, p, 1)
res = {
    "n": len(h),
    "spearman": round(rho, 3),
    "ci95": [
        round(float(np.percentile(boots, 2.5)), 3),
        round(float(np.percentile(boots, 97.5)), 3),
    ],
    "r2_linear": round(float(np.corrcoef(h, p)[0, 1] ** 2), 3),
    "pm25_est = a*haze + b": {"a": float(a), "b": float(b)},
}
json.dump(res, open(sys.argv[2], "w"), indent=2)
print(res)
