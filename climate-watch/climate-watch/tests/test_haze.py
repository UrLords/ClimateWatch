import numpy as np, cv2
from climatewatch.haze import compute_haze
from climatewatch.sky import heuristic_sky


def scene(haze: float):
    h, w = 360, 640
    img = np.zeros((h, w, 3), np.float32)
    img[:150] = (0.9, 0.75, 0.55)
    rng = np.random.default_rng(1)
    img[150:] = rng.uniform(0.05, 0.6, (h - 150, w, 3))
    A = np.array([0.8, 0.8, 0.8], np.float32)
    img = img * (1 - haze) + A * haze
    return (img * 255).astype(np.uint8)


def test_haze_index_monotonic():
    lo, hi = compute_haze(scene(0.0)).haze_index, compute_haze(scene(0.7)).haze_index
    assert hi > lo + 10


def test_sky_heuristic_finds_top():
    m = heuristic_sky(scene(0.1))
    assert m[:100].mean() > 0.5 and m[300:].mean() < 0.05
