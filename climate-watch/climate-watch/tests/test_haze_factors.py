import time
import cv2, numpy as np, pytest
from synth import *
from climatewatch.haze import compute_haze, relative_haze, ensure_bgr
from climatewatch.sky import heuristic_sky

BETAS = [5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 2e-3, 3e-3]
H = lambda img, m=None: compute_haze(img, m)


def idx(beta, **kw):
    img, sky = make_scene(beta, **kw)
    return H(img, sky).haze_index


def test_monotonic_and_spread_true_mask():
    v = [idx(b) for b in BETAS]
    assert all(b > a for a, b in zip(v, v[1:])), v
    assert v[-1] - v[0] >= 20, v


def test_monotonic_without_sky():
    v = []
    for b in BETAS:
        img, sky = make_scene(b)
        v.append(H(img[int(480 * 0.4) :], None).haze_index)
    assert all(b > a for a, b in zip(v, v[1:])), v


def test_no_sky_flagged():
    img, _ = make_scene(2e-4)
    r = H(img[int(480 * 0.4) :], None)
    assert "no_usable_sky" in r.flags and r.confidence <= 0.8


@pytest.mark.parametrize("g", [0.5, 0.7, 1.3])
def test_exposure_invariance(g):
    img, sky = make_scene(1e-3)
    assert abs(H(gain(img, g), sky).haze_index - H(img, sky).haze_index) <= 3.0


def test_saturation_flagged():
    img, sky = make_scene(1e-3)
    r = H(gain(img, 2.6), sky)
    assert "saturated_clipping" in r.flags or "overexposed" in r.flags


@pytest.mark.parametrize("sigma", [5, 10, 20])
def test_noise_robust(sigma):
    img, sky = make_scene(1e-3)
    assert abs(H(noise(img, sigma), sky).haze_index - H(img, sky).haze_index) <= 4.0


@pytest.mark.parametrize("q", [90, 50, 20])
def test_jpeg_robust(q):
    img, sky = make_scene(1e-3)
    assert abs(H(jpeg(img, q), sky).haze_index - H(img, sky).haze_index) <= 3.0


@pytest.mark.parametrize("w", [320, 1280, 1920])
def test_resolution_invariant(w):
    img, _ = make_scene(1e-3)
    ref = H(img).haze_index
    assert abs(H(cv2.resize(img, (w, int(w * 0.75)))).haze_index - ref) <= 2.0


def test_night_and_blur_lower_confidence():
    img, sky = make_scene(1e-3)
    night = H(gain(img, 0.07), sky)
    assert "night_or_dark" in night.flags and night.confidence < 0.5
    blur = H(cv2.GaussianBlur(img, (0, 0), 12), sky)
    assert "blurry_or_lens_fog" in blur.flags


def test_smoke_yellower_than_fog():
    s = H(*(lambda im: (im[0], im[1]))(make_scene(1e-3, A=SMOKE_A))).sky_yellowness
    f = H(*(lambda im: (im[0], im[1]))(make_scene(1e-3))).sky_yellowness
    assert s - f >= 0.5


def test_overcast_still_monotonic():
    v = [idx(b, sky0=OVERCAST) for b in (1e-4, 8e-4, 2e-3)]
    assert v[0] < v[1] < v[2]


def test_wb_drift_bounded_and_baseline_cancels_it():
    """Known limitation: camera WB shifts raw index (+-5). Per-camera baseline must cancel it."""
    clear, sky = make_scene(1e-4)
    hazy, _ = make_scene(2e-3)
    for f in (dict(b=0.85, r=1.15), dict(b=1.15, r=0.85)):
        raw_drift = abs(
            H(wb_shift(hazy, **f), sky).haze_index - H(hazy, sky).haze_index
        )
        assert raw_drift <= 9
        r0 = relative_haze(H(hazy, sky).haze_index, H(clear, sky).haze_index)
        r1 = relative_haze(
            H(wb_shift(hazy, **f), sky).haze_index,
            H(wb_shift(clear, **f), sky).haze_index,
        )
        assert abs(r0 - r1) <= 8, (r0, r1)


def test_temporal_stability():
    img, sky = make_scene(1e-3)
    v = [H(noise(jpeg(img, 70), 6, seed=i), sky).haze_index for i in range(30)]
    assert np.std(v) < 1.0


@pytest.mark.parametrize(
    "make",
    [
        lambda: np.zeros((240, 320, 3), np.uint8),
        lambda: np.full((240, 320, 3), 255, np.uint8),
        lambda: np.full((240, 320, 3), 128, np.uint8),
        lambda: np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8),
        lambda: np.random.default_rng(0).integers(0, 255, (240, 320), dtype=np.uint8),
        lambda: np.random.default_rng(0).integers(
            0, 255, (240, 320, 4), dtype=np.uint8
        ),
        lambda: np.random.default_rng(0).random((240, 320, 3)).astype(np.float32),
        lambda: np.random.default_rng(0).integers(
            0, 65535, (240, 320, 3), dtype=np.uint16
        ),
    ],
)
def test_degenerate_inputs_no_nan(make):
    r = H(make())
    assert (
        np.isfinite(r.haze_index)
        and 0 <= r.haze_index <= 100
        and 0 <= r.confidence <= 1
    )


def test_relative_haze():
    assert relative_haze(50, None) is None
    assert (
        relative_haze(30, 30) == 0
        and relative_haze(65, 30) == 100
        and relative_haze(20, 30) == 0
    )


def test_performance():
    img, sky = make_scene(1e-3)
    t = time.perf_counter()
    [H(img, sky) for _ in range(10)]
    per = (time.perf_counter() - t) / 10
    t = time.perf_counter()
    [heuristic_sky(img) for _ in range(10)]
    per_s = (time.perf_counter() - t) / 10
    assert per < 0.15 and per_s < 0.05, (per, per_s)
