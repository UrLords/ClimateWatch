import numpy as np, pytest, types
from synth import *
from climatewatch.sky import heuristic_sky, SkySegmenter, SKY_CLASS_ID


@pytest.mark.parametrize(
    "kw",
    [
        dict(beta=1e-4),
        dict(beta=8e-4),
        dict(beta=1e-4, sky0=OVERCAST),
        dict(beta=8e-4, A=SMOKE_A),
        dict(beta=4e-4, A=SMOKE_A, sky0=OVERCAST),
    ],
)
def test_heuristic_iou(kw):
    img, sky = make_scene(**kw)
    assert iou(sky, heuristic_sky(img)) >= 0.8


def test_no_sky_camera_gives_empty_mask():
    img, _ = make_scene(2e-4)
    assert heuristic_sky(img[int(480 * 0.4) :]).mean() < 0.02


@pytest.mark.xfail(
    reason="Known limit: extreme haze flattens texture so buildings merge with sky; expect yolo26-sem to do better",
    strict=False,
)
def test_heuristic_dense_haze():
    img, sky = make_scene(3e-3)
    assert iou(sky, heuristic_sky(img)) >= 0.8


def _seg():
    s = SkySegmenter.__new__(SkySegmenter)
    s.model = None
    return s


def test_parse_semantic_map_attr():
    m = np.zeros((10, 20), np.uint8)
    m[:4] = SKY_CLASS_ID
    out = _seg()._parse(types.SimpleNamespace(sem_mask=m), (20, 40))
    assert out.shape == (20, 40) and out[:8].all() and not out[12:].any()


def test_parse_unknown_result_returns_none():
    assert (
        _seg()._parse(types.SimpleNamespace(masks=None, boxes=None), (20, 40)) is None
    )


def test_missing_model_falls_back():
    img, _ = make_scene(2e-4)
    m, src = SkySegmenter().mask(img)
    assert src == "heuristic" and m.shape == img.shape[:2]
