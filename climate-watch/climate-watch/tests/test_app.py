import numpy as np, pytest

pytest.importorskip("gradio")
from synth import make_scene
import cv2


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def test_analyze_contract():
    import app

    ann, res = app.analyze(rgb(make_scene(2e-3)[0]), "t1", 0.0, 0.0, -1.0)
    assert ann.shape[2] == 3 and {
        "camera_id",
        "status",
        "haze",
        "detections",
        "flood_risk",
        "advice",
        "haze_calibrated",
    } <= set(res)
    assert res["haze_calibrated"] is False and res["haze_relative"] is None


def test_baseline_changes_status():
    import app

    img = rgb(make_scene(2e-3)[0])
    _, raw = app.analyze(img, "t", 0, 0, -1)
    _, rel = app.analyze(img, "t", 0, 0, 30.0)
    assert (
        raw["status"] == "clear"
        and rel["status"] == "alert"
        and rel["advice"]["actions"]
    )


@pytest.mark.parametrize("shape", [(120, 160), (120, 160, 4), (120, 160, 3)])
def test_analyze_odd_inputs(shape):
    import app

    ann, res = app.analyze(
        np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8), "x", 0, 0, -1
    )
    assert res["status"] in ("clear", "alert", "low_confidence")


def test_analyze_none():
    import app

    assert app.analyze(None)[1] == {"error": "no image"}
