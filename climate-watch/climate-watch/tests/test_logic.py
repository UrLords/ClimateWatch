import numpy as np, cv2
from climatewatch.privacy import blur_boxes
from climatewatch.advice import build_advice, PLAYBOOK
from climatewatch.detector import HAZARD_CLASS_MAP
from climatewatch.risk import flood_risk

LOW = {"score": 0, "level": "low", "rain": None}


def test_blur_hides_detail_only_inside_box_and_does_not_mutate():
    img = np.random.default_rng(0).integers(0, 255, (200, 300, 3), dtype=np.uint8)
    orig = img.copy()
    out = blur_boxes(img, [[50, 40, 150, 160]])
    lap = lambda a: cv2.Laplacian(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert lap(out[40:160, 50:150]) < 0.2 * lap(img[40:160, 50:150])
    assert (
        (out[:15] == img[:15]).all()
        and (out[:, 250:] == img[:, 250:]).all()
        and (img == orig).all()
    )


def test_blur_edge_cases():
    img = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    out = blur_boxes(
        img,
        [[-20, -20, 30, 30], [90, 90, 400, 400], [50, 50, 50, 50], [60, 60, 40, 40]],
    )
    assert out.shape == img.shape


def test_playbook_covers_every_detectable_hazard():
    for h in set(HAZARD_CLASS_MAP.values()):
        assert h in PLAYBOOK, f"hazard '{h}' can be detected but has no playbook entry"


def test_advice_rules():
    assert any(
        "limit outdoor" in a for a in build_advice(80, 0.9, set(), LOW)["actions"]
    )
    assert build_advice(80, 0.3, set(), LOW)["actions"] == []
    assert build_advice(45, 0.9, set(), LOW, relative=True)["actions"]
    assert build_advice(45, 0.9, set(), LOW, relative=False)["actions"] == []
    a = build_advice(
        0,
        1,
        {"waste_pile", "drain_blocked"},
        {"score": 0.6, "level": "medium", "rain": {"mm": 8, "prob_max": 90}},
    )
    assert a["actions"][0].startswith("Flood risk") and len(a["actions"]) == len(
        set(a["actions"])
    )
    assert all(
        x in sum(PLAYBOOK.values(), []) or x.startswith("Flood risk")
        for x in a["actions"]
    )


def test_flood_risk_monotonic():
    s = lambda hz, rain: flood_risk(hz, rain)["score"]
    dry, wet = {"mm": 0, "prob_max": 0}, {"mm": 10, "prob_max": 90}
    assert s({"drain_blocked"}, wet) > s({"drain_blocked"}, dry) > 0
    assert s({"drain_blocked", "waste_pile"}, wet) > s({"drain_blocked"}, wet)
    assert (
        s(set(), wet) == 0
        and flood_risk({"flood_water", "drain_blocked"}, wet)["level"] == "high"
    )
    assert flood_risk(set(), None)["level"] == "low"
