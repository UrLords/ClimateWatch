import cv2, numpy as np
from synth import make_scene
from climatewatch.haze import compute_haze
import worker.ingest as W
import test_worker as TW


def test_ir_mono_flagged_not_for_color_or_dense_haze():
    for beta in (1e-4, 2e-3):
        img, sky = make_scene(beta)
        assert "ir_mono_mode" not in compute_haze(img, sky).flags
    g = cv2.cvtColor(make_scene(1e-4)[0], cv2.COLOR_BGR2GRAY)
    r = compute_haze(cv2.merge([g, g, g]))
    assert "ir_mono_mode" in r.flags and r.confidence <= 0.5


def test_poll_sec_limits_fetch_rate(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("PUBLIC_DIR", str(tmp_path / "p"))
    (tmp_path / "i").mkdir()
    cv2.imwrite(str(tmp_path / "i" / "a.jpg"), make_scene(1e-4)[0])
    cam = TW.mkcam(tmp_path / "i", poll_sec=20)
    db = W.init_db()
    n = [0]
    orig = cam.grab
    cam.grab = lambda: (n.__setitem__(0, n[0] + 1), orig())[1]
    for t in range(60):
        W.tick(cam, db, None, {}, 30, now=1_000_000 + t)
    assert n[0] == 3
