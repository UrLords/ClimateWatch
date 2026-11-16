import json, os, shutil, sqlite3, time
import cv2, numpy as np, pytest
from synth import make_scene, gain
import worker.ingest as W


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PUBLIC_DIR", str(tmp_path / "public"))
    for d in ("clear", "hazy"):
        (tmp_path / d).mkdir()
    for i in range(60):
        cv2.imwrite(
            str(tmp_path / "clear" / f"{i:03d}.jpg"), make_scene(1e-4, seed=i % 5)[0]
        )
        cv2.imwrite(
            str(tmp_path / "hazy" / f"{i:03d}.jpg"), make_scene(2e-3, seed=i % 5)[0]
        )
    return tmp_path


def mkcam(path, cid="c1", **kw):
    return W.Cam(
        {
            "id": cid,
            "name": cid,
            "lat": -2.2,
            "lon": 113.9,
            "type": "folder",
            "url": str(path),
            "source_license": "own",
            **kw,
        }
    )


class FakeClient:
    def __init__(self, status="alert", fail=False):
        self.calls, self.status, self.fail = 0, status, fail

    def predict(self, path, cid, lat, lon, base, api_name=None):
        self.calls += 1
        path = path["path"] if isinstance(path, dict) else path
        if self.fail:
            raise RuntimeError("space asleep")
        out = path + ".ann.jpg"
        shutil.copy(path, out)
        return out, {
            "status": self.status,
            "advice": {
                "actions": [
                    "Clear debris from the drainage grate (wear gloves; avoid traffic)."
                ]
            },
            "flood_risk": {"score": 0.5, "level": "medium", "rain": None},
        }


def run(cam, db, client, state, n, t0=1_000_000, interval=1):
    for i in range(n):
        W.tick(cam, db, client, state, interval, now=t0 + i)
    return t0 + n


def test_license_guard(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "cameras:\n- {id: x, name: x, lat: 0, lon: 0, type: snapshot, url: http://a/b.jpg}\n"
    )
    with pytest.raises(SystemExit):
        W.load_cameras(str(p))
    p.write_text(
        "cameras:\n- {id: x, name: x, lat: 0, lon: 0, type: snapshot, url: http://a/b.jpg, source_license: ' '}\n"
    )
    with pytest.raises(SystemExit):
        W.load_cameras(str(p))


def test_ring_buffer_is_60_seconds_and_interval_respected(env):
    cam, db = mkcam(env / "clear"), W.init_db()
    run(cam, db, None, {}, 100, interval=30)
    assert len(cam.buf) == 60
    assert db.execute("SELECT count(*) FROM observations").fetchone()[0] == 4


def test_baseline_warmup_then_relative_then_alert_and_privacy(env):
    cam, db, state = mkcam(env / "clear"), W.init_db(), {}
    t = run(cam, db, None, state, 49)
    assert state["c1"]["haze_mode"] == "warming_up" and state["c1"]["status"] != "alert"
    t = run(cam, db, None, state, 20, t0=t)
    assert state["c1"]["haze_mode"] == "relative" and state["c1"]["status"] == "clear"
    cam.files = sorted(str(p) for p in (env / "hazy").glob("*.jpg"))
    run(cam, db, None, state, 3, t0=t)
    assert state["c1"]["status"] == "alert" and state["c1"]["haze_relative"] >= 40
    W.export_nodes([cam], state)
    pub = json.load(open(env / "public" / "nodes.json"))
    assert pub["nodes"][0]["status"] == "alert" and "context" not in json.dumps(pub)
    ctx = list((env / "data" / "private" / "context" / "c1").glob("*/*.jpg"))
    assert 0 < len(ctx) <= 60
    assert not any(
        p.suffix == ".jpg" and "private" in str(p) for p in (env / "public").rglob("*")
    )


def test_offline_after_three_failures_and_recovery(env):
    cam, db, state = mkcam(env / "empty"), W.init_db(), {}
    os.makedirs(env / "empty")
    t = run(cam, db, None, state, 2)
    assert "c1" not in state
    t = run(cam, db, None, state, 2, t0=t)
    assert state["c1"]["status"] == "offline"
    cam.files = sorted(str(p) for p in (env / "clear").glob("*.jpg"))
    run(cam, db, None, state, 1, t0=t)
    assert state["c1"]["status"] != "offline"


def test_global_lighting_change_is_not_an_event_but_new_object_is(env):
    cam = mkcam(env / "clear")
    base = make_scene(1e-4)[0]
    for _ in range(20):
        W.change_fraction(cam, base)
    assert W.change_fraction(cam, gain(base, 1.5)) < W.CHANGE_FRAC
    obj = base.copy()
    obj[300:380, 200:300] = (20, 20, 20)
    assert W.change_fraction(cam, obj) >= W.CHANGE_FRAC


def test_space_integration_heartbeat_and_public_frame(env):
    (env / "one").mkdir()
    shutil.copy(env / "clear" / "000.jpg", env / "one" / "a.jpg")
    cam, db, state, cl = mkcam(env / "one"), W.init_db(), {}, FakeClient("alert")
    run(cam, db, cl, state, 1)
    assert cl.calls == 1 and state["c1"]["status"] == "alert" and state["c1"]["advice"]
    assert (env / "public" / state["c1"]["frame"]).exists()
    run(cam, db, cl, state, 5, t0=1_000_001)
    assert cl.calls == 1
    W.tick(cam, db, cl, state, 1, now=1_000_000 + W.HEARTBEAT + 5)
    assert cl.calls == 2


def test_clear_space_result_does_not_publish_frame(env):
    cam, db, state = mkcam(env / "clear"), W.init_db(), {}
    run(cam, db, FakeClient("clear"), state, 1)
    assert (
        state["c1"]["frame"] is None
        and not (env / "public" / "frames" / "c1.jpg").exists()
    )


def test_space_failure_does_not_crash(env):
    cam, db, state = mkcam(env / "clear"), W.init_db(), {}
    run(cam, db, FakeClient(fail=True), state, 3)
    assert state["c1"]["status"] in ("clear", "low_confidence")


def test_purge_old_context(env):
    root = env / "data" / "private" / "context" / "c1"
    for ts in (1_000, 1_000_000):
        (root / str(ts)).mkdir(parents=True)
        (root / str(ts) / "a.jpg").write_bytes(b"x")
    W.purge_old_context(1_000_000 + 10, hours=24)
    assert not (root / "1000").exists() and (root / "1000000").exists()


def test_baseline_persists_across_restart(env):
    cam, db = mkcam(env / "clear"), W.init_db()
    run(cam, db, None, {}, 60)
    cam2 = mkcam(env / "clear")
    W.load_baseline(cam2, db, days=100 * 365)
    assert cam2.baseline.value is not None


def test_nodes_json_atomic_schema(env):
    cam = mkcam(env / "clear")
    W.export_nodes([cam], {})
    d = json.load(open(env / "public" / "nodes.json"))
    assert d["nodes"][0]["status"] == "offline" and {"id", "name", "lat", "lon"} <= set(
        d["nodes"][0]
    )
    assert not (env / "public" / "nodes.json.tmp").exists()
