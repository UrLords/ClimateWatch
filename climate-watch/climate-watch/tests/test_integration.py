"""End-to-end: real Gradio Space (in-process, CPU) <- real gradio_client <- worker.tick. Catches contract bugs mocks cannot."""

import os, cv2, pytest

pytest.importorskip("gradio")
pytest.importorskip("gradio_client")
from gradio_client import Client
from synth import make_scene
import worker.ingest as W


@pytest.fixture(scope="module")
def client():
    import app

    app.demo.launch(prevent_thread_lock=True, server_port=7899, quiet=True)
    yield Client("http://127.0.0.1:7899", verbose=False)
    app.demo.close()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PUBLIC_DIR", str(tmp_path / "public"))
    (tmp_path / "hazy").mkdir()
    cv2.imwrite(str(tmp_path / "hazy" / "a.jpg"), make_scene(2e-3)[0])
    return tmp_path


def test_call_space_contract(client, env):
    img, res = W.call_space(
        client,
        str(env / "hazy" / "a.jpg"),
        {"id": "c1", "lat": -2.2, "lon": 113.9},
        30.0,
    )
    assert (
        os.path.exists(img)
        and res["status"] == "alert"
        and res["haze_relative"] is not None
    )
    assert res["advice"]["source"] == "playbook"


def test_worker_to_space_full_tick(client, env):
    cam = W.Cam(
        {
            "id": "c1",
            "name": "c1",
            "lat": -2.2,
            "lon": 113.9,
            "type": "folder",
            "url": str(env / "hazy"),
            "source_license": "own",
        }
    )
    cam.baseline.vals.extend([31.0] * 60)
    state, db = {}, W.init_db()
    W.tick(cam, db, client, state, 1, now=1_000_000)
    assert (
        state["c1"]["status"] == "alert"
        and state["c1"]["advice"]
        and (env / "public" / state["c1"]["frame"]).exists()
    )
    W.export_nodes([cam], state)
    assert (env / "public" / "nodes.json").exists()
