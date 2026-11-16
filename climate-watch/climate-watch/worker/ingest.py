"""Always-on CPU worker: 1 fps ring buffer -> sky mask + haze (every interval) -> candidate -> HF Space -> SQLite + public/nodes.json

Run:  python -m worker.ingest --config worker/cameras.yaml [--once]
Env:  SPACE_ID=UrLord/climate-watch  HF_TOKEN=...   (unset => CPU-only mode, haze alerts only)
Design notes (found by testing):
  * ring buffer fills at BUF_FPS (1 fps) so 60 frames == 60 s of context; analysis runs every `interval_sec`.
  * persistent hazards (old garbage pile) cause NO frame change => detectors also run on a heartbeat.
  * change screen ignores global lighting changes (mean-subtracted, blob-size based).
"""

import argparse, glob, json, os, shutil, sqlite3, time
from collections import deque

import cv2
import numpy as np
import requests
import yaml

from climatewatch.baseline import Baseline
from climatewatch.haze import compute_haze, relative_haze
from climatewatch.sky import SkySegmenter

BUF_SECONDS, BUF_FPS = 60, 1
HAZE_ALERT_REL, CHANGE_FRAC, COOLDOWN, HEARTBEAT, OFFLINE_AFTER = (
    40.0,
    0.004,
    300,
    600,
    3,
)
DATA = lambda: os.getenv("DATA_DIR", "data")
PUBLIC = lambda: os.getenv("PUBLIC_DIR", "public")
RETENTION_H = float(os.getenv("PRIVATE_RETENTION_H", "24"))
UA = {"User-Agent": "ClimateWatch/0.1 (hackathon; contact in repo)"}
_sky = None


def sky_seg():
    global _sky
    _sky = _sky or SkySegmenter()
    return _sky


def load_cameras(path):
    cfg = yaml.safe_load(open(path))
    for c in cfg["cameras"]:
        if not str(c.get("source_license", "")).strip():
            raise SystemExit(
                f"Camera {c.get('id')} has no source_license. Refusing: only legal, published feeds."
            )
    return cfg


class Cam:
    def __init__(self, c):
        self.c = c
        self.buf = deque(maxlen=BUF_SECONDS * BUF_FPS)
        self.bg = self.mask = None
        self.last_grab = -1e9
        self.mask_ts = self.last_analysis = self.last_alert = self.last_detect = (
            self.i
        ) = self.fail = 0
        self.baseline = Baseline()
        self.files = (
            sorted(
                glob.glob(os.path.join(c["url"], "*.jp*g"))
                + glob.glob(os.path.join(c["url"], "*.png"))
            )
            if c["type"] == "folder"
            else []
        )

    def grab(self):
        t, u = self.c["type"], self.c["url"]
        if t == "folder":
            if not self.files:
                return None
            f = cv2.imread(self.files[self.i % len(self.files)])
            self.i += 1
            return f
        if t == "snapshot":
            r = requests.get(u, timeout=10, headers=UA)
            r.raise_for_status()
            return cv2.imdecode(np.frombuffer(r.content, np.uint8), 1)
        cap = cv2.VideoCapture(u)
        ok, f = cap.read()
        cap.release()
        return f if ok else None


def init_db():
    os.makedirs(DATA(), exist_ok=True)
    db = sqlite3.connect(os.path.join(DATA(), "climatewatch.db"))
    db.execute("""CREATE TABLE IF NOT EXISTS observations(
        camera_id TEXT, ts INTEGER, haze_index REAL, confidence REAL, status TEXT, result_json TEXT, frame TEXT)""")
    return db


def load_baseline(cam, db, days=14):
    rows = db.execute(
        "SELECT haze_index, confidence FROM observations WHERE camera_id=? AND ts>?",
        (cam.c["id"], time.time() - days * 86400),
    ).fetchall()
    for h, conf in rows:
        cam.baseline.update(h, conf)


def change_fraction(cam, frame):
    """Fraction of pixels that changed by >25 levels vs running background, after normalising global brightness."""
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(
        np.float32
    )
    g = g / (g.mean() + 1.0) * 128.0
    frac = 0.0
    if cam.bg is not None:
        m = (np.abs(g - cam.bg) > 25).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        frac = float(m.mean())
    cam.bg = g if cam.bg is None else cv2.addWeighted(cam.bg, 0.98, g, 0.02, 0)
    return frac


def purge_old_context(now, hours=None):
    root, hours = (
        os.path.join(DATA(), "private", "context"),
        RETENTION_H if hours is None else hours,
    )
    if not os.path.isdir(root):
        return
    for cam_dir in os.listdir(root):
        for ev in os.listdir(os.path.join(root, cam_dir)):
            p = os.path.join(root, cam_dir, ev)
            if ev.isdigit() and now - int(ev) > hours * 3600:
                shutil.rmtree(p, ignore_errors=True)


def call_space(client, jpg_path, cam, baseline):
    from gradio_client import handle_file

    out = client.predict(
        handle_file(jpg_path),
        cam["id"],
        cam["lat"],
        cam["lon"],
        -1 if baseline is None else baseline,
        api_name="/analyze",
    )
    return out[0], out[1]


def export_nodes(cams, state):
    nodes = [
        {
            "id": c.c["id"],
            "name": c.c["name"],
            "lat": c.c["lat"],
            "lon": c.c["lon"],
            **state.get(c.c["id"], {"status": "offline"}),
        }
        for c in cams
    ]
    os.makedirs(PUBLIC(), exist_ok=True)
    tmp = os.path.join(PUBLIC(), "nodes.json.tmp")
    json.dump({"updated": int(time.time()), "nodes": nodes}, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(PUBLIC(), "nodes.json"))


def tick(cam, db, client, state, interval, now=None):
    """One grab (buffer at 1 fps); full analysis only every `interval` seconds."""
    cid, now = cam.c["id"], now or int(time.time())
    if now - cam.last_grab < cam.c.get("poll_sec", 1.0 / BUF_FPS):
        return
    cam.last_grab = now
    frame = cam.grab()
    if frame is None:
        cam.fail += 1
        if cam.fail >= OFFLINE_AFTER:
            state[cid] = {"status": "offline", "ts": now}
        return
    cam.fail = 0
    frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
    cam.buf.append(
        (now, cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes())
    )
    if now - cam.last_analysis < interval:
        return
    cam.last_analysis = now

    if cam.c.get("has_sky", True) and (cam.mask is None or now - cam.mask_ts > 1800):
        cam.mask, _ = sky_seg().mask(frame)
        cam.mask_ts = now
    haze = compute_haze(frame, cam.mask if cam.c.get("has_sky", True) else None)
    cam.baseline.update(haze.haze_index, haze.confidence)
    base = cam.baseline.value
    rel = relative_haze(haze.haze_index, base)
    changed = change_fraction(cam, frame) >= CHANGE_FRAC

    haze_alert = rel is not None and rel >= HAZE_ALERT_REL and haze.confidence >= 0.5
    status = (
        "low_confidence"
        if haze.confidence < 0.5
        else "alert"
        if haze_alert
        else "clear"
    )
    node = {
        "status": status,
        "haze_index": round(haze.haze_index, 1),
        "haze_relative": None if rel is None else round(rel, 1),
        "haze_mode": "relative" if rel is not None else "warming_up",
        "confidence": round(haze.confidence, 2),
        "flags": haze.flags,
        "advice": [],
        "frame": None,
        "ts": now,
    }
    row_json, frame_rel = {"haze": haze.to_dict()}, None

    run_detectors = changed or haze_alert or now - cam.last_detect >= HEARTBEAT
    if run_detectors and client and haze.confidence >= 0.3:
        cam.last_detect = now
        tmp = os.path.join(DATA(), "tmp")
        os.makedirs(tmp, exist_ok=True)
        latest = os.path.join(tmp, f"{cid}.jpg")
        cv2.imwrite(latest, frame)
        try:
            img_path, res = call_space(client, latest, cam.c, base)
            if res["status"] == "alert" or status != "alert":
                node.update(
                    status=res["status"]
                    if res["status"] != "clear" or status == "clear"
                    else status
                )
            node.update(advice=res["advice"]["actions"], flood_risk=res["flood_risk"])
            if res["status"] == "alert":
                frame_rel = f"frames/{cid}.jpg"
                os.makedirs(os.path.join(PUBLIC(), "frames"), exist_ok=True)
                shutil.copy(img_path, os.path.join(PUBLIC(), frame_rel))
                node["frame"] = frame_rel
            row_json = res
        except Exception as e:
            print(f"[space] {cid}: {e}")

    if node["status"] == "alert" and now - cam.last_alert > COOLDOWN:
        cam.last_alert = now
        ev = os.path.join(DATA(), "private", "context", cid, str(now))
        os.makedirs(ev, exist_ok=True)
        for ts, jpg in cam.buf:
            open(os.path.join(ev, f"{ts}.jpg"), "wb").write(jpg)
    purge_old_context(now)
    state[cid] = node
    db.execute(
        "INSERT INTO observations VALUES (?,?,?,?,?,?,?)",
        (
            cid,
            now,
            haze.haze_index,
            haze.confidence,
            node["status"],
            json.dumps(row_json),
            frame_rel,
        ),
    )
    db.commit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="worker/cameras.yaml")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    cfg = load_cameras(a.config)
    db, state = init_db(), {}
    cams = [Cam(c) for c in cfg["cameras"]]
    for c in cams:
        load_baseline(c, db)
    client = None
    if os.getenv("SPACE_ID"):
        from gradio_client import Client

        client = Client(os.environ["SPACE_ID"], hf_token=os.getenv("HF_TOKEN"))
    interval = cfg.get("interval_sec", 30)
    while True:
        t0 = time.time()
        for cam in cams:
            try:
                tick(cam, db, client, state, interval)
            except Exception as e:
                print(f"[{cam.c['id']}] {e}")
        export_nodes(cams, state)
        if a.once:
            break
        time.sleep(max(0.0, 1.0 / BUF_FPS - (time.time() - t0)))
