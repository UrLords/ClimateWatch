"""CPU background camera checks and local, privacy-filtered evidence records."""

from datetime import datetime, timezone
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time

from climatewatch.evidence import get_engine, repeated_threat_frames
from live_evidence import MANIFESTS, fetch_camera


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class CameraMonitor:
    def __init__(
        self,
        directory,
        analysis_lock,
        database,
        engine_factory=get_engine,
        fetcher=fetch_camera,
    ):
        self.directory = Path(directory)
        self.analysis_lock = analysis_lock
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.engine_factory, self.fetcher = engine_factory, fetcher
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.records = {}
        for manifest in MANIFESTS:
            path = self.directory / manifest
            if path.is_file():
                for record in json.loads(path.read_text(encoding="utf-8-sig")):
                    self.records[record["id"]] = record
        self.states = {
            identifier: {
                "id": identifier,
                "status": "waiting",
                "checked_at": None,
                "next_check_at": 0,
            }
            for identifier in self.records
        }
        self.previous = {}
        self.events = {}
        self.thread = None
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS camera_events (camera_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, result TEXT NOT NULL)"
            )
            for identifier, created, text in db.execute(
                "SELECT camera_id,created_at,result FROM camera_events"
            ):
                if identifier in self.records:
                    result = json.loads(text)

                    if result.get("summary", {}).get("hazards") and set(
                        result["summary"]["hazards"]
                    ) <= {"smoke", "fire", "waste"}:
                        self.events[identifier] = {
                            "created_at": created,
                            "result": result,
                        }

    def start(self):
        self.thread = threading.Thread(
            target=self.run, daemon=True, name="ClimateWatch-camera-monitor"
        )
        self.thread.start()

    def stop(self):
        with self.lock:
            self.stop_event.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=5)

    def interval(self, record):
        return max(
            60,
            record.get(
                "refreshSeconds",
                record.get("liveSource", {}).get("collectionIntervalSeconds", 60),
            ),
        )

    def snapshot(self):
        with self.lock:
            return {
                "enabled": bool(self.thread and self.thread.is_alive()),
                "policy": "Only persistent smoke, fire, or waste creates evidence; ordinary water and objects are ignored.",
                "minimum_interval_seconds": 60,
                "confirmation_checks": 2,
                "sampled_frames_per_check": 3,
                "cameras": [dict(state) for state in self.states.values()],
                "events": [
                    {
                        "camera_id": identifier,
                        "created_at": event["created_at"],
                        "hazards": event["result"]["summary"]["hazards"],
                        "preview": event["result"]["frames"][0]["thumbnail"],
                    }
                    for identifier, event in self.events.items()
                ],
            }

    def result(self, identifier):
        with self.lock:
            event = self.events.get(identifier)
            return event["result"] if event else None

    def process(self, identifier):
        record = self.records[identifier]
        with self.lock:
            self.states[identifier].update(status="scanning", error=None)
        try:
            record, body, mime, capture = self.fetcher(self.directory, identifier)
            if self.stop_event.is_set():
                return
            digest = hashlib.sha256(body).hexdigest()
            prior = self.previous.get(identifier)
            if prior and prior["hash"] == digest:
                with self.lock:
                    self.states[identifier].update(
                        status="waiting_for_new_frame", checked_at=utc_now()
                    )
                return
            engine = self.engine_factory()
            roi = record.get("drain_roi")
            if isinstance(roi, dict):
                roi = roi.get("xyxy")
            result = engine.analyze(body, mime, drain_roi=roi, max_frames=3)
            hazards = set(result["summary"]["hazards"]) & {"smoke", "fire", "waste"}
            matched = []
            if prior and time.time() - prior["time"] <= max(
                900, self.interval(record) * 3
            ):
                for frame in result.get("frames", []):
                    pair = [
                        {"time_s": -1, "detections": prior["detections"]},
                        {**frame, "time_s": 0},
                    ]
                    current = next(
                        (
                            item
                            for item in repeated_threat_frames(pair, True)
                            if item["time_s"] == 0
                        ),
                        None,
                    )
                    if current:
                        matched.append({**current, "time_s": frame["time_s"]})
            confirmed = {d["hazard"] for frame in matched for d in frame["detections"]}
            self.previous[identifier] = {
                "hash": digest,
                "hazards": hazards,
                "time": time.time(),
                "detections": [
                    d for frame in result.get("frames", []) for d in frame["detections"]
                ],
            }
            state = "no_target_detected" if not hazards else "candidate_pending"
            if confirmed and result.get("frames"):
                result["source"] = {
                    "id": identifier,
                    "sourceUrl": record.get("sourceUrl"),
                    "recordedAt": None,
                    "footageKind": record.get("footageKind"),
                }
                result["capture"] = capture
                result["monitor"] = {
                    "automatic": True,
                    "confirmation_checks": 2,
                    "confirmed_hazards": sorted(confirmed),
                }
                result["frames"] = matched
                from climatewatch.evidence import summarize

                result["summary"] = summarize(result["frames"])
                result["summary"]["hazards"] = sorted(confirmed)
                created = utc_now()
                with self.lock:
                    if self.stop_event.is_set():
                        return
                    with closing(sqlite3.connect(self.database)) as db, db:
                        db.execute(
                            "INSERT OR REPLACE INTO camera_events VALUES (?,?,?)",
                            (identifier, created, json.dumps(result, allow_nan=False)),
                        )
                    self.events[identifier] = {"created_at": created, "result": result}
                state = "evidence_captured"
            with self.lock:
                self.states[identifier].update(status=state, checked_at=utc_now())
        except Exception as error:
            self.previous.pop(identifier, None)
            with self.lock:
                self.states[identifier].update(
                    status="offline", checked_at=utc_now(), error=str(error)[:200]
                )
        finally:
            with self.lock:
                self.states[identifier]["next_check_at"] = time.time() + self.interval(
                    record
                )

    def run(self):
        while not self.stop_event.is_set():
            with self.lock:
                due = [
                    state
                    for state in self.states.values()
                    if state["next_check_at"] <= time.time()
                ]
            if not due or not self.analysis_lock.acquire(blocking=False):
                self.stop_event.wait(2)
                continue
            try:
                identifier = min(due, key=lambda state: state["next_check_at"])["id"]
                self.process(identifier)
            finally:
                self.analysis_lock.release()
            self.stop_event.wait(0.1)
