"""Automatic monitoring checks without external feeds or model weights."""

import json
from pathlib import Path
import tempfile
import threading
import unittest
import sys
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_monitor import CameraMonitor


def output(hazard=None):
    detection = {
        "hazard": hazard,
        "label": hazard,
        "box": [0.1, 0.1, 0.7, 0.7],
        "confidence": 0.8,
    }
    return {
        "frames": [
            {
                "time_s": 0,
                "detections": [detection],
                "thumbnail": "data:image/jpeg;base64,eA==",
            }
        ]
        if hazard
        else [],
        "summary": {"hazards": [hazard] if hazard else []},
        "provenance": {},
    }


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        folder = self.root / "live-cameras"
        folder.mkdir()
        (folder / "cameras.json").write_text(
            json.dumps(
                [{"id": "one", "mediaType": "hls"}, {"id": "two", "mediaType": "hls"}]
            )
        )
        self.body = b"first"
        self.fetch = Mock(
            side_effect=lambda directory, id: (
                {"id": id},
                self.body,
                "image/png",
                {"retrieved_at": "now"},
            )
        )
        self.engine = Mock()
        self.monitor = CameraMonitor(
            self.root,
            threading.Lock(),
            self.root / "events.sqlite",
            lambda: self.engine,
            self.fetch,
        )

    def tearDown(self):
        self.monitor.stop()
        self.temp.cleanup()

    def test_normal_checks_create_no_evidence_without_ui(self):
        self.engine.analyze.return_value = output()
        self.monitor.process("one")
        self.assertEqual(self.monitor.states["one"]["status"], "no_target_detected")
        self.assertEqual(self.monitor.snapshot()["events"], [])
        self.engine.analyze.assert_called_once()
        self.assertEqual(self.engine.analyze.call_args.kwargs["max_frames"], 3)

    def test_two_distinct_positive_checks_required_and_result_persists(self):
        self.engine.analyze.return_value = output("smoke")
        self.monitor.process("one")
        self.assertEqual(self.monitor.states["one"]["status"], "candidate_pending")
        self.monitor.process("one")
        self.assertEqual(self.monitor.snapshot()["events"], [])
        self.body = b"new"
        self.monitor.process("one")
        self.assertEqual(self.monitor.states["one"]["status"], "evidence_captured")
        self.assertTrue(self.monitor.result("one")["monitor"]["automatic"])
        restored = CameraMonitor(
            self.root,
            threading.Lock(),
            self.root / "events.sqlite",
            lambda: self.engine,
            self.fetch,
        )
        self.assertEqual(restored.snapshot()["events"][0]["hazards"], ["smoke"])

    def test_normal_or_failed_check_breaks_confirmation(self):
        self.engine.analyze.return_value = output("waste")
        self.monitor.process("one")
        self.body = b"normal"
        self.engine.analyze.return_value = output()
        self.monitor.process("one")
        self.body = b"candidate"
        self.engine.analyze.return_value = output("waste")
        self.monitor.process("one")
        self.assertEqual(self.monitor.snapshot()["events"], [])
        self.fetch.side_effect = RuntimeError("operator unavailable")
        self.monitor.process("one")
        self.assertEqual(self.monitor.states["one"]["status"], "offline")
        self.assertNotIn("one", self.monitor.previous)

    def test_unrelated_boxes_and_stop_do_not_confirm_an_event(self):
        first = output("waste")
        first["frames"][0]["detections"][0]["box"] = [0, 0, 0.1, 0.1]
        self.engine.analyze.return_value = first
        self.monitor.process("one")
        second = output("waste")
        second["frames"][0]["detections"][0]["box"] = [0.8, 0.8, 1, 1]
        self.body = b"other-region"
        self.engine.analyze.return_value = second
        self.monitor.process("one")
        self.assertEqual(self.monitor.snapshot()["events"], [])
        self.body = b"stopped"
        self.monitor.stop()
        self.monitor.process("one")
        self.assertEqual(self.monitor.snapshot()["events"], [])

    def test_ordinary_water_never_becomes_threat_even_if_engine_regresses(self):
        self.engine.analyze.return_value = output("water")
        self.monitor.process("one")
        self.body = b"new-water"
        self.monitor.process("one")
        self.assertEqual(self.monitor.snapshot()["events"], [])
        self.assertEqual(self.monitor.states["one"]["status"], "no_target_detected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
