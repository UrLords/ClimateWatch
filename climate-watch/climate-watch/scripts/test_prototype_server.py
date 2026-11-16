"""Run with python scripts/test_prototype_server.py; no weights or network services required."""

from __future__ import annotations

import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from serve_prototype import MAX_BYTES, PrototypeServer
from climatewatch.evidence import (
    EvidenceEngine,
    InvalidMedia,
    drain_overlap,
    validate_roi,
)


class DecoderOnlyEngine(EvidenceEngine):
    """Exercise real input decoding and response assembly without loading weights in API unit tests."""

    def __init__(self):
        self.models = []
        self.waste = None

    def frame(self, bgr, time_s, roi=None):
        return {
            "time_s": round(float(time_s), 3),
            "width": bgr.shape[1],
            "height": bgr.shape[0],
            "detections": [],
            "thumbnail": None,
            "privacy_regions_obscured": 0,
        }


def photo(width=320, height=180, image_format="PNG"):
    image = Image.new("RGB", (width, height), (140, 180, 210))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, height // 2, width, height), fill=(80, 100, 60))
    for x in range(0, width, 16):
        draw.rectangle((x, height // 3, x + 8, height), fill=(55, 60, 65))
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.dist = cls.root / "dist"
        cls.dist.mkdir()
        (cls.dist / "index.html").write_text("<title>ClimateWatch</title>")
        (cls.root / "private.txt").write_text("must not be exposed")
        evidence = cls.dist / "evidence"
        evidence.mkdir()
        (evidence / "test.png").write_bytes(photo())
        (evidence / "sources.json").write_text(
            json.dumps(
                [
                    {
                        "id": "test",
                        "mediaPath": "/evidence/test.png",
                        "drain_roi": {"xyxy": [0, 0, 1, 1]},
                    }
                ]
            )
        )
        cls.server = PrototypeServer(0, cls.dist, engine=DecoderOnlyEngine())
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def request(self, method="GET", path="/api/health", body=None, headers=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=10
        )
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        raw = response.read()
        result = (
            json.loads(raw)
            if raw
            and response.getheader("Content-Type", "").startswith("application/json")
            else raw
        )
        connection.close()
        return response.status, result

    def test_health_truthfully_reports_model_availability(self):
        with patch(
            "serve_prototype.model_status",
            return_value=[{"id": "missing", "available": False}],
        ):
            status, data = self.request()
            self.assertEqual(status, 200)
            self.assertFalse(data["capabilities"]["hazard_detection"])
            self.assertFalse(data["detectors_available"])
            self.assertEqual(data["raw_upload_retention"], "none")

    def test_valid_png_and_jpeg_return_only_measurements(self):
        before = sorted(str(p) for p in self.root.rglob("*"))
        for fmt, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg")):
            with self.subTest(fmt=fmt):
                status, data = self.request(
                    "POST",
                    "/api/analyze-image",
                    photo(image_format=fmt),
                    {"Content-Type": mime},
                )
                self.assertEqual(status, 200)
                self.assertEqual(data["kind"], "model_evidence")
                self.assertEqual(data["media"]["width"], 320)
                self.assertEqual(data["media"]["height"], 180)
                self.assertEqual(data["media"]["sampled_frames"], 1)
                self.assertEqual(data["frames"], [])
                self.assertEqual(data["summary"]["actions"], [])
                self.assertFalse(data["summary"]["review_required"])
                self.assertGreater(len(data["summary"]["limitations"]), 3)
                self.assertEqual(len(data["provenance"]["input_sha256"]), 64)
                self.assertEqual(data["provenance"]["raw_upload_retention"], "none")
        self.assertEqual(before, sorted(str(p) for p in self.root.rglob("*")))

    def test_invalid_input(self):
        cases = [
            (b"", {"Content-Type": "image/png"}, 400),
            (b"not an image", {"Content-Type": "image/png"}, 400),
            (photo(), {"Content-Type": "text/plain"}, 415),
            (photo(), {"Content-Type": "image/jpeg"}, 400),
            (
                b"x",
                {"Content-Type": "image/png", "Content-Length": str(MAX_BYTES + 1)},
                413,
            ),
            (b"x", {"Content-Type": "image/png", "Content-Length": "bad"}, 400),
            (photo(4097, 32), {"Content-Type": "image/png"}, 413),
            (photo(16, 16), {"Content-Type": "image/png"}, 400),
        ]
        for body, headers, expected in cases:
            with self.subTest(headers=headers, length=len(body)):
                status, data = self.request("POST", "/api/analyze-image", body, headers)
                self.assertEqual(status, expected)
                self.assertIn("error", data)

    def test_origins_and_host(self):
        status, _ = self.request(
            "POST",
            "/api/analyze-image",
            photo(),
            {"Content-Type": "image/png", "Origin": "https://other.example"},
        )
        self.assertEqual(status, 403)
        status, _ = self.request(headers={"Host": "untrusted.example"})
        self.assertEqual(status, 403)
        status, _ = self.request(
            "POST",
            "/api/analyze-image",
            photo(),
            {
                "Content-Type": "image/png",
                "Origin": f"http://127.0.0.1:{self.server.server_port}",
            },
        )
        self.assertEqual(status, 200)

    def test_static_files_and_traversal(self):
        self.assertEqual(self.request(path="/")[0], 200)
        self.assertEqual(self.request("HEAD", "/")[1], b"")
        for path in (
            "/../private.txt",
            "/%2e%2e/private.txt",
            "/..%5cprivate.txt",
            "/%00",
            "/missing",
            "/api/unknown",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.request(path=path)[0], 404)

    def test_methods_and_busy_worker(self):
        self.assertEqual(self.request("PUT", "/api/health")[0], 405)
        self.assertEqual(self.request("OPTIONS", "/api/analyze-image")[0], 405)
        self.server.analysis_lock.acquire()
        try:
            self.assertEqual(
                self.request(
                    "POST", "/api/analyze-image", photo(), {"Content-Type": "image/png"}
                )[0],
                429,
            )
        finally:
            self.server.analysis_lock.release()

    def test_manifest_is_allowlisted_and_roi_is_validated(self):
        status, data = self.request(
            "POST",
            "/api/analyze-evidence",
            json.dumps({"id": "test"}),
            {"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["source"]["id"], "test")
        self.assertEqual(data["provenance"]["drain_roi"], [0.0, 0.0, 1.0, 1.0])
        for request in (
            {"id": "../private.txt"},
            {"id": "http://localhost/private"},
            {"id": "test", "url": "https://example.com"},
            [],
            {"id": "test", "drain_roi": [1, 0, 0, 1]},
        ):
            status, _ = self.request(
                "POST",
                "/api/analyze-evidence",
                json.dumps(request),
                {"Content-Type": "application/json"},
            )
            self.assertIn(status, (400, 404))
        status, _ = self.request(
            "POST",
            "/api/analyze-image",
            photo(),
            {"Content-Type": "image/png", "X-Drain-ROI": "[0,0,2,1]"},
        )
        self.assertEqual(status, 400)

    def test_video_sampling_uses_actual_time_and_limits(self):
        import cv2
        import numpy as np

        video = self.root / "test.mp4"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64)
        )
        self.assertTrue(writer.isOpened())
        for shade in range(20):
            writer.write(np.full((64, 64, 3), shade * 10, np.uint8))
        writer.release()
        status, data = self.request(
            "POST",
            "/api/analyze-clip",
            video.read_bytes(),
            {"Content-Type": "video/mp4"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["media"]["sampled_frames"], 12)
        self.assertAlmostEqual(data["media"]["duration_s"], 2, places=2)
        self.assertEqual(data["provenance"]["sample_times_s"][0], 0)
        self.assertAlmostEqual(data["provenance"]["sample_times_s"][-1], 1.9, places=2)
        for mime in ["video/mp4", "video/webm"]:
            status, _ = self.request(
                "POST", "/api/analyze-clip", b"not video", {"Content-Type": mime}
            )
            self.assertEqual(status, 400)
        video.unlink()

    def test_mask_overlap_never_claims_confirmed_blockage(self):
        roi = validate_roi([0.25, 0.25, 0.75, 0.75])
        mask = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]
        result = drain_overlap([{"hazard": "waste", "mask": mask}], roi)
        self.assertGreater(result["waste_mask_coverage"], 0.95)
        self.assertEqual(result["status"], "potential_obstruction")
        self.assertEqual(
            drain_overlap([{"hazard": "water", "mask": mask}], roi)[
                "waste_mask_coverage"
            ],
            0,
        )
        self.assertIsNone(drain_overlap([], None))
        for value in (
            [True, 0, 1, 1],
            [0, 0, float("nan"), 1],
            [0, 0, 0, 1],
            {"left": 0},
        ):
            with self.assertRaises(InvalidMedia):
                validate_roi(value)

    def test_camera_capture_routes_model_results_and_rejects_unknown_ids(self):
        record = {
            "id": "approved-camera",
            "sourceUrl": "https://example.org/operator",
            "recordedAt": None,
        }
        with patch(
            "live_evidence.fetch_camera",
            return_value=(
                record,
                photo(),
                "image/png",
                {"retrieved_at": "2026-11-14T00:00:00Z"},
            ),
        ):
            status, result = self.request(
                "POST",
                "/api/analyze-camera",
                json.dumps({"id": "approved-camera"}),
                {"Content-Type": "application/json"},
            )
        self.assertEqual(status, 200)
        self.assertEqual(result["source"]["id"], "approved-camera")
        self.assertEqual(result["capture"]["retrieved_at"], "2026-11-14T00:00:00Z")
        status, _ = self.request(
            "POST",
            "/api/analyze-camera",
            json.dumps({"id": "https://localhost/private"}),
            {"Content-Type": "application/json"},
        )
        self.assertEqual(status, 404)
        self.assertEqual(self.request(path="/api/camera-frame?id=unapproved")[0], 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
