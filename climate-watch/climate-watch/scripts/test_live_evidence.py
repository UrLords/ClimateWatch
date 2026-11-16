"""Allowlist and bounded live camera capture checks, with no external network."""

import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climatewatch.evidence import InvalidMedia
from live_evidence import (
    camera_record,
    fetch_camera,
    scope_for,
    validate_url,
    hls_sample,
)


class LiveCameraTests(unittest.TestCase):
    def test_hosts_credentials_and_paths_cannot_escape_camera(self):
        base = "https://video.deldot.gov/live/SCAM098.stream/playlist.m3u8"
        scope = "/live/SCAM098.stream/"
        validate_url(base, base, scope)
        for url in [
            "http://video.deldot.gov/live/SCAM098.stream/a.ts",
            "https://127.0.0.1/private",
            "https://video.deldot.gov:8443/live/SCAM098.stream/a.ts",
            "https://user@video.deldot.gov/live/SCAM098.stream/a.ts",
            "https://video.deldot.gov/live/SCAM098.stream/%2e%2e/a.ts",
            "https://video.deldot.gov/live/OTHER.stream/a.ts",
        ]:
            with self.subTest(url=url), self.assertRaises(InvalidMedia):
                validate_url(url, base, scope)
        with self.assertRaises(InvalidMedia):
            scope_for(
                {"streamUrl": "https://s3-eu-west-1.amazonaws.com/private-bucket/a.mp4"}
            )

    def test_hls_fetches_only_two_recent_segments_and_rejects_encryption(self):
        base = "https://video.deldot.gov/live/SCAM098.stream/playlist.m3u8"
        master = b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=100\nchunk.m3u8\n"
        playlist = b"#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:20\n#EXTINF:2,\nold.ts\n#EXTINF:2,\nnew1.ts\n#EXTINF:2,\nnew2.ts\n"
        fetched = []

        def fake(url, *args):
            fetched.append(url)
            if url.endswith("playlist.m3u8"):
                return master, {}
            if url.endswith("chunk.m3u8"):
                return playlist, {}
            return b"media", {}

        with patch("live_evidence.fetch_bytes", side_effect=fake):
            body, meta = hls_sample(base, "/live/SCAM098.stream/")
        self.assertEqual(body, b"mediamedia")
        self.assertEqual(meta["segments_sampled"], 2)
        self.assertEqual(meta["media_sequence"], 20)
        self.assertFalse(any(url.endswith("old.ts") for url in fetched))
        with patch(
            "live_evidence.fetch_bytes",
            return_value=(b"#EXTM3U\n#EXT-X-KEY:METHOD=AES\nfile.ts\n", {}),
        ):
            with self.assertRaises(InvalidMedia):
                hls_sample(base, "/live/SCAM098.stream/")

    def test_unknown_camera_never_triggers_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "live-cameras").mkdir()
            (root / "live-cameras/cameras.json").write_text(
                json.dumps(
                    [
                        {
                            "id": "known",
                            "mediaType": "video",
                            "streamUrl": "https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/test.mp4",
                        }
                    ]
                )
            )
            with patch(
                "live_evidence.fetch_bytes",
                return_value=(
                    b"video",
                    {"Last-Modified": "Fri, 13 Nov 2026 10:00:00 GMT"},
                ),
            ) as fetch:
                with self.assertRaises(InvalidMedia):
                    fetch_camera(root, "https://localhost/private")
                fetch.assert_not_called()
                record, body, mime, meta = fetch_camera(root, "known")
                self.assertEqual(mime, "video/mp4")
                self.assertEqual(body, b"video")
                self.assertFalse(meta["capture_time_known"])
                self.assertIn("retrieved_at", meta)


if __name__ == "__main__":
    unittest.main(verbosity=2)
