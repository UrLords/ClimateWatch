"""Local ClimateWatch preview and real frame/clip inference; never a public API server."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
from urllib.parse import unquote, urlsplit, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from climatewatch.evidence import (
    MAX_BYTES,
    MAX_CLIP_BYTES,
    InvalidMedia,
    decode_image,
    get_engine,
    model_status,
    validate_roi,
)


def evidence_source(directory, identifier):
    if not isinstance(identifier, str) or len(identifier) > 100:
        raise InvalidMedia("Choose a known evidence identifier.")
    for manifest in ("evidence/sources.json", "europe-cameras/cameras.json"):
        file = directory / manifest
        if not file.is_file():
            continue
        records = json.loads(file.read_text(encoding="utf-8-sig"))
        for record in records:
            if record.get("id") != identifier:
                continue
            media_path = record.get("mediaPath") or record.get("media", {}).get("url")
            if not isinstance(media_path, str) or not media_path.startswith(
                ("/evidence/", "/europe-cameras/")
            ):
                raise InvalidMedia("This source has no approved local media.", 404)
            target = (directory / media_path.lstrip("/")).resolve()
            if not target.is_relative_to(directory) or not target.is_file():
                raise InvalidMedia("Source media is not available.", 404)
            if target.stat().st_size > MAX_CLIP_BYTES:
                raise InvalidMedia("Source media exceeds the scan size limit.", 413)
            mime = mimetypes.guess_type(target.name)[0]
            if mime not in ("image/jpeg", "image/png", "video/mp4", "video/webm"):
                raise InvalidMedia("Unsupported source media format.", 415)
            return record, target.read_bytes(), mime
    raise InvalidMedia("Unknown evidence identifier.", 404)


class PrototypeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int = 8766, directory: Path | None = None, engine=None):
        self.directory = (directory or ROOT / "frontend" / "dist").resolve()
        self.analysis_lock = threading.Lock()
        self.engine = engine
        self.monitor = None
        super().__init__(("127.0.0.1", port), PrototypeHandler)

    def server_close(self):
        if self.monitor:
            self.monitor.stop()
        super().server_close()


class PrototypeHandler(BaseHTTPRequestHandler):
    server_version = "ClimateWatchLocal/1"

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *_args):
        pass

    def reply(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def json(self, status: int, payload: dict):
        self.reply(
            status,
            json.dumps(payload, allow_nan=False).encode(),
            "application/json; charset=utf-8",
        )

    def local_request(self) -> bool:
        port = self.server.server_port
        host = self.headers.get("Host", "")
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if host not in hosts:
            self.json(403, {"error": "Use this server through its local address."})
            return False
        return True

    def do_GET(self):
        if not self.local_request():
            return
        path = urlsplit(self.path).path
        if path == "/api/health":
            self.json(
                200,
                {
                    "service": "ClimateWatch local prototype",
                    "status": "ready",
                    "capabilities": {
                        "hazard_detection": all(m["available"] for m in model_status()),
                        "instance_segmentation": True,
                        "clip_scanning": True,
                        "drain_roi_overlap": True,
                    },
                    "models": model_status(),
                    "supported_hazards": ["smoke", "fire", "waste"],
                    "automatic_monitoring": bool(self.server.monitor),
                    "detectors_available": all(m["available"] for m in model_status()),
                    "limits": {
                        "image_bytes": MAX_BYTES,
                        "clip_bytes": MAX_CLIP_BYTES,
                        "clip_seconds": 120,
                        "sampled_frames": 12,
                    },
                    "raw_upload_retention": "none",
                    "retention": {
                        "raw_uploads": "none",
                        "raw_camera_clips": "none",
                        "confirmed_evidence": "latest privacy-filtered model evidence per camera, stored locally",
                    },
                },
            )
            return
        if path == "/api/monitor":
            self.json(
                200,
                self.server.monitor.snapshot()
                if self.server.monitor
                else {"enabled": False, "cameras": [], "events": []},
            )
            return
        if path == "/api/camera-evidence":
            identifier = parse_qs(urlsplit(self.path).query).get("id", [None])[0]
            result = (
                self.server.monitor.result(identifier) if self.server.monitor else None
            )
            self.json(200, result or {"frames": [], "summary": {"hazards": []}})
            return
        if path == "/api/camera-frame":
            from live_evidence import camera_record, fetch_camera

            try:
                identifier = parse_qs(urlsplit(self.path).query).get("id", [None])[0]
                record = camera_record(self.server.directory, identifier)
                if record.get("mediaType") != "image":
                    raise InvalidMedia("Use direct playback for video cameras.")
                _record, body, mime, _metadata = fetch_camera(
                    self.server.directory, identifier
                )
                decode_image(body, mime)
                self.reply(200, body, mime)
            except InvalidMedia as error:
                self.json(error.status, {"error": str(error)})
            return
        if path.startswith("/api/"):
            self.json(404, {"error": "Unknown API endpoint."})
            return
        try:
            decoded = unquote(path, errors="strict")
            if (
                "\\" in decoded
                or "\x00" in decoded
                or any(p in (".", "..") for p in decoded.split("/"))
            ):
                raise ValueError
            relative = decoded.lstrip("/") or "index.html"
            target = (self.server.directory / relative).resolve()
            if not target.is_relative_to(self.server.directory) or not target.is_file():
                raise ValueError
            body = target.read_bytes()
        except (ValueError, OSError):
            self.json(
                404,
                {
                    "error": "File not found. Build the frontend before starting the preview."
                },
            )
            return
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.reply(200, body, mime)

    do_HEAD = do_GET

    def do_POST(self):
        if not self.local_request():
            return
        endpoint = urlsplit(self.path).path
        if endpoint not in (
            "/api/analyze-image",
            "/api/analyze-clip",
            "/api/analyze-evidence",
            "/api/analyze-camera",
        ):
            self.json(404, {"error": "Unknown API endpoint."})
            return
        origin = self.headers.get("Origin")
        if (origin and origin != f"http://{self.headers['Host']}") or self.headers.get(
            "Sec-Fetch-Site"
        ) == "cross-site":
            self.json(403, {"error": "Scans must come from the same local origin."})
            return
        if (
            self.headers.get("Transfer-Encoding")
            or len(self.headers.get_all("Content-Length", [])) != 1
        ):
            self.json(411, {"error": "A single Content-Length header is required."})
            return
        try:
            size = int(self.headers["Content-Length"])
        except ValueError:
            self.json(400, {"error": "Invalid Content-Length."})
            return
        limit = {
            "/api/analyze-image": MAX_BYTES,
            "/api/analyze-clip": MAX_CLIP_BYTES,
            "/api/analyze-evidence": 4096,
            "/api/analyze-camera": 4096,
        }[endpoint]
        if size <= 0 or size > limit:
            self.json(
                413 if size > limit else 400,
                {"error": f"Send a nonempty request of at most {limit} bytes."},
            )
            return
        content_type = self.headers.get_content_type()
        accepted = {
            "/api/analyze-image": ("image/jpeg", "image/png"),
            "/api/analyze-clip": ("video/mp4", "video/webm"),
            "/api/analyze-evidence": ("application/json",),
            "/api/analyze-camera": ("application/json",),
        }[endpoint]
        if content_type not in accepted:
            self.json(415, {"error": "Accepted Content-Type: " + ", ".join(accepted)})
            return
        if not self.server.analysis_lock.acquire(blocking=False):
            self.json(429, {"error": "Another scan is running. Try again shortly."})
            return
        try:
            body = self.rfile.read(size)
            if len(body) != size:
                raise InvalidMedia("Incomplete upload.")
            roi = (
                validate_roi(json.loads(self.headers["X-Drain-ROI"]))
                if self.headers.get("X-Drain-ROI")
                else None
            )
            archived, record, capture_metadata = False, None, None
            if endpoint in ("/api/analyze-evidence", "/api/analyze-camera"):
                request = json.loads(body)
                if not isinstance(request, dict) or set(request) - {
                    "id",
                    "drain_roi",
                    "refresh",
                }:
                    raise InvalidMedia(
                        "Use an evidence id with optional drain_roi or refresh."
                    )
                if endpoint == "/api/analyze-camera":
                    from live_evidence import fetch_camera

                    record, body, content_type, capture_metadata = fetch_camera(
                        self.server.directory, request.get("id")
                    )
                else:
                    record, body, content_type = evidence_source(
                        self.server.directory, request.get("id")
                    )
                    archived = not record.get("liveSource")
                    if request.get("refresh"):
                        raise InvalidMedia(
                            "Use the live camera scan endpoint to refresh an approved camera.",
                            400,
                        )
                default_roi = record.get("drain_roi")
                if isinstance(default_roi, dict):
                    default_roi = default_roi.get("xyxy")
                roi = validate_roi(request.get("drain_roi", default_roi))
            if content_type.startswith("image/"):
                decode_image(body, content_type)
            engine = self.server.engine or get_engine()
            result = engine.analyze(
                body, content_type, drain_roi=roi, archived=archived
            )
            if record:
                result["source"] = {
                    "id": record["id"],
                    "sourceUrl": record.get("sourceUrl"),
                    "recordedAt": record.get("recordedAt"),
                    "footageKind": record.get("footageKind"),
                }
            if capture_metadata:
                result["capture"] = capture_metadata
            self.json(200, result)
        except InvalidMedia as error:
            self.json(error.status, {"error": str(error)})
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.json(400, {"error": "Use valid JSON."})
        except TimeoutError:
            self.json(408, {"error": "The upload timed out."})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            print(f"Scan failed: {type(error).__name__}: {error}", file=sys.stderr)
            self.json(
                500,
                {
                    "error": "The model scan failed. Check the local server output and model files."
                },
            )
        finally:
            self.server.analysis_lock.release()

    def unsupported(self):
        self.json(
            405, {"error": "Only GET, HEAD and the scan POST endpoints are supported."}
        )

    do_OPTIONS = do_PUT = do_PATCH = do_DELETE = unsupported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable background checks for isolated API tests.",
    )
    args = parser.parse_args()
    with PrototypeServer(args.port) as server:
        if not args.no_monitor:
            from camera_monitor import CameraMonitor

            server.monitor = CameraMonitor(
                server.directory,
                server.analysis_lock,
                ROOT / "data" / "prototype-evidence.sqlite",
            )
            server.monitor.start()
        print(
            f"ClimateWatch local preview: http://127.0.0.1:{server.server_port}",
            flush=True,
        )
        print(
            "Automatic camera checks enabled."
            if server.monitor
            else "Automatic checks disabled.",
            flush=True,
        )
        print(
            "Only persistent smoke, fire, or waste creates evidence. Raw footage is not retained.",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if server.monitor:
                server.monitor.stop()
