"""Bounded retrieval from published, allowlisted camera records; no user-supplied URLs."""

from datetime import datetime, timezone
from pathlib import Path
import json
import re
from urllib.parse import urljoin, urlsplit, unquote
from urllib.request import Request, HTTPRedirectHandler, build_opener
from urllib.error import URLError, HTTPError

from climatewatch.evidence import InvalidMedia, MAX_CLIP_BYTES

MANIFESTS = ("live-cameras/cameras.json", "europe-cameras/cameras.json")


def camera_record(directory: Path, identifier: str):
    if not isinstance(identifier, str) or len(identifier) > 100:
        raise InvalidMedia("Choose a known camera identifier.")
    for name in MANIFESTS:
        path = directory / name
        if path.is_file():
            for record in json.loads(path.read_text(encoding="utf-8-sig")):
                if record.get("id") == identifier:
                    return record
    raise InvalidMedia("Unknown camera identifier.", 404)


def scope_for(record):
    url = record.get("streamUrl") or record.get("liveSource", {}).get("imageUrl")
    parsed = urlsplit(url or "")
    if parsed.hostname == "video.deldot.gov" and re.fullmatch(
        r"/live/[A-Z0-9]+\.stream/playlist\.m3u8", parsed.path
    ):
        return url, parsed.path.rsplit("/", 1)[0] + "/"
    if parsed.hostname == "s3-eu-west-1.amazonaws.com" and re.fullmatch(
        r"/jamcams\.tfl\.gov\.uk/[\w.-]+\.mp4", parsed.path
    ):
        return url, parsed.path
    if parsed.hostname == "weathercam.digitraffic.fi" and re.fullmatch(
        r"/C\d+\.jpg", parsed.path
    ):
        return url, parsed.path
    raise InvalidMedia("Camera provider or URL is not approved for capture.", 400)


def validate_url(url, base, path_scope):
    candidate, origin = urlsplit(url), urlsplit(base)
    if (
        candidate.scheme != "https"
        or candidate.hostname != origin.hostname
        or candidate.port not in (None, 443)
        or candidate.username
        or candidate.password
    ):
        raise InvalidMedia("Camera media left the approved HTTPS origin.", 502)
    path = unquote(candidate.path)
    if (
        "\\" in path
        or any(part in (".", "..") for part in path.split("/"))
        or not (
            path.startswith(path_scope)
            if path_scope.endswith("/")
            else path == path_scope
        )
    ):
        raise InvalidMedia("Camera media left its approved path.", 502)


def fetch_bytes(url, base, scope, limit):
    validate_url(url, base, scope)

    class ApprovedRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            validate_url(newurl, base, scope)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    try:
        request = Request(
            url,
            headers={
                "User-Agent": "ClimateWatch/0.2 local-camera-viewer",
                "Digitraffic-User": "ClimateWatch/hackathon-prototype",
            },
        )
        with build_opener(ApprovedRedirect).open(request, timeout=12) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > limit:
                raise InvalidMedia("Camera response exceeds the capture limit.", 413)
            body = response.read(limit + 1)
            if len(body) > limit:
                raise InvalidMedia("Camera response exceeds the capture limit.", 413)
            return body, dict(response.headers)
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise InvalidMedia(
            "The camera operator is unavailable. Try another camera or retry shortly.",
            502,
        ) from error


def hls_sample(url, scope):
    playlist_url = url
    for _ in range(3):
        body, _headers = fetch_bytes(playlist_url, url, scope, 128_000)
        try:
            text = body.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise InvalidMedia(
                "Camera returned an invalid HLS playlist.", 502
            ) from error
        if (
            not text.startswith("#EXTM3U")
            or "#EXT-X-KEY" in text
            or "#EXT-X-MAP" in text
        ):
            raise InvalidMedia(
                "This camera format is not supported for local capture.", 502
            )
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if not lines:
            raise InvalidMedia("Camera playlist has no media.", 502)
        if "#EXT-X-STREAM-INF" in text:
            playlist_url = urljoin(playlist_url, lines[0])
            continue
        if "#EXT-X-ENDLIST" in text:
            raise InvalidMedia("The camera stream has ended.", 502)
        sequence = re.search(r"#EXT-X-MEDIA-SEQUENCE:(\d+)", text)
        segments = lines[-2:]
        media = bytearray()
        for segment in segments:
            target = urljoin(playlist_url, segment)
            if not urlsplit(target).path.endswith(".ts"):
                raise InvalidMedia("Only MPEG-TS camera segments are supported.", 502)
            media.extend(
                fetch_bytes(
                    target, url, scope, min(12_000_000, MAX_CLIP_BYTES - len(media))
                )[0]
            )
        return bytes(media), {
            "media_sequence": int(sequence.group(1)) if sequence else None,
            "segments_sampled": len(segments),
            "capture_time_known": False,
        }
    raise InvalidMedia("Camera playlist nesting exceeds the capture limit.", 502)


def fetch_camera(directory: Path, identifier: str):
    record = camera_record(directory, identifier)
    url, scope = scope_for(record)
    if record.get("mediaType") == "hls":
        body, metadata = hls_sample(url, scope)
        mime = "video/mp2t"
    else:
        body, headers = fetch_bytes(
            url,
            url,
            scope,
            MAX_CLIP_BYTES if record.get("mediaType") == "video" else 8 * 1024 * 1024,
        )
        mime = "video/mp4" if record.get("mediaType") == "video" else "image/jpeg"
        metadata = {
            "provider_last_modified": headers.get("Last-Modified"),
            "capture_time_known": False,
        }
    metadata.update(
        {
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "source_url": url,
            "transport": record.get("mediaType", "image"),
        }
    )
    return record, body, mime, metadata
