"""Verify owner-published camera transport, without retaining camera footage."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

import cv2

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "tmc.deldot.gov",
    "video.deldot.gov",
    "api.tfl.gov.uk",
    "s3-eu-west-1.amazonaws.com",
}
DELDOT_IDS = [
    "SCAM098",
    "SCAM099",
    "NCAM256",
    "NCAM249",
    "NCAM263",
    "NCAM292",
    "NCAM344",
    "KCAM110",
    "KCAM003",
    "KCAM001",
]
CATALOG = "https://tmc.deldot.gov/json/videocamera.json"


def fetch(url, limit=2_000_000):
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED
        or parsed.port not in (None, 443)
    ):
        raise ValueError("Only verified official HTTPS hosts are permitted.")
    with urlopen(
        Request(url, headers={"User-Agent": "ClimateWatch/0.2 camera-research"}),
        timeout=15,
    ) as response:
        if urlsplit(response.url).hostname not in ALLOWED:
            raise ValueError("Unexpected redirect host.")
        body = response.read(limit + 1)
        if len(body) > limit:
            raise ValueError("Response exceeded bounded research limit.")
        return body, dict(response.headers)


def decode_check(body, suffix):
    with tempfile.TemporaryDirectory(prefix="climatewatch-camera-") as folder:
        path = Path(folder) / ("sample" + suffix)
        path.write_bytes(body)
        cap = cv2.VideoCapture(str(path))
        try:
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Video sample could not decode.")
            return {
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "sampleDecoded": True,
            }
        finally:
            cap.release()


def media_playlist(stream):
    master, headers = fetch(stream)
    text = master.decode("utf8")
    if not text.startswith("#EXTM3U"):
        raise ValueError("Invalid HLS manifest.")
    variant = next(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    )
    playlist_url = urljoin(stream, variant)
    playlist = fetch(playlist_url)[0].decode("utf8")
    if "#EXT-X-ENDLIST" in playlist or "#EXT-X-KEY" in playlist:
        raise ValueError("Stream is ended or encrypted.")
    return playlist_url, playlist, headers


def hls_check(camera):
    stream = camera["urls"]["m3u8s"]
    playlist_url, playlist, headers = media_playlist(stream)
    sequence = int(re.search(r"#EXT-X-MEDIA-SEQUENCE:(\d+)", playlist).group(1))
    target_duration = int(re.search(r"#EXT-X-TARGETDURATION:(\d+)", playlist).group(1))
    segment = next(
        line.strip()
        for line in reversed(playlist.splitlines())
        if line.strip() and not line.startswith("#")
    )
    dimensions = decode_check(
        fetch(urljoin(playlist_url, segment), 8_000_000)[0], ".ts"
    )
    record = {
        "id": "deldot-" + camera["id"].lower(),
        "title": camera["title"],
        "name": camera["title"],
        "city": camera["county"] + " County, Delaware",
        "country": "United States",
        "continent": "North America",
        "coordinates": [camera["lon"], camera["lat"]],
        "locationAccuracy": "Official DelDOT camera catalog coordinates.",
        "topic": "visibility",
        "mode": "live",
        "mediaType": "hls",
        "streamUrl": stream,
        "sourceUrl": "https://deldot.gov/map/index.shtml?tab=TrafficCameras",
        "metadataUrl": CATALOG,
        "sourceName": "Delaware Department of Transportation",
        "credit": "DelDOT — Delaware Department of Transportation",
        "license": "Owner-published public live traffic stream; blanket redistribution licence not established.",
        "rightsStatus": "public-viewing-confirmed; hosted-redistribution-unconfirmed",
        "recordedAt": None,
        "footageKind": "Continuous HLS traffic camera video",
        "hazardStatus": "unassessed",
        "verification": {
            "checkedAt": datetime.now(timezone.utc).isoformat(),
            "cors": headers.get("Access-Control-Allow-Origin"),
            "initialMediaSequence": sequence,
            "targetDurationSeconds": target_duration,
            "playlistUrlAtCheck": playlist_url,
            **dimensions,
        },
        "limitations": [
            "Camera may pan or be temporarily unavailable.",
            "No environmental hazard is asserted without a scan and review.",
            "Capture UTC time has not been independently verified; sequence advance establishes ongoing publication.",
            "Use direct owner playback locally; public redistribution rights need confirmation.",
        ],
    }
    print("Decoded HLS", camera["id"], dimensions, flush=True)
    return record


def main():
    catalog = json.loads(fetch(CATALOG)[0])
    rows = {
        c["id"]: c
        for c in catalog["videoCameras"]
        if c.get("enabled") and c.get("status") == "Active"
    }
    for id in DELDOT_IDS:
        if id not in rows:
            print("Unavailable in official catalog:", id, flush=True)
    candidates = [rows[id] for id in DELDOT_IDS if id in rows]
    candidates.extend(c for id, c in rows.items() if id not in DELDOT_IDS)
    candidates = candidates[:15]
    records = []
    for batch in (candidates[:10], candidates[10:]):
        if len(records) >= 10:
            break
        decoded = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            pending = [
                (camera["id"], pool.submit(hls_check, camera)) for camera in batch
            ]
            for id, future in pending:
                try:
                    decoded.append(future.result())
                except Exception as error:
                    print(
                        "Decode failed:",
                        id,
                        type(error).__name__,
                        str(error),
                        flush=True,
                    )

        if decoded:
            time.sleep(
                min(
                    45,
                    max(r["verification"]["targetDurationSeconds"] for r in decoded)
                    + 2,
                )
            )
        for record in decoded:
            try:
                verification = record["verification"]
                playlist_url, playlist, _ = media_playlist(record["streamUrl"])
                current = int(
                    re.search(r"#EXT-X-MEDIA-SEQUENCE:(\d+)", playlist).group(1)
                )
                if current <= verification["initialMediaSequence"]:
                    raise ValueError("Media sequence did not advance.")
                verification.update(
                    laterMediaSequence=current,
                    sequenceAdvanced=True,
                    laterPlaylistUrlAtCheck=playlist_url,
                )
                records.append(record)
            except Exception as error:
                print(
                    "Live progression failed:",
                    record["id"],
                    type(error).__name__,
                    str(error),
                    flush=True,
                )
    records = records[:10]
    if len(records) < 10:
        raise SystemExit(
            f"Only {len(records)}/10 live streams verified after at most 15 candidates. Existing manifest left unchanged."
        )
    tfl = json.loads(fetch("https://api.tfl.gov.uk/Place/Type/JamCam", 4_000_000)[0])
    clips = []
    attempted_clips = 0
    for camera in tfl:
        properties = {p["key"]: p["value"] for p in camera["additionalProperties"]}
        if (
            properties.get("available") != "true"
            or "Bridge" not in camera["commonName"]
            or not properties.get("videoUrl")
        ):
            continue
        attempted_clips += 1
        try:
            body, headers = fetch(properties["videoUrl"], 8_000_000)
            dimensions = decode_check(body, ".mp4")
        except Exception as error:
            print("Skipped TfL", camera["id"], type(error).__name__, flush=True)
            if attempted_clips >= 15:
                break
            continue
        clips.append(
            {
                "id": "tfl-" + camera["id"].removeprefix("JamCams_"),
                "title": camera["commonName"],
                "name": camera["commonName"],
                "city": "London",
                "country": "United Kingdom",
                "continent": "Europe",
                "coordinates": [camera["lon"], camera["lat"]],
                "locationAccuracy": "Official TfL JamCam coordinates.",
                "topic": "visibility",
                "mode": "rolling-clip",
                "mediaType": "video",
                "streamUrl": properties["videoUrl"],
                "imageUrl": properties.get("imageUrl"),
                "sourceUrl": "https://tfl.gov.uk/info-for/open-data-users/our-open-data",
                "metadataUrl": "https://api.tfl.gov.uk/Place/" + camera["id"],
                "sourceName": "Transport for London",
                "credit": "Powered by TfL Open Data. Contains OS data © Crown copyright and database rights 2016; Geomni UK Map data © and database rights 2019.",
                "license": "TfL Transport Data Service licence",
                "licenseUrl": "https://tfl.gov.uk/corporate/terms-and-conditions/transport-data-service",
                "recordedAt": None,
                "footageKind": "Short CCTV video clip replaced by the provider periodically; not continuous live video.",
                "refreshSeconds": 300,
                "hazardStatus": "unassessed",
                "verification": {
                    "checkedAt": datetime.now(timezone.utc).isoformat(),
                    "lastModified": headers.get("Last-Modified"),
                    "etag": headers.get("ETag"),
                    **dimensions,
                },
                "limitations": [
                    "Short periodically replaced footage, not continuous realtime video.",
                    "No environmental hazard is asserted.",
                    "Last-Modified is the provider object update time; exact capture timestamp is not verified.",
                    "Credit TfL, preserve notices, and do not imply endorsement.",
                ],
            }
        )
        print("Decoded European rolling clip", camera["commonName"], flush=True)
        if len(clips) == 5:
            break
        if attempted_clips >= 15:
            break
    if len(clips) < 5:
        print(
            f"Only {len(clips)}/5 European rolling clips verified; unavailable clips are not presented as current.",
            flush=True,
        )
    records.extend(clips)
    assert len(records) >= 10 and all(len(c["coordinates"]) == 2 for c in records)
    output = ROOT / "frontend/public/live-cameras/cameras.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf8"
    )
    print(
        "Saved",
        len(records),
        "records:",
        len(DELDOT_IDS),
        "advancing HLS,",
        len(clips),
        "rolling clips.",
        flush=True,
    )


if __name__ == "__main__":
    main()
