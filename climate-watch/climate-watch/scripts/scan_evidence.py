"""Reproduce bundled model results from the licensed local evidence manifest."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from climatewatch.evidence import get_engine
from serve_prototype import evidence_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--id", action="append", help="Scan only these manifest identifiers."
    )
    args = parser.parse_args()
    public = ROOT / "frontend" / "public"
    output = public / "evidence" / "results"
    output.mkdir(exist_ok=True)
    records = json.loads(
        (public / "evidence" / "sources.json").read_text(encoding="utf-8-sig")
    )
    camera_manifest = public / "europe-cameras" / "cameras.json"
    if camera_manifest.exists():
        records += json.loads(camera_manifest.read_text(encoding="utf-8-sig"))
    engine = get_engine()
    for record in records:
        if args.id and record["id"] not in args.id:
            continue
        record, body, mime = evidence_source(public, record["id"])
        roi = record.get("drain_roi")
        if isinstance(roi, dict):
            roi = roi.get("xyxy")
        result = engine.analyze(body, mime, drain_roi=roi, archived=True)
        result["source"] = {
            key: record.get(key)
            for key in ("id", "sourceUrl", "recordedAt", "footageKind")
        }
        result["provenance"]["precomputed"] = True
        result["provenance"]["capture_mode"] = (
            "bundled dated source; rescan does not make an archive live"
        )
        (output / (record["id"] + ".json")).write_text(
            json.dumps(result, separators=(",", ":")), encoding="utf-8"
        )
        print(
            record["id"],
            "frames=",
            len(result["frames"]),
            "hazards=",
            result["summary"]["hazards"],
            "seconds=",
            result["provenance"]["elapsed_s"],
            flush=True,
        )


if __name__ == "__main__":
    main()
