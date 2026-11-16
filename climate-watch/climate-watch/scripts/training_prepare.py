"""Prepare a bounded, batch-disjoint TACO segmentation experiment from human annotations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from ultralytics.data.converter import merge_multi_segment

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".local" / "training"
DATA = WORK / "taco-small-v1"
SEED = 31031
ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
)
LICENSE_URL = "https://zenodo.org/api/records/3587843"


def download(url, limit=10_000_000):
    with urlopen(
        Request(url, headers={"User-Agent": "ClimateWatch-research/1.0"}), timeout=25
    ) as response:
        body = response.read(limit + 1)
    if len(body) > limit:
        raise ValueError("Download exceeded the experiment limit.")
    return body


def prepare():
    WORK.mkdir(exist_ok=True, parents=True)
    annotations = WORK / "taco-annotations.json"
    if not annotations.exists():
        annotations.write_bytes(download(ANNOTATIONS_URL))
    raw = annotations.read_bytes()
    annotation = json.loads(raw)
    license_record = json.loads(download(LICENSE_URL))
    assert license_record["metadata"]["license"]["id"] == "cc-by-4.0"
    (WORK / "taco-publisher-license.json").write_text(
        json.dumps(license_record, indent=2)
    )
    grouped = {}
    for image in annotation["images"]:
        grouped.setdefault(image["file_name"].split("/")[0], []).append(image)
    groups = sorted(grouped)
    rng = random.Random(SEED)
    rng.shuffle(groups)
    assignments = {"train": groups[:9], "val": groups[9:12], "test": groups[12:]}
    targets = {"train": 180, "val": 30, "test": 40}
    ann_by_image = {}
    for ann in annotation["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)
    records, hashes = [], {}
    failures = []

    def fetch(image):
        url = image["flickr_640_url"]
        if (
            urlsplit(url).hostname != "img.tacodataset.org"
            or urlsplit(url).scheme != "https"
        ):
            raise ValueError("Unexpected image host.")
        path = WORK / "download-cache" / f"{image['id']}.jpg"
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            path.write_bytes(download(url, 2_000_000))
        body = path.read_bytes()
        with Image.open(io.BytesIO(body)) as photo:
            photo.load()
            assert (
                abs(photo.width / photo.height - image["width"] / image["height"])
                < 0.025
            )
            pixels = np.asarray(photo.convert("L").resize((9, 8)))
            dhash = sum(
                int(v) << i
                for i, v in enumerate((pixels[:, 1:] > pixels[:, :-1]).ravel())
            )
        return image, body, dhash

    for split, split_groups in assignments.items():
        for kind in ("images", "labels"):
            (DATA / kind / split).mkdir(parents=True, exist_ok=True)
        pool = [image for group in split_groups for image in grouped[group]]
        rng.shuffle(pool)
        accepted = 0
        for offset in range(0, len(pool), 16):
            with ThreadPoolExecutor(max_workers=4) as executor:
                jobs = [
                    (image, executor.submit(fetch, image))
                    for image in pool[offset : offset + 16]
                ]
                for original, job in jobs:
                    if accepted >= targets[split]:
                        break
                    try:
                        image, body, dhash = job.result()
                        digest = hashlib.sha256(body).hexdigest()
                        if digest in hashes or any(
                            other_split != split
                            and (dhash ^ other_hash).bit_count() <= 3
                            for other_split, other_hash in hashes.values()
                        ):
                            failures.append(
                                {
                                    "id": image["id"],
                                    "reason": "duplicate or cross-split perceptual near-duplicate",
                                }
                            )
                            continue
                        lines = []
                        for ann in ann_by_image.get(image["id"], []):
                            if ann.get("iscrowd") or not isinstance(
                                ann["segmentation"], list
                            ):
                                continue
                            segments = ann["segmentation"]
                            polygon = (
                                np.concatenate(merge_multi_segment(segments), axis=0)
                                if len(segments) > 1
                                else np.array(segments[0]).reshape(-1, 2)
                            )
                            normalized = (
                                polygon / [image["width"], image["height"]]
                            ).clip(0, 1)
                            if len(normalized) < 3:
                                continue
                            lines.append(
                                "0 "
                                + " ".join(
                                    f"{value:.6f}" for value in normalized.ravel()
                                )
                            )
                        if not lines:
                            raise ValueError("No usable human instance polygons.")
                        name = f"taco_{image['id']:04d}"
                        (DATA / "images" / split / (name + ".jpg")).write_bytes(body)
                        (DATA / "labels" / split / (name + ".txt")).write_text(
                            "\n".join(lines) + "\n"
                        )
                        hashes[digest] = (split, dhash)
                        records.append(
                            {
                                "id": image["id"],
                                "split": split,
                                "group": image["file_name"].split("/")[0],
                                "filename": name + ".jpg",
                                "image_url": image["flickr_640_url"],
                                "original_url": image["flickr_url"],
                                "original_size": [image["width"], image["height"]],
                                "sha256": digest,
                                "dhash": str(dhash),
                                "human_instance_count": len(lines),
                                "annotation_ids": [
                                    a["id"] for a in ann_by_image[image["id"]]
                                ],
                                "publisher_license": "CC BY 4.0",
                                "individual_license_field": image.get("license"),
                            }
                        )
                        accepted += 1
                    except Exception as error:
                        failures.append({"id": original["id"], "reason": str(error)})
            print(split, accepted, "/", targets[split], flush=True)
            if accepted >= targets[split]:
                break
        assert accepted == targets[split], f"Incomplete {split}: {accepted}"
    assert not set(assignments["train"]) & set(assignments["test"])
    manifest = {
        "dataset": "TACO official human annotations",
        "seed": SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "annotations_url": ANNOTATIONS_URL,
        "annotations_sha256": hashlib.sha256(raw).hexdigest(),
        "publisher_license_url": "https://zenodo.org/records/3587843",
        "publisher_license": "CC BY 4.0",
        "attribution": "Pedro F. Proença and Pedro Simões, TACO: Trash Annotations in Context, 2020",
        "license_note": "Publisher dataset record declares CC BY 4.0. Individual image license fields in official annotations are blank; no separate per-image license is inferred.",
        "split_method": "Disjoint original TACO batches, fixed seed; SHA256 exact and 64-bit dHash<=3 cross-split duplicate exclusion. Batches are not verified camera/site identities.",
        "class_mapping": "All 60 human-annotated litter categories -> waste (0); disconnected polygons linked with Ultralytics merge_multi_segment.",
        "groups": assignments,
        "counts": targets,
        "records": records,
        "excluded": failures,
        "limitations": [
            "Field-photograph litter domain, not annotated blocked drains or elevated CCTV.",
            "No demo images or model-generated labels used.",
        ],
    }
    (DATA / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (DATA / "dataset.yaml").write_text(
        f"path: {DATA.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: waste\n"
    )
    print(
        "Prepared",
        len(records),
        "images and",
        sum(r["human_instance_count"] for r in records),
        "human polygons",
        flush=True,
    )


if __name__ == "__main__":
    prepare()
