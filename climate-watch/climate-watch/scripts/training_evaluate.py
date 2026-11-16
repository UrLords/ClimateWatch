"""Compare the frozen candidate and existing waste fallback on untouched test data."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from climatewatch.evidence import get_engine

DATA = ROOT / ".local/training/taco-small-v1"
OUT = ROOT / "models/candidate-waste-taco-v1"
SIZE = 256


def boxes_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def truth(image):
    labels = DATA / "labels" / image.parent.name / (image.stem + ".txt")
    result = []
    for line in labels.read_text().splitlines():
        polygon = np.array([float(v) for v in line.split()[1:]]).reshape(-1, 2)
        result.append(
            {
                "box": [*polygon.min(axis=0), *polygon.max(axis=0)],
                "mask": polygon.tolist(),
            }
        )
    return result


def union_mask(items):
    mask = np.zeros((SIZE, SIZE), np.uint8)
    for item in items:
        if len(item.get("mask", [])) >= 3:
            polygon = (
                (np.array(item["mask"]) * (SIZE - 1)).clip(0, SIZE - 1).astype(np.int32)
            )
            cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def score(rows, threshold=0.0):
    tp = fp = fn = 0
    ious = []
    for row in rows:
        predictions = [
            item for item in row["prediction"] if item["confidence"] >= threshold
        ]
        matched = set()
        for item in sorted(
            predictions, key=lambda item: item["confidence"], reverse=True
        ):
            candidates = [
                (boxes_iou(item["box"], target["box"]), j)
                for j, target in enumerate(row["truth"])
                if j not in matched
            ]
            overlap, index = max(candidates, default=(0, -1))
            if overlap >= 0.5:
                tp += 1
                matched.add(index)
            else:
                fp += 1
        fn += len(row["truth"]) - len(matched)
        prediction, target = union_mask(predictions), union_mask(row["truth"])
        union = np.logical_or(prediction, target).sum()
        ious.append(
            float(np.logical_and(prediction, target).sum() / union) if union else 1.0
        )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "f0_5": 1.25 * precision * recall / (0.25 * precision + recall)
        if precision + recall
        else 0.0,
        "mean_union_mask_iou": float(np.mean(ious)),
        "images": len(rows),
    }


def candidate_prediction(model, image):
    result = model.predict(
        str(image), imgsz=416, conf=0.001, max_det=100, device="cpu", verbose=False
    )[0]
    return [
        {
            "box": box.xyxyn[0].cpu().numpy().tolist(),
            "confidence": float(box.conf),
            "mask": result.masks.xyn[i].tolist() if result.masks is not None else [],
        }
        for i, box in enumerate(result.boxes)
    ]


def main():
    os.environ.pop("WASTE_WEIGHTS", None)
    torch.set_num_threads(8)
    model = YOLO(OUT / "run/weights/best.pt")
    assert model.names == {0: "waste"}
    val_rows = [
        {
            "image": path.name,
            "truth": truth(path),
            "prediction": candidate_prediction(model, path),
        }
        for path in sorted((DATA / "images/val").glob("*.jpg"))
    ]
    validation = {
        str(threshold): score(val_rows, threshold)
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    }
    threshold = max(validation, key=lambda key: (validation[key]["f0_5"], float(key)))
    threshold = float(threshold)
    print("Frozen validation-selected confidence", threshold, flush=True)
    engine = get_engine()
    candidate_rows, fallback_rows = [], []
    timings = {"candidate_s": [], "fallback_s": []}
    for path in sorted((DATA / "images/test").glob("*.jpg")):
        targets = truth(path)
        start = time.perf_counter()
        candidate = candidate_prediction(model, path)
        timings["candidate_s"].append(time.perf_counter() - start)
        start = time.perf_counter()
        fallback = [
            d
            for d in engine.frame(cv2.imread(str(path)), 0)["detections"]
            if d["hazard"] == "waste"
        ]
        timings["fallback_s"].append(time.perf_counter() - start)
        candidate_rows.append(
            {"image": path.name, "truth": targets, "prediction": candidate}
        )
        fallback_rows.append(
            {"image": path.name, "truth": targets, "prediction": fallback}
        )
    controls = json.loads(
        (ROOT / ".local/training/camera-controls-manifest.json").read_text()
    )
    negatives = []
    for record in controls:
        if not record.get("included"):
            continue
        path = ROOT / ".local/training/camera-controls" / record["filename"]
        candidate = [
            p for p in candidate_prediction(model, path) if p["confidence"] >= threshold
        ]
        fallback = [
            d
            for d in engine.frame(cv2.imread(str(path)), 0)["detections"]
            if d["hazard"] == "waste"
        ]
        negatives.append(
            {
                "id": record["id"],
                "candidate_detections": len(candidate),
                "fallback_detections": len(fallback),
            }
        )
    demos = []
    for name in ("bali-river-waste", "awka-waste-drain"):
        path = ROOT / "frontend/public/evidence" / (name + ".jpg")
        predicted = [
            p for p in candidate_prediction(model, path) if p["confidence"] >= threshold
        ]
        demos.append(
            {
                "id": name,
                "candidate_detections": len(predicted),
                "note": "Regression observation only; never used for training, threshold selection or numerical test metrics.",
            }
        )
    candidate_metrics, fallback_metrics = (
        score(candidate_rows, threshold),
        score(fallback_rows),
    )
    negative_summary = {
        "images": len(negatives),
        "candidate_false_positive_frames": sum(
            r["candidate_detections"] > 0 for r in negatives
        ),
        "fallback_false_positive_frames": sum(
            r["fallback_detections"] > 0 for r in negatives
        ),
        "per_frame": negatives,
    }
    improved = (
        candidate_metrics["f0_5"] > fallback_metrics["f0_5"] + 0.03
        and candidate_metrics["recall"] >= fallback_metrics["recall"]
        and candidate_metrics["precision"] >= 0.5
        and negative_summary["candidate_false_positive_frames"]
        <= negative_summary["fallback_false_positive_frames"]
        and all(d["candidate_detections"] > 0 for d in demos)
    )
    report = {
        "experiment": "TACO batch-disjoint 180/30/40, single-class waste",
        "validation_thresholds": validation,
        "evaluation_protocol": {
            "instance_match": "Greedy one-to-one box IoU >= 0.5, not mask AP.",
            "mask_metric": "Per-image foreground union-mask IoU, averaged across images, rasterized at 256 square.",
            "candidate_inference": {"imgsz": 416, "max_det": 100},
            "runtime_waste_inference": {"imgsz": 640, "max_det": 40},
        },
        "confidence_threshold_selected_on_val_only": threshold,
        "test_candidate": candidate_metrics,
        "test_candidate_at_existing_integration_threshold_0_5": score(
            candidate_rows, 0.5
        ),
        "test_existing_fallback": fallback_metrics,
        "camera_controls": negative_summary,
        "demo_regression": demos,
        "timing_mean_seconds": {
            key: float(np.mean(values)) for key, values in timings.items()
        },
        "promotion_criteria_passed": improved,
        "deployed": False,
        "decision": "Candidate passes pilot comparison; camera-domain validation still required before activation."
        if improved
        else "Keep existing fallback: candidate did not pass all predefined improvement gates.",
        "limitations": [
            "Only 40 positive field photographs and 10 visually reviewed road-camera negatives.",
            "Candidate timing measures only the waste model; fallback timing includes privacy/environment/fire models and is not a like-for-like speed comparison.",
            "Candidate comparison uses 416-pixel inference; activation at the current 640-pixel runtime settings would require a separate validation.",
            "No blocked-drain or hydraulic-obstruction training labels.",
            "Original batches are separated, but author/site identities are not verified.",
            "All 60 TACO categories merged into waste. This experiment does not establish production CCTV accuracy.",
        ],
    }
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2))
    (ROOT / ".local/training/test-predictions.json").write_text(
        json.dumps(
            {"candidate": candidate_rows, "fallback": fallback_rows},
            default=lambda x: float(x),
        )
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
