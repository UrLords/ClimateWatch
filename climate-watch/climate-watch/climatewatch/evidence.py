"""Local model inference on bounded media, with auditable frame-level evidence."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import io
import math
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 8 * 1024 * 1024
MAX_CLIP_BYTES = 40 * 1024 * 1024
MAX_PIXELS = 12_000_000
MAX_SECONDS = 120
MAX_FRAMES = 12
PROMPTS = [
    "trash",
    "garbage pile",
    "plastic bag",
    "plastic bottle",
    "drain grate",
    "water",
    "smoke",
    "fire",
    "person",
    "car",
    "tree",
    "road",
]
SEGMENT_PROMPTS = ["garbage"]
THREATS = {"smoke", "fire", "waste"}
WASTE_LABELS = {
    "garbage",
    "garbage_pile",
    "trash",
    "trash_bag",
    "waste",
    "waste_pile",
    "litter",
    "debris",
    "plastic_bag",
    "plastic_bottle",
}
MODEL_SPECS = [
    {
        "id": "Ultralytics/YOLOE-26s-seg",
        "file": "environment-yoloe-26s-seg.pt",
        "license": "AGPL-3.0",
        "url": "https://docs.ultralytics.com/models/yoloe/",
        "task": "open-vocabulary instance segmentation",
    },
    {
        "id": "rabahdev/fire-smoke-yolov8n",
        "file": "fire-smoke-yolov8n.pt",
        "license": "AGPL-3.0",
        "url": "https://huggingface.co/rabahdev/fire-smoke-yolov8n",
        "task": "D-Fire fine-tuned smoke and flame detection",
    },
    {
        "id": "CIDAS/clipseg-rd64-refined",
        "file": "clipseg/model.safetensors",
        "license": "Apache-2.0",
        "url": "https://huggingface.co/CIDAS/clipseg-rd64-refined",
        "task": "text-conditioned semantic segmentation",
    },
]
LIMITATIONS = [
    "Model scores are not calibrated probabilities. Review every finding against the frame.",
    "Open-vocabulary waste masks are proposals, not a site-trained hazard classifier.",
    "CLIPSeg scores are mean pixel activations within a proposed mask; they are not detection probabilities or measured accuracy.",
    "Smoke does not establish wildfire cause or PM2.5. Plain water and ordinary scene objects are not threat detections.",
    "A manually marked drain region plus waste overlap suggests inspection, not confirmed hydraulic blockage.",
    "At most 12 frames are sampled. Clip evidence requires overlapping same-hazard boxes in two distinct samples; this is not calibrated object tracking.",
    "People and vehicles detected by the model are obscured in thumbnails. Review before public sharing.",
]


class InvalidMedia(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def validate_roi(value):
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            isinstance(v, bool)
            or not isinstance(v, (float, int))
            or not math.isfinite(v)
            or not 0 <= v <= 1
            for v in value
        )
        or value[2] <= value[0]
        or value[3] <= value[1]
    ):
        raise InvalidMedia(
            "drain_roi must be [left, top, right, bottom], with ordered normalized coordinates from 0 to 1."
        )
    return [float(v) for v in value]


def decode_image(body: bytes, content_type: str) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(body)) as image:
            expected = {"image/jpeg": "JPEG", "image/png": "PNG"}.get(content_type)
            if not expected or image.format != expected:
                raise InvalidMedia("Image format does not match Content-Type.")
            width, height = image.size
            if min(width, height) < 32:
                raise InvalidMedia("Use an image at least 32 pixels on each side.")
            if max(width, height) > 4096 or width * height > MAX_PIXELS:
                raise InvalidMedia(
                    "Image exceeds the 4096px side or 12 megapixel limit.", 413
                )
            if max(width, height) / min(width, height) > 8:
                raise InvalidMedia(
                    "Use a conventional photo with aspect ratio at most 8:1."
                )
            if getattr(image, "n_frames", 1) != 1:
                raise InvalidMedia("Upload a single still image.")
            image.load()
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((960, 960))
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise InvalidMedia(
            "The uploaded bytes are not a readable JPEG or PNG."
        ) from error


def model_status():
    active_specs = MODEL_SPECS[:2] if os.environ.get("WASTE_WEIGHTS") else MODEL_SPECS
    models = [
        {**spec, "available": (ROOT / "models" / spec["file"]).is_file()}
        for spec in active_specs
    ]
    if configured := os.environ.get("WASTE_WEIGHTS"):
        path = Path(configured).expanduser()
        models.append(
            {
                "id": "configured/waste-detector",
                "file": path.name,
                "license": "Not specified by configuration",
                "task": "user-trained waste detection",
                "available": path.is_file(),
            }
        )
    return models


def validate_max_frames(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_FRAMES
    ):
        raise InvalidMedia("max_frames must be an integer from 1 to 12.")
    return value


def validate_waste_labels(names, task):
    labels = list(names.values()) if isinstance(names, dict) else list(names)
    if (
        task not in ("detect", "segment")
        or not labels
        or any(
            not isinstance(name, str)
            or name.lower().strip().replace(" ", "_").replace("-", "_")
            not in WASTE_LABELS
            for name in labels
        )
    ):
        raise InvalidMedia(
            "Configured waste model must detect only these explicit waste labels: "
            + ", ".join(sorted(WASTE_LABELS)),
            503,
        )


def repeated_threat_frames(frames, is_clip):
    """Keep spatially repeated threat proposals; ordinary objects never become evidence."""
    output = []
    for index, frame in enumerate(frames):
        detections = []
        for detection in frame["detections"]:
            if detection["hazard"] not in THREATS:
                continue
            repeated = not is_clip
            for other_index, other in enumerate(frames):
                if repeated:
                    break
                if other_index == index or other["time_s"] == frame["time_s"]:
                    continue
                for candidate in other["detections"]:
                    if candidate["hazard"] != detection["hazard"]:
                        continue
                    a, b = detection["box"], candidate["box"]
                    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                        0, min(a[3], b[3]) - max(a[1], b[1])
                    )
                    union = (
                        (a[2] - a[0]) * (a[3] - a[1])
                        + (b[2] - b[0]) * (b[3] - b[1])
                        - intersection
                    )
                    if union > 0 and intersection / union >= 0.1:
                        repeated = True
                        break
            if repeated:
                detections.append(detection)
        if detections:
            selected = {**frame, "detections": detections}
            if frame.get("drain_assessment"):
                selected["drain_assessment"] = drain_overlap(
                    detections, frame["drain_assessment"]["drain_roi"]
                )
            output.append(selected)
    return output


def drain_overlap(detections, roi):
    """Measure projected mask coverage of a user-defined reference, not flow capacity."""
    if roi is None:
        return None
    size = 512
    mask = np.zeros((size, size), dtype=np.uint8)
    for detection in detections:
        if detection["hazard"] != "waste" or len(detection.get("mask", [])) < 3:
            continue
        polygon = np.clip(
            np.asarray(detection["mask"]) * (size - 1), 0, size - 1
        ).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)
    x1, y1, x2, y2 = [min(size, max(0, round(v * size))) for v in roi]
    if x2 <= x1 or y2 <= y1:
        raise InvalidMedia("The marked drain region is too small.")
    coverage = float(mask[y1:y2, x1:x2].mean())
    return {
        "drain_roi": roi,
        "reference_source": "manual",
        "waste_mask_coverage": round(coverage, 4),
        "status": "potential_obstruction"
        if coverage >= 0.1
        else "no_substantial_overlap",
        "interpretation": "Projected waste-mask overlap only; does not measure water flow or confirm blockage.",
    }


def summarize(frames, archived=False):
    candidates = sorted(
        {d["hazard"] for f in frames for d in f["detections"] if d["hazard"] in THREATS}
    )
    counts = {
        hazard: sum(any(d["hazard"] == hazard for d in f["detections"]) for f in frames)
        for hazard in candidates
    }
    hazards = candidates
    potential = any(
        f.get("drain_assessment", {}).get("status") == "potential_obstruction"
        for f in frames
    )
    actions, prevention, sources = [], [], []
    if "waste" in hazards:
        actions += [
            "Send the dated frame and location to the local sanitation or drainage service for inspection; keep clear of traffic, sharp waste and contaminated water."
        ]
        prevention += [
            "Arrange regular waste collection and drain inspections before heavy rain; compare repeat frames to check whether litter returns."
        ]
        sources += [
            {
                "title": "Report a blocked drain or flooding",
                "url": "https://www.gov.uk/report-flood-cause",
            }
        ]
    if potential:
        actions.insert(
            0,
            "Waste masks overlap the marked drainage region. Ask the drainage operator to verify and safely remove any obstruction; do not enter the drain.",
        )
    if any(h in hazards for h in ["smoke", "fire"]):
        actions += [
            "If smoke or flames are present now, keep away, notify the local fire service, and follow official evacuation instructions. Use local air-quality guidance for smoke exposure."
        ]
        prevention += [
            "Check repeated observations, weather and official incident records; arrange site-specific fire prevention and avoid open waste burning."
        ]
        sources += [
            {
                "title": "Outdoor air pollution guidance",
                "url": "https://www.nhsinform.scot/healthy-living/outdoor-health/outdoor-air-pollution/",
            }
        ]
    return {
        "hazards": hazards,
        "candidate_hazards": candidates,
        "frames_with_hazard": counts,
        "review_required": bool(hazards),
        "potential_drain_obstruction": potential,
        "actions": actions,
        "prevention": prevention,
        "sources": sources,
        "time_context": "Archived evidence: these actions apply if similar conditions recur, not as a report of a current emergency."
        if archived
        else "Check the capture time and current conditions before taking action.",
        "limitations": LIMITATIONS,
    }


class EvidenceEngine:
    def __init__(self):
        from ultralytics import YOLO, YOLOE
        import torch

        torch.set_num_threads(min(8, torch.get_num_threads()))
        configured = os.environ.get("WASTE_WEIGHTS")
        active_specs = MODEL_SPECS[:2] if configured else MODEL_SPECS
        missing = [
            spec["file"]
            for spec in active_specs
            if not (ROOT / "models" / spec["file"]).is_file()
        ]
        if missing:
            raise InvalidMedia(
                "Local model files are missing: " + ", ".join(missing), 503
            )
        self.environment = YOLOE(ROOT / "models" / MODEL_SPECS[0]["file"])
        self.fire = YOLO(ROOT / "models" / MODEL_SPECS[1]["file"])
        self.segment_processor = self.segment_model = None
        if not configured:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

            self.segment_processor = CLIPSegProcessor.from_pretrained(
                ROOT / "models" / "clipseg", local_files_only=True
            )
            self.segment_model = CLIPSegForImageSegmentation.from_pretrained(
                ROOT / "models" / "clipseg", local_files_only=True
            ).eval()
        self.models = [
            {
                **spec,
                "sha256": hashlib.sha256(
                    (ROOT / "models" / spec["file"]).read_bytes()
                ).hexdigest(),
                "device": "cpu",
            }
            for spec in active_specs
        ]
        self.waste = None
        if configured:
            path = Path(configured).expanduser()
            if not path.is_file() or path.suffix.lower() not in (".pt", ".onnx"):
                raise InvalidMedia(
                    "WASTE_WEIGHTS must point to an existing local .pt or .onnx model.",
                    503,
                )
            self.waste = YOLO(path)
            validate_waste_labels(self.waste.names, self.waste.task)
            self.models.append(
                {
                    "id": "configured/waste-detector",
                    "file": path.name,
                    "license": "Not specified by configuration",
                    "task": "user-trained waste detection",
                    "device": "cpu",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )

    def frame(self, bgr, time_s, roi=None):
        h, w = bgr.shape[:2]
        if max(h, w) > 960:
            bgr = cv2.resize(
                bgr, (round(w * 960 / max(h, w)), round(h * 960 / max(h, w)))
            )
        h, w = bgr.shape[:2]
        detections, privacy_boxes = [], []
        predictions = [
            self.environment.predict(
                bgr, imgsz=640, conf=0.25, max_det=40, device="cpu", verbose=False
            )[0],
            self.fire.predict(
                bgr, imgsz=640, conf=0.55, max_det=12, device="cpu", verbose=False
            )[0],
        ]
        if self.waste is not None:
            predictions.append(
                self.waste.predict(
                    bgr, imgsz=640, conf=0.5, max_det=40, device="cpu", verbose=False
                )[0]
            )
        for model_index, result in enumerate(predictions):
            for i, box in enumerate(result.boxes):
                label = result.names[int(box.cls)]
                coords = box.xyxyn[0].cpu().numpy().clip(0, 1)
                if model_index == 0 and label in ("person", "car"):
                    privacy_boxes.append(coords)
                    continue
                hazard = (
                    "waste"
                    if model_index == 2
                    else {
                        "trash": "waste",
                        "garbage pile": "waste",
                        "smoke": "smoke",
                        "fire": "fire",
                    }.get(label.lower())
                )
                if hazard is None or (model_index == 0 and hazard in ("smoke", "fire")):
                    continue
                if self.waste is not None and model_index == 0:
                    continue
                model_id = (
                    self.models[-1]["id"]
                    if model_index == 2
                    else self.models[model_index]["id"]
                )
                item = {
                    "label": label,
                    "hazard": hazard,
                    "confidence": round(float(box.conf), 4),
                    "score_type": "detection_confidence",
                    "box": coords.round(5).tolist(),
                    "model_id": model_id,
                    "method": "open_vocabulary_segmentation"
                    if model_index == 0
                    else "fine_tuned_detection",
                }
                if result.masks is not None:
                    polygon = np.asarray(result.masks.xyn[i], dtype=np.float32)
                    if len(polygon) >= 3:
                        polygon = cv2.approxPolyDP(polygon[:, None, :], 0.002, True)[
                            :, 0, :
                        ]
                        item["mask"] = polygon.clip(0, 1).round(5).tolist()
                detections.append(item)
        import torch

        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        activations = []
        if self.waste is None:
            inputs = self.segment_processor(
                text=SEGMENT_PROMPTS,
                images=[rgb] * len(SEGMENT_PROMPTS),
                padding=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                activations = self.segment_model(**inputs).logits.sigmoid().numpy()
        for label, activation in zip(SEGMENT_PROMPTS, activations):
            binary = (activation >= 0.5).astype(np.uint8)
            if float(binary.mean()) < 0.01:
                continue
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            ah, aw = binary.shape
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
                if cv2.contourArea(contour) / (ah * aw) < 0.005:
                    continue
                region = np.zeros_like(binary)
                cv2.drawContours(region, [contour], -1, 1, -1)
                score = float(activation[(region == 1) & (binary == 1)].mean())
                polygon = cv2.approxPolyDP(contour, 1.4, True)[:, 0, :] / [aw, ah]
                x, y, width, height = cv2.boundingRect(contour)
                detections.append(
                    {
                        "label": label,
                        "hazard": "waste",
                        "confidence": round(score, 4),
                        "score_type": "mean_mask_activation",
                        "box": [
                            round(x / aw, 5),
                            round(y / ah, 5),
                            round((x + width) / aw, 5),
                            round((y + height) / ah, 5),
                        ],
                        "mask": polygon.round(5).tolist(),
                        "model_id": MODEL_SPECS[2]["id"],
                        "method": "text_conditioned_segmentation",
                    }
                )
        safe = bgr.copy()
        for box in privacy_boxes:
            x1, y1, x2, y2 = (box * [w, h, w, h]).astype(int)
            region = safe[y1:y2, x1:x2]
            if region.size:
                safe[y1:y2, x1:x2] = cv2.resize(
                    cv2.resize(region, (6, 6)),
                    (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_NEAREST,
                )
        ok, jpeg = cv2.imencode(".jpg", safe, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise InvalidMedia("Could not prepare an evidence frame.", 500)
        frame = {
            "time_s": round(time_s, 3),
            "width": w,
            "height": h,
            "detections": detections,
            "thumbnail": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
            "privacy_regions_obscured": len(privacy_boxes),
        }
        assessment = drain_overlap(detections, roi)
        if assessment is not None:
            frame["drain_assessment"] = assessment
        return frame

    def analyze(
        self,
        body: bytes,
        content_type: str,
        drain_roi=None,
        archived=False,
        max_frames=12,
    ):
        start = time.perf_counter()
        max_frames = validate_max_frames(max_frames)
        roi = validate_roi(drain_roi)
        if content_type.startswith("image/"):
            bgr = decode_image(body, content_type)
            h, w = bgr.shape[:2]
            frames = [self.frame(bgr, 0, roi)]
            duration = 0
        else:
            if content_type not in ("video/mp4", "video/webm", "video/mp2t"):
                raise InvalidMedia("Only JPEG, PNG, MP4 and WebM are supported.", 415)
            if (content_type == "video/mp4" and body[4:8] != b"ftyp") or (
                content_type == "video/webm" and not body.startswith(b"\x1aE\xdf\xa3")
            ):
                raise InvalidMedia("Video format does not match Content-Type.")
            if content_type == "video/mp2t" and (
                len(body) < 376 or body[0] != 0x47 or body[188] != 0x47
            ):
                raise InvalidMedia("Camera segment is not MPEG-TS.")
            with tempfile.TemporaryDirectory(prefix="climatewatch-scan-") as temp:
                path = (
                    Path(temp)
                    / {
                        "video/mp4": "clip.mp4",
                        "video/webm": "clip.webm",
                        "video/mp2t": "clip.ts",
                    }[content_type]
                )
                path.write_bytes(body)
                cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
                try:
                    fps, count = (
                        cap.get(cv2.CAP_PROP_FPS),
                        cap.get(cv2.CAP_PROP_FRAME_COUNT),
                    )
                    w, h = (
                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    )
                    if not cap.isOpened() or not all(
                        math.isfinite(n) and n > 0 for n in (fps, count, w, h)
                    ):
                        raise InvalidMedia(
                            "Cannot read video dimensions, frame rate and duration."
                        )
                    duration = count / fps
                    if (
                        duration > MAX_SECONDS
                        or max(w, h) > 4096
                        or w * h > MAX_PIXELS
                        or min(w, h) < 32
                    ):
                        raise InvalidMedia(
                            "Use a clip up to 120 seconds, 4096px per side and 12 megapixels per frame.",
                            413,
                        )
                    frames = []
                    for frame_number in np.unique(
                        np.linspace(0, count - 1, min(max_frames, int(count))).astype(
                            int
                        )
                    ):
                        if frames and time.perf_counter() - start > 100:
                            break
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
                        ok, bgr = cap.read()
                        if not ok:
                            continue
                        frames.append(self.frame(bgr, frame_number / fps, roi))
                    if not frames:
                        raise InvalidMedia("No video frame could be decoded.")
                finally:
                    cap.release()
        sampled_times = [frame["time_s"] for frame in frames]
        frames = repeated_threat_frames(
            frames, is_clip=not content_type.startswith("image/")
        )
        digest = hashlib.sha256(body).hexdigest()
        return {
            "analysis_id": digest[:16] + "-" + str(time.time_ns()),
            "kind": "model_evidence",
            "model": {
                "id": (
                    "Configured waste model" if self.waste is not None else "CLIPSeg"
                )
                + " + YOLOE-26s + D-Fire YOLOv8n",
                "license": "Configured waste license unspecified / AGPL-3.0"
                if self.waste is not None
                else "Apache-2.0 / AGPL-3.0",
                "device": "cpu",
            },
            "models": self.models,
            "media": {
                "width": w,
                "height": h,
                "duration_s": round(duration, 3),
                "sampled_frames": len(sampled_times),
                "evidence_frames": len(frames),
            },
            "frames": frames,
            "summary": summarize(frames, archived),
            "detectors_available": True,
            "provenance": {
                "mode": "model_inference",
                "input_sha256": digest,
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": round(time.perf_counter() - start, 2),
                "sample_times_s": sampled_times,
                "config": {
                    "imgsz": 640,
                    "environment_confidence": 0.25,
                    "fire_confidence": 0.55,
                    "waste_confidence": 0.5 if self.waste is not None else None,
                    "max_sampled_frames": max_frames,
                    "prompts": PROMPTS,
                    "segmentation_prompts": SEGMENT_PROMPTS
                    if self.waste is None
                    else [],
                    "pixel_activation_threshold": 0.5,
                    "minimum_component_fraction": 0.005,
                    "clip_minimum_samples": 2,
                    "clip_match_iou": 0.1,
                },
                "raw_upload_retention": "none",
                "drain_roi": roi,
            },
        }


@lru_cache(maxsize=1)
def get_engine():
    return EvidenceEngine()
