import json
import cv2
import gradio as gr
import numpy as np

from climatewatch.advice import build_advice
from climatewatch.detector import Detector
from climatewatch.haze import compute_haze, ensure_bgr, relative_haze
from climatewatch.privacy import blur_boxes
from climatewatch.risk import flood_risk, rain_next_hours
from climatewatch.sky import SkySegmenter

try:
    import spaces

    gpu = spaces.GPU(duration=30)
except Exception:

    def gpu(fn):
        return fn


sky_seg = SkySegmenter()
detector = Detector()


@gpu
def run_detectors(bgr):
    return detector.hazards(bgr), detector.person_boxes(bgr)


def analyze(
    image,
    camera_id: str = "demo",
    lat: float = 0.0,
    lon: float = 0.0,
    haze_baseline: float = -1.0,
):
    if image is None:
        return None, {"error": "no image"}
    bgr = ensure_bgr(cv2.cvtColor(ensure_bgr(image), cv2.COLOR_RGB2BGR))

    mask, sky_src = sky_seg.mask(bgr)
    haze = compute_haze(bgr, mask)
    dets, persons = run_detectors(bgr)

    hazards = {d["hazard"] for d in dets}
    flood = flood_risk(hazards, rain_next_hours(lat, lon) if (lat or lon) else None)
    base = haze_baseline if (haze_baseline is not None and haze_baseline >= 0) else None
    rel = relative_haze(
        haze.haze_index, base
    )
    eff = (
        rel if rel is not None else haze.haze_index
    )
    advice = build_advice(
        eff, haze.confidence, hazards, flood, relative=rel is not None
    )

    out = blur_boxes(bgr, persons)
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            out,
            f"{d['label']} {d['conf']:.2f}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    status = (
        "alert"
        if (
            hazards
            or (eff >= (40 if rel is not None else 75) and haze.confidence >= 0.5)
        )
        else "low_confidence"
        if haze.confidence < 0.5
        else "clear"
    )
    result = {
        "camera_id": camera_id,
        "status": status,
        "sky_source": sky_src,
        "haze": haze.to_dict(),
        "haze_relative": None if rel is None else round(rel, 1),
        "haze_calibrated": False,
        "detections": dets,
        "flood_risk": flood,
        "advice": advice,
    }
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), result


demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Image(type="numpy", label="CCTV frame"),
        gr.Textbox(value="demo", label="camera_id"),
        gr.Number(value=0, label="lat"),
        gr.Number(value=0, label="lon"),
        gr.Number(value=-1, label="haze_baseline (-1 = none)"),
    ],
    outputs=[gr.Image(label="Annotated (people blurred)"), gr.JSON(label="Result")],
    api_name="analyze",
    title="Climate Watch inference API",
    description="Haze index + hazard detection + playbook advice. Haze index is uncalibrated unless stated.",
)

if __name__ == "__main__":
    demo.launch()
