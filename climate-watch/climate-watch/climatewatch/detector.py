"""Object detection using configured hazard and privacy models.

Env: WASTE_WEIGHTS = path of fine-tuned model (classes mapped via HAZARD_CLASS_MAP)
     PERSON_WEIGHTS = general COCO model used only to blur people (default yolo26n.pt)
"""

from __future__ import annotations
import os


HAZARD_CLASS_MAP = {
    "garbage": "waste_pile",
    "trash": "waste_pile",
    "waste": "waste_pile",
    "debris": "waste_pile",
    "blocked_drain": "drain_blocked",
    "clogged_drain": "drain_blocked",
    "flood_water": "flood_water",
    "standing_water": "flood_water",
    "smoke": "smoke",
    "fire": "open_burning",
}


class Detector:
    def __init__(self):
        self.waste = self.person = None
        try:
            from ultralytics import YOLO

            w = os.getenv("WASTE_WEIGHTS")
            if w and os.path.exists(w):
                self.waste = YOLO(w)
            if os.getenv("PRIVACY_BLUR", "1") == "1":
                self.person = YOLO(os.getenv("PERSON_WEIGHTS", "yolo26n.pt"))
        except Exception as e:
            print(f"[detector] unavailable ({e})")

    def hazards(self, bgr, conf: float = 0.4) -> list[dict]:
        if self.waste is None:
            return []
        r = self.waste.predict(bgr, conf=conf, verbose=False)[0]
        out = []
        for b in r.boxes:
            name = r.names[int(b.cls)]
            if name in HAZARD_CLASS_MAP:
                out.append(
                    {
                        "label": name,
                        "hazard": HAZARD_CLASS_MAP[name],
                        "conf": round(float(b.conf), 3),
                        "xyxy": [round(float(v), 1) for v in b.xyxy[0]],
                    }
                )
        return out

    def person_boxes(self, bgr) -> list[list[float]]:
        if self.person is None:
            return []
        r = self.person.predict(bgr, classes=[0], conf=0.25, verbose=False)[0]
        return [[float(v) for v in b.xyxy[0]] for b in r.boxes]
