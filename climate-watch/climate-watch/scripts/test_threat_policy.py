"""Focused checks for threat-only evidence; no weights, browser or network required."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climatewatch.evidence import (
    InvalidMedia,
    repeated_threat_frames,
    summarize,
    validate_max_frames,
    validate_waste_labels,
)


def frame(time_s, *hazards):
    return {
        "time_s": time_s,
        "detections": [
            {"hazard": hazard, "box": [0.1, 0.1, 0.5, 0.5]} for hazard in hazards
        ],
        "thumbnail": "test fixture",
    }


class ThreatPolicyTests(unittest.TestCase):
    def test_ordinary_objects_never_become_evidence_or_advice(self):
        selected = repeated_threat_frames(
            [frame(0, "water", "drain", "tree", "car", "person")], is_clip=False
        )
        self.assertEqual(selected, [])
        summary = summarize(selected)
        for key in ("hazards", "candidate_hazards", "actions", "prevention", "sources"):
            self.assertEqual(summary[key], [])
        self.assertFalse(summary["review_required"])

    def test_clip_removes_isolated_threats_but_keeps_repeated_boxes(self):
        original = [frame(0, "smoke", "waste"), frame(1, "smoke"), frame(2, "water")]
        selected = repeated_threat_frames(original, is_clip=True)
        self.assertEqual([f["time_s"] for f in selected], [0, 1])
        self.assertEqual(
            [[d["hazard"] for d in f["detections"]] for f in selected],
            [["smoke"], ["smoke"]],
        )
        self.assertEqual(len(original[0]["detections"]), 2)

    def test_two_different_places_or_duplicate_time_are_not_stable(self):
        first, second = frame(0, "waste"), frame(1, "waste")
        second["detections"][0]["box"] = [0.7, 0.7, 0.9, 0.9]
        self.assertEqual(repeated_threat_frames([first, second], is_clip=True), [])
        self.assertEqual(
            repeated_threat_frames(
                [frame(0, "smoke"), frame(0, "smoke")], is_clip=True
            ),
            [],
        )
        self.assertEqual(repeated_threat_frames([frame(0, "smoke")], is_clip=True), [])

    def test_still_threat_remains_candidate_for_monitor_temporal_gate(self):
        selected = repeated_threat_frames([frame(0, "waste")], is_clip=False)
        self.assertEqual(len(selected), 1)
        self.assertEqual(summarize(selected)["hazards"], ["waste"])

    def test_sampling_budget_rejects_booleans_and_out_of_range(self):
        for valid in (1, 3, 4, 12):
            self.assertEqual(validate_max_frames(valid), valid)
        for invalid in (True, False, 0, 13, 3.5, "4", None):
            with self.assertRaises(InvalidMedia):
                validate_max_frames(invalid)

    def test_configured_weights_cannot_relabel_coco_as_waste(self):
        validate_waste_labels({0: "garbage", 1: "plastic bag"}, "segment")
        for names, task in [
            ({0: "person", 1: "car"}, "detect"),
            ({0: "waste"}, "classify"),
            ({0: "water"}, "segment"),
            ({}, "detect"),
        ]:
            with self.assertRaises(InvalidMedia):
                validate_waste_labels(names, task)


if __name__ == "__main__":
    unittest.main(verbosity=2)
