"""Offline checks for unit conversion, causal alignment and exclusion boundaries."""

import unittest
import math

from robot_calibration.analysis.prepare_recordings import (
    ARM,
    JOINTS,
    match_frame,
    normalized_to_degrees,
    segments_for,
)
from robot_calibration.analysis.compare_joint_response import hold_displacements
from robot_calibration.analysis.audit_joint_alignment import out_of_limits


class PreparationTests(unittest.TestCase):
    def test_pose_audit_reports_limits_without_clamping_measurements(self):
        definitions = {
            j: {"limit": {"lower": str(-math.pi / 2), "upper": str(math.pi / 2)}}
            for j in ARM
        }
        measured = dict.fromkeys(ARM, 0.0) | {"wrist_flex": 98.0}

        self.assertEqual(out_of_limits(measured, definitions), ["wrist_flex"])
        self.assertEqual(measured["wrist_flex"], 98.0)

        measured["wrist_flex"] = 90.0

        self.assertEqual(out_of_limits(measured, definitions), [])

    def test_normalization_matches_encoder_formula_and_inversion(self):
        calibration = {
            j: {"range_min": 1000, "range_max": 3000, "drive_mode": 0} for j in JOINTS
        }
        positions = dict.fromkeys(ARM, 20.0) | {"gripper": 30.0}
        degrees, censored = normalized_to_degrees(positions, calibration)

        # +20 normalized -> encoder 2200; midpoint 2000. Homing is already applied.
        self.assertAlmostEqual(degrees["elbow_flex"], 200 * 360 / 4095)
        self.assertEqual(censored, [])

        calibration["elbow_flex"]["drive_mode"] = 1
        inverse, _ = normalized_to_degrees(positions, calibration)

        self.assertEqual(inverse["elbow_flex"], -degrees["elbow_flex"])

        positions["shoulder_pan"] = 100

        self.assertIn("shoulder_pan", normalized_to_degrees(positions, calibration)[1])

        positions["gripper"] = float("nan")

        with self.assertRaises(ValueError):
            normalized_to_degrees(positions, calibration)

    def test_alignment_never_uses_future_or_silently_accepts_stale_frames(self):
        frames = [
            {"frame": i, "read_start_s": end - 0.02, "read_end_s": end}
            for i, end in enumerate((0.10, 0.17, 0.24))
        ]
        ends = [f["read_end_s"] for f in frames]

        self.assertIsNone(match_frame(frames, ends, 0.09, 0.15))
        self.assertEqual(match_frame(frames, ends, 0.23, 0.15)["frame"], 1)
        self.assertFalse(match_frame(frames, ends, 0.5, 0.15)["fresh"])

    def test_intervention_partition_preserves_failed_and_retry_outcomes(self):
        meta = {"outcome": "failed", "status": "complete"}
        review = {
            "segments": [
                {"start_s": 0, "end_s": 18, "imitation_candidate": False},
                {"start_s": 18, "end_s": 25, "imitation_candidate": False},
                {"start_s": 25, "end_s": None, "imitation_candidate": True},
            ]
        }
        result = segments_for(meta, review, 50)

        self.assertEqual([s["end_s"] for s in result], [18, 25, 50])
        self.assertFalse(result[1]["imitation_candidate"])

        review["segments"][1]["start_s"] = 19

        with self.assertRaises(ValueError):
            segments_for(meta, review, 50)

    def test_completed_recording_is_not_automatically_a_success(self):
        segment = segments_for({"outcome": "uncertain", "status": "complete"}, {}, 10)[
            0
        ]

        self.assertFalse(segment["imitation_candidate"])

    def test_hold_metrics_exclude_ramps_and_use_last_ten_samples(self):
        rows = [{"phase": "cycle_1_positive_ramp"}] * 5
        rows += [{"phase": "cycle_1_positive_hold"}] * 20
        values = [100] * 5 + [2] * 10 + [1.5] * 10
        result = hold_displacements(rows, values, 1)

        self.assertEqual(result["positive"], 0.5)
        self.assertIsNone(result["negative"])


if __name__ == "__main__":
    unittest.main()
