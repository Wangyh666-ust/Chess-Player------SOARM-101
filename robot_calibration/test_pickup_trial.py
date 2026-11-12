"""Pickup/telemetry/video failure tests; no real arm or camera is accessed."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import collect_robot_calibration as collector
from pickup_trial import PHASES, VideoRecorder, load_pickup
from annotate_separation import annotate


def taught_data():
    start = collector.MockArm().read()
    # Software fixture, not coordinates suitable for any physical arm.
    approach = start | {"wrist_flex": 1.0}
    closed = approach | {"gripper": 45.0}
    lifted = closed | {"elbow_flex": 1.0}
    positions = (start, approach, closed, lifted, lifted, closed, approach, start)
    return {"units": collector.UNITS, "waypoints": [
        {"phase": phase, "time_s": i * 2.0, "positions": pos}
        for i, (phase, pos) in enumerate(zip(PHASES, positions))]}


class PickupTests(unittest.TestCase):
    def test_taught_schedule_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"
            data = taught_data()
            path.write_text(json.dumps(data))
            _, schedule = load_pickup(path, 10, collector.JOINTS, None)
            self.assertEqual(schedule[-1][1], schedule[0][1])
            for phase, command in schedule:
                collector.check_positions(command)
            data["waypoints"][2]["positions"] = data["waypoints"][2]["positions"] | {"shoulder_pan": 10}
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "fixed"):
                load_pickup(path, 10, collector.JOINTS, None)

    def test_motion_requires_video_and_valid_trial_labels(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            collector.parse_args(["--backend", "lerobot", "--mode", "pickup", "--execute",
                                  "--port", "fake", "--robot-id", "fake", "--limits", "fake.json",
                                  "--trajectory", "fake.json", "--piece-id", "pawn", "--condition", "board",
                                  "--output", "unused"])

    def test_repeated_mock_pickups_keep_same_trajectory_and_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "trajectory.json"
            path.write_text(json.dumps(taught_data()))
            collector.main(["--mode", "pickup", "--trajectory", str(path), "--piece-id", "pawn-1",
                            "--condition", "mock", "--trials", "2", "--hz", "10", "--output", str(root / "trials")])
            metas = [json.loads(p.read_text()) for p in sorted((root / "trials").glob("*/metadata.json"))]
            self.assertEqual(len(metas), 2)
            self.assertEqual(metas[0]["trajectory_sha256"], metas[1]["trajectory_sha256"])
            self.assertTrue(all(m["status"] == "completed" for m in metas))
            self.assertTrue(all(m["pickup_outcome"] == "unreviewed" for m in metas))
            for path in (root / "trials").glob("*/samples.jsonl"):
                row = json.loads(path.read_text().splitlines()[0])
                self.assertEqual(set(row["telemetry"]), set(collector.TELEMETRY_REGISTERS))
                self.assertEqual(row["telemetry"]["Present_Current"]["units"], "register_value")

    def test_telemetry_failure_happens_before_torque_enable(self):
        class UnsupportedArm(collector.MockArm):
            did_enable = False

            def read_register(self, register):
                raise ValueError("Unsupported register")

            def enable(self, initial):
                self.did_enable = True

        with tempfile.TemporaryDirectory() as directory:
            args = collector.parse_args(["--mode", "joint_response", "--telemetry",
                                         "--output", str(Path(directory) / "failed")])
            arm = UnsupportedArm()
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                collector.run(args, arm, None, [("hold", 0)])
            self.assertFalse(arm.did_enable)

    def test_camera_failure_before_motion_and_cleanup(self):
        class BrokenVideo:
            closed = False

            def start(self, *_args):
                raise RuntimeError("Camera failed")

            def close(self):
                self.closed = True
                return {"frames": 0}

        with tempfile.TemporaryDirectory() as directory:
            args = collector.parse_args(["--mode", "joint_response", "--output", str(Path(directory) / "failed")])
            arm = collector.MockArm()
            video = BrokenVideo()
            with self.assertRaisesRegex(RuntimeError, "Camera failed"):
                collector.run(args, arm, None, [("hold", 0)], video)
            self.assertFalse(arm.enabled)
            self.assertTrue(video.closed)

    def test_video_clock_and_resource_cleanup(self):
        class Capture:
            released = False

            def isOpened(self): return True
            def set(self, *_args): return False
            def read(self):
                time.sleep(0.001)
                return True, SimpleNamespace(shape=(48, 64, 3))
            def release(self): self.released = True

        class Writer:
            released = False

            def isOpened(self): return True
            def write(self, _frame): pass
            def release(self): self.released = True

        capture, writer = Capture(), Writer()
        cv2 = SimpleNamespace(VideoCapture=lambda _source: capture, VideoWriter=lambda *_args: writer,
                              VideoWriter_fourcc=lambda *_args: 0, CAP_PROP_FPS=5, CAP_PROP_BUFFERSIZE=38)
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"cv2": cv2}):
            video = VideoRecorder("0", 30)
            video.start(directory, time.monotonic())
            result = video.close()
            frames = [json.loads(line) for line in (Path(directory) / "frames.jsonl").read_text().splitlines()]
            self.assertGreaterEqual(result["frames"], 1)
            self.assertEqual(len(frames), result["frames"])
            self.assertTrue(all(f["read_end_s"] >= f["read_start_s"] >= 0 for f in frames))
            self.assertTrue(capture.released and writer.released)

    def test_separation_annotation_and_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.json").write_text(json.dumps({"status": "completed", "backend": "mock"}))
            (root / "frames.jsonl").write_text('\n'.join(json.dumps({"frame": i, "read_start_s": i / 10,
                                                                     "read_end_s": (i + .5) / 10}) for i in range(5)))
            (root / "samples.jsonl").write_text(json.dumps({"sample": 3, "read_start_s": .24, "read_end_s": .26}))
            with self.assertRaises(ValueError):
                annotate(root, "test", "separated", 3, 2)
            result = annotate(root, "test", "separated", 2, 3)
            self.assertEqual(result["host_interval_s"], [.2, .35])
            self.assertEqual(result["overlapping_samples"], [3])
            with self.assertRaises(FileExistsError):
                annotate(root, "test", "occluded")


if __name__ == "__main__":
    unittest.main()
