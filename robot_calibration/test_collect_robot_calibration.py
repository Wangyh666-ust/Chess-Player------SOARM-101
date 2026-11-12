"""Run with: python3 -m unittest discover -s robot_calibration -v"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import collect_robot_calibration as collector


class FakeBus:
    def __init__(self):
        self.events = []
        self.position = collector.MockArm().read()
        self.torque = 0
        self.mode = 0

    def sync_read(self, register, normalize=True):
        self.events.append(("read", register))
        if register == "Present_Position":
            return self.position.copy()
        value = self.torque if register == "Torque_Enable" else self.mode
        return dict.fromkeys(collector.JOINTS, value)

    def sync_write(self, register, values):
        self.events.append(("write", register, values.copy()))

    def enable_torque(self):
        self.events.append(("enable",))

    def disconnect(self, disable_torque=True):
        self.events.append(("disconnect", disable_torque))


class CollectorTests(unittest.TestCase):
    def arguments(self, output, *extra):
        return collector.parse_args(["--output", str(output), *extra])

    def test_motion_needs_explicit_activation_and_limits(self):
        for extra in ([], ["--execute"], ["--limits", "some.json"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.arguments("unused", "--backend", "lerobot", "--port", "unused",
                                   "--robot-id", "arm", "--mode", "joint_response", *extra)

    def test_invalid_numbers_rejected(self):
        for flag, value in (("--hz", "nan"), ("--ramp-s", "0"),
                            ("--amplitude-deg", "inf"), ("--cycles", "0")):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.arguments("unused", flag, value)

    def test_limits_require_measured_values(self):
        with self.assertRaises(ValueError):
            collector.read_limits(Path(__file__).with_name("limits.example.json"))
        with self.assertRaises(ValueError):
            collector.check_positions(collector.MockArm().read() | {"shoulder_pan": float("nan")})

    def test_sweep_endpoints_and_speed(self):
        schedule = collector.response_schedule(30, 3, 3, 1, 2)
        offsets = [offset for _, offset in schedule]
        self.assertEqual((min(offsets), max(offsets), offsets[-1]), (-3, 3, 0))
        self.assertLessEqual(max(abs(b - a) for a, b in zip(offsets, offsets[1:])), 1 / 30 + 1e-12)

    def test_complete_mock_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            collector.main(["--output", str(output), "--mode", "joint_response", "--joint", "wrist_roll"])
            meta = json.loads((output / "metadata.json").read_text())
            rows = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
            self.assertEqual(meta["status"], "completed")
            self.assertEqual(meta["backend"], "mock")
            self.assertEqual(len(rows), meta["samples"])
            self.assertEqual(rows[-1]["command"]["wrist_roll"], 0)
            for row in rows:
                self.assertLessEqual(row["read_start_s"], row["read_end_s"])
                for joint in collector.JOINTS:
                    if joint != "wrist_roll":
                        self.assertEqual(row["command"][joint], meta["initial_positions"][joint])

    def test_partial_trace_and_cleanup_on_read_failure(self):
        class BrokenArm(collector.MockArm):
            calls = 0
            closed = False

            def read(self):
                self.calls += 1
                if self.calls == 4:
                    raise OSError("Lost connection")
                return super().read()

            def close(self):
                self.closed = True
                super().close()

        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory) / "broken", "--mode", "joint_response")
            arm = BrokenArm()
            with self.assertRaises(OSError):
                collector.run(args, arm, None, [("hold", 0)] * 10)
            meta = json.loads((args.output / "metadata.json").read_text())
            self.assertEqual(meta["status"], "failed")
            self.assertEqual(meta["samples"], 2)
            self.assertTrue(arm.closed)

    def test_out_of_limits_sweep_never_enables_motors(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory) / "bounds", "--mode", "joint_response")
            arm = collector.MockArm()
            limits = dict.fromkeys(collector.ARM_JOINTS, [-1, 1]) | {"gripper": [0, 100]}
            with self.assertRaises(ValueError):
                collector.run(args, arm, limits, [("hold", 0)])
            self.assertFalse(arm.enabled)

    def test_tracking_failure_keeps_offending_sample(self):
        class DriftingArm(collector.MockArm):
            def send(self, command):
                self.position["elbow_flex"] = 12.0

        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory) / "drift", "--mode", "joint_response")
            arm = DriftingArm()
            with self.assertRaisesRegex(RuntimeError, "Tracking guard"):
                collector.run(args, arm, None, [("hold", 0)])
            self.assertFalse(arm.enabled)
            row = json.loads((args.output / "samples.jsonl").read_text())
            self.assertEqual(row["measured"]["elbow_flex"], 12)

    def test_slow_reads_do_not_produce_command_bursts(self):
        class Clock:
            now = 0.0

            def monotonic(self):
                return self.now

            def sleep(self, duration):
                self.now += duration

        clock = Clock()

        class SlowArm(collector.MockArm):
            sent = []

            def read(self):
                clock.now += 0.04
                return super().read()

            def send(self, command):
                self.sent.append(clock.now)
                super().send(command)

        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory) / "slow", "--mode", "joint_response")
            args.backend = "lerobot"  # Exercise scheduling with a fake clock/bus only.
            arm = SlowArm()
            with patch.object(collector.time, "monotonic", clock.monotonic), \
                    patch.object(collector.time, "sleep", clock.sleep):
                collector.run(args, arm, None, [("hold", 0)] * 3)
            for a, b in zip(arm.sent, arm.sent[1:]):
                self.assertGreaterEqual(b - a, 1 / args.hz - 1e-9)

    def test_interruption_disables_motion_and_marks_record(self):
        class InterruptedArm(collector.MockArm):
            def send(self, command):
                raise KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory) / "interrupt", "--mode", "joint_response")
            arm = InterruptedArm()
            with self.assertRaises(KeyboardInterrupt):
                collector.run(args, arm, None, [("hold", 0)])
            self.assertFalse(arm.enabled)
            meta = json.loads((args.output / "metadata.json").read_text())
            self.assertEqual(meta["status"], "interrupted")

    def test_existing_recording_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            with self.assertRaises(FileExistsError):
                collector.run(args, collector.MockArm(), None, [("read", 0)])

    def test_hardware_read_is_passive(self):
        arm = collector.LeRobotArm.__new__(collector.LeRobotArm)
        arm.bus = FakeBus()
        arm.enabled = False
        arm.read()
        arm.close()
        self.assertEqual(arm.bus.events, [("read", "Present_Position"), ("disconnect", False)])

    def test_hardware_goal_seed_precedes_torque(self):
        arm = collector.LeRobotArm.__new__(collector.LeRobotArm)
        arm.bus = FakeBus()
        arm.enabled = False
        arm.enable(arm.bus.position.copy())
        self.assertEqual(arm.bus.events[-2][0:2], ("write", "Goal_Position"))
        self.assertEqual(arm.bus.events[-1], ("enable",))
        arm.close()
        self.assertEqual(arm.bus.events[-1], ("disconnect", True))

    def test_hardware_refuses_torque_on_and_wrong_mode(self):
        for attribute in ("torque", "mode"):
            arm = collector.LeRobotArm.__new__(collector.LeRobotArm)
            arm.bus = FakeBus()
            arm.enabled = False
            setattr(arm.bus, attribute, 1)
            with self.assertRaises(RuntimeError):
                arm.enable(arm.bus.position.copy())
            self.assertFalse(any(event[0] in ("write", "enable") for event in arm.bus.events))


if __name__ == "__main__":
    unittest.main()
