"""Record SO-101 joint responses or taught pickups, with optional telemetry/video.

Defaults to a fast mock run with no dependencies or hardware access. Hardware
uses LeRobot's bus directly: its normal robot.connect() configures controllers,
which would change the response we want to measure. No gains are changed here.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import signal
import sys
import time


ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
JOINTS = (*ARM_JOINTS, "gripper")
UNITS = {"arm": "deg", "gripper": "normalized_0_100"}
SDK_VERSION = "0.7.0"
TELEMETRY_REGISTERS = ("Present_Current", "Present_Load", "Present_Voltage", "Present_Temperature",
                       "Goal_Position")


def read_telemetry(arm, origin):
    """Keep register values and separate acquisition intervals; do not infer force."""
    result = {}
    for register in TELEMETRY_REGISTERS:
        before = time.monotonic() - origin
        values = arm.read_register(register)
        after = time.monotonic() - origin
        if set(values) != set(JOINTS) or not all(map(finite, values.values())):
            raise ValueError(f"Missing/non-finite telemetry for {register}")
        result[register] = {"values": values, "read_start_s": before, "read_end_s": after,
                            "units": UNITS if register == "Goal_Position" else "register_value"}
    return result


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_limits(path):
    data = json.loads(path.read_text())
    if data.get("units") != UNITS or set(data.get("limits", {})) != set(JOINTS):
        raise ValueError("Limits must specify the six joints and the documented units")
    for joint, pair in data["limits"].items():
        if not isinstance(pair, list) or len(pair) != 2 or not all(map(finite, pair)):
            raise ValueError(f"Fill measured numeric limits for {joint}; null is not a limit")
        if pair[0] >= pair[1]:
            raise ValueError(f"Invalid interval for {joint}")
        if joint == "gripper" and not 0 <= pair[0] < pair[1] <= 100:
            raise ValueError("Gripper limits must be within normalized 0..100")
    return data["limits"]


def check_positions(positions, limits=None):
    if set(positions) != set(JOINTS) or not all(map(finite, positions.values())):
        raise ValueError("Missing joint position or non-finite position")
    if not 0 <= positions["gripper"] <= 100:
        raise ValueError("Gripper reading outside normalized 0..100")
    if limits:
        for joint, value in positions.items():
            if not limits[joint][0] <= value <= limits[joint][1]:
                raise ValueError(f"{joint}={value:.3f} outside selected limits {limits[joint]}")


def response_schedule(hz, amplitude, ramp_s, hold_s, cycles):
    """Linear ramps: start -> positive -> start -> negative -> start, with holds.

    Every cycle returns to its initial joint angle. Only this experiment's
    tested joint changes; the other joints hold their initial measurements.
    """
    ramp_ticks = math.ceil(ramp_s * hz)
    hold_ticks = math.ceil(hold_s * hz)
    samples = [("initial_hold", 0.0)] * hold_ticks
    previous = 0.0
    for cycle in range(cycles):
        for label, target in (("positive", amplitude), ("centre", 0.0),
                              ("negative", -amplitude), ("return", 0.0)):
            phase = f"cycle_{cycle + 1}_{label}"
            samples.extend((phase + "_ramp", previous + (target - previous) * i / ramp_ticks)
                           for i in range(1, ramp_ticks + 1))
            samples.extend([(phase + "_hold", target)] * hold_ticks)
            previous = target
    return samples


class MockArm:
    """A synthetic first-order response for testing files and guards, not physics."""

    def __init__(self):
        self.position = dict.fromkeys(ARM_JOINTS, 0.0) | {"gripper": 50.0}
        self.enabled = False
        self.goal = self.position.copy()

    def connect(self):
        return {"model": "MOCK first-order response; not physical evidence"}

    def read(self):
        return self.position.copy()

    def read_register(self, register):
        if register == "Goal_Position":
            return self.goal.copy()
        if register in ("Present_Current", "Present_Load"):
            return {j: round(abs(self.goal[j] - self.position[j]) * 10) for j in JOINTS}
        return dict.fromkeys(JOINTS, 25 if register == "Present_Temperature" else 120)

    def enable(self, initial):
        self.enabled = True

    def send(self, command):
        if not self.enabled:
            raise RuntimeError("Mock torque not enabled")
        self.goal = command.copy()
        for joint in JOINTS:
            self.position[joint] += 0.4 * (command[joint] - self.position[joint])

    def close(self):
        self.enabled = False


class LeRobotArm:
    def __init__(self, port, robot_id):
        installed = importlib.metadata.version("lerobot")
        if installed != SDK_VERSION:
            raise RuntimeError(f"Expected LeRobot {SDK_VERSION}, found {installed}")
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        self.robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id,
                                                       use_degrees=True, cameras={}))
        if set(self.robot.calibration) != set(JOINTS):
            raise RuntimeError("Run lerobot-calibrate with this robot ID before collecting")
        self.bus = self.robot.bus
        self.enabled = False

    def connect(self):
        self.bus.connect()
        if not self.bus.is_calibrated:
            raise RuntimeError("Saved calibration does not match motors; calibrate separately")
        registers = ("Operating_Mode", "Torque_Enable", "P_Coefficient", "I_Coefficient",
                     "D_Coefficient", "Max_Torque_Limit")
        return {"lerobot_version": SDK_VERSION,
                "calibration": {name: asdict(value) for name, value in self.robot.calibration.items()},
                "registers": {name: self.bus.sync_read(name, normalize=False) for name in registers}}

    def read(self):
        return {name: float(value) for name, value in self.bus.sync_read("Present_Position").items()}

    def read_register(self, register):
        return self.bus.sync_read(register, normalize=(register == "Goal_Position"))

    def enable(self, initial):
        # Start supported, with torque off. Seed every goal BEFORE enabling
        # motors, so an old stored goal cannot trigger an unexpected movement.
        if any(self.bus.sync_read("Torque_Enable", normalize=False).values()):
            raise RuntimeError("Motion requires all torque OFF initially; support the arm first")
        if any(self.bus.sync_read("Operating_Mode", normalize=False).values()):
            raise RuntimeError("Expected position mode (0); configure with LeRobot separately")
        current = self.read()
        check_positions(current)
        if any(abs(current[j] - initial[j]) > 0.5 for j in JOINTS):
            raise RuntimeError("Arm moved during preparation; reposition and start a new run")
        self.bus.sync_write("Goal_Position", initial)
        self.enabled = True  # Cleanup also covers partially successful torque enabling.
        self.bus.enable_torque()

    def send(self, command):
        self.bus.sync_write("Goal_Position", command)

    def close(self):
        # Passive recording must not change torque. Motion always releases it.
        # Support the arm: disabling torque can let it fall under gravity.
        self.bus.disconnect(disable_torque=self.enabled)


def run(args, arm, limits, schedule, video=None):
    """Keep the partial trace and an explicit failure status on any interruption."""
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {"schema_version": 2, "status": "running", "backend": args.backend,
            "mode": args.mode, "units": UNITS, "joint_order": JOINTS,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "limits": limits, "samples": 0,
            "timestamp_note": "Host monotonic read intervals; not device acquisition timestamps"}
    error = None
    squared_error = 0.0
    max_error = 0.0

    def save():
        (args.output / "metadata.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")

    save()
    try:
        origin = time.monotonic()
        meta["monotonic_origin_s"] = origin
        meta["hardware"] = arm.connect()
        initial = arm.read()
        check_positions(initial, limits)
        meta["initial_positions"] = initial
        if args.telemetry:
            meta["initial_telemetry"] = read_telemetry(arm, origin)
        if args.trajectory:
            meta["trajectory"] = json.loads(args.trajectory.read_text())
            meta["trajectory_sha256"] = hashlib.sha256(args.trajectory.read_bytes()).hexdigest()
            if meta["trajectory_sha256"] != args.trajectory_sha256:
                raise ValueError("Trajectory changed after validation; start a new experiment")
        if args.mode == "pickup":
            for _, command in schedule:
                check_positions(command, limits)
            if any(abs(initial[j] - schedule[0][1][j]) > 0.5 for j in JOINTS):
                raise ValueError("Position the arm at the taught starting pose before this trial")
            target = schedule[0][1]
            ticks = math.ceil(args.hz)
            alignment = [("align_start", {j: initial[j] + (target[j] - initial[j]) * i / ticks
                                          for j in JOINTS}) for i in range(ticks + 1)]
            schedule = alignment + schedule
        if video:
            video.start(args.output, origin)  # Fail camera setup before enabling motion.
        if args.mode == "joint_response":
            for sign in (-1, 1):
                check_positions(initial | {args.joint: initial[args.joint] + sign * args.amplitude_deg}, limits)
        if args.mode != "read":
            save()  # Persist the configuration before enabling motion.
            arm.enable(initial)
        previous = initial.copy()
        started = time.monotonic()
        clock_offset = started - origin
        meta["motion_start_s"] = clock_offset
        last_tick = started - 1 / args.hz
        with (args.output / "samples.jsonl").open("x", buffering=1) as trace:
            for index, (phase, offset) in enumerate(schedule):
                if video:
                    video.check()
                planned_s = index / args.hz
                if args.backend == "lerobot":
                    # A delayed sample must not cause a burst of rapid commands.
                    deadline = max(started + planned_s, last_tick + 1 / args.hz)
                    time.sleep(max(0.0, deadline - time.monotonic()))
                    if time.monotonic() - started - planned_s > args.max_lateness_s:
                        raise RuntimeError("Sampling deadline missed; aborted without catching up")
                    last_tick = time.monotonic()
                command = None
                sent_at = None
                sent_end = None
                if args.mode != "read":
                    command = (offset.copy() if args.mode == "pickup" else
                               initial | {args.joint: initial[args.joint] + offset})
                    check_positions(command, limits)
                    sent_at = time.monotonic() - origin
                    arm.send(command)
                    sent_end = time.monotonic() - origin
                read_start = time.monotonic() - origin
                measured = arm.read()
                read_end = time.monotonic() - origin
                row = {"sample": index, "planned_s": planned_s, "phase": phase,
                       "command_start_s": sent_at, "command_end_s": sent_end,
                       "read_start_s": read_start, "read_end_s": read_end,
                       "command": command, "measured": measured}
                telemetry_error = None
                if args.telemetry:
                    try:
                        row["telemetry"] = read_telemetry(arm, origin)
                    except Exception as exc:
                        row["telemetry_error"] = f"{type(exc).__name__}: {exc}"
                        telemetry_error = exc
                # Preserve even a bad observation for diagnosis; the checker
                # below will reject it. repr handles non-finite values strictly.
                try:
                    encoded = json.dumps(row, allow_nan=False)
                except ValueError:
                    encoded = json.dumps({"sample": index, "invalid_observation": repr(row)})
                trace.write(encoded + "\n")
                meta["samples"] += 1
                if telemetry_error:
                    raise telemetry_error
                check_positions(measured, limits)
                if args.backend == "lerobot" and time.monotonic() - started - planned_s > args.max_lateness_s:
                    raise RuntimeError("Communication/read deadline missed")
                if command:
                    for joint in JOINTS:
                        bound = args.max_tracking_error_deg if joint != "gripper" else 5.0
                        # Accept either adjacent command during sampling, but
                        # never allow the measured arm to depart from both.
                        if min(abs(measured[joint] - command[joint]),
                               abs(measured[joint] - previous[joint])) > bound:
                            raise RuntimeError(f"Tracking guard exceeded for {joint}")
                    err = measured[args.joint] - command[args.joint]
                    squared_error += err * err
                    max_error = max(max_error, abs(err))
                    previous = command
        meta["status"] = "completed"
        if args.mode == "pickup":
            # Completion is playback completion, not proof of a successful grasp.
            meta["pickup_outcome"] = "unreviewed"
        if args.mode == "joint_response":
            meta["tracking_summary"] = {"rmse_deg": math.sqrt(squared_error / meta["samples"]),
                                        "max_abs_error_deg": max_error,
                                        "note": "Sampled tracking only; not an identified delay or friction model"}
    except BaseException as exc:
        error = exc
        meta["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            arm.close()
            meta["cleanup"] = "completed"
        except BaseException as exc:
            meta["cleanup"] = f"failed: {type(exc).__name__}: {exc}"
            meta["status"] = "failed"
            error = error or exc
        if video:
            try:
                meta["video"] = video.close()
            except BaseException as exc:
                meta["video_error"] = str(exc)
                meta["status"] = "failed"
                error = error or exc
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    if error:
        raise error
    return meta


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mock", "lerobot"), default="mock")
    parser.add_argument("--mode", choices=("read", "joint_response", "pickup"), default="read")
    parser.add_argument("--output", type=Path, required=True, help="New run directory; never overwritten")
    parser.add_argument("--port")
    parser.add_argument("--robot-id")
    parser.add_argument("--execute", action="store_true", help="Explicitly enable real joint motion")
    parser.add_argument("--limits", type=Path, help="Measured local joint envelope JSON")
    parser.add_argument("--joint", choices=ARM_JOINTS, default="shoulder_pan")
    parser.add_argument("--amplitude-deg", type=float, default=3.0)
    parser.add_argument("--ramp-s", type=float, default=3.0)
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=5.0, help="Read-only recording duration")
    parser.add_argument("--max-tracking-error-deg", type=float, default=5.0)
    parser.add_argument("--max-lateness-s", type=float, default=0.2)
    parser.add_argument("--telemetry", action="store_true", help="Read current/load/voltage/temperature and motor goals")
    parser.add_argument("--camera", help="OpenCV camera index or device path; optional outside hardware pickup")
    parser.add_argument("--camera-fps", type=float, default=30)
    parser.add_argument("--trajectory", type=Path, help="Locally taught pickup waypoints JSON")
    parser.add_argument("--piece-id", help="Physical piece identifier, e.g. white-pawn-1")
    parser.add_argument("--condition", help="Recorded condition label, e.g. magnetic-board")
    parser.add_argument("--trials", type=int, default=1, help="Pickup repetitions with a manual reset between trials")
    args = parser.parse_args(argv)
    bounds = {"amplitude_deg": (0.1, 5), "ramp_s": (1, 10), "hold_s": (0.1, 5),
              "hz": (10, 100), "duration_s": (0.1, 300),
              "max_tracking_error_deg": (0.1, 10), "max_lateness_s": (0.01, 0.5),
              "camera_fps": (5, 120)}
    for key, (lo, hi) in bounds.items():
        if not finite(getattr(args, key)) or not lo <= getattr(args, key) <= hi:
            parser.error(f"{key} must be finite and between {lo} and {hi}")
    if not 1 <= args.cycles <= 5 or args.amplitude_deg / args.ramp_s > 5:
        parser.error("Use 1..5 cycles and command speed at most 5 deg/s")
    if args.backend == "lerobot":
        if not args.port or not args.robot_id:
            parser.error("Hardware requires --port and --robot-id")
        if args.mode != "read" and (not args.execute or not args.limits):
            parser.error("Real movement requires --execute and a filled --limits file")
    elif args.execute:
        parser.error("--execute is for the lerobot backend; mock never accesses hardware")
    if not 1 <= args.trials <= 20 or (args.trials != 1 and args.mode != "pickup"):
        parser.error("Use 1..20 trials; repetitions require pickup mode")
    if args.mode == "pickup":
        if not args.trajectory or not args.piece_id or not args.condition:
            parser.error("Pickup needs --trajectory, --piece-id and --condition")
        if args.backend == "lerobot" and not args.camera:
            parser.error("Hardware pickups require --camera to review separation")
        args.telemetry = True
    elif args.trajectory:
        parser.error("--trajectory is only used in pickup mode")
    return args


def main(argv=None):
    args = parse_args(argv)
    limits = read_limits(args.limits) if args.limits else None
    if args.backend == "mock" and limits is None:
        limits = {name: [-20, 20] for name in ARM_JOINTS} | {"gripper": [0, 100]}
    schedule = (response_schedule(args.hz, args.amplitude_deg, args.ramp_s, args.hold_s, args.cycles)
                if args.mode == "joint_response" else [("read", 0.0)] * math.ceil(args.duration_s * args.hz))
    if args.mode == "pickup":
        from pickup_trial import load_pickup
        args.trajectory_sha256 = hashlib.sha256(args.trajectory.read_bytes()).hexdigest()
        _, schedule = load_pickup(args.trajectory, args.hz, JOINTS, limits)
    if len(schedule) / args.hz > 300:
        raise ValueError("Experiment exceeds the 300 second budget")
    if args.backend == "lerobot":
        print("Hardware mode: close all other robot controllers. Joint bounds do not check collisions.", flush=True)
        if args.mode != "read":
            print("Start with the arm supported, gripper empty and torque OFF. Clear the taught path. "
                  "Torque will be disabled on exit; support it against falling. Keep physical power-off accessible.", flush=True)
    output_root = args.output
    if args.trials > 1:
        output_root.mkdir(parents=True, exist_ok=False)
    for trial in range(args.trials):
        if args.trials > 1:
            args.output = output_root / f"trial-{trial + 1:03d}"
        if args.backend == "lerobot" and args.mode == "pickup":
            answer = input(f"Trial {trial + 1}: reset the SAME piece/pose, verify the taught path, "
                           "support the arm at start with torque OFF. Type ready (anything else stops): ")
            if answer.strip().lower() != "ready":
                raise KeyboardInterrupt("Operator stopped trial series")
        arm = LeRobotArm(args.port, args.robot_id) if args.backend == "lerobot" else MockArm()
        video = None
        if args.camera:
            from pickup_trial import VideoRecorder
            video = VideoRecorder(args.camera, args.camera_fps)
        result = run(args, arm, limits, schedule, video)
        print(f"{result['status']}: {result['samples']} samples in {args.output} ({args.backend})")


if __name__ == "__main__":
    def stop(_signum, _frame):
        raise KeyboardInterrupt("Stop requested")

    signal.signal(signal.SIGTERM, stop)
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; inspect the partial recording and support the arm.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
