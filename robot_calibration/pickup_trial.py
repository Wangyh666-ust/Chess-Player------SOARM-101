"""Validate a taught pickup and record camera frames on the telemetry clock."""

import json
import math
from pathlib import Path
import threading
import time


PHASES = ("start", "approach", "close", "lift", "hold", "lower", "release", "return")


def load_pickup(path, hz, joints, limits):
    from collect_robot_calibration import UNITS, check_positions, finite

    data = json.loads(path.read_text())
    if data.get("units") != UNITS:
        raise ValueError("Pickup trajectory units must match the collector")
    points = data.get("waypoints", [])
    if [p.get("phase") for p in points] != list(PHASES):
        raise ValueError(f"Pickup phases must be {PHASES}")
    for point in points:
        check_positions(point["positions"], limits)
        if not finite(point.get("time_s")) or not 0 <= point["time_s"] <= 300:
            raise ValueError("Waypoint time must be finite and within 0..300 s")
    if points[0]["time_s"] != 0 or points[-1]["positions"] != points[0]["positions"]:
        raise ValueError("Start at t=0 and return to exactly the initial pose")
    if points[2]["positions"]["gripper"] >= points[1]["positions"]["gripper"]:
        raise ValueError("Close must reduce the calibrated gripper opening")
    if points[6]["positions"]["gripper"] <= points[5]["positions"]["gripper"]:
        raise ValueError("Release must increase the calibrated gripper opening")
    for previous, point in zip(points, points[1:]):
        duration = point["time_s"] - previous["time_s"]
        if duration < 1 / hz:
            raise ValueError("Waypoints must be at least one sample apart")
        for joint in joints:
            delta = abs(point["positions"][joint] - previous["positions"][joint])
            cap = 20.0 if joint == "gripper" else 5.0
            if delta / duration > cap:
                raise ValueError(f"{point['phase']}: {joint} exceeds {cap} units/s")
            if joint != "gripper" and abs(point["positions"][joint] - points[0]["positions"][joint]) > 20:
                raise ValueError("Pickup arm excursion must stay within 20 degrees of start")
            if point["phase"] in ("close", "release") and joint != "gripper" and delta != 0:
                raise ValueError("Keep arm angles fixed during closing and release")
            if point["phase"] == "hold" and delta != 0:
                raise ValueError("Hold must keep every joint unchanged")
            if point["phase"] in ("approach", "lift", "lower") and joint == "gripper" and delta != 0:
                raise ValueError("Keep gripper opening unchanged during approach/lift/lower")
    schedule = []
    segment = 1
    for tick in range(math.ceil(points[-1]["time_s"] * hz) + 1):
        t = min(tick / hz, points[-1]["time_s"])
        while segment < len(points) - 1 and t > points[segment]["time_s"]:
            segment += 1
        a, b = points[segment - 1], points[segment]
        fraction = (t - a["time_s"]) / (b["time_s"] - a["time_s"])
        positions = {j: a["positions"][j] + fraction * (b["positions"][j] - a["positions"][j])
                     for j in joints}
        schedule.append((b["phase"], positions))
    # Confirm final placement/return tracking for one second before torque-off.
    schedule.extend([("final_hold", points[-1]["positions"].copy())] * math.ceil(hz))
    return data, schedule


class VideoRecorder:
    """Camera in its own thread: disk/camera I/O never sends robot commands.

    Frames and motor reads share a host monotonic origin, not a hardware trigger.
    Camera buffering/exposure latency is unknown and must not be called exact sync.
    """

    def __init__(self, source, fps):
        self.source = source
        self.fps = fps
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.count = 0
        self.last_frame = None
        self.thread = None

    def start(self, directory, origin):
        self.directory = Path(directory)
        self.origin = origin
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()
        if not self.ready.wait(5):
            raise RuntimeError("Camera did not produce its first frame within 5 seconds")
        self.check()

    def _capture(self):
        capture = None
        writer = None
        try:
            import cv2

            source = int(self.source) if self.source.isdigit() else self.source
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open camera {self.source}")
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # A request; not supported by every backend.
            with (self.directory / "frames.jsonl").open("x", buffering=1) as frames:
                while not self.stop.is_set():
                    before = time.monotonic() - self.origin
                    ok, frame = capture.read()
                    after = time.monotonic() - self.origin
                    if not ok:
                        raise RuntimeError("Camera frame acquisition failed")
                    if writer is None:
                        size = (frame.shape[1], frame.shape[0])
                        writer = cv2.VideoWriter(str(self.directory / "video.avi"),
                                                 cv2.VideoWriter_fourcc(*"MJPG"), self.fps, size)
                        if not writer.isOpened():
                            raise RuntimeError("Cannot create MJPEG video")
                    elif (frame.shape[1], frame.shape[0]) != size:
                        raise RuntimeError("Camera frame dimensions changed during recording")
                    writer.write(frame)
                    frames.write(json.dumps({"frame": self.count, "read_start_s": before,
                                             "read_end_s": after}) + "\n")
                    self.count += 1
                    self.last_frame = time.monotonic()
                    self.ready.set()
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready.set()
        finally:
            if writer is not None:
                writer.release()
            if capture is not None:
                capture.release()

    def check(self):
        if self.error:
            raise RuntimeError(self.error)
        if self.last_frame is not None and time.monotonic() - self.last_frame > 1:
            raise RuntimeError("No fresh camera frame for over one second")

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError("Camera worker did not exit; video may be incomplete")
        if self.error:
            raise RuntimeError(self.error)
        return {"frames": self.count, "file": "video.avi", "timestamps": "frames.jsonl",
                "encoding_fps": self.fps,
                "timing": "Use frames.jsonl, not AVI playback time; camera buffer latency unknown"}
