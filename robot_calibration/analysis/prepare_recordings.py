"""Prepare the manual SO-101 handoff offline; never connect to a robot.

Keep source timestamps and raw registers. Frame matching is causal on the host
clock, not a claim of exposure synchronization. Outputs are new directories.
"""

import argparse
from bisect import bisect_right
from collections import Counter
import json
import math
from pathlib import Path

from robot_calibration.records import ARM, JOINTS, read_json, read_rows, sha256, write_json

NORMALIZATION_SOURCE = "https://raw.githubusercontent.com/huggingface/lerobot/v0.7.0/src/lerobot/motors/motors_bus.py"


def interval(value):
    if (
        len(value) != 2
        or not all(math.isfinite(x) for x in value)
        or not 0 <= value[0] <= value[1]
    ):
        raise ValueError(f"Invalid acquisition interval: {value}")

    return value


def normalized_to_degrees(values, calibration):
    """LeRobot 0.7.0 STS3215 range normalization -> calibration-centred degrees.

    DEGREES uses (raw - calibration midpoint) * 360 / 4095. The homing
    offset is already applied by the motor, so applying it again is incorrect.
    Endpoint observations were clipped upstream: report them as censored.
    """

    if set(values) != set(JOINTS) or set(calibration) != set(JOINTS):
        raise ValueError("Expected all six joints and their calibration")

    result, censored = {}, []

    for joint in JOINTS:
        v, c = values[joint], calibration[joint]
        low, high = (0, 100) if joint == "gripper" else (-100, 100)

        if not math.isfinite(v) or not low <= v <= high:
            raise ValueError(f"Invalid normalized value for {joint}: {v}")

        span = c["range_max"] - c["range_min"]

        if span <= 0 or c["drive_mode"] not in (0, 1):
            raise ValueError(f"Invalid calibration for {joint}")

        if v in (low, high):
            censored.append(joint)

        if joint != "gripper":
            sign = -1 if c["drive_mode"] else 1
            result[joint] = sign * v * span * 360 / (200 * 4095)

    return result, censored


def load_frames(path):
    frames = read_rows(path)
    previous = -1

    for i, frame in enumerate(frames):
        start, end = interval([frame["read_start_s"], frame["read_end_s"]])

        if frame["frame"] != i or start < previous:
            raise ValueError(f"Invalid frame ordering: {path}:{i}")

        previous = end

    return frames, [f["read_end_s"] for f in frames]


def match_frame(frames, ends, cutoff, max_age):
    """Latest acquisition completed before observation starts; never future frames."""

    i = bisect_right(ends, cutoff) - 1

    if i < 0:
        return None

    frame = frames[i]

    return {**frame, "age_s": cutoff - ends[i], "fresh": cutoff - ends[i] <= max_age}


def segments_for(meta, review, end):
    """Partition the whole recording, retaining uncertainty and excluded material."""

    if review.get("segments"):
        segments = [dict(s) for s in review["segments"]]

        for s in segments:
            if s["end_s"] is None:
                s["end_s"] = end
    else:
        outcome = meta["outcome"]
        segments = [
            {
                "start_s": 0,
                "end_s": end,
                "kind": "attempt",
                "outcome": outcome,
                "imitation_candidate": meta["status"] == "complete"
                and outcome == "success",
            }
        ]

    previous = 0

    for i, s in enumerate(segments):
        if (
            s["start_s"] != previous
            or not math.isfinite(s["end_s"])
            or not s["start_s"] < s["end_s"] <= end
        ):
            raise ValueError("Segments must be a contiguous partition of the recording")

        if type(s["imitation_candidate"]) is not bool:
            raise ValueError("Segment eligibility must be explicit")

        s["segment_id"] = i
        previous = s["end_s"]

    if previous != end:
        raise ValueError("Segments do not cover the recording tail")

    return segments


def prepare(source, output, review_path, max_frame_age=0.15):
    if not math.isfinite(max_frame_age) or max_frame_age <= 0:
        raise ValueError("Frame age bound must be positive and finite")

    source = source.resolve()
    review = read_json(review_path)
    trial_paths = sorted(source.glob("manual-recordings/*/trial-*/metadata.json"))

    if not trial_paths:
        raise ValueError("No manual recordings found")

    known = {p.parent.relative_to(source).as_posix() for p in trial_paths}

    if set(review["trials"]) - known:
        raise ValueError("Review names a missing trial")

    output.mkdir(parents=True, exist_ok=False)

    summary = {
        "schema_version": 1,
        "status": "running",
        "source_root": str(source),
        "review_sha256": sha256(review_path),
        "processor_sha256": sha256(Path(__file__)),
        "normalization_source": NORMALIZATION_SOURCE,
        "trials": [],
        "units": {
            "arm": "calibration_centred_deg",
            "gripper": "normalized_0_100",
            "current_load": "register_value",
            "time": "host_monotonic_s_from_trial_origin",
        },
        "max_frame_age_s": max_frame_age,
        "limitations": [
            "Motor-to-URDF joint zero/sign mapping is unvalidated",
            "Camera exposure/buffering latency is unknown",
            "Commands are subsampled; no exact hardware replay",
            "Candidate segments retain idle time and require training review",
            "All segments of a recording/session must share a dataset split",
        ],
    }

    write_json(output / "manifest.json", summary)

    try:
        for mp in trial_paths:
            trial = mp.parent
            key = trial.relative_to(source).as_posix()
            meta = read_json(mp)

            if meta.get("method") != "manual_teleop" or "normalized" not in meta.get(
                "units_note", ""
            ):
                raise ValueError(f"Unknown recording unit convention: {key}")

            calibration = read_json(trial / "calibration.json")

            if sha256(trial / "calibration.json") != meta["calibration_sha256"]:
                raise ValueError(f"Calibration hash mismatch: {key}")

            rows = read_rows(trial / "telemetry.jsonl")

            if not rows or len(rows) != meta["samples"]:
                raise ValueError(f"Sample count mismatch: {key}")

            videos = {
                v: load_frames(trial / v / "frames.jsonl") for v in ("fixed", "wrist")
            }
            end = (
                max(
                    rows[-1]["register_time_s"][k][1]
                    for k in rows[-1]["register_time_s"]
                )
                + 1e-6
            )
            detail = review["trials"].get(key, {})
            separation = read_json(trial / "separation.json")

            for name, expected in separation["original_sha256"].items():
                path = (trial / name).resolve()

                if not path.is_relative_to(trial.resolve()) or sha256(path) != expected:
                    raise ValueError(f"Separation evidence mismatch: {key}/{name}")

            segments = segments_for(meta, detail, end)

            for segment in segments:
                segment["source_sample_indices"] = []
                segment["candidate_sample_indices"] = []

            dest = output / key

            dest.mkdir(parents=True)

            inputs = [
                mp,
                trial / "calibration.json",
                trial / "telemetry.jsonl",
                trial / "separation.json",
                *(trial / v / "frames.jsonl" for v in videos),
            ]
            counts = Counter()
            previous = -1

            with (dest / "samples.jsonl").open("x") as stream:
                for i, row in enumerate(rows):
                    obs = interval(row["observation_time_s"])
                    sent = interval(row["send_time_s"])

                    if obs[0] <= previous or obs[1] > sent[0]:
                        raise ValueError(f"Invalid observation/action order: {key}:{i}")

                    previous = obs[0]
                    q, censored = normalized_to_degrees(row["observation"], calibration)
                    command, _ = normalized_to_degrees(row["goal"], calibration)

                    for reg, times in row["register_time_s"].items():
                        interval(times)

                        vals = row["registers"][reg]

                        if set(vals) != set(JOINTS) or not all(
                            math.isfinite(v) for v in vals.values()
                        ):
                            raise ValueError(f"Invalid register values: {key}:{i}")

                    segment = next(
                        s for s in segments if s["start_s"] <= obs[0] < s["end_s"]
                    )
                    frames = {
                        v: match_frame(*data, obs[0], max_frame_age)
                        for v, data in videos.items()
                    }
                    reasons = []

                    if not segment["imitation_candidate"]:
                        reasons.append("segment_not_imitation_candidate")

                    if censored:
                        reasons.append("normalized_position_at_clipped_endpoint")

                    if sent[1] >= segment["end_s"]:
                        reasons.append("action_crosses_segment_boundary")

                    for v, frame in frames.items():
                        if frame is None or not frame["fresh"]:
                            reasons.append(f"{v}_frame_missing_or_stale")
                        elif frame["read_start_s"] < segment["start_s"]:
                            reasons.append(f"{v}_frame_crosses_segment_boundary")

                    result = {
                        "source_sample": i,
                        "time_s": obs[0],
                        "observation_interval_s": obs,
                        "command_interval_s": sent,
                        "arm_position_deg": q,
                        "arm_command_deg": command,
                        "gripper_position_norm": row["observation"]["gripper"],
                        "gripper_command_norm": row["goal"]["gripper"],
                        "censored_joints": censored,
                        "registers": row["registers"],
                        "register_intervals_s": row["register_time_s"],
                        "frames": frames,
                        "segment_id": segment["segment_id"],
                        "imitation_candidate": not reasons,
                        "exclusion_reasons": reasons,
                    }

                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                    segment["source_sample_indices"].append(i)

                    if not reasons:
                        segment["candidate_sample_indices"].append(i)

                    counts["samples"] += 1
                    counts["imitation_candidates"] += not reasons

                    counts.update(reasons)

            record = {
                "trial": key,
                "source_status": meta["status"],
                "source_outcome": meta["outcome"],
                "counts": dict(counts),
                "segments": segments,
                "review": detail,
                "origin_monotonic_s": meta["origin_monotonic_s"],
                "started_utc": meta["started_utc"],
                "split_group": "recording-session-2026-11-12",
                "condition_original": meta["condition"],
                "condition_reported_after_capture": "magnetic-board",
                "source_square_original": meta["square"],
                "source_square": (
                    None if detail.get("source_square_unreliable") else meta["square"]
                ),
                "calibration": calibration,
                "inputs_sha256": {
                    p.relative_to(source).as_posix(): sha256(p) for p in inputs
                },
                "video_paths": {v: str(trial / v / "video.avi") for v in videos},
                "source_separation": separation,
            }

            write_json(dest / "metadata.json", record)
            summary["trials"].append(record)

        write_json(
            output / "segments.json",
            [
                {"trial": t["trial"], "split_group": t["split_group"], **s}
                for t in summary["trials"]
                for s in t["segments"]
            ],
        )

        summary["status"] = "completed"
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "manifest.json", summary)

    print(f"Prepared {len(summary['trials'])} recordings in {output}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--source", type=Path, required=True, help="Extracted handoff root"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "recording_review_20261112.json",
    )
    parser.add_argument("--max-frame-age-s", type=float, default=0.15)

    args = parser.parse_args()

    prepare(args.source, args.output, args.review, args.max_frame_age_s)
