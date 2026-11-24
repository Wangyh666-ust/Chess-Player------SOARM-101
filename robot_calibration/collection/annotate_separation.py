"""Export numbered frames for inspection, then record a human-reviewed separation interval.

This does not infer separation from motor current. Choose the last frame with
board contact and the first frame with a visible gap, or record uncertainty.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def annotate(trial, reviewer, outcome, last_contact=None, first_clear=None):
    metadata_path = trial / "metadata.json"
    frames_path = trial / "frames.jsonl"
    metadata = json.loads(metadata_path.read_text())
    frames = [json.loads(line) for line in frames_path.read_text().splitlines()]
    if not frames or [f["frame"] for f in frames] != list(range(len(frames))):
        raise ValueError("Missing or non-contiguous frame log")
    if not reviewer.strip():
        raise ValueError("Reviewer name is required")
    result = {"schema_version": 1, "reviewer": reviewer, "outcome": outcome,
              "reviewed_utc": datetime.now(timezone.utc).isoformat(),
              "run_status": metadata["status"], "backend": metadata["backend"],
              "frames_sha256": hashlib.sha256(frames_path.read_bytes()).hexdigest(),
              "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
              "timing_note": "Host acquisition interval only; exposure/buffer latency is unknown"}
    if outcome == "separated":
        if (type(last_contact) is not int or type(first_clear) is not int
                or not 0 <= last_contact < first_clear < len(frames)):
            raise ValueError("Provide ordered valid last-contact and first-clear frame indices")
        lower = frames[last_contact]["read_start_s"]
        upper = frames[first_clear]["read_end_s"]
        if not 0 <= lower < upper:
            raise ValueError("Invalid acquisition timestamps")
        result.update(last_contact_frame=last_contact, first_clear_frame=first_clear,
                      host_interval_s=[lower, upper])
        # Identify records overlapping the visual interval, without conflating
        # each register's acquisition time with the position sampling time.
        matches = []
        for line in (trial / "samples.jsonl").read_text().splitlines():
            row = json.loads(line)
            intervals = [(row.get("read_start_s"), row.get("read_end_s"))]
            intervals.extend((v["read_start_s"], v["read_end_s"])
                             for v in row.get("telemetry", {}).values())
            if any(a is not None and b is not None and a <= upper and b >= lower for a, b in intervals):
                matches.append(row["sample"])
        result["overlapping_samples"] = matches
    elif outcome not in ("no-separation", "occluded"):
        raise ValueError("Unknown outcome")
    elif last_contact is not None or first_clear is not None:
        raise ValueError("Only a confirmed separation can have event frame indices")
    with (trial / "separation.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def export_frames(trial, start, end):
    import cv2

    frames = [json.loads(line) for line in (trial / "frames.jsonl").read_text().splitlines()]
    if not 0 <= start <= end < len(frames):
        raise ValueError("Frame range is outside the recorded video")
    directory = trial / f"review-frames-{start}-{end}"
    directory.mkdir(exist_ok=False)
    capture = cv2.VideoCapture(str(trial / "video.avi"))
    try:
        if not capture.isOpened():
            raise RuntimeError("Cannot decode recorded video")
        # Sequential decode avoids inaccurate seeking in compressed video.
        for index in range(end + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended before frame {index}; do not annotate it as complete")
            if index >= start:
                if not cv2.imwrite(str(directory / f"frame-{index:06d}.jpg"), frame):
                    raise RuntimeError("Failed to export review frame")
    finally:
        capture.release()
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--export-frames", nargs=2, type=int, metavar=("FIRST", "LAST"))
    parser.add_argument("--reviewer")
    parser.add_argument("--outcome", choices=("separated", "no-separation", "occluded"))
    parser.add_argument("--last-contact-frame", type=int)
    parser.add_argument("--first-clear-frame", type=int)
    args = parser.parse_args()
    if args.export_frames:
        if args.outcome:
            parser.error("Export and inspect frames before recording the annotation")
        print(export_frames(args.trial, *args.export_frames))
    else:
        if not args.reviewer or not args.outcome:
            parser.error("Annotation requires --reviewer and --outcome")
        print(json.dumps(annotate(args.trial, args.reviewer, args.outcome,
                                  args.last_contact_frame, args.first_clear_frame), indent=2))


if __name__ == "__main__":
    main()
