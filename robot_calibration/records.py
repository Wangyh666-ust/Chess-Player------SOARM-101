"""Joint order and JSON/hash helpers shared by calibration and manipulation."""

import hashlib
import json

ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
JOINTS = (*ARM, "gripper")


def read_json(path):
    """Read JSON, accepting a UTF-8 byte-order mark from external exports."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path):
    """Read nonempty JSON-lines records in their original order."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    """Hash file contents for provenance and change detection."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    """Write finite JSON; callers own output-directory/overwrite protection."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
