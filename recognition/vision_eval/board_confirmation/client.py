"""Invoke the checked vision operation without passing detection metadata."""

import base64
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RUNTIME = BASE / "implementation" / "runtime_v2"
STEREO_RUNTIME = BASE / "implementation" / "runtime_field_v2"
FIELD_FOLLOWUP_RUNTIME = BASE / "implementation" / "runtime_field_followup_v1"


def _encode(jpeg):
    return base64.b64encode(jpeg).decode("ascii")


def infer(views, reference, *, preflight=False):
    return _infer(views, reference, preflight=preflight, runtime=RUNTIME, max_tokens=1500)


def infer_stereo(views, reference, *, preflight=False):
    return _infer(views, reference, preflight=preflight, runtime=STEREO_RUNTIME, max_tokens=16000, timeout=165)


def infer_field_followup(views, reference, *, preflight=False):
    return _infer(views, reference, preflight=preflight, runtime=FIELD_FOLLOWUP_RUNTIME,
                  max_tokens=16000, timeout=165)


def _infer(views, reference, *, preflight, runtime, max_tokens, timeout=50):
    args = [
        sys.executable, "-X", "utf8", str(ROOT / "_model_tools/deepseek_codegen.py"),
        "--vision", "--task", str(runtime / "task.txt"),
        "--output", str(runtime / "unused.py"), "--max-tokens", str(max_tokens),
        "--contract", str(runtime / "contract.json"),
    ]
    if preflight:
        args.append("--preflight")
        payload = ""
    else:
        payload = json.dumps({
            "requests": {
                square: {
                    camera: {"jpeg_b64": _encode(jpeg), "center": center}
                    for camera, (jpeg, center) in pair.items()
                }
                for square, pair in views.items()
            },
            "reference_jpeg_b64": _encode(reference),
        })
    try:
        completed = subprocess.run(
            args, input=payload, capture_output=True, encoding="utf8",
            cwd=ROOT, timeout=15 if preflight else timeout,
        )
        result = json.loads(completed.stdout)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except (ValueError, OSError):
        return {"error": "dispatch_failed"}
    if completed.returncode or not isinstance(result, dict):
        return {"error": "dispatch_failed"}
    return result
