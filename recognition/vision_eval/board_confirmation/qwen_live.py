import hashlib
import json
import time
from pathlib import Path

import requests

from client import BASE, ROOT
from qwen_trial import infer as infer_qwen

_PROMPT_NAMES = ("prompt_v6.txt", "common_mistakes_v6.md")
_MANIFEST_REL = Path("field_references") / "manifest.json"
_REFERENCE_REL = Path("field_references") / "reference.jpg"
_CRED_NAME = "DashScopeAPI.txt"


def _load_prompt():
    parts = []
    for name in _PROMPT_NAMES:
        try:
            parts.append((BASE / name).read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            return None
    return "\n\n".join(parts)


def _manifest():
    try:
        data = json.loads((BASE / _MANIFEST_REL).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    sha = data.get("combined_sha256") if isinstance(data, dict) else None
    if not isinstance(sha, str):
        return None
    return sha.strip().lower()


def _load_key():
    try:
        text = (ROOT / _CRED_NAME).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return None
    token = text.strip()
    if not token:
        return None
    if not token.isascii() or any(not 33 <= ord(ch) <= 126 for ch in token):
        return None
    return token


def infer(views, reference, *, preflight=False):
    key = _load_key()
    if key is None:
        return {"preflight_ok": False, "error": "credential_unavailable"} if preflight else {
            "error": "credential_unavailable"
        }

    manifest_sha = _manifest()
    if manifest_sha is None:
        return {"preflight_ok": False, "error": "reference_unavailable"} if preflight else {
            "error": "reference_unavailable"
        }

    if preflight:
        try:
            digest = hashlib.sha256((BASE / _REFERENCE_REL).read_bytes()).hexdigest()
        except (OSError, ValueError):
            return {"preflight_ok": False, "error": "reference_unavailable"}
        if digest.strip().lower() != manifest_sha:
            return {"preflight_ok": False, "error": "reference_unavailable"}
        if _load_prompt() is None:
            return {"preflight_ok": False, "error": "preflight_failed"}
        return {"preflight_ok": True}

    prompt = _load_prompt()
    if prompt is None:
        return {"error": "preflight_failed"}

    if not isinstance(reference, (bytes, bytearray)):
        return {"error": "reference_unavailable"}
    if hashlib.sha256(bytes(reference)).hexdigest().strip().lower() != manifest_sha:
        return {"error": "reference_unavailable"}

    started = time.monotonic()
    try:
        provider = infer_qwen(prompt, views, reference, key, post=requests.post)
    except Exception:
        elapsed = round(time.monotonic() - started, 3)
        return {"error": "invalid_response", "elapsed_seconds": elapsed}
    elapsed = round(time.monotonic() - started, 3)

    if not isinstance(provider, dict):
        return {"error": "invalid_response", "elapsed_seconds": elapsed}
    status = provider.get("status")
    out = dict(provider, elapsed_seconds=elapsed)
    result = out.pop("parsed", None)

    if provider.get("ok") is True:
        if not isinstance(result, dict) or result.keys() != views.keys():
            return dict(out, error="invalid_response")
        out["result"] = result
        return out

    if status in (401, 403):
        out["error"] = "authentication_failed"
        return out
    if provider.get("error_code") == "timeout" or status == 408:
        out["error"] = "timeout"
        return out

    code = provider.get("error_code")
    if isinstance(code, str) and code:
        out["error"] = code
        return out
    category = provider.get("error_category")
    if isinstance(category, str) and category:
        out["error"] = category
        return out
    out["error"] = "invalid_response"
    return out
