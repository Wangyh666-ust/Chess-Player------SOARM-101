import hashlib
import json

from budget1024_replay_v3 import _run_dashscope
from compact24_replay import build_request as _build_request
from qwen_live import BASE, _load_key

REFERENCE_REL = "field_references/reference_compact24.jpg"
MANIFEST_REL = "field_references/manifest_compact24.json"
PROMPT_REL = "prompt_low_rook_v8_budget1024.txt"
REFERENCE_COUNT = 24
THINKING_BUDGET = 1024

_OLD_ATLAS_LINE = "参考总图上面是近照，下面是带编号的 C1/C2 实拍样本。"
_NEW_ATLAS_LINE = "参考总图由 24 张带编号的 C1/C2 实拍参考照组成。"


def _error(code, message):
    return {"error": code, "message": message, "preflight_ok": False}


def _load_manifest():
    try:
        manifest = json.loads((BASE / MANIFEST_REL).read_text(encoding="utf-8"))
        count = manifest["count"]
        digest = manifest["jpeg_sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        return None, _error("reference_unavailable", "manifest unavailable")
    if count != REFERENCE_COUNT or not isinstance(digest, str) or not digest:
        return None, _error("reference_unavailable", "manifest count/digest invalid")
    return digest, None


def _load_prompt():
    try:
        return (BASE / PROMPT_REL).read_text(encoding="utf-8").strip(), None
    except (OSError, ValueError):
        return None, _error("preflight_failed", "prompt unavailable")


def _render_prompt(prompt, views):
    n = len(views)
    prompt = prompt.replace("本次五格共用", f"本次 {n} 格共用")
    prompt = prompt.replace("完整的五格 JSON", f"完整的 {n} 格 JSON")
    return prompt.replace(_OLD_ATLAS_LINE, _NEW_ATLAS_LINE)


def _normalize(resp):
    provider_error = resp.get("error")
    code = provider_error.get("code") if isinstance(provider_error, dict) else provider_error
    status = resp.get("http_status")
    if code == "credential_unavailable":
        mapped = "credential_unavailable"
    elif status in (401, 403):
        mapped = "authentication_failed"
    elif code in ("timeout", "deadline", "deadline_exceeded"):
        mapped = "timeout"
    elif code == "unfinished":
        mapped = "unfinished_response"
    elif code == "schema":
        mapped = "schema_error"
    elif status in (400, 422):
        mapped = "invalid_provider_request"
    elif status == 429:
        mapped = "provider_rate_limit"
    else:
        mapped = "network_error"
    out = dict(resp, error=mapped, thinking_budget=THINKING_BUDGET, reference_count=REFERENCE_COUNT)
    if isinstance(provider_error, dict):
        out["provider_error"] = provider_error
    return out


def infer(views, reference, *, preflight=False):
    digest, err = _load_manifest()
    if err:
        return err
    prompt, err = _load_prompt()
    if not prompt:
        return err or _error("preflight_failed", "prompt is empty")
    if preflight:
        try:
            payload = (BASE / REFERENCE_REL).read_bytes()
        except OSError:
            return _error("reference_unavailable", "reference unavailable")
        if hashlib.sha256(payload).hexdigest() != digest:
            return _error("reference_unavailable", "reference digest mismatch")
        if not _load_key():
            return _error("credential_unavailable", "no credential for preflight")
        return {"preflight_ok": True}
    if not isinstance(views, dict) or not 1 <= len(views) <= 5:
        return _error("invalid_views", "views must be a dict with 1..5 entries")
    if not isinstance(reference, (bytes, bytearray)):
        return _error("invalid_reference", "reference must be raw bytes")
    if hashlib.sha256(bytes(reference)).hexdigest() != digest:
        return _error("reference_unavailable", "reference digest mismatch")
    prompt = _render_prompt(prompt, views)
    try:
        messages = _build_request(views, bytes(reference), prompt, REFERENCE_COUNT)
    except (ValueError, TypeError, KeyError):
        return _error("schema_error", "invalid square crops")
    for message in messages:
        for part in message["content"]:
            if part["type"] == "image_url":
                part["image_url"].pop("detail", None)
    resp = _run_dashscope("max", messages, list(views))
    if resp.get("status") != "success":
        return _normalize(resp)
    opinions = resp.get("opinions") or {}
    if opinions.keys() != views.keys():
        out = dict(resp)
        out["status"] = "error"
        out["error"] = "schema_error"
        out["provider_error"] = {"code": "schema", "message": "opinion keys mismatch"}
        return out
    out = dict(resp)
    out["result"] = opinions
    out["thinking_budget"] = THINKING_BUDGET
    out["reference_count"] = REFERENCE_COUNT
    return out
