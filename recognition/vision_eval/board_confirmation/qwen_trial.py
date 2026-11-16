import hashlib
import json

import requests

from client import BASE, ROOT
from review_field import build_packet, parse_packet

MODEL = "qwen3.8-max"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
TIMEOUT = (10, 150)


def packet(prompt, views, reference):
    body = build_packet(prompt, views, reference)
    body.pop("thinking", None)

    body["model"] = MODEL
    body["enable_thinking"] = True
    body["reasoning_effort"] = "xhigh"
    body["vl_high_resolution_images"] = True
    body["max_completion_tokens"] = 16000
    body.pop("max_tokens", None)
    body["stream"] = False
    body["response_format"] = {"type": "json_object"}
    return body


def _safe_error(status, resp, credential):
    out = {"ok": False, "status": status}
    code = None
    try:
        if resp is not None:
            j = resp.json()
            if isinstance(j, dict):
                err = j.get("error")
                if isinstance(err, dict):
                    code = err.get("code")
                if code is None and isinstance(j.get("code"), str):
                    code = j.get("code")
    except Exception:
        code = None
    if isinstance(code, str) and code and credential not in code and len(code) <= 80 and all(x.isascii() and (x.isalnum() or x in "_.-") for x in code):
        out["error_code"] = code
    else:
        out["error_code"] = "http_error"
    return out


def infer(prompt, views, reference, key, *, post=requests.post):
    body = packet(prompt, views, reference)
    headers = {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
    }
    try:
        resp = post(URL, json=body, headers=headers, timeout=TIMEOUT, allow_redirects=False)
    except requests.exceptions.Timeout:
        return {"ok": False, "status": None, "error_code": "timeout"}
    except requests.exceptions.RequestException:
        return {"ok": False, "status": None, "error_code": "network_error"}

    status = getattr(resp, "status_code", None)
    if status is None:
        return {"ok": False, "status": None, "error_code": "invalid_response"}
    if status != 200:
        return _safe_error(status, resp, key)

    try:
        data = resp.json()
    except Exception:
        return {"ok": False, "status": status, "error_code": "invalid_json"}

    try:
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content")
    except Exception:
        return {"ok": False, "status": status, "error_code": "invalid_response"}
    if not isinstance(content, str):
        content = ""
    content = content.replace(key, "[redacted]")

    reasoning = message.get("reasoning_content")
    has_reasoning = isinstance(reasoning, str) and len(reasoning) > 0

    finish_reason = choice.get("finish_reason")
    model = data.get("model")
    usage = data.get("usage")

    result = {
        "ok": False,
        "status": status,
        "content": content,
        "has_reasoning": has_reasoning,
        "finish_reason": finish_reason,
        "model": model,
        "usage": usage,
    }

    if finish_reason == "stop":
        try:
            result["parsed"] = parse_packet(content, list(views))
            result["ok"] = True
        except Exception as exc:
            result["parsed"] = None
            result["error_category"] = "schema_error"
            result["validation_reason"] = str(exc).replace(key, "[redacted]")[:250]

    else:
        result["error_category"] = "unfinished_response"
    return result


def main():
    key_path = ROOT / "DashScopeAPI.txt"
    key = key_path.read_text(encoding="utf-8-sig").strip()
    if not key:
        raise SystemExit("missing credential")
    if any(c.isspace() for c in key):
        raise SystemExit("invalid credential")

    sent_path = BASE / "pilot/run/sent_images.json"
    sent = json.loads(sent_path.read_text(encoding="utf-8"))
    squares = sent["squares"]

    views = {}
    for sid, sq in squares.items():
        views[sid] = {}
        for cname in ("C1", "C2"):
            ent = sq[cname]
            p = sent_path.parent / ent["file"]
            b = p.read_bytes()
            if hashlib.sha256(b).hexdigest() != ent["sha256"]:
                raise SystemExit("image sha mismatch")
            views[sid][cname] = b, tuple(ent["center"])

    ref_path = BASE / "field_references/reference.jpg"
    ref_bytes = ref_path.read_bytes()
    manifest = json.loads((BASE / "field_references/manifest.json").read_text(encoding="utf-8"))
    expected_ref = manifest["combined_sha256"]
    if hashlib.sha256(ref_bytes).hexdigest() != expected_ref:
        raise SystemExit("reference sha mismatch")

    prompt = (BASE / "prompt_v6.txt").read_text(encoding="utf-8")
    prompt += "\n\n" + (BASE / "common_mistakes_v6.md").read_text(encoding="utf-8")
    image_count = sum(part["type"] == "image_url" for part in packet(prompt, views, ref_bytes)["messages"][0]["content"])

    out_dir = BASE / "pilot/qwen38_rook_v1"
    out_dir.mkdir(parents=True, exist_ok=False)

    result = infer(prompt, views, ref_bytes, key)

    (out_dir / "response.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    meta = {
        "model": MODEL,
        "squares": list(views),
        "reference_sha256": expected_ref,
        "image_blocks": image_count,
        "prompt_source": "prompt_v6.txt + common_mistakes_v6.md",
        "finish_reason": result.get("finish_reason"),
        "has_reasoning": result.get("has_reasoning"),
        "error_category": result.get("error_category"),
    }
    (out_dir / "input_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
