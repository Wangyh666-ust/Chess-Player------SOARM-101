"""Independent 1024-token thinking-budget replay experiment for board confirmation.

Reuses the compact24_replay helpers (load_inputs, build_request, OUT_JPG,
OUT_MANIFEST) and the multimodel_review_trial transport, but requests
enable_thinking=True with thinking_budget=1024 instead of reasoning_effort.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests

from multimodel_review_trial import (
    DASHSCOPE,
    _model_for,
    _load_key,
    _http_error,
    _strip_credentials,
    parse_generated,
)

from compact24_replay import (
    load_inputs,
    build_request,
    OUT_JPG,
    OUT_MANIFEST,
)


def _run_dashscope(mode: str, msgs: list, squares: list) -> dict:
    url = DASHSCOPE
    model = _model_for(mode)
    body = {
        "model": model,
        "messages": msgs,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": True,
        "thinking_budget": 1024,
        "max_completion_tokens": 5000,
    }
    key = _load_key()
    result: dict[str, Any] = {
        "mode": mode, "model": model, "elapsed_seconds": None, "first_token_seconds": None,
        "first_answer_seconds": None, "status": "error", "finish_reason": None,
        "error": None, "raw_content": None, "opinions": None, "usage": None,
        "reasoning_chars": 0, "image_blocks": 0, "reference_sha256": None,
    }
    if not key:
        return dict(result, error={"code": "credential_unavailable"})
    start = time.monotonic()
    deadline = start + 80
    first_reason = None
    first_answer = None
    answer_parts: list[str] = []
    reasoning_chars = 0
    finish = None
    usage = None
    try:
        with requests.Session() as s:
            s.trust_env = True
            hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            r = s.post(url, json=body, headers=hdr, timeout=(8, 65), stream=True, allow_redirects=False)
            result["http_status"] = r.status_code
            if r.status_code != 200:
                result["error"] = _http_error(r, key)
                result["elapsed_seconds"] = round(time.monotonic() - start, 3)
                return result
            for raw in r.iter_lines():
                if time.monotonic() > deadline:
                    raise TimeoutError("deadline")
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line:
                    continue
                if line == "[DONE]":
                    break
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if isinstance(ev, dict) and ev.get("error"):
                    error = ev["error"]
                    result["error"] = {"code": "provider_stream_error",
                                       "message": _strip_credentials(str(error), key)[:400]}
                    break
                if isinstance(ev, dict) and ev.get("usage"):
                    usage = ev["usage"]
                ch = ev.get("choices", []) if isinstance(ev, dict) else []
                if not isinstance(ch, list) or not ch:
                    continue
                c = ch[0]
                if c.get("finish_reason"):
                    finish = c["finish_reason"]
                delta = c.get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                rc = delta.get("reasoning_content") or ""
                ct = delta.get("content") or ""
                if rc:
                    reasoning_chars += len(rc)
                    if first_reason is None:
                        first_reason = time.monotonic() - start
                if ct:
                    if first_answer is None:
                        first_answer = time.monotonic() - start
                    answer_parts.append(ct)
        result["raw_content"] = _strip_credentials("".join(answer_parts), key)
        result["reasoning_chars"] = reasoning_chars
        result["usage"] = usage
        result["finish_reason"] = finish
        if result["error"] is None:
            if finish != "stop":
                result["error"] = {"code": "unfinished", "message": f"finish_reason={finish}"}
            else:
                try:
                    result["opinions"] = parse_generated(result["raw_content"], squares)
                    result["status"] = "success"
                except ValueError as e:
                    result["error"] = {"code": "schema", "message": str(e)}
        first = first_reason if first_reason is not None else first_answer
        result["first_token_seconds"] = round(first, 3) if first is not None else None
        result["first_answer_seconds"] = round(first_answer, 3) if first_answer is not None else None
        result["elapsed_seconds"] = round(time.monotonic() - start, 3)
        return result
    except requests.exceptions.Timeout as e:
        result["error"] = {"code": "timeout", "message": str(e)[:400]}
    except TimeoutError:
        result["error"] = {"code": "deadline", "message": "absolute deadline exceeded"}
    except Exception as e:
        result["error"] = {"code": "exception", "message": str(e)[:400]}
    result["first_token_seconds"] = round(first_reason, 3) if first_reason is not None else None
    result["first_answer_seconds"] = round(first_answer, 3) if first_answer is not None else None
    result["elapsed_seconds"] = round(time.monotonic() - start, 3)
    result["reasoning_chars"] = reasoning_chars
    result["raw_content"] = _strip_credentials("".join(answer_parts), key) if answer_parts else None
    if result["error"] and "message" in result["error"]:
        result["error"]["message"] = _strip_credentials(result["error"]["message"], key)
    return result


def _validate_manifest(reference_count: int, actual_sha: str) -> None:
    manifest = json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))
    count = manifest.get("count")
    if count != 24:
        raise ValueError(f"manifest count mismatch: {count!r} != 24")
    recorded = manifest.get("jpeg_sha256")
    if recorded != actual_sha:
        raise ValueError(
            f"manifest jpeg_sha256 mismatch: {recorded!r} != {actual_sha!r}"
        )
    if reference_count != 24:
        raise ValueError(f"reference_count mismatch: {reference_count!r} != 24")


def _count_image_blocks(messages: list) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                total += 1
    return total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="board confirmation 1024 thinking-budget replay")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    prompt = args.prompt.read_text(encoding="utf-8").strip()

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=False)

    views, _original_prompt, _original_atlas_bytes, _original_atlas_sha256 = load_inputs()
    atlas_bytes = OUT_JPG.read_bytes()
    atlas_sha = hashlib.sha256(atlas_bytes).hexdigest()

    _validate_manifest(24, atlas_sha)

    messages = build_request(views, atlas_bytes, prompt, 24)
    if not isinstance(messages, list):
        raise TypeError(f"build_request must return a list, got {type(messages)!r}")

    squares = list(views)
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    image_blocks = _count_image_blocks(messages)

    result = _run_dashscope("max", messages, squares)

    result["thinking_budget"] = 1024
    result["max_completion_tokens"] = 5000
    result["reference_sha256"] = atlas_sha
    result["reference_count"] = 24
    result["image_blocks"] = image_blocks
    result["prompt_sha256"] = prompt_sha256

    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
