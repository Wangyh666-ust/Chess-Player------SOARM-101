from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

import requests

from client import BASE
from live_capture import SQUARES
from review_field import build_packet, parse_packet
from server import create_server

EXPECTED_SQUARES = ("b5", "e4", "a6", "d5", "e6")
CAMERAS = ("C1", "C2")

FIXTURE_LABELS = {
    "b5": "unsure",
    "e4": "wP",
    "a6": "wN",
    "d5": "bP",
    "e6": "wP",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_prompt() -> str:
    prompt = (BASE / "prompt_v6.txt").read_text(encoding="utf-8")
    mistakes = (BASE / "common_mistakes_v6.md").read_text(encoding="utf-8")
    return prompt + "\n\n" + mistakes


def _verify_reference() -> tuple[bytes, str]:
    ref_path = BASE / "field_references" / "reference.jpg"
    ref_bytes = ref_path.read_bytes()
    manifest = json.loads(
        (ref_path.parent / "manifest.json").read_text(encoding="utf-8")
    )
    expected = manifest["combined_sha256"]
    actual = _sha256(ref_bytes)
    if actual != expected:
        raise ValueError("reference sha256 mismatch")
    return ref_bytes, actual


def load_inputs():
    sent = json.loads(
        (BASE / "pilot" / "run" / "sent_images.json").read_text(encoding="utf-8")
    )
    squares = sent["squares"]
    if sorted(squares) != sorted(EXPECTED_SQUARES):
        raise ValueError("unexpected square set in sent_images.json")

    image_dir = BASE / "pilot" / "run"
    views: dict = {}
    for square in EXPECTED_SQUARES:
        cams = squares[square]
        if sorted(cams) != sorted(CAMERAS):
            raise ValueError(f"unexpected cameras for {square}")
        views[square] = {}
        for camera in CAMERAS:
            entry = cams[camera]
            data = (image_dir / entry["file"]).read_bytes()
            if _sha256(data) != entry["sha256"]:
                raise ValueError(f"sha256 mismatch for {square}/{camera}")
            views[square][camera] = (data, tuple(entry["center"]))

    reference, reference_sha = _verify_reference()
    prompt = _read_prompt()
    return views, prompt, reference, reference_sha


def _image_blocks(prompt: str, views, reference: bytes) -> int:
    packet = build_packet(prompt, views, reference)
    return sum(
        1
        for block in packet["messages"][0]["content"]
        if block.get("type") == "image_url"
    )


def build_local_payload(prompt: str, views, reference: bytes) -> dict:
    packet = build_packet(prompt, views, reference)
    return {
        "model": "qwen3.5-4b-local",
        "messages": packet["messages"],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": 2300,
        "stream": False,
        "response_format": {"type": "json_object"},
        "cache_prompt": False,
    }


def parse_local_reply(data: dict, squares) -> dict:
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    if choice.get("finish_reason") != "stop":
        return {"error": "finish_reason_not_stop", "opinions": None}
    if not isinstance(content, str) or not content.strip():
        return {"error": "empty_content", "opinions": None}
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        return {"error": "invalid_json", "opinions": None}
    if not isinstance(raw, dict) or not isinstance(raw.get("results"), list):
        return {"error": "schema_error", "opinions": None}
    try:
        opinions = parse_packet(content, list(squares))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"schema_error: {exc}", "opinions": None}
    if not isinstance(opinions, dict):
        return {"error": "schema_error", "opinions": None}
    return {"error": None, "opinions": opinions}


def run_cloud(output: Path, prompt: str, views, reference: bytes) -> dict:
    page_runs = output / "page_runs"

    def fixture_collector(_upstream, _cropper, progress, *, previous=None):
        progress("collecting", 6.0)
        local = dict.fromkeys(SQUARES, "empty")
        local.update(FIXTURE_LABELS)
        return {
            "local": local,
            "requests": views,
            "selected": dict.fromkeys(views, "archived benchmark"),
            "windows": [{"frames": 1}] * 3,
            "observed_seconds": 6,
            "skipped": {},
            "trace": {},
        }

    server = create_server(
        None,
        0,
        runs_dir=page_runs,
        collector=fixture_collector,
        reference_bytes=reference,
    )
    port = server.server_address[1]
    origin = f"http://127.0.0.1:{port}"
    url = f"{origin}/api"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    phases: list[str] = []
    state = None
    try:
        with requests.Session() as session:
            session.trust_env = False
            start = time.monotonic()
            resp = session.get(f"{url}/state", timeout=5)
            resp.raise_for_status()
            state = resp.json()
            token = state.get("token")
            phases.append("state")

            resp = session.post(
                f"{url}/start",
                json={},
                headers={"Origin": origin, "X-Board-Token": token},
                timeout=5,
                allow_redirects=False,
            )
            if resp.status_code != 202:
                raise RuntimeError(f"unexpected start status {resp.status_code}")
            phases.append("start")

            deadline = time.monotonic() + 173.0
            while True:
                if state.get("phase") == "ready" and state.get("draft"):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for ready")
                time.sleep(0.4)
                resp = session.get(f"{url}/state", timeout=5)
                resp.raise_for_status()
                state = resp.json()
                if state.get("phase") not in phases:
                    phases.append(state.get("phase"))
            elapsed = time.monotonic() - start
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    draft = state["draft"]
    draft_id = draft.get("id")
    response = None
    if draft_id:
        reply_path = page_runs / draft_id / "response.json"
        if reply_path.is_file():
            response = json.loads(reply_path.read_text(encoding="utf-8"))

    error = None
    if isinstance(response, dict) and response.get("error"):
        error = response.get("error")
    if draft.get("review_error"):
        error = draft.get("review_error")

    return {
        "mode": "cloud",
        "elapsed_seconds": elapsed,
        "trace": phases,
        "model": state.get("review_model"),
        "page_review_seconds": draft.get("review_seconds"),
        "opinions": draft.get("opinions"),
        "error": error,
        "response": response,
    }


def run_local(prompt: str, views, reference: bytes) -> dict:
    url = "http://127.0.0.1:8768/v1/chat/completions"
    payload = build_local_payload(prompt, views, reference)
    result: dict = {
        "mode": "local",
        "elapsed_seconds": None,
        "model": None,
        "opinions": None,
        "error": None,
        "response": {},
    }

    with requests.Session() as session:
        session.trust_env = False
        start = time.monotonic()
        try:
            resp = session.post(
                url, json=payload, timeout=(5, 115), allow_redirects=False
            )
            elapsed = time.monotonic() - start
            status = resp.status_code
            result["elapsed_seconds"] = elapsed
            try:
                data = resp.json()
            except ValueError:
                result["error"] = f"http_{status}_non_json"
                result["response"] = {"status": status, "content": resp.text}
                return result
            if not isinstance(data, dict):
                result["error"] = f"http_{status}_unexpected_shape"
                result["response"] = {"status": status}
                return result

            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            reasoning = message.get("reasoning_content")
            was_reasoning = isinstance(reasoning, str) and bool(reasoning.strip())
            finish = choice.get("finish_reason")

            result["model"] = data.get("model")
            result["response"] = {
                "status": status,
                "content": message.get("content"),
                "usage": data.get("usage"),
                "timings": data.get("timings"),
                "reasoning_seen": was_reasoning,
                "model": data.get("model"),
                "finish_reason": finish,
            }
            if status != 200:
                result["error"] = f"http_{status}"
                return result
            if data.get("error"):
                result["error"] = "provider_error"
                result["response"]["provider_error"] = data.get("error")
                return result

            parsed = parse_local_reply(data, list(views))
            result["opinions"] = parsed["opinions"]
            result["error"] = parsed["error"]
        except requests.Timeout:
            result["elapsed_seconds"] = time.monotonic() - start
            result["error"] = "timeout"
        except requests.RequestException as exc:
            result["elapsed_seconds"] = time.monotonic() - start
            result["error"] = f"request_error: {exc.__class__.__name__}"

    return result


def _sanitize(value):
    if isinstance(value, dict):
        return {
            key: _sanitize(item)
            for key, item in value.items()
            if key.lower() != "token"
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cloud", "local"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    output: Path = args.output
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)

    views, prompt, reference, reference_sha = load_inputs()
    blocks = _image_blocks(prompt, views, reference)

    started = time.monotonic()
    if args.mode == "cloud":
        result = run_cloud(output, prompt, views, reference)
    else:
        result = run_local(prompt, views, reference)
    end_to_end = time.monotonic() - started

    result["elapsed_seconds"] = end_to_end
    result["image_blocks"] = blocks
    result["reference_sha256"] = reference_sha

    result = _sanitize(result)

    (output / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
