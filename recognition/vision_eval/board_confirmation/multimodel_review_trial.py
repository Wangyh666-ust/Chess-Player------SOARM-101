from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

from qwen_speed_trial import load_inputs
from review_field import build_packet
from qwen_live import _load_key

LABELS = ["wP", "wR", "wN", "wB", "wQ", "wK", "bP", "bR", "bN", "bB", "bQ", "bK", "empty", "unsure"]
HUMAN = {
    "wP": "white pawn", "wR": "white rook", "wN": "white knight", "wB": "white bishop",
    "wQ": "white queen", "wK": "white king", "bP": "black pawn", "bR": "black rook",
    "bN": "black knight", "bB": "black bishop", "bQ": "black queen", "bK": "black king",
    "empty": "clearly empty square", "unsure": "unresolved or conflicting evidence",
}
BASE = "严格国际象棋裁判。每个格子独立判断，不参考邻居或初始局面。先看同一组参考照片，再看两张目标裁图；从底座依次检查轮廓、躯干、顶部。"
INSTR = BASE + "圆形投影、反光或中心按钮不能单独证明是兵，除非可见连续球体接在颈部。车：短身、宽平圆顶、低中央按钮。王：宽顶盘上有独立直立顶饰。后：冠沿。象：帽或缺口。马：不对称马头。不猜测不可见细节，证据不清或冲突时弃权。"
VALEN_INSTR = INSTR + '只能从以下14项选择：wP,wR,wN,wB,wQ,wK,bP,bR,bN,bB,bQ,bK,empty,unsure。'
GEN_INSTR = VALEN_INSTR + '只输出JSON，不要其他文字：{"results":[{"square":"a1","label":"unsure","C1":"简短可见线索","C2":"看不清"}]}，每个目标格恰一行，C1/C2各1–44字符。'
DASHSCOPE = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
VALEN_URL = "https://api.valen-team.workers.dev/v1/decisions"
STEP_URL = "http://127.0.0.1:8769/v1/chat/completions"


def _square_center(views: Any, sq: str) -> str:
    return "，".join(f"{camera}中心({center[0]},{center[1]})"
                     for camera, (_, center) in views[sq].items())


def build_messages(views: Any, reference: Any) -> list:
    return build_packet(GEN_INSTR, views, reference)["messages"]


def make_valen_request(views: Any, reference: Any) -> dict:
    packet = build_packet(VALEN_INSTR, views, reference)
    msgs = packet["messages"]
    content = []
    for b in msgs[0]["content"]:
        if b.get("type") == "text":
            content.append({"type": "input_text", "text": b["text"]})
        elif b.get("type") == "image_url":
            content.append({"type": "input_image", "image_url": b["image_url"]["url"]})
    questions = []
    for sq in views:
        questions.append({
            "type": "choice",
            "name": sq,
            "instructions": f"格子{sq}（{_square_center(views, sq)}）的同一枚目标棋子是什么？",
            "choices": [{"value": l, "description": HUMAN[l]} for l in LABELS],
        })
    return {
        "model": "valen-4b",
        "temperature": 1.0,
        "input": [{"role": "user", "content": content}],
        "questions": questions,
    }


def _valid_clue(s: Any) -> bool:
    return isinstance(s, str) and 1 <= len(s) <= 44


def _extract_json(content: str) -> Any:
    s = content.strip()
    m = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", s, re.S)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except Exception as e:
        raise ValueError(f"invalid JSON: {e}")


def parse_generated(content: str, squares: list | None = None) -> dict:
    if not isinstance(content, str):
        raise ValueError("content must be string")
    obj = _extract_json(content)
    if not isinstance(obj, dict) or set(obj.keys()) != {"results"}:
        raise ValueError("expected exactly results")
    res = obj["results"]
    if not isinstance(res, list):
        raise ValueError("results must be list")
    expected = list(squares) if squares is not None else [r.get("square") for r in res if isinstance(r, dict)]
    out = {}
    seen = set()
    for r in res:
        if not isinstance(r, dict) or set(r.keys()) != {"square", "label", "C1", "C2"}:
            raise ValueError("bad row keys")
        sq = r["square"]
        if not isinstance(sq, str) or sq not in expected:
            raise ValueError("bad square")
        if sq in seen:
            raise ValueError("duplicate square")
        seen.add(sq)
        if r["label"] not in LABELS:
            raise ValueError("bad label")
        if not _valid_clue(r["C1"]) or not _valid_clue(r["C2"]):
            raise ValueError("bad clue")
        out[sq] = {"label": r["label"], "C1": r["C1"], "C2": r["C2"]}
    if set(out.keys()) != set(expected):
        raise ValueError("missing squares")
    return out


def parse_valen(data: Any, squares: list | None = None) -> dict:
    if not isinstance(data, dict):
        raise ValueError("data must be object")
    answers = data.get("answers")
    if not isinstance(answers, list):
        raise ValueError("answers must be list")
    expected = list(squares) if squares is not None else [a.get("name") for a in answers if isinstance(a, dict)]
    out = {}
    seen = set()
    for a in answers:
        if not isinstance(a, dict):
            raise ValueError("bad answer")
        name = a.get("name")
        if not isinstance(name, str) or name not in expected:
            raise ValueError("bad name")
        if name in seen:
            raise ValueError("duplicate name")
        seen.add(name)
        if a.get("type") != "choice":
            raise ValueError("bad type")
        choice = a.get("choice")
        if choice not in LABELS:
            raise ValueError("bad choice")
        probs = a.get("probabilities")
        if not isinstance(probs, list):
            raise ValueError("bad probabilities")
        clean_probs = []
        scored = set()
        for p in probs:
            if not isinstance(p, dict) or "value" not in p or "probability" not in p:
                raise ValueError("bad prob entry")
            if not isinstance(p["value"], str) or p["value"] not in LABELS or p["value"] in scored:
                raise ValueError("bad prob label")
            scored.add(p["value"])
            prob = p["probability"]
            if not isinstance(prob, (int, float)) or not math.isfinite(prob) or not (0 <= prob <= 1):
                raise ValueError("bad probability")
            clean_probs.append({"value": p["value"], "probability": float(prob)})
        conf = a.get("confidence")
        if conf is not None:
            if not isinstance(conf, (int, float)) or not math.isfinite(conf) or not (0 <= conf <= 1):
                raise ValueError("bad confidence")
            conf = float(conf)
        out[name] = {"label": choice, "probabilities": clean_probs, "confidence": conf}
        if choice not in scored:
            raise ValueError("chosen label is missing from scores")
    if set(out.keys()) != set(expected):
        raise ValueError("square set mismatch")
    return out


def _http_error(resp: requests.Response, key="") -> dict:
    try:
        j = resp.json()
        err = j.get("error", j) if isinstance(j, dict) else {}
        if not isinstance(err, dict):
            err = {"message": str(err)}
    except Exception:
        err = {"message": resp.text}
    return {part: _strip_credentials(str(err[part]), key)[:400]
            for part in ("code", "message") if part in err}


def _model_for(mode: str) -> str:
    if mode == "flash":
        return "qwen3.8-flash"
    if mode == "max":
        return "qwen3.8-max"
    if mode == "valen":
        return "valen-4b"
    if mode == "step":
        return "step3-vl-10b"
    return ""


def _strip_credentials(obj: Any, key: str) -> Any:
    if key:
        obj = obj.replace(key, "***")
    return obj


def _run_dashscope(mode: str, msgs: list, squares: list) -> dict:
    url = DASHSCOPE
    model = _model_for(mode)
    body = {
        "model": model,
        "messages": msgs,
        "stream": True,
        "stream_options": {"include_usage": True},
        "reasoning_effort": "low",
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


def _run_valen(body: dict, squares: list) -> dict:
    model = "valen-4b"
    result: dict[str, Any] = {
        "mode": "valen", "model": model, "elapsed_seconds": None, "first_token_seconds": None,
        "first_answer_seconds": None, "status": "error", "finish_reason": None,
        "error": None, "raw_content": None, "opinions": None, "usage": None,
        "reasoning_chars": 0, "image_blocks": 0, "reference_sha256": None,
    }
    start = time.monotonic()
    try:
        with requests.Session() as s:
            r = s.post(VALEN_URL, json=body, timeout=(8, 75), allow_redirects=False)
            result["http_status"] = r.status_code
            if r.status_code != 200:
                result["error"] = _http_error(r)
                result["elapsed_seconds"] = round(time.monotonic() - start, 3)
                return result
            data = r.json()
        result["raw_content"] = {"model_id": data.get("model_id"), "answers": data.get("answers")}
        try:
            opinions = parse_valen(data, squares)
            result["opinions"] = opinions
            result["status"] = "success"
        except ValueError as e:
            result["error"] = {"code": "schema", "message": str(e)}
        result["first_answer_seconds"] = round(time.monotonic() - start, 3)
        result["elapsed_seconds"] = round(time.monotonic() - start, 3)
        return result
    except Exception as e:
        result["error"] = {"code": "exception", "message": str(e)[:400]}
        result["elapsed_seconds"] = round(time.monotonic() - start, 3)
        return result


def _run_step(msgs: list, squares: list) -> dict:
    model = "step3-vl-10b"
    result: dict[str, Any] = {
        "mode": "step", "model": model, "elapsed_seconds": None, "first_token_seconds": None,
        "first_answer_seconds": None, "status": "error", "finish_reason": None,
        "error": None, "raw_content": None, "opinions": None, "usage": None,
        "reasoning_chars": 0, "image_blocks": 0, "reference_sha256": None,
    }
    start = time.monotonic()
    try:
        body = {"model": "step3-vl-10b", "messages": msgs, "temperature": 0, "max_tokens": 900, "stream": False}
        with requests.Session() as s:
            s.trust_env = False
            r = s.post(STEP_URL, json=body, timeout=(8, 75), allow_redirects=False)
            result["http_status"] = r.status_code
            if r.status_code != 200:
                result["error"] = _http_error(r)
                result["elapsed_seconds"] = round(time.monotonic() - start, 3)
                return result
            data = r.json()
        ch = data.get("choices", [])
        result["usage"] = data.get("usage")
        if not ch:
            raise ValueError("no choices")
        c = ch[0]
        finish = c.get("finish_reason")
        result["finish_reason"] = finish
        msg = c.get("message") or {}
        rc = msg.get("reasoning_content") or ""
        if rc:
            result["reasoning_chars"] = len(rc)
        content = msg.get("content") or ""
        result["raw_content"] = content
        if finish != "stop":
            result["error"] = {"code": "unfinished", "message": f"finish_reason={finish}"}
            result["elapsed_seconds"] = round(time.monotonic() - start, 3)
            return result
        try:
            result["opinions"] = parse_generated(content, squares)
            result["status"] = "success"
        except ValueError as e:
            result["error"] = {"code": "schema", "message": str(e)}
        result["first_answer_seconds"] = round(time.monotonic() - start, 3)
        result["elapsed_seconds"] = round(time.monotonic() - start, 3)
        return result
    except Exception as e:
        result["error"] = {"code": "exception", "message": str(e)[:400]}
        result["elapsed_seconds"] = round(time.monotonic() - start, 3)
        return result


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["flash", "max", "valen", "step"])
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)
    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=False)
    views, _, reference, reference_sha = load_inputs()
    squares = list(views)
    mode = args.mode
    if mode in ("flash", "max", "step"):
        msgs = build_messages(views, reference)
        image_blocks = sum(1 for b in msgs[0]["content"] if isinstance(b, dict) and b.get("type") == "image_url")
        if mode == "step":
            result = _run_step(msgs, squares)
        else:
            result = _run_dashscope(mode, msgs, squares)
    else:
        body = make_valen_request(views, reference)
        image_blocks = sum(part["type"] == "input_image" for part in body["input"][0]["content"])
        result = _run_valen(body, squares)
    result["image_blocks"] = image_blocks
    result["reference_sha256"] = reference_sha
    with open(outdir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
