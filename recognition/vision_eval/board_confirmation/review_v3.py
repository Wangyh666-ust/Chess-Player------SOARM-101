"""Strict packet builder and response parser for adjacent review v3."""

from review import (
    _validate_views,
    _validate_reference,
    _validate_square,
    _load_strict,
    _has_control_chars,
    _text_block,
    _image_block,
    LABELS,
)

MODEL = "deepseek-flash"
MAX_TOKENS = 2400
MIN_SQUARES = 1
MAX_SQUARES = 5

BASE_VALUES = ("target", "empty", "unclear")
TOP_VALUES = (
    "upright_finial",
    "crown_rim",
    "flat_top",
    "sphere",
    "bishop_cap",
    "horse_head",
    "unclear",
)
TOP_TO_TYPE = {
    "upright_finial": "K",
    "crown_rim": "Q",
    "flat_top": "R",
    "sphere": "P",
    "bishop_cap": "B",
    "horse_head": "N",
}
RESULT_LABELS = tuple(LABELS) + ("empty", "unsure")

RECORD_KEYS = {"square", "observations", "label", "evidence"}
OBS_KEYS = {"C1", "C2"}
OBSERVATION_KEYS = {"base", "top", "detail"}


def _clean_text(value, name, max_len=None):
    if not isinstance(value, str):
        raise ValueError("%s must be a string" % name)
    if value == "":
        raise ValueError("%s must be nonempty" % name)
    if _has_control_chars(value):
        raise ValueError("%s must not contain control characters" % name)
    if value != value.strip():
        raise ValueError("%s must not have surrounding whitespace" % name)
    if max_len is not None and len(value) > max_len:
        raise ValueError("%s exceeds %d characters" % (name, max_len))
    return value


def _format_center(center):
    x, y = center
    return "(%d,%d)" % (int(x), int(y))


def build_packet(prompt, requests, reference):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if not isinstance(requests, dict):
        raise ValueError("requests must be a mapping")
    if not (MIN_SQUARES <= len(requests) <= MAX_SQUARES):
        raise ValueError("requests must contain 1-5 squares")

    reference = _validate_reference(reference)
    content = [_text_block(prompt.strip())]
    for square, views in requests.items():
        for camera, (jpeg, center) in _validate_views(square, views).items():
            content.append(_text_block(f"格 {square}｜{camera}｜目标格中心 {_format_center(center)}"))
            content.append(_image_block(jpeg))

    content.append(_text_block("下图是同款棋子的类别参考照，与任何现场方格无关。"))
    content.append(_image_block(reference))

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    return payload


def _parse_observation(square, view_name, observation):
    if not isinstance(observation, dict):
        raise ValueError("%s %s observation must be an object" % (square, view_name))
    keys = set(observation.keys())
    if keys != OBSERVATION_KEYS:
        raise ValueError(
            "%s %s observation must have exactly base, top, detail"
            % (square, view_name)
        )

    base = observation["base"]
    top = observation["top"]
    detail = observation["detail"]

    if not isinstance(base, str) or base not in BASE_VALUES:
        raise ValueError("%s %s base invalid" % (square, view_name))
    if not isinstance(top, str) or top not in TOP_VALUES:
        raise ValueError("%s %s top invalid" % (square, view_name))
    detail = _clean_text(detail, "%s %s detail" % (square, view_name), 100)

    if base == "empty" and top != "unclear":
        raise ValueError("%s %s empty base requires unclear top" % (square, view_name))

    return {"base": base, "top": top, "detail": detail}


def _parse_record(record, expected_squares, seen_squares):
    if not isinstance(record, dict):
        raise ValueError("each result record must be an object")
    if set(record.keys()) != RECORD_KEYS:
        raise ValueError("result record must have exactly square, observations, label, evidence")

    square = record["square"]
    observations = record["observations"]
    label = record["label"]
    evidence = record["evidence"]

    if not isinstance(square, str):
        raise ValueError("square must be a string")
    if square not in expected_squares:
        raise ValueError("unexpected square %r" % (square,))
    if square in seen_squares:
        raise ValueError("duplicate square %r" % (square,))

    if not isinstance(observations, dict):
        raise ValueError("%s observations must be an object" % square)
    if set(observations.keys()) != OBS_KEYS:
        raise ValueError("%s observations must have exactly C1 and C2" % square)

    parsed_obs = {}
    for view_name in ("C1", "C2"):
        parsed_obs[view_name] = _parse_observation(
            square, view_name, observations[view_name]
        )

    if not isinstance(label, str) or label not in RESULT_LABELS:
        raise ValueError("%s label invalid" % square)

    evidence = _clean_text(evidence, "%s evidence" % square, 150)

    seen_squares.add(square)
    return {
        "square": square,
        "observations": parsed_obs,
        "label": label,
        "evidence": evidence,
    }


def _apply_guard(record):
    observations = record["observations"]
    label = record["label"]

    if label == "unsure":
        return record
    bases = {obs["base"] for obs in observations.values()}
    types = {TOP_TO_TYPE[obs["top"]] for obs in observations.values()
             if obs["base"] == "target" and obs["top"] in TOP_TO_TYPE}
    reason = None
    if label == "empty":
        if "empty" not in bases or "target" in bases:
            reason = "观察结果不支持空位判断"
    elif "empty" in bases or len(types) > 1:
        reason = "两路观察有冲突"
    elif not types:
        reason = "缺少落在目标格且看得清的区分特征"
    elif types != {label[1]}:
        reason = "可见顶饰与所报棋子类型不一致"

    if reason is not None:
        record["label"] = "unsure"
        record["guard_reason"] = reason

    return record


def parse_packet(content, squares):
    if not isinstance(squares, (list, tuple)):
        raise ValueError("squares must be a list or tuple")
    if not (MIN_SQUARES <= len(squares) <= MAX_SQUARES):
        raise ValueError("squares must contain 1-5 entries")

    expected_squares = []
    for square in squares:
        _validate_square(square)
        if square in expected_squares:
            raise ValueError("duplicate input square %r" % (square,))
        expected_squares.append(square)

    data = _load_strict(content)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON must be an object")
    if set(data.keys()) != {"results"}:
        raise ValueError("top-level JSON must have exactly results")

    results = data["results"]
    if not isinstance(results, list):
        raise ValueError("results must be a list")
    if len(results) != len(expected_squares):
        raise ValueError("results count must match requested squares")

    seen_squares = set()
    parsed = {}
    for record in results:
        parsed_record = _parse_record(record, expected_squares, seen_squares)
        parsed[parsed_record["square"]] = parsed_record

    if set(parsed.keys()) != set(expected_squares):
        raise ValueError("missing results for requested squares")

    return {square: _apply_guard(parsed[square]) for square in expected_squares}
