import base64
import json

LABELS = ("wP", "wR", "wN", "wB", "wQ", "wK",
          "bP", "bR", "bN", "bB", "bQ", "bK")

_SQUARES = tuple(f + r for f in "abcdefgh" for r in "12345678")


def _validate_square(square):
    if not isinstance(square, str) or square not in _SQUARES:
        raise ValueError("bad square")


def _validate_views(square, views):
    _validate_square(square)
    if not isinstance(views, dict):
        raise ValueError("views must be dict")
    if set(views.keys()) != {"C1", "C2"}:
        raise ValueError("views must have exactly C1 and C2")
    out = {}
    for key in ("C1", "C2"):
        item = views[key]
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("each view must be (bytes, (x,y))")
        data, coord = item
        if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
            raise ValueError("view bytes must be nonempty")
        if not isinstance(coord, tuple) or len(coord) != 2:
            raise ValueError("coordinate must be (x,y)")
        x, y = coord
        if isinstance(x, bool) or isinstance(y, bool):
            raise ValueError("coordinate must be int")
        if not isinstance(x, int) or not isinstance(y, int):
            raise ValueError("coordinate must be int")
        if x < 0 or y < 0:
            raise ValueError("coordinate must be nonnegative")
        out[key] = (bytes(data), (x, y))
    return out


def _validate_reference(reference):
    if not isinstance(reference, (bytes, bytearray)) or len(reference) == 0:
        raise ValueError("reference must be nonempty bytes")
    ref = bytes(reference)
    if len(ref) < 2 or ref[0] != 0xFF or ref[1] != 0xD8:
        raise ValueError("reference must be JPEG")
    return ref


def _data_url(data):
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def _image_block(data):
    return {
        "type": "image_url",
        "image_url": {"url": _data_url(data), "detail": "original"},
    }


def _text_block(text):
    return {"type": "text", "text": text}


def _square_pair_text(square, tag=""):
    return (
        f"{tag}方形: {square}。请识别方格 {square} 上的棋子。\n"
        f"现场图 C1 与 C2 均为方格 {square} 的现场照片。"
        f"C1 和 C2 编号仅在本方格内有效（局部编号），不同方格之间互不相关。\n"
        f"C1 的目标格中心（从左上角起的像素坐标）见其图像块；"
        f"C2 的坐标同理。坐标仅供定位参考。\n"
        f"请严格按 JSON 返回: {{\"results\":[{{\"square\":\"{square}\","
        f"\"label\":\"...\",\"image\":\"C1或C2\",\"evidence\":\"...\"}}]}}。"
        f"label 取 {', '.join(LABELS)}、empty 或 unsure；evidence 为不超过 180 字的中文说明。\n"
    )


def build_payload(prompt, square, views, reference):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt required")
    v = _validate_views(square, views)
    ref = _validate_reference(reference)

    content = []
    content.append(_text_block(
        f"{prompt.strip()}\n\n请识别方格: {square}\n"
    ))
    content.append(_text_block(_square_pair_text(square)))
    (c1, xy1) = v["C1"]
    (c2, xy2) = v["C2"]
    content.append(_text_block(
        f"现场图 C1，目标格中心（从左上角起的像素坐标）: ({xy1[0]}, {xy1[1]})"
    ))
    content.append(_image_block(c1))
    content.append(_text_block(
        f"现场图 C2，目标格中心（从左上角起的像素坐标）: ({xy2[0]}, {xy2[1]})"
    ))
    content.append(_image_block(c2))
    content.append(_text_block(
        "下图是与现场图分离的参考图（非现场照片），仅用于棋子外观参考，"
        "不得当作现场帧 C1/C2。"
    ))
    content.append(_image_block(ref))

    return {
        "model": "deepseek-flash",
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content}],
    }


def _reject_duplicates(pairs):
    seen = set()
    out = {}
    for k, v in pairs:
        if k in seen:
            raise ValueError("duplicate key")
        seen.add(k)
        out[k] = v
    return out


def _reject_constant(x):
    raise ValueError("bad constant")


def _load_strict(content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty content")
    try:
        return json.loads(
            content,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError("bad json") from e


def _has_control_chars(s):
    for ch in s:
        o = ord(ch)
        if o < 0x20 or o == 0x7F:
            return True
    return False


def _validate_record(rec, square):
    if not isinstance(rec, dict):
        raise ValueError("record must be object")
    if set(rec.keys()) != {"square", "label", "image", "evidence"}:
        raise ValueError("record fields mismatch")
    if rec["square"] != square:
        raise ValueError("square mismatch")
    label = rec["label"]
    if not isinstance(label, str):
        raise ValueError("label must be string")
    if label not in LABELS and label not in ("empty", "unsure"):
        raise ValueError("bad label")
    image = rec["image"]
    if image not in ("C1", "C2"):
        raise ValueError("bad image")
    ev = rec["evidence"]
    if not isinstance(ev, str):
        raise ValueError("evidence must be string")
    if len(ev) == 0 or len(ev) > 180:
        raise ValueError("bad evidence length")
    if ev.strip() != ev:
        raise ValueError("evidence must not have surrounding whitespace")
    if not ev.strip():
        raise ValueError("evidence blank")
    if _has_control_chars(ev):
        raise ValueError("evidence has control chars")
    return {"square": square, "label": label, "image": image, "evidence": ev}


def parse_result(content, square):
    _validate_square(square)
    data = _load_strict(content)
    if not isinstance(data, dict) or set(data.keys()) != {"results"}:
        raise ValueError("top-level must be {results}")
    results = data["results"]
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("results must have exactly one entry")
    return _validate_record(results[0], square)


def build_packet(prompt, requests, reference):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt required")
    if not isinstance(requests, dict):
        raise ValueError("requests must be dict")
    if len(requests) == 0:
        raise ValueError("requests empty")
    if len(requests) > 5:
        raise ValueError("too many squares")
    if len(set(requests.keys())) != len(requests):
        raise ValueError("duplicate squares")
    ref = _validate_reference(reference)

    ordered = list(requests.keys())
    for sq in ordered:
        _validate_square(sq)
    validated = {sq: _validate_views(sq, requests[sq]) for sq in ordered}

    content = []
    content.append(_text_block(
        f"{prompt.strip()}\n\n请一次性识别以下方格: {', '.join(ordered)}。\n"
        f"注意: 每个方格的 C1 和 C2 均为局部编号，仅在该方格内有效；"
        f"不同方格之间的 C1/C2 不相关。\n"
    ))
    for sq in ordered:
        v = validated[sq]
        content.append(_text_block(_square_pair_text(sq, tag=f"[{sq}] ")))
        (c1, xy1) = v["C1"]
        (c2, xy2) = v["C2"]
        content.append(_text_block(
            f"[{sq}] 现场图 C1，目标格中心（从左上角起的像素坐标）: "
            f"({xy1[0]}, {xy1[1]})"
        ))
        content.append(_image_block(c1))
        content.append(_text_block(
            f"[{sq}] 现场图 C2，目标格中心（从左上角起的像素坐标）: "
            f"({xy2[0]}, {xy2[1]})"
        ))
        content.append(_image_block(c2))
    content.append(_text_block(
        f"以下为公共参考图（非现场照片），与所有现场帧分离，仅用于棋子外观参考，"
        f"不得当作任一方格的 C1/C2。请为所有方格 {', '.join(ordered)} 各返回一条记录。"
    ))
    content.append(_image_block(ref))

    return {
        "model": "deepseek-flash",
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content}],
    }


def parse_packet(content, squares):
    if not isinstance(squares, (list, tuple)) or len(squares) == 0:
        raise ValueError("squares required")
    if len(squares) > 5:
        raise ValueError("too many squares")
    ordered = list(squares)
    if len(set(ordered)) != len(ordered):
        raise ValueError("duplicate squares")
    for sq in ordered:
        _validate_square(sq)
    data = _load_strict(content)
    if not isinstance(data, dict) or set(data.keys()) != {"results"}:
        raise ValueError("top-level must be {results}")
    results = data["results"]
    if not isinstance(results, list) or len(results) != len(ordered):
        raise ValueError("results count mismatch")
    seen = {}
    for rec in results:
        if not isinstance(rec, dict):
            raise ValueError("record must be object")
        if "square" not in rec:
            raise ValueError("missing square")
        sq = rec["square"]
        if sq not in ordered:
            raise ValueError("unknown square")
        if sq in seen:
            raise ValueError("duplicate square in results")
        seen[sq] = _validate_record(rec, sq)
    if set(seen.keys()) != set(ordered):
        raise ValueError("missing squares")
    return {sq: seen[sq] for sq in ordered}
