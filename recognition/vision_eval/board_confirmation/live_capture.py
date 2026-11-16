"""Adapt the live status and raw camera routes to a six-second board snapshot."""

from collections import Counter, defaultdict
import http.client
import json
import math
import time
from urllib.parse import urlsplit

from review import LABELS


SQUARES = tuple(f + r for r in "87654321" for f in "abcdefgh")
LEASE = ("/view/0.jpg?rotation=0&yolo=1", "/view/1.jpg?rotation=270&yolo=1")
CLEAN = ("/view/0.jpg?rotation=0&yolo=0", "/view/1.jpg?rotation=0&yolo=0")
MAX_BODY = 4 * 1024 * 1024
SKEW = 0.35
WINDOW = 2.0
WINDOWS = 3
SPAN = WINDOW * WINDOWS


def _http_get(upstream):
    url = urlsplit(upstream or "")
    if url.scheme != "http" or not url.hostname or url.path not in ("", "/") or url.query:
        raise RuntimeError("双摄服务地址无效。")

    def get(path):
        conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=1.5)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            if response.status != 200:
                return b"", {}
            if path in LEASE:
                return b"", {}
            data = response.read(MAX_BODY + 1)
            headers = {key.lower(): val for key, val in response.getheaders()}
            return (data, headers) if len(data) <= MAX_BODY else (b"", {})
        except (OSError, http.client.HTTPException):
            return b"", {}
        finally:
            conn.close()

    return get


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _status(get):
    body, _ = get("/status")
    try:
        grid = json.loads(body)["collection"]["grid_pipeline"]
        voting = grid["voting"]
    except (ValueError, KeyError, TypeError):
        return None, None
    if voting.get("publish_period_s") != WINDOW:
        raise RuntimeError("双摄预览的投票周期需要设为 2 秒。")
    return grid, voting.get("last_completed_window")


def _usable(grid):
    cameras = grid.get("cameras", [])
    skew_ms = grid.get("pair_skew_ms")
    return (
        len(cameras) == 2 and [cam.get("camera_index") for cam in cameras] == [0, 1]
        and all(cam.get("fresh") is True and cam.get("grid_valid") is True
                and type(cam.get("seq")) is int for cam in cameras)
        and _number(skew_ms) and 0 <= skew_ms <= SKEW * 1000
    )


def _rows(grid):
    items = grid.get("squares", ())
    result = {row.get("square"): row for row in items}
    if len(items) != 64 or result.keys() != set(SQUARES):
        raise RuntimeError("双摄服务返回的棋盘格信息不完整。")
    return result


def _raw_pair(get, anchor, last_seq):
    frames = []
    for camera, path in enumerate(CLEAN):
        jpg, hdrs = get(path)
        try:
            seq = int(hdrs["x-frame-sequence"])
            ts = float(hdrs["x-capture-time"])
        except (KeyError, ValueError):
            return None
        if not jpg.startswith(b"\xff\xd8") or not math.isfinite(ts) or seq <= last_seq[camera]:
            return None
        if not anchor <= ts < anchor + SPAN:
            return None
        frames.append((jpg, seq, ts))
    a, b = frames
    index = int((a[2] - anchor) // WINDOW)
    if int((b[2] - anchor) // WINDOW) != index or abs(a[2] - b[2]) > SKEW:
        return None
    return index, a, b


def _label(signal):
    if isinstance(signal, dict):
        index = signal.get("class_id")
        if type(index) is int and 0 <= index < len(LABELS):
            return LABELS[index]
    return None


def _summarize(windows, samples, previous):
    local, reasons = {}, {}
    ranked, stable = [], []
    for square in SQUARES:
        counts = Counter(_label(rows[square].get("final")) for _, rows in windows)
        counts.pop(None, None)
        votes = counts.most_common(1)
        samples_for_square = samples.get(square, ())
        conflicts = [index for index, flag in samples_for_square if flag]
        if votes and votes[0][1] >= 2:
            local[square], count = votes[0]
            reasons[square] = f"{count}/3 个窗口一致"
        elif counts or samples_for_square:
            local[square] = "unsure"
            reasons[square] = "三个窗口内未形成一致类别"
        else:
            local[square] = "empty"
            reasons[square] = "窗口内未检测到棋子"
        flags = []
        score = 0
        if local[square] == "unsure":
            score += 4
            flags.append("本地结果未确认")
        if len(conflicts) >= 3 and len(set(conflicts)) >= 2:
            score += 5
            flags.append("两路在多个窗口给出不同类别")
        elif conflicts:
            score += 2
            flags.append("出现双摄类别分歧")
        if previous and square in previous and local[square] not in ("empty", previous[square]):
            score += 2
            flags.append("此格与上次定格不同")
        if score:
            ranked.append((-score, square, "；".join(flags)))
        elif not previous and local[square] in LABELS and local[square][1] in "PRQK":
            # Surface a stable king, rook, queen or pawn as a control when capacity permits.
            stable.append(("KRQP".index(local[square][1]), square, "抽查稳定识别"))
    return local, reasons, sorted(ranked), sorted(stable)


def collect(upstream, cropper, progress=lambda *_: None, *, get=None,
            clock=time.monotonic, sleep=time.sleep, previous=None):
    get = get if get is not None else _http_get(upstream)
    progress("warmup", 0.0)
    _, last = _status(get)
    initial_end = (last or {}).get("window_end", float("-inf"))
    last_lease = float("-inf")

    def renew():
        nonlocal last_lease
        if clock() - last_lease >= 0.5:
            for path in LEASE:
                get(path)
            last_lease = clock()

    anchor = None
    warmup_by = clock() + 15.0
    while clock() < warmup_by:
        renew()
        grid, win = _status(get)
        if grid and win and _usable(grid) and win["window_end"] > initial_end and win["frames"] > 0:
            anchor = win["window_end"]
            break
        sleep(0.15)
    if anchor is None:
        raise RuntimeError("双摄预热超时。请确认 Ubuntu 的双摄预览可用、棋盘编码清晰可见。")

    start = clock()
    windows, pairs = [], {}
    observations = defaultdict(list)
    seen = set()
    last_seq = (-1, -1)
    while clock() - start < 8.0:
        renew()
        elapsed = clock() - start
        progress("collecting", min(SPAN, elapsed))
        grid, win = _status(get)
        if grid and win and _usable(grid):
            rows = _rows(grid)
            expected = anchor + len(windows) * WINDOW
            actual = win["window_start"]
            if abs(actual - expected) < 1e-5:
                if abs(win["window_end"] - actual - WINDOW) > 1e-5 or win["frames"] <= 0:
                    raise RuntimeError("投票窗口的时间或帧数不完整。")
                windows.append(({k: win[k] for k in ("window_start", "window_end", "frames")}, rows))
            elif actual > expected + 1e-5:
                break
            if len(windows) == WINDOWS:
                break
            seq = tuple(cam["seq"] for cam in grid["cameras"])
            captured = [cam.get("captured_at") for cam in grid["cameras"]]
            if all(_number(ts) and anchor <= ts < anchor + SPAN for ts in captured) and seq not in seen:
                seen.add(seq)
                index = int((min(captured) - anchor) // WINDOW)
                for square, row in rows.items():
                    main, aux = _label(row.get("primary")), _label(row.get("auxiliary"))
                    if main or aux or row.get("temporal_state") == "pending":
                        observations[square].append((index, bool(main and aux and main != aux)))
            if len(pairs) < WINDOWS and len(windows) not in pairs:
                captured_pair = _raw_pair(get, anchor, last_seq)
                if captured_pair and captured_pair[0] not in pairs:
                    index, a, b = captured_pair
                    pairs[index] = a, b
                    last_seq = a[1], b[1]
        sleep(0.15)

    if len(windows) != WINDOWS:
        raise RuntimeError(f"只收齐 {len(windows)}/3 个连续投票窗口，请重新识别。")
    progress("collecting", SPAN)
    local, reasons, ranked, stable = _summarize(windows, observations, previous)
    ordered = ranked[:8] + stable[:6]
    candidates = [square for _, square, _ in ordered]
    available = {}
    pair_trace = []
    selected_pairs = {}
    for index in (1, 2, 0):
        if index not in pairs:
            continue
        first, second = pairs[index]
        trace = {"window": index + 1, "sequences": [first[1], second[1]],
                 "capture_times": [first[2], second[2]]}
        pair_trace.append(trace)
        if candidates:
            left = cropper.crops(first[0], 1, candidates)
            right = cropper.crops(second[0], 2, candidates)
            for square in left.keys() & right.keys():
                if square not in available:
                    available[square] = {"C1": left[square], "C2": right[square]}
                    selected_pairs[square] = trace
    requests, selected, skipped = {}, {}, {}
    for pool, limit in ((ranked[:8], 5), (stable[:6], 2)):
        start_count = len(requests)
        for _, square, reason in pool:
            if len(requests) == 5 or len(requests) - start_count == limit:
                break
            if square in available:
                requests[square], selected[square] = available[square], reason
            elif reason != "抽查稳定识别":
                skipped[square] = "缺少可用双摄裁图"
    return {
        "local": local, "reasons": reasons, "windows": [meta for meta, _ in windows],
        "observed_seconds": SPAN, "requests": requests, "selected": selected, "skipped": skipped,
        "trace": {"pairs": pair_trace, "unique_polls": len(seen),
                  "selected_pairs": {square: selected_pairs[square] for square in requests}},
    }
