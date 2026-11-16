import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
if str(BASE.parent) not in sys.path:
    sys.path.insert(0, str(BASE.parent))
os.environ["CHESS_ARUCO_ASSETS"] = str(BASE / "calibration")

from aruco_live_service_v10 import GridLocator

_CODES = ["wP", "wR", "wK", "wQ", "wB", "wN",
          "bP", "bR", "bK", "bQ", "bB", "bN"]
_ORIENT = {1: "files_desc_ranks_asc", 2: "ranks_desc_files_desc"}
_CODE_CHARS = set("wbbPRKQBN")


def _code_ok(code):
    if not isinstance(code, str):
        return False
    if len(code) != 2:
        return False
    if code[0] not in ("w", "b"):
        return False
    return code[1] in _CODE_CHARS


def _square_ok(sq):
    if not isinstance(sq, str) or len(sq) != 2:
        return False
    return sq[0] in "abcdefgh" and sq[1] in "12345678"


class Cropper:
    def __init__(self):
        self._locators = {}

    def _locator(self, camera):
        if camera not in _ORIENT:
            return None
        loc = self._locators.get(camera)
        if loc is None:
            try:
                loc = GridLocator(None, {"orientation": _ORIENT[camera]})
            except Exception:
                return None
            self._locators[camera] = loc
        return loc

    def crops(self, raw, camera, squares):
        if camera not in _ORIENT:
            return {}
        if not isinstance(raw, (bytes, bytearray)) or not raw:
            return {}
        if not isinstance(squares, (list, tuple)):
            return {}
        want = set()
        for s in squares:
            if not _square_ok(s):
                return {}
            want.add(s)
        if not want:
            return {}
        try:
            arr = np.frombuffer(bytes(raw), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return {}
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            return {}
        if camera == 2:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        loc = self._locator(camera)
        if loc is None:
            return {}
        try:
            res = loc.detect(img)
        except Exception:
            return {}
        if not isinstance(res, dict) or res.get("valid") is not True:
            return {}
        cells = res.get("cells")
        if not isinstance(cells, (list, tuple)):
            return {}
        H, W = img.shape[:2]
        out = {}
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            sq = cell.get("square")
            if sq not in want or sq in out:
                continue
            if not cell.get("visible"):
                continue
            try:
                center = cell["center"]
                quad = cell["quad"]
                cx = float(center[0]); cy = float(center[1])
                xs = [float(p[0]) for p in quad]
                ys = [float(p[1]) for p in quad]
            except Exception:
                continue
            bw = max(xs) - min(xs)
            bh = max(ys) - min(ys)
            if bw <= 0 or bh <= 0:
                continue
            left = min(cx - 0.70 * bw, min(xs))
            right = max(cx + 0.70 * bw, max(xs))
            top = min(cy - 2.1 * bh, min(ys))
            bottom = max(cy + 0.65 * bh, max(ys))
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            if min(xs) < 0 or max(xs) >= W or min(ys) < 0 or max(ys) >= H:
                continue
            x0 = int(np.floor(max(0.0, left)))
            y0 = int(np.floor(max(0.0, top)))
            x1 = int(np.ceil(min(float(W), right)))
            y1 = int(np.ceil(min(float(H), bottom)))
            if x1 <= x0 or y1 <= y0:
                continue
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            m = max(cw, ch)
            if m <= 0:
                continue
            scale = min(3.0, 520.0 / float(m))
            if scale <= 0:
                continue
            nw = max(1, int(round(cw * scale)))
            nh = max(1, int(round(ch * scale)))
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            big = cv2.resize(crop, (nw, nh), interpolation=interp)
            ok, buf = cv2.imencode(".jpg", big, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                continue
            tc_x = int(round((cx - x0) * nw / cw))
            tc_y = int(round((cy - y0) * nh / ch))
            tc_x = min(max(tc_x, 0), nw - 1)
            tc_y = min(max(tc_y, 0), nh - 1)
            out[sq] = (bytes(buf.tobytes()), (tc_x, tc_y))
        return out


def _rect_ok(r):
    if not isinstance(r, (list, tuple)) or len(r) != 4:
        return False
    try:
        l, t, rr, b = (float(v) for v in r)
    except Exception:
        return False
    return rr > l and b > t


def reference():
    with open(str(BASE / "reference_layout.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    src = data.get("source")
    image_size = data.get("image_size")
    pieces = data.get("pieces")
    if not isinstance(src, str) or not src:
        raise ValueError("bad source")
    if (not isinstance(image_size, (list, tuple)) or len(image_size) != 2
            or not all(isinstance(v, (int, float)) and v > 0 for v in image_size)):
        raise ValueError("bad image_size")
    if not isinstance(pieces, dict):
        raise ValueError("bad pieces")
    for code in _CODES:
        r = pieces.get(code)
        if not _rect_ok(r):
            raise ValueError("bad rectangle for " + code)
    path = (ROOT / src).resolve()
    path.relative_to(ROOT)
    with open(str(path), "rb") as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError("bad reference image")
    H, W = img.shape[:2]
    IW, IH = int(image_size[0]), int(image_size[1])
    if (W, H) != (IW, IH):
        raise ValueError("reference image size changed")
    cell_w, cell_h = 150, 150
    cols, rows = 6, 2
    label_h = 22
    pad = 8
    tile_w = cell_w + pad
    tile_h = cell_h + label_h + pad
    sheet_w = cols * tile_w + pad
    sheet_h = rows * tile_h + pad
    sheet = np.full((sheet_h, sheet_w, 3), 200, dtype=np.uint8)
    for idx, code in enumerate(_CODES):
        l, t, r, b = (float(v) for v in pieces[code])
        x0, y0, x1, y1 = map(round, (l, t, r, b))
        if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
            raise ValueError("reference rectangle outside the source image")
        sub = img[y0:y1, x0:x1]
        ch, cw = sub.shape[:2]
        if ch <= 0 or cw <= 0:
            continue
        s = min(cell_w / float(cw), cell_h / float(ch))
        nw = max(1, int(round(cw * s)))
        nh = max(1, int(round(ch * s)))
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
        rim = cv2.resize(sub, (nw, nh), interpolation=interp)
        col = idx % cols
        row = idx // cols
        ox = pad + col * tile_w + (cell_w - nw) // 2
        oy = pad + row * tile_h + (cell_h - nh) // 2
        sheet[oy:oy + nh, ox:ox + nw] = rim
        lx = pad + col * tile_w + 2
        ly = pad + row * tile_h + cell_h + 16
        cv2.putText(sheet, code, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError("encode failed")
    return bytes(buf.tobytes())
