"""Minimal known-corner adapter that maps detections to board symbols.

This module intentionally supports only the known-corner synthetic scope.
It does not load models and does not reorder corner semantics.
"""

import cv2
import numpy as np

import bench_common as B


def load(*args, **kwargs):
    return {
        "scope": "synthetic",
        "knowncorners": True,
        "modelsloaded": False,
    }


def _project_point(matrix, x, y):
    point = np.array([[[float(x), float(y)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(point, matrix)[0][0]
    return float(out[0]), float(out[1])


def _clip_index(value, limit=7):
    idx = int(value // 100)
    if idx < 0:
        idx = 0
    if idx > limit:
        idx = limit
    return idx


def map_board(detections, corners, native_H=None):
    if corners is None and native_H is None:
        return ["_" for _ in range(64)]

    if native_H is not None:
        H = np.asarray(native_H, dtype=np.float64)
    else:
        src = np.asarray(corners, dtype=np.float32)
        dst = np.array(
            [[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32
        )
        H = cv2.getPerspectiveTransform(src, dst)

    squares = ["_" for _ in range(64)]
    confidence = [-1.0 for _ in range(64)]

    if detections is None:
        detections = []

    for det in detections:
        if "symbol" not in det:
            continue
        symbol = det["symbol"]
        if symbol is None or symbol == "_":
            continue

        box = det.get("box")
        if box is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        h = max(1.0, y2 - y1)
        strip_top = y2 - 0.25 * h

        poly = np.array(
            [
                [x1, strip_top],
                [x2, strip_top],
                [x2, y2],
                [x1, y2],
            ],
            dtype=np.float32,
        )
        warped = cv2.perspectiveTransform(poly[None, :, :], H)[0]

        min_x = float(np.min(warped[:, 0]))
        max_x = float(np.max(warped[:, 0]))
        min_y = float(np.min(warped[:, 1]))
        max_y = float(np.max(warped[:, 1]))

        min_x = max(0.0, min(799.0, min_x))
        max_x = max(0.0, min(799.0, max_x))
        min_y = max(0.0, min(799.0, min_y))
        max_y = max(0.0, min(799.0, max_y))

        area_warp = max(0.0, (max_x - min_x)) * max(0.0, (max_y - min_y))

        best_iou = -1.0
        best_index = -1

        for row in range(8):
            for col in range(8):
                rx0 = col * 100.0
                ry0 = row * 100.0
                rx1 = (col + 1) * 100.0
                ry1 = (row + 1) * 100.0

                inter_x0 = max(min_x, rx0)
                inter_y0 = max(min_y, ry0)
                inter_x1 = min(max_x, rx1)
                inter_y1 = min(max_y, ry1)

                inter_w = max(0.0, inter_x1 - inter_x0)
                inter_h = max(0.0, inter_y1 - inter_y0)
                inter = inter_w * inter_h

                iou = inter / (area_warp + 10000.0 - inter + 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_index = row * 8 + col

        if best_iou < 1e-6:
            cx = (x1 + x2) / 2.0
            cy = y2
            proj = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
            out = cv2.perspectiveTransform(proj, H)[0][0]
            col = _clip_index(float(out[0]))
            row = _clip_index(float(out[1]))
            best_index = row * 8 + col

        if best_index < 0:
            continue

        conf = float(det.get("confidence", 0.0))
        if conf > confidence[best_index]:
            confidence[best_index] = conf
            squares[best_index] = symbol

    return squares
