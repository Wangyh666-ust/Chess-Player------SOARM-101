import copy
import threading
import time
import sys

import cv2
import numpy as np

import guided_grid_recovery_v1 as core
import live_grid_locator_v5 as old


_ORIGINAL_RELOCALIZE = old.relocalize


_FACADE_ORIENTATION = "files_desc_ranks_asc"


def _is_finite_xyxy(rect):
    try:
        arr = np.asarray(rect, dtype=np.float64)
    except Exception:
        return False
    if arr.shape != (4,):
        return False
    if not np.all(np.isfinite(arr)):
        return False
    x0, y0, x1, y1 = arr.tolist()
    return (x1 - x0) > 0.0 and (y1 - y0) > 0.0


def _validate_mask(cal):
    if not isinstance(cal, dict):
        return False
    rects = cal.get("occlusion_rects", None)
    if rects is None:
        return True
    try:
        rects = list(rects)
    except Exception:
        return False
    for rect in rects:
        if not _is_finite_xyxy(rect):
            return False
    return True


def _mask_rects(cal):
    rects = cal.get("occlusion_rects", None) if isinstance(cal, dict) else None
    if not rects:
        return []
    out = []
    for rect in rects:
        arr = np.asarray(rect, dtype=np.float64).reshape(4)
        out.append(arr)
    return out


def _quad_is_valid(quad, shape):
    pts = np.asarray(quad, dtype=np.float64)
    if pts.shape != (4, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if cv2.contourArea(pts.astype(np.float32)) <= 1e-6:
        return False
    cross = []
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        c = pts[(i + 2) % 4]
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    cross = np.asarray(cross, dtype=np.float64)
    if np.any(cross <= 1e-9):
        return False
    return True


def _center_from_quad(quad):
    pts = np.asarray(quad, dtype=np.float64)
    p0, p1, p2, p3 = pts[0], pts[1], pts[2], pts[3]
    d1 = p2 - p0
    d2 = p3 - p1
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) <= 1e-9:
        return tuple(np.mean(pts, axis=0).tolist())
    t = ((p1[0] - p0[0]) * d2[1] - (p1[1] - p0[1]) * d2[0]) / denom
    center = p0 + t * d1
    return (float(center[0]), float(center[1]))


def _intersects_rect(quad, rect):
    qx0 = min(float(p[0]) for p in quad)
    qx1 = max(float(p[0]) for p in quad)
    qy0 = min(float(p[1]) for p in quad)
    qy1 = max(float(p[1]) for p in quad)
    rx0, ry0, rx1, ry1 = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
    if qx1 < rx0 or qx0 > rx1:
        return False
    if qy1 < ry0 or qy0 > ry1:
        return False
    return True


def _classify(quad, shape, rects):
    h, w = int(shape[0]), int(shape[1])
    for p in quad:
        x = float(p[0])
        y = float(p[1])
        if x < 0.0 or y < 0.0 or x > float(w - 1) or y > float(h - 1):
            return "out_of_frame", False, True
    for rect in rects:
        if _intersects_rect(quad, rect):
            return "occluded", False, True
    return "visible", True, False


def _normalize_failure(result):
    if isinstance(result, dict):
        return result
    try:
        grid, _ = result
    except Exception:
        return {"valid": False, "reason": "core_failure"}
    if isinstance(grid, dict):
        return dict(grid)
    return {"valid": False, "reason": "core_failure"}


def _rebuild_success(grid, newcal, shape, orientation, mask_rects):
    nodes = np.asarray(grid.get("grid_nodes"), dtype=np.float64)
    if nodes.shape == (81, 2):
        nodes = nodes.reshape(9, 9, 2)
    elif nodes.shape != (9, 9, 2):
        return None
    cells = []
    for r in range(8):
        for c in range(8):
            quad = [
                nodes[r, c].tolist(),
                nodes[r, c + 1].tolist(),
                nodes[r + 1, c + 1].tolist(),
                nodes[r + 1, c].tolist(),
            ]
            if not _quad_is_valid(quad, shape):
                return None
            center = _center_from_quad(quad)
            cellkey = old.square_name(orientation, c, r)
            vis, visible, occluded = _classify(quad, shape, mask_rects)
            cells.append({
                "square": cellkey,
                "row": int(r),
                "col": int(c),
                "quad": [[float(p[0]), float(p[1])] for p in quad],
                "center": [center[0], center[1]],
                "visible": bool(visible),
                "visibility": vis,
                "occluded_or_estimated": bool(occluded),
            })
    out = dict(grid)
    out["cells"] = cells
    out["valid"] = True
    grid_cal = copy.deepcopy(newcal)
    if isinstance(grid_cal, dict):
        grid_cal["cells"] = copy.deepcopy(cells)
        grid_cal["heldout_uv"] = []
        grid_cal["heldout_pixel"] = []
    out["grid_nodes"] = [[float(nodes[r, c, 0]), float(nodes[r, c, 1])]
                         for r in range(9) for c in range(9)]
    out["orientation"] = orientation
    out["width"] = int(shape[1])
    out["height"] = int(shape[0])
    return out, grid_cal


def relocalize(image, cal):
    if not _validate_mask(cal):
        return old._blank_result("bad_mask"), None
    orientation = cal.get("orientation") if isinstance(cal, dict) else None
    if orientation != _FACADE_ORIENTATION:
        result = _ORIGINAL_RELOCALIZE(image, cal)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return _normalize_failure(result), None
    result = core.recover(image, cal)
    if isinstance(result, tuple) and len(result) == 2:
        grid, newcal = result
        if isinstance(grid, dict) and grid.get("valid"):
            shape = image.shape[:2]
            rebuilt = _rebuild_success(grid, newcal, shape, orientation, _mask_rects(cal))
            if rebuilt is not None:
                out_grid, out_cal = rebuilt
                return out_grid, out_cal
            return old._blank_result("invalid_rebuilt_grid"), None
        return _normalize_failure(grid), None
    failure = _normalize_failure(result)
    return failure, None


class GridLocator(object):
    _REACQUISITION_INTERVAL = 60.0

    def __init__(self, reference_bgr, calibration):
        orientation = calibration.get("orientation") if isinstance(calibration, dict) else None
        self._delegate = None
        if orientation != _FACADE_ORIENTATION:
            self._delegate = old.GridLocator(reference_bgr, calibration)
            return
        self._lock = threading.RLock()
        self.reference_bgr = np.array(reference_bgr, copy=True)
        self.calibration = copy.deepcopy(calibration)
        self.reacquisition_successes = 0
        self._last_success_time = None

    def detect(self, image):
        if self._delegate is not None:
            return self._delegate.detect(image)
        with self._lock:
            result, newcal = relocalize(image, copy.deepcopy(self.calibration))
            valid = isinstance(result, dict) and bool(result.get("valid"))
            if valid and newcal is not None:
                self.calibration = copy.deepcopy(newcal)
                self.reference_bgr = np.array(image, copy=True)
                now = time.time()
                if self._last_success_time is None or \
                        (now - self._last_success_time) >= self._REACQUISITION_INTERVAL:
                    self.reacquisition_successes += 1
                    self._last_success_time = now
            return result

    def __getattr__(self, name):
        delegate = self.__dict__.get("_delegate", None)
        if delegate is not None:
            return getattr(delegate, name)
        raise AttributeError(name)


def _cli():
    old_relocalize = old.relocalize
    old_legacy = old._LegacyGridLocator
    old.relocalize = relocalize
    old._LegacyGridLocator = GridLocator
    try:
        return old._cli_main(sys.argv[1:])
    finally:
        old.relocalize = old_relocalize
        old._LegacyGridLocator = old_legacy


if __name__ == "__main__":
    sys.exit(_cli())
