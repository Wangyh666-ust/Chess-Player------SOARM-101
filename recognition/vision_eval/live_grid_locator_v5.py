"""Live grid locator v4: asynchronous checkerboard relocalization.

Coordinate convention: calibration "grid_nodes" is 9x9x2 in image pixels
(rows are down, columns right).  Orientation is fixed to the physical
camera axes (not arbitrary board rotation).  Relocalization requires a
clearly visible 7x7 internal corner chessboard target; heavily occluded
boards are not reliably handled.
"""

import copy
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

try:
    from vision_eval.live_grid_locator_v2 import (
        associate as _legacy_associate,
        rotate_points as _legacy_rotate_points,
        GridLocator as _LegacyGridLocator,
        _checker_ok as _legacy__checker_ok,
    )
    from vision_eval import live_grid_locator_v2 as _legacy
except Exception:  # pragma: no cover - fallback import for flat layout
    from live_grid_locator_v2 import (  # type: ignore
        associate as _legacy_associate,
        rotate_points as _legacy_rotate_points,
        GridLocator as _LegacyGridLocator,
        _checker_ok as _legacy__checker_ok,
    )
    import live_grid_locator_v2 as _legacy  # type: ignore

associate = _legacy_associate
rotate_points = _legacy_rotate_points

_ORIENTATIONS = ("files_desc_ranks_asc", "ranks_desc_files_desc")
_SQ_MIN_AREA = 36.0


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def square_name(orientation, col, row):
    if orientation not in _ORIENTATIONS:
        raise ValueError("bad orientation")
    for v in (col, row):
        if not isinstance(v, (int, np.integer)) or v < 0 or v > 7:
            raise ValueError("col/row must be int in 0..7")
    if orientation == "files_desc_ranks_asc":
        return chr(ord('h') - int(col)) + str(int(row) + 1)
    return chr(ord('h') - int(row)) + str(8 - int(col))


_GRID_UV = np.array([[c + 1.0, r + 1.0] for r in range(7) for c in range(7)],
                    dtype=np.float64).reshape(49, 2)


def _order_quad(pts):
    """Return ordered 4 corners (tl,tr,br,bl) or None for handedness/axis."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if p.shape != (4, 2):
        return None
    best = None
    best_score = -1.0
    for i in range(4):
        tl = p[i]
        tr, br, bl = p[(i + 1) % 4], p[(i + 2) % 4], p[(i + 3) % 4]
        right = tr - tl
        down = bl - tl
        hand = right[0] * down[1] - right[1] * down[0]
        if not (right[0] > 0 and down[1] > 0 and hand > 0):
            continue
        rn = float(np.hypot(right[0], right[1]))
        dn = float(np.hypot(down[0], down[1]))
        if rn <= 1e-9 or dn <= 1e-9:
            continue
        score = right[0] / rn + down[1] / dn
        if score > best_score:
            best_score = score
            best = np.stack([tl, tr, br, bl])
    return best


def _quad_valid(quad, image_area):
    q = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    if q.shape != (4, 2) or not np.all(np.isfinite(q)):
        return False
    area = cv2.contourArea(q, oriented=True)
    if area < _SQ_MIN_AREA or not cv2.isContourConvex(q):
        return False
    return image_area <= 0 or image_area * 0.015 <= area <= image_area * 0.85


def _quad_median(gray, x, y, r):
    x0 = int(round(x)) - r
    y0 = int(round(y)) - r
    x1 = x0 + 2 * r + 1
    y1 = y0 + 2 * r + 1
    h, w = gray.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return float(np.median(patch))


def _checker_ok(gray, pt, du, dv):
    """Sample four quadrant patches and validate a checker crossing."""
    q = []
    for su, sv in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        c = pt + su * 0.16 * du + sv * 0.16 * dv
        m = _quad_median(gray, c[0], c[1], 2)
        if m is None:
            return False
        q.append(m)
    q00, q01, q10, q11 = q[0], q[1], q[2], q[3]
    diag = abs((q00 + q11 - q01 - q10) / 2.0)
    if diag < 35.0:
        return False
    if abs(q00 - q11) > 65.0 or abs(q01 - q10) > 65.0:
        return False
    return True


def _local_derivs(pts7, c, r):
    """Local du,dv 2D pixel vectors at corner (r,c) in 7x7 lattice."""
    grid = pts7.reshape(7, 7, 2)
    pt = grid[r, c]
    if 0 < c < 6:
        du = 0.5 * (grid[r, c + 1] - grid[r, c - 1])
    elif c == 0:
        du = grid[r, 1] - grid[r, 0]
    else:
        du = grid[r, 6] - grid[r, 5]
    if 0 < r < 6:
        dv = 0.5 * (grid[r + 1, c] - grid[r - 1, c])
    elif r == 0:
        dv = grid[1, c] - grid[0, c]
    else:
        dv = grid[6, c] - grid[5, c]
    return pt, du, dv


def _blank_result(reason, elapsed_ms=0.0, method='checkerboard_relocalization',
                  checker_count=0, residual=None):
    return {
        "valid": False,
        "grid_nodes": [],
        "cells": [],
        "width": 0,
        "height": 0,
        "matched": 0,
        "inliers": 0,
        "checker_count": int(checker_count),
        "median_fb_px": None,
        "p95_reprojection_px": residual,
        "elapsed_ms": float(elapsed_ms),
        "method": method,
        "reference_assisted": False,
        "reason": str(reason),
    }


def _build_from_uv7(gray, pts7, calibration, t0):
    """Given 7x7 measured corners, fit homography and build grid/cal."""
    pts7 = np.asarray(pts7, dtype=np.float64).reshape(49, 2)
    if not np.all(np.isfinite(pts7)):
        return None, "nonfinite_corners"

    H, _ = cv2.findHomography(_GRID_UV, pts7, 0)
    if H is None or not np.all(np.isfinite(H)):
        return None, "homography_failed"

    proj = cv2.perspectiveTransform(_GRID_UV.reshape(-1, 1, 2), H).reshape(-1, 2)
    resid = np.linalg.norm(proj - pts7, axis=1)
    p95 = float(np.percentile(resid, 95))

    diffs = np.diff(pts7.reshape(7, 7, 2), axis=1)
    spacings = np.linalg.norm(diffs.reshape(-1, 2), axis=1)
    spacings = spacings[spacings > 0]
    if spacings.size == 0:
        return None, "no_spacing"
    median_spacing = float(np.median(spacings))
    if p95 > max(3.0, 0.08 * median_spacing):
        return None, "high_residual"

    uv_grid = np.array([[c, r] for r in range(9) for c in range(9)],
                       dtype=np.float64).reshape(9, 9, 2)
    nodes = cv2.perspectiveTransform(uv_grid.reshape(-1, 1, 2), H).reshape(9, 9, 2)

    h, w = gray.shape[:2]
    if not np.all(np.isfinite(nodes)):
        return None, "nonfinite_nodes"
    over = np.maximum(0.0, np.maximum(-nodes, nodes - np.array([w - 1, h - 1])))
    if float(np.max(np.linalg.norm(over, axis=2))) > 4.0:
        return None, "extrapolation"

    image_area = float(w * h)
    if not _quad_valid(np.array([nodes[0,0],nodes[0,8],nodes[8,8],nodes[8,0]]), image_area):
        return None, "invalid_board_area"
    mask = {(int(x["col"]), int(x["row"])): bool(x.get("occluded_or_estimated", False)) for x in calibration.get("cells", [])}
    cells = []
    for r in range(8):
        for c in range(8):
            quad = np.array([nodes[r, c], nodes[r, c + 1],
                             nodes[r + 1, c + 1], nodes[r + 1, c]], dtype=np.float64)
            if not _quad_valid(quad, 0):
                return None, "invalid_cell"
            center = quad.mean(axis=0)
            cells.append({
                "square": square_name(calibration["orientation"], c, r),
                "col": c,
                "row": r,
                "quad": quad.tolist(),
                "center": [float(center[0]), float(center[1])],
                "occluded_or_estimated": mask.get((c, r), False),
            })

    checker_count = 0
    for r in range(7):
        for c in range(7):
            pt, du, dv = _local_derivs(pts7, c, r)
            if _legacy__checker_ok(gray, pt, du, dv):
                checker_count += 1

    if checker_count < 32:
        return None, "checker_count_low"

    newcal = copy.deepcopy(calibration)
    newcal["grid_nodes"] = nodes.reshape(9, 9, 2).tolist()
    newcal["width"] = int(w)
    newcal["height"] = int(h)
    newcal["anchors_uv"] = _GRID_UV.tolist()
    newcal["anchors_pixel"] = pts7.tolist()
    newcal["heldout_uv"] = []
    newcal["heldout_pixel"] = []
    newcal["cells"] = cells

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    result = {
        "valid": True,
        "grid_nodes": nodes.reshape(81, 2).tolist(),
        "cells": cells,
        "width": int(w),
        "height": int(h),
        "matched": int(checker_count),
        "inliers": 49,
        "checker_count": int(checker_count),
        "median_fb_px": None,
        "p95_reprojection_px": p95,
        "elapsed_ms": elapsed_ms,
        "method": "checkerboard_relocalization",
        "reference_assisted": False,
        "reason": "ok_relocalized",
    }
    return result, newcal


def relocalize(image_bgr, calibration):
    """Pure relocalization of current chessboard into a fresh grid dict."""
    t0 = time.perf_counter()

    def fail(reason):
        return _blank_result(reason, (time.perf_counter() - t0) * 1000.0), None

    if calibration is None or not isinstance(calibration, dict):
        return fail("invalid_calibration")
    if calibration.get("orientation") not in _ORIENTATIONS:
        return fail("bad_orientation")
    if image_bgr is None:
        return fail("invalid_image")
    try:
        img = np.asarray(image_bgr)
    except Exception:
        return fail("invalid_image")
    if img.ndim != 3 or img.shape[2] != 3 or img.shape[0] < 10 or img.shape[1] < 10:
        return fail("invalid_image")
    if img.dtype != np.uint8:
        try:
            img = img.astype(np.uint8)
        except Exception:
            return fail("invalid_image")

    try:
        gray_full = _gray(img)
    except Exception:
        return fail("invalid_image")

    h, w = gray_full.shape[:2]
    scale = min(1.0, 800.0 / float(max(h, w)))
    if scale < 1.0:
        small = cv2.resize(gray_full, (int(round(w * scale)), int(round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = gray_full

    try:
        ok, corners = cv2.findChessboardCornersSB(
            small, (7, 7), flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    except Exception:
        return fail("detector_error")
    if not ok or corners is None:
        return fail("no_chessboard")
    corners_native = np.asarray(corners, dtype=np.float64).reshape(7, 7, 2)
    if scale < 1.0:
        corners_native = corners_native / scale
    pts7 = corners_native.reshape(49, 2).astype(np.float32).reshape(-1, 1, 2)
    try:
        cv2.cornerSubPix(gray_full, pts7, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    except Exception:
        pass
    pts7 = pts7.reshape(7, 7, 2).astype(np.float64)

    # Reorder 7x7 lattice to canonical (columns right, rows down, positive handedness).
    best = None
    best_score = -1.0
    lattice = pts7
    for flip_r in (False, True):
        for flip_c in (False, True):
            for transpose in (False, True):
                g = lattice
                if flip_r:
                    g = g[::-1, :]
                if flip_c:
                    g = g[:, ::-1]
                if transpose:
                    g = np.transpose(g, (1, 0, 2))
                right = g[0, 1] - g[0, 0]
                down = g[1, 0] - g[0, 0]
                hand = right[0] * down[1] - right[1] * down[0]
                if not (right[0] > 0 and down[1] > 0 and hand > 0):
                    continue
                rn = float(np.hypot(right[0], right[1]))
                dn = float(np.hypot(down[0], down[1]))
                if rn <= 1e-9 or dn <= 1e-9:
                    continue
                score = right[0] / rn + down[1] / dn
                if score > best_score:
                    best_score = score
                    best = g
    if best is None:
        return fail("ordering_failed")

    try:
        result, err = _build_from_uv7(gray_full, best, calibration, t0)
    except Exception:
        return fail("build_error")
    if result is None:
        return fail(err or "build_failed")
    return result, err


class GridLocator:
    """Async relocalization wrapper around legacy GridLocator."""

    def __init__(self, reference_bgr, calibration):
        if reference_bgr is None or calibration is None:
            raise ValueError("reference and calibration required")
        self.calibration = copy.deepcopy(calibration)
        self.tracker = _LegacyGridLocator(reference_bgr, self.calibration)
        self.reference_bgr = np.array(reference_bgr, copy=True)
        self._lock = threading.RLock()
        self._thread = None
        self._pending = None
        self._candidate = None
        self._last_error = None
        self._last_job_time = 0.0
        self.reacquisition_attempts = 0
        self.reacquisition_successes = 0

    # ---- worker -------------------------------------------------------
    def _worker(self, frame_copy, calibration):
        try:
            res, newcal = relocalize(frame_copy, calibration)
        except Exception as exc:  # capture diagnostic, never silent
            res, newcal = None, None
            reason = "worker_exception:%s" % type(exc).__name__
        else:
            reason = res.get("reason", "") if res else "worker_none"
        with self._lock:
            if res is not None and res.get("valid") and newcal is not None:
                self._candidate = {
                    "image": frame_copy,
                    "calibration": newcal,
                    "result": res,
                }
                self._last_error = None
            else:
                self._candidate = None
                self._last_error = reason or "relocalize_failed"
            self._pending = None
            self._thread = None

    def _maybe_schedule(self, frame):
        if self._pending is not None or self._candidate is not None:
            return
        now = time.monotonic()
        if now - self._last_job_time < 2.0:
            return
        self._last_job_time = now
        self.reacquisition_attempts += 1
        frame_copy = np.array(frame, copy=True)
        cal = copy.deepcopy(self.calibration)
        t = threading.Thread(target=self._worker, args=(frame_copy, cal), daemon=True)
        self._thread = t
        self._pending = t
        t.start()

    # ---- detect -------------------------------------------------------
    def detect(self, image_bgr):
        total_t0 = time.perf_counter()
        if image_bgr is None:
            return _blank_result("invalid_image",
                                 (time.perf_counter() - total_t0) * 1000.0,
                                 method="legacy")
        try:
            img = np.asarray(image_bgr)
        except Exception:
            return _blank_result("invalid_image",
                                 (time.perf_counter() - total_t0) * 1000.0,
                                 method="legacy")
        if img.ndim != 3 or img.shape[2] != 3:
            return _blank_result("invalid_image",
                                 (time.perf_counter() - total_t0) * 1000.0,
                                 method="legacy")
        if img.dtype != np.uint8:
            try:
                img = img.astype(np.uint8)
            except Exception:
                return _blank_result("invalid_image",
                                     (time.perf_counter() - total_t0) * 1000.0,
                                     method="legacy")

        with self._lock:
            adopted = self._consume_candidate(img)
            if adopted is not None:
                adopted["elapsed_ms"] = (time.perf_counter() - total_t0) * 1000.0
                adopted["reacquisition_attempts"] = self.reacquisition_attempts
                adopted["reacquisition_successes"] = self.reacquisition_successes
                return adopted

            try:
                out = self.tracker.detect(img)
            except Exception as exc:
                out = _blank_result("tracker_exception:%s" % type(exc).__name__,
                                    (time.perf_counter() - total_t0) * 1000.0,
                                    method="legacy")
            if out.get("valid"):
                out["method"] = "legacy_relocalized" if self.reacquisition_successes else "legacy"
                out["reason"] = "ok_relocalized" if self.reacquisition_successes else "ok"
                out["elapsed_ms"] = (time.perf_counter() - total_t0) * 1000.0
                out["reacquisition_attempts"] = self.reacquisition_attempts
                out["reacquisition_successes"] = self.reacquisition_successes
                return out

            self._maybe_schedule(img)
            base_reason = out.get("reason", "") or "detect_failed"
            if self._last_error:
                reason = base_reason + ";relocalizing;" + self._last_error
            else:
                reason = base_reason + ";relocalizing"
            res = _blank_result(reason,
                                (time.perf_counter() - total_t0) * 1000.0,
                                method="legacy")
            res["reacquisition_attempts"] = self.reacquisition_attempts
            res["reacquisition_successes"] = self.reacquisition_successes
            return res

    def _consume_candidate(self, img):
        cand = self._candidate
        if cand is None:
            return None
        self._candidate = None
        try:
            new_tracker = _LegacyGridLocator(cand["image"], cand["calibration"])
        except Exception as exc:
            self._last_error = "tracker_init:%s" % type(exc).__name__
            return None
        try:
            live = new_tracker.detect(img)
        except Exception as exc:
            self._last_error = "validate_exception:%s" % type(exc).__name__
            return None
        if not live.get("valid"):
            self._last_error = "validation_failed:" + str(live.get("reason", ""))
            return None
        self.tracker = new_tracker
        self.calibration = cand["calibration"]
        self.reference_bgr = cand["image"]
        self.reacquisition_successes += 1
        self._last_error = None
        live["method"] = "legacy_relocalized"
        live["reason"] = "ok_relocalized"
        return live


# ---- CLI ---------------------------------------------------------------
def _load_mapping(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_camera(mapping, cam_index):
    cams = mapping.get("cameras")
    if isinstance(cams, dict):
        cands = list(cams.items())
        items = []
        for k, v in cands:
            items.append((k, v))
    elif isinstance(cams, list):
        items = [(str(v.get("id", v.get("camera_id", ""))), v) for v in cams]
    else:
        return None
    want = str(cam_index + 1)
    for key, val in items:
        cid = str(val.get("id", val.get("camera_id", key)))
        if cid == want:
            return val
    return None


def _annotate(image, result):
    out = np.array(image, copy=True)
    nodes = result.get("grid_nodes", [])
    if nodes:
        nodes = np.asarray(nodes, dtype=np.float64).reshape(9, 9, 2)
        for r in range(9):
            for c in range(8):
                cv2.line(out, tuple(int(v) for v in nodes[r, c]),
                         tuple(int(v) for v in nodes[r, c + 1]), (0, 255, 0), 1)
        for r in range(8):
            for c in range(9):
                cv2.line(out, tuple(int(v) for v in nodes[r, c]),
                         tuple(int(v) for v in nodes[r + 1, c]), (0, 255, 0), 1)
    for cell in result.get("cells", []):
        cx, cy = cell["center"]
        cv2.putText(out, cell["square"], (int(cx) - 10, int(cy) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out


def _cli_main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-config", required=True)
    args = ap.parse_args(argv)
    with open(args.audit_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    inputs = cfg.get("inputs", [])
    if len(inputs) > 2:
        print("max 2 cases", file=sys.stderr)
        return 2
    mapping = _load_mapping(cfg["mapping_path"])
    outdir = cfg["output_dir"]
    if os.path.exists(outdir):
        print("output_dir exists", file=sys.stderr)
        return 2
    os.makedirs(outdir, exist_ok=False)
    results = []
    for case in inputs:
        ci = case.get("camera_index")
        if ci not in (0, 1):
            results.append({"camera_index": ci, "error": "bad_camera_index"})
            continue
        cam = _find_camera(mapping, ci)
        if cam is None:
            results.append({"camera_index": ci, "error": "camera_not_found"})
            continue
        img = cv2.imread(case["image"], cv2.IMREAD_COLOR)
        if img is None:
            results.append({"camera_index": ci, "error": "image_read_failed"})
            continue
        t0 = time.perf_counter()
        grid, newcal = relocalize(img, cam)
        elapsed = (time.perf_counter() - t0) * 1000.0
        tracking_valid = None
        if grid.get("valid") and newcal is not None:
            try:
                tl = _LegacyGridLocator(img, newcal)
                tr = tl.detect(img)
                tracking_valid = bool(tr.get("valid"))
            except Exception:
                tracking_valid = False
        ann = _annotate(img, grid)
        cv2.imwrite(os.path.join(outdir, "case_%d_annotated.png" % ci), ann)
        if grid.get("valid") and newcal is not None:
            cv2.imwrite(os.path.join(outdir, "case_%d_newref.png" % ci), img)
        results.append({
            "camera_index": ci,
            "grid": grid,
            "newcal": newcal,
            "tracking_valid": tracking_valid,
            "elapsed_ms": elapsed,
        })
    with open(os.path.join(outdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main(sys.argv[1:]))
