import copy
import json
import sys

import cv2
import numpy as np

import guided_grid_adapter_v2 as base

_UNSCALED = base.relocalize

_WORKING_MAX_WIDTH = 640


def _scale_geometry(obj, sx, sy):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = _scale_geometry(v, sx, sy)
        return out
    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        return [_scale_geometry(v, sx, sy) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, tuple):
        return tuple(_scale_geometry(v, sx, sy) for v in obj)
    return obj


def _scale_points(arr, sx, sy):
    if arr is None:
        return None
    if isinstance(arr, list) and len(arr) == 0:
        return []
    a = np.asarray(arr, dtype=np.float64).copy()
    if a.size == 0:
        return []
    if a.shape[-1] != 2:
        raise ValueError("last dimension must be 2")
    a = a * np.array([sx, sy], dtype=np.float64)
    return a.tolist()


def _scale_cal(cal, sx, sy):
    out = copy.deepcopy(cal)
    if not isinstance(out, dict):
        return out

    if "grid_nodes" in out:
        out["grid_nodes"] = _scale_points(out["grid_nodes"], sx, sy)

    if "anchors_pixel" in out:
        out["anchors_pixel"] = _scale_points(out["anchors_pixel"], sx, sy)
    if "heldout_pixel" in out:
        out["heldout_pixel"] = _scale_points(out["heldout_pixel"], sx, sy)

    if "cells" in out and isinstance(out["cells"], list):
        cells = []
        for c in out["cells"]:
            if not isinstance(c, dict):
                cells.append(c)
                continue
            nc = dict(c)
            if "quad" in nc:
                nc["quad"] = _scale_points(nc["quad"], sx, sy)
            if "center" in nc:
                center = nc["center"]
                if isinstance(center, (list, tuple)) and len(center) == 2:
                    nc["center"] = [float(center[0]) * sx, float(center[1]) * sy]
                elif isinstance(center, np.ndarray) and center.shape[-1] == 2:
                    q = center.astype(np.float64, copy=True)
                    q[..., 0] *= sx
                    q[..., 1] *= sy
                    nc["center"] = q
            cells.append(nc)
        out["cells"] = cells

    if "occlusion_rects" in out and isinstance(out["occlusion_rects"], list):
        rects = []
        for r in out["occlusion_rects"]:
            if isinstance(r, (list, tuple)) and len(r) == 4:
                rl = list(r)
                rects.append([float(rl[0]) * sx, float(rl[1]) * sy,
                              float(rl[2]) * sx, float(rl[3]) * sy])
            elif isinstance(r, np.ndarray) and r.shape == (4,):
                q = r.astype(np.float64, copy=True)
                q[0] *= sx
                q[1] *= sy
                q[2] *= sx
                q[3] *= sy
                rects.append(q)
            else:
                rects.append(r)
        out["occlusion_rects"] = rects

    if "width" in out:
        try:
            out["width"] = float(out["width"]) * sx
        except (TypeError, ValueError):
            pass
    if "height" in out:
        try:
            out["height"] = float(out["height"]) * sy
        except (TypeError, ValueError):
            pass

    return out


def _inverse_scale_geometry(obj, sx, sy):
    if isinstance(obj, dict):
        return {k: _inverse_scale_geometry(v, sx, sy) for k, v in obj.items()}
    return obj


def _inverse_points(arr, sx, sy):
    return _scale_points(arr, 1.0 / sx, 1.0 / sy)


def _inverse_scale_cal(cal, sx, sy, orig_w, orig_h):
    out = copy.deepcopy(cal)
    if not isinstance(out, dict):
        return out

    if "grid_nodes" in out:
        nodes = out["grid_nodes"]
        if isinstance(nodes, list):
            nodes = _inverse_points(nodes, sx, sy)
            out["grid_nodes"] = nodes
        elif isinstance(nodes, np.ndarray):
            n = nodes.astype(np.float64, copy=True)
            if n.shape[-1] == 2:
                n[..., 0] /= sx
                n[..., 1] /= sy
            out["grid_nodes"] = n

    if "anchors_pixel" in out:
        out["anchors_pixel"] = _inverse_points(out["anchors_pixel"], sx, sy)
    if "heldout_pixel" in out:
        out["heldout_pixel"] = _inverse_points(out["heldout_pixel"], sx, sy)

    if "cells" in out and isinstance(out["cells"], list):
        cells = []
        for c in out["cells"]:
            if not isinstance(c, dict):
                cells.append(c)
                continue
            nc = dict(c)
            if "quad" in nc:
                nc["quad"] = _inverse_points(nc["quad"], sx, sy)
            if "center" in nc:
                center = nc["center"]
                if isinstance(center, (list, tuple)) and len(center) == 2:
                    nc["center"] = [float(center[0]) / sx, float(center[1]) / sy]
                elif isinstance(center, np.ndarray) and center.shape[-1] == 2:
                    q = center.astype(np.float64, copy=True)
                    q[..., 0] /= sx
                    q[..., 1] /= sy
                    nc["center"] = q
            cells.append(nc)
        out["cells"] = cells

    if "occlusion_rects" in out and isinstance(out["occlusion_rects"], list):
        rects = []
        for r in out["occlusion_rects"]:
            if isinstance(r, (list, tuple)) and len(r) == 4:
                rl = list(r)
                rects.append([float(rl[0]) / sx, float(rl[1]) / sy,
                              float(rl[2]) / sx, float(rl[3]) / sy])
            elif isinstance(r, np.ndarray) and r.shape == (4,):
                q = r.astype(np.float64, copy=True)
                q[0] /= sx
                q[1] /= sy
                q[2] /= sx
                q[3] /= sy
                rects.append(q)
            else:
                rects.append(r)
        out["occlusion_rects"] = rects

    out["width"] = orig_w
    out["height"] = orig_h
    return out


def relocalize(image, cal):
    if cal is None:
        cal = {}

    orientation = None
    if isinstance(cal, dict):
        orientation = cal.get("orientation")

    if orientation != "files_desc_ranks_asc":
        return _UNSCALED(image, cal)

    if image is None:
        return {"valid": False}, None
    arr = image
    if not isinstance(arr, np.ndarray):
        return {"valid": False}, None
    if arr.ndim < 2:
        return {"valid": False}, None

    h, w = arr.shape[0], arr.shape[1]
    if w is None or w <= 0 or h is None or h <= 0:
        return {"valid": False}, None

    new_w = min(_WORKING_MAX_WIDTH, int(w))
    if new_w <= 0:
        new_w = int(w)
    new_h = int(round(h * (new_w / float(w))))
    if new_h <= 0:
        new_h = int(h)

    sx = new_w / float(w)
    sy = new_h / float(h)

    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    scaled_cal = _scale_cal(cal, sx, sy)
    scaled_cal["width"] = new_w
    scaled_cal["height"] = new_h

    result = _UNSCALED(resized, scaled_cal)

    if not isinstance(result, tuple) or len(result) != 2:
        return base.old._blank_result("bad_result"), None

    grid, newcal = result

    if not isinstance(grid, dict):
        return base.old._blank_result("bad_result"), None

    if not bool(grid.get("valid")):
        return grid, None

    if newcal is None:
        return base.old._blank_result("no_calibration"), None

    out_grid = copy.deepcopy(grid)
    out_cal = copy.deepcopy(newcal)

    for key in ("grid_nodes", "grid_nodes_pixel", "anchors_pixel", "heldout_pixel"):
        if key in out_grid:
            out_grid[key] = _inverse_points(out_grid[key], sx, sy)

    if isinstance(out_grid.get("observed_anchors"), dict):
        oa = out_grid["observed_anchors"]
        if "pixel" in oa:
            oa["pixel"] = _inverse_points(oa["pixel"], sx, sy)

    if isinstance(out_grid.get("cells"), list):
        cells = []
        for c in out_grid["cells"]:
            if isinstance(c, dict):
                nc = dict(c)
                if "quad" in nc:
                    nc["quad"] = _inverse_points(nc["quad"], sx, sy)
                if "center" in nc:
                    nc["center"] = _inverse_points(nc["center"], sx, sy)
                cells.append(nc)
            else:
                cells.append(c)
        out_grid["cells"] = cells

    if "occlusion_rects" in out_grid and isinstance(out_grid["occlusion_rects"], list):
        rects = []
        for r in out_grid["occlusion_rects"]:
            if isinstance(r, (list, tuple)) and len(r) == 4:
                rects.append([float(r[0]) / sx, float(r[1]) / sy,
                              float(r[2]) / sx, float(r[3]) / sy])
            elif isinstance(r, np.ndarray) and r.shape == (4,):
                q = r.astype(np.float64, copy=True)
                q[0] /= sx
                q[1] /= sy
                q[2] /= sx
                q[3] /= sy
                rects.append(q)
            else:
                rects.append(r)
        out_grid["occlusion_rects"] = rects

    out_grid["width"] = w
    out_grid["height"] = h

    p95 = out_grid.get("p95_reprojection_px")
    if isinstance(p95, (int, float)) and not isinstance(p95, bool):
        out_grid["p95_reprojection_px"] = float(p95) * max(1.0 / sx, 1.0 / sy)
    median_fb_px = out_grid.get("median_fb_px")
    if isinstance(median_fb_px, (int, float)) and not isinstance(median_fb_px, bool):
        out_grid["median_fb_px"] = float(median_fb_px) * max(1.0 / sx, 1.0 / sy)

    out_grid["working_resolution"] = [int(new_w), int(new_h)]
    out_grid["method"] = "guided_visible_corners_scale640"

    out_cal = _inverse_scale_cal(out_cal, sx, sy, w, h)
    out_cal["working_resolution"] = [int(new_w), int(new_h)]
    out_cal["method"] = "guided_visible_corners_scale640"

    return out_grid, out_cal


GridLocator = base.GridLocator

base.relocalize = relocalize


def _run_cli():
    old = base.old
    old_relocalize = old.relocalize
    old_legacy = old._LegacyGridLocator
    try:
        old.relocalize = relocalize
        old._LegacyGridLocator = GridLocator
        return old._cli_main(sys.argv[1:])
    finally:
        old.relocalize = old_relocalize
        old._LegacyGridLocator = old_legacy


if __name__ == "__main__":
    sys.exit(_run_cli())
