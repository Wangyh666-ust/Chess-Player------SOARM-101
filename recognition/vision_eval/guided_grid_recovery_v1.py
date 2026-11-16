import copy
import time

import cv2
import numpy as np

try:
    from . import live_grid_locator_v5 as _old
except ImportError:
    import live_grid_locator_v5 as _old

_METHOD = "guided_visible_corners"
_ORIENTATIONS = ("files_desc_ranks_asc", "files_asc_ranks_asc")
_GUIDED_REASON = "ok_guided"


def _blank(reason, elapsed_ms=0.0, matched=0, inliers=0, residual=None):
    try:
        out = _old._blank_result(
            reason,
            elapsed_ms=elapsed_ms,
            method=_METHOD,
            checker_count=int(inliers),
            residual=residual,
        )
    except Exception:
        out = {
            "valid": False,
            "grid_nodes": [],
            "cells": [],
            "width": 0,
            "height": 0,
            "matched": 0,
            "inliers": 0,
            "checker_count": 0,
            "median_fb_px": None,
            "p95_reprojection_px": residual,
            "elapsed_ms": float(elapsed_ms),
            "method": _METHOD,
            "reference_assisted": False,
            "reason": str(reason),
        }
    out["valid"] = False
    out["matched"] = int(matched)
    out["inliers"] = int(inliers)
    out["checker_count"] = int(inliers)
    out["median_fb_px"] = None
    out["p95_reprojection_px"] = residual
    out["method"] = _METHOD
    out["reference_assisted"] = True
    out["reason"] = str(reason)
    return out


def _rect_mask(shape, rects, pad=0.0):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if not rects:
        return mask.astype(bool)
    for r in rects:
        try:
            x0, y0, x1, y1 = [float(v) for v in r]
        except Exception:
            continue
        if pad:
            x0 -= pad
            y0 -= pad
            x1 += pad
            y1 += pad
        x0 = int(np.floor(max(0.0, min(w, x0))))
        y0 = int(np.floor(max(0.0, min(h, y0))))
        x1 = int(np.ceil(max(0.0, min(w, x1))))
        y1 = int(np.ceil(max(0.0, min(h, y1))))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1
    return mask.astype(bool)


def _inside(pt, w, h, margin=0.0):
    x, y = float(pt[0]), float(pt[1])
    return (x >= margin and y >= margin and x <= (w - 1 - margin)
            and y <= (h - 1 - margin))


def _quad_median(gray, x, y, half):
    h, w = gray.shape[:2]
    x0 = int(np.floor(x - half))
    x1 = int(np.ceil(x + half))
    y0 = int(np.floor(y - half))
    y1 = int(np.ceil(y + half))
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h or x1 <= x0 or y1 <= y0:
        return None
    patch = gray[y0:y1 + 1, x0:x1 + 1]
    if patch.size == 0:
        return None
    return float(np.median(patch))


def _border_probes(orientation):
    probes = []
    for d in np.arange(0.5, 8.0, 1.0):
        probes.append((-0.22, d))
        probes.append((8.22, d))
        probes.append((d, -0.22))
        probes.append((d, 8.22))
    return probes


def _soft_reject(mask_pad, w, h, pt, margin=3.0):
    if not _inside(pt, w, h, margin=0.0):
        return True
    xi = int(round(float(pt[0])))
    yi = int(round(float(pt[1])))
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return True
    return bool(mask_pad[yi, xi])


def recover(image_bgr, calibration):
    t0 = time.time()
    matched = 0
    inliers = 0

    if image_bgr is None:
        return _blank("no_image", (time.time() - t0) * 1000.0)
    if not isinstance(image_bgr, np.ndarray):
        return _blank("image_not_ndarray", (time.time() - t0) * 1000.0)
    if image_bgr.dtype != np.uint8:
        return _blank("image_dtype_not_uint8", (time.time() - t0) * 1000.0)
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return _blank("image_not_bgr3", (time.time() - t0) * 1000.0)
    h, w = image_bgr.shape[:2]
    if h < 16 or w < 16:
        return _blank("image_too_small", (time.time() - t0) * 1000.0)

    if calibration is None or not isinstance(calibration, dict):
        return _blank("calibration_missing", (time.time() - t0) * 1000.0)

    orientation = calibration.get("orientation", None)
    if orientation is None:
        return _blank("orientation_missing", (time.time() - t0) * 1000.0)
    if orientation not in _ORIENTATIONS:
        return _blank("orientation_invalid", (time.time() - t0) * 1000.0)

    nodes_in = calibration.get("grid_nodes", None)
    if nodes_in is None:
        return _blank("grid_nodes_missing", (time.time() - t0) * 1000.0)
    try:
        nodes = np.asarray(nodes_in, dtype=np.float64)
    except Exception:
        return _blank("grid_nodes_not_numeric", (time.time() - t0) * 1000.0)
    if nodes.shape != (9, 9, 2):
        return _blank("grid_nodes_bad_shape", (time.time() - t0) * 1000.0)
    if not np.all(np.isfinite(nodes)):
        return _blank("grid_nodes_not_finite", (time.time() - t0) * 1000.0)

    occlusion_rects = calibration.get("occlusion_rects", None)
    if occlusion_rects is not None and not isinstance(occlusion_rects, (list, tuple)):
        return _blank("occlusion_rects_bad_type", (time.time() - t0) * 1000.0)

    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return _blank("gray_conversion_failed", (time.time() - t0) * 1000.0)

    mask_pad = _rect_mask((h, w), occlusion_rects, pad=3.0)

    try:
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=1500,
            qualityLevel=0.008,
            minDistance=5,
            blockSize=5,
        )
    except cv2.error:
        return _blank("good_features_failed", (time.time() - t0) * 1000.0)

    if corners is None or len(corners) == 0:
        return _blank("no_corners", (time.time() - t0) * 1000.0)

    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)

    try:
        refined = cv2.cornerSubPix(
            gray,
            corners.copy(),
            (3, 3),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.03),
        )
    except cv2.error:
        refined = corners

    pts = refined.reshape(-1, 2).astype(np.float64)

    keep = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        if not (_inside(p, w, h, margin=0.0)):
            keep[i] = False
            continue
        xi = int(round(p[0]))
        yi = int(round(p[1]))
        if xi < 0 or yi < 0 or xi >= w or yi >= h or mask_pad[yi, xi]:
            keep[i] = False
    pts = pts[keep]

    if len(pts) == 0:
        return _blank("no_usable_corners", (time.time() - t0) * 1000.0)

    used = np.zeros(len(pts), dtype=bool)
    uv_list = []
    px_list = []

    cell_du_ref = None
    spans = []

    for v in range(1, 8):
        for u in range(1, 8):
            seed = nodes[v, u]
            du_vec = (nodes[v, u + 1] - nodes[v, u - 1]) / 2.0
            dv_vec = (nodes[v + 1, u] - nodes[v - 1, u]) / 2.0
            ndu = float(np.hypot(du_vec[0], du_vec[1]))
            ndv = float(np.hypot(dv_vec[0], dv_vec[1]))
            if not (np.isfinite(ndu) and np.isfinite(ndv)):
                continue
            if ndu < 1.0 or ndv < 1.0:
                continue
            scale = min(ndu, ndv)
            radius = min(0.40 * scale, 40.0)
            if radius <= 0:
                continue
            d = np.hypot(pts[:, 0] - seed[0], pts[:, 1] - seed[1])
            near_idx = np.where((d <= radius) & (~used))[0]
            if len(near_idx) == 0:
                continue
            order = near_idx[np.argsort(d[near_idx])]
            picked = -1
            for idx in order:
                cand = pts[idx]
                if mask_pad[int(round(cand[1])), int(round(cand[0]))]:
                    continue
                ok = True
                for su, sv in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                    c = cand + su * 0.16 * du_vec + sv * 0.16 * dv_vec
                    if not _inside(c, w, h, margin=0.0):
                        ok = False
                        break
                    xi = int(round(c[0]))
                    yi = int(round(c[1]))
                    if xi < 0 or yi < 0 or xi >= w or yi >= h or mask_pad[yi, xi]:
                        ok = False
                        break
                if not ok:
                    continue
                if not _old._checker_ok(gray, cand, du_vec, dv_vec):
                    continue
                picked = int(idx)
                break
            if picked < 0:
                continue
            used[picked] = True
            uv_list.append((float(u), float(v)))
            px_list.append((float(pts[picked, 0]), float(pts[picked, 1])))

    matched = len(uv_list)
    if matched < 12:
        return _blank(
            "insufficient_matches",
            (time.time() - t0) * 1000.0,
            matched=matched,
            inliers=0,
        )

    uv_arr = np.asarray(uv_list, dtype=np.float64)
    px_arr = np.asarray(px_list, dtype=np.float64)
    all_pts = uv_arr.reshape(-1, 1, 2)
    all_px = px_arr.reshape(-1, 1, 2)

    if (uv_arr[:, 0].max() - uv_arr[:, 0].min()) < 4.0 or \
       (uv_arr[:, 1].max() - uv_arr[:, 1].min()) < 4.0:
        return _blank(
            "insufficient_uv_span",
            (time.time() - t0) * 1000.0,
            matched=matched,
            inliers=0,
        )

    try:
        H, mask = cv2.findHomography(all_pts, all_px, cv2.RANSAC, 4.0)
    except cv2.error:
        return _blank(
            "homography_failed",
            (time.time() - t0) * 1000.0,
            matched=matched,
            inliers=0,
        )

    if H is None or mask is None:
        return _blank(
            "homography_none",
            (time.time() - t0) * 1000.0,
            matched=matched,
            inliers=0,
        )
    if not np.all(np.isfinite(H)):
        return _blank(
            "homography_not_finite",
            (time.time() - t0) * 1000.0,
            matched=matched,
            inliers=0,
        )

    mask = mask.reshape(-1).astype(bool)
    inliers = int(np.sum(mask))
    if inliers < 12:
        return _blank(
            "insufficient_inliers",
            (time.time() - t0) * 1000.0,
            matched=matched,
            inliers=inliers,
        )
    if matched > 0 and (inliers / float(matched)) < 0.65:
        return _blank(
            "low_inlier_ratio",
            (time.time() - t0) * 1000.0,
            matched=matched,
            inliers=inliers,
        )

    in_uv = uv_arr[mask]
    in_px = px_arr[mask]

    if (in_uv[:, 0].max() - in_uv[:, 0].min()) < 4.0 or \
       (in_uv[:, 1].max() - in_uv[:, 1].min()) < 4.0:
        return _blank(
            "inlier_uv_span_small",
            (time.time() - t0) * 1000.0,
            matched=matched,
            inliers=inliers,
        )

    proj = cv2.perspectiveTransform(in_uv.reshape(-1, 1, 2), H).reshape(-1, 2)
    resid = np.hypot(proj[:, 0] - in_px[:, 0], proj[:, 1] - in_px[:, 1])
    p95 = float(np.percentile(resid, 95.0)) if len(resid) else None
    if p95 is None or p95 > 4.0:
        return _blank(
            "reprojection_too_large",
            (time.time() - t0) * 1000.0,
            matched=matched,
            inliers=inliers,
            residual=p95,
        )

    uu, vv = np.meshgrid(np.arange(9, dtype=np.float64), np.arange(9, dtype=np.float64))
    lattice = np.stack([uu, vv], axis=-1).reshape(-1, 1, 2)
    nodes_proj = cv2.perspectiveTransform(lattice, H).reshape(9, 9, 2)

    if not np.all(np.isfinite(nodes_proj)):
        return _blank(
            "projected_nodes_not_finite",
            (time.time() - t0) * 1000.0,
            matched=matched,
            inliers=inliers,
            residual=p95,
        )

    cell_areas = []
    for vy in range(8):
        for vx in range(8):
            q = np.array([
                nodes_proj[vy, vx],
                nodes_proj[vy, vx + 1],
                nodes_proj[vy + 1, vx + 1],
                nodes_proj[vy + 1, vx],
            ], dtype=np.float64)
            area = 0.0
            sign = 0
            convex = True
            for k in range(4):
                a = q[k]
                b = q[(k + 1) % 4]
                c = q[(k + 2) % 4]
                cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
                s = 1 if cross > 0 else (-1 if cross < 0 else 0)
                if s == 0:
                    convex = False
                elif sign == 0:
                    sign = s
                elif sign != s:
                    convex = False
                area += (a[0] * b[1] - b[0] * a[1])
            area = abs(area) / 2.0
            if not convex or area < 36.0:
                return _blank(
                    "bad_cell_geometry",
                    (time.time() - t0) * 1000.0,
                    matched=matched,
                    inliers=inliers,
                    residual=p95,
                )
            cell_areas.append(area)

    cell_areas = np.asarray(cell_areas, dtype=np.float64)
    med_area = float(np.median(cell_areas))
    frame_area = float(h * w)
    total_area = float(np.sum(cell_areas))
    if frame_area <= 0:
        return _blank("frame_area_zero", (time.time() - t0) * 1000.0)
    frac = total_area / frame_area
    if frac < 0.015 or frac > 0.85:
        return _blank(
            "board_area_out_of_range",
            (time.time() - t0) * 1000.0,
            matched=matched,
            inliers=inliers,
            residual=p95,
        )

    med_side = float(np.sqrt(med_area))
    clip_allow = 0.75 * med_side
    for vy in range(9):
        for vx in range(9):
            p = nodes_proj[vy, vx]
            dx = max(0.0, -p[0], p[0] - (w - 1))
            dy = max(0.0, -p[1], p[1] - (h - 1))
            dist_out = float(np.hypot(dx, dy))
            if dist_out > clip_allow:
                return _blank(
                    "nodes_outside_frame",
                    (time.time() - t0) * 1000.0,
                    matched=matched,
                    inliers=inliers,
                    residual=p95,
                )

    disp = np.hypot(nodes_proj[..., 0] - nodes[..., 0], nodes_proj[..., 1] - nodes[..., 1])
    med_prior_side_vals = []
    for vy in range(8):
        for vx in range(8):
            q = np.array([
                nodes[vy, vx],
                nodes[vy, vx + 1],
                nodes[vy + 1, vx + 1],
                nodes[vy + 1, vx],
            ], dtype=np.float64)
            a = 0.0
            for k in range(4):
                j = (k + 1) % 4
                a += q[k, 0] * q[j, 1] - q[j, 0] * q[k, 1]
            med_prior_side_vals.append(abs(a) / 2.0)
    med_prior_area = float(np.median(np.asarray(med_prior_side_vals)))
    med_prior_side = float(np.sqrt(max(med_prior_area, 1e-6)))
    if float(np.max(disp)) > 0.6 * med_prior_side:
        return _blank(
            "displacement_guard",
            (time.time() - t0) * 1000.0,
            matched=matched,
            inliers=inliers,
            residual=p95,
        )

    bright_vals = []
    dark_vals = []
    quad_samples = []
    for k in range(len(in_uv)):
        u_val, v_val = in_uv[k]
        seed = nodes[int(round(v_val)), int(round(u_val))] if (
            0 <= int(round(v_val)) <= 8 and 0 <= int(round(u_val)) <= 8) else None
        du_vec = None
        dv_vec = None
        ui = int(round(u_val))
        vi = int(round(v_val))
        if 0 <= ui - 1 <= 8 and ui + 1 <= 8 and 0 <= vi - 1 <= 8 and vi + 1 <= 8:
            du_vec = np.array([nodes[vi, ui + 1, 0] - nodes[vi, ui - 1, 0],
                               nodes[vi, ui + 1, 1] - nodes[vi, ui - 1, 1]]) / 2.0
            dv_vec = np.array([nodes[vi + 1, ui, 0] - nodes[vi - 1, ui, 0],
                               nodes[vi + 1, ui, 1] - nodes[vi - 1, ui, 1]]) / 2.0
        if du_vec is None or dv_vec is None:
            continue
        cand = in_px[k]
        qs = []
        ok = True
        for su, sv in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            c = cand + su * 0.16 * du_vec + sv * 0.16 * dv_vec
            m = _quad_median(gray, c[0], c[1], 2)
            if m is None:
                ok = False
                break
            qs.append(m)
        if not ok:
            continue
        quad_samples.append(qs)
        srt = sorted(qs)
        dark_vals.append(srt[0])
        dark_vals.append(srt[1])
        bright_vals.append(srt[2])
        bright_vals.append(srt[3])

    if len(bright_vals) < 8 or len(dark_vals) < 8:
        return _blank(
            "insufficient_quadrant_samples",
            (time.time() - t0) * 1000.0,
            matched=matched,
            inliers=inliers,
            residual=p95,
        )

    bright_med = float(np.median(bright_vals))
    dark_med = float(np.median(dark_vals))
    contrast = bright_med - dark_med
    if contrast < 35.0:
        return _blank(
            "low_checker_contrast",
            (time.time() - t0) * 1000.0,
            matched=matched,
            inliers=inliers,
            residual=p95,
        )
    mid = 0.5 * (bright_med + dark_med)

    probe_uv = []
    for uu_p, vv_p in _border_probes(orientation):
        uv_pt = np.array([[[float(uu_p), float(vv_p)]]], dtype=np.float64)
        proj_pt = cv2.perspectiveTransform(uv_pt, H).reshape(2)
        if not _inside(proj_pt, w, h, margin=0.0):
            continue
        xi = int(round(proj_pt[0]))
        yi = int(round(proj_pt[1]))
        if xi < 0 or yi < 0 or xi >= w or yi >= h or mask_pad[yi, xi]:
            continue
        val = _quad_median(gray, proj_pt[0], proj_pt[1], 2)
        if val is None:
            continue
        probe_uv.append((uu_p, vv_p, val))

    if len(probe_uv) < 10:
        return _blank(
            "insufficient_border_probes",
            (time.time() - t0) * 1000.0,
            matched=matched,
            inliers=inliers,
            residual=p95,
        )

    edges_hit = set()
    n_above = 0
    for uu_p, vv_p, val in probe_uv:
        if val > mid:
            n_above += 1
            if abs(uu_p + 0.22) < 1e-6:
                edges_hit.add("u_min")
            if abs(uu_p - 8.22) < 1e-6:
                edges_hit.add("u_max")
            if abs(vv_p + 0.22) < 1e-6:
                edges_hit.add("v_min")
            if abs(vv_p - 8.22) < 1e-6:
                edges_hit.add("v_max")
    frac_above = n_above / float(len(probe_uv))
    if len(edges_hit) < 2 or frac_above < 0.75:
        return _blank(
            "border_evidence_insufficient",
            (time.time() - t0) * 1000.0,
            matched=matched,
            inliers=inliers,
            residual=p95,
        )

    cells = []
    for col in range(8):
        for row in range(8):
            quad = [
                nodes_proj[col, row],
                nodes_proj[col, row + 1],
                nodes_proj[col + 1, row + 1],
                nodes_proj[col + 1, row],
            ]
            center_uv = np.array([[[col + 0.5, row + 0.5]]], dtype=np.float64)
            center = cv2.perspectiveTransform(center_uv, H).reshape(2)

            out_of_frame = False
            occluded = False
            for qp in quad:
                if not _inside(qp, w, h, margin=0.0):
                    out_of_frame = True
                    break
            if occlusion_rects:
                xs = [q[0] for q in quad]
                ys = [q[1] for q in quad]
                bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
                for r in occlusion_rects:
                    try:
                        rx0, ry0, rx1, ry1 = [float(val) for val in r]
                    except Exception:
                        continue
                    if bx0 < rx1 and bx1 > rx0 and by0 < ry1 and by1 > ry0:
                        occluded = True
                        break

            if out_of_frame:
                visibility = "out_of_frame"
                occluded_or_estimated = True
            elif occluded:
                visibility = "occluded"
                occluded_or_estimated = True
            else:
                visibility = "visible"
                occluded_or_estimated = False

            name = _old.square_name(orientation, col, row)
            cells.append({
                "square": name,
                "col": int(col),
                "row": int(row),
                "quad": [[float(q[0]), float(q[1])] for q in quad],
                "center": [float(center[0]), float(center[1])],
                "occluded_or_estimated": bool(occluded_or_estimated),
                "visibility": visibility,
                "visible": bool(visibility == "visible"),
            })

    if len(cells) != 64:
        return _blank(
            "cell_count_mismatch",
            (time.time() - t0) * 1000.0,
            matched=matched,
            inliers=inliers,
            residual=p95,
        )

    elapsed_ms = (time.time() - t0) * 1000.0

    anchors_uv = in_uv.reshape(-1, 2).astype(float).tolist()
    anchors_pixel = in_px.reshape(-1, 2).astype(float).tolist()

    grid = {
        "valid": True,
        "width": int(w),
        "height": int(h),
        "grid_nodes": nodes_proj.reshape(81, 2).astype(float).tolist(),
        "cells": cells,
        "matched": int(matched),
        "inliers": int(inliers),
        "checker_count": int(inliers),
        "median_fb_px": None,
        "p95_reprojection_px": float(p95),
        "elapsed_ms": float(elapsed_ms),
        "method": _METHOD,
        "reference_assisted": True,
        "reason": _GUIDED_REASON,
        "border_support": {
            "contrast": float(contrast),
            "midpoint": float(mid),
            "probes_total": int(len(probe_uv)),
            "probes_above": int(n_above),
            "fraction_above": float(frac_above),
            "edges": sorted(edges_hit),
        },
        "observed_anchors": {
            "count": int(inliers),
            "uv": anchors_uv,
            "pixel": anchors_pixel,
        },
        "border_coordinate_assumption": (
            "board_outer_border_lies_at_uv_u_equals_0_or_8_and_v_equals_0_or_8"
        ),
    }

    newcal = copy.deepcopy(calibration)
    newcal["grid_nodes"] = nodes_proj.reshape(9, 9, 2).astype(float).tolist()
    newcal["width"] = int(w)
    newcal["height"] = int(h)
    newcal["cells"] = copy.deepcopy(cells)
    newcal["anchors_uv"] = anchors_uv
    newcal["anchors_pixel"] = anchors_pixel
    newcal["heldout"] = []
    newcal["allow_partial_grid"] = True

    return grid, newcal
