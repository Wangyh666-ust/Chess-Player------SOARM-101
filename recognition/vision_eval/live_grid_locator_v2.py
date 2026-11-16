"""Reference-assisted grid registration via optical flow.

Pure image-array library. No I/O, device, network, or CLI.
"""
import math
import cv2
import numpy as np


def square_name(orientation, col, row):
    """Return chess square name for integer col,row and camera orientation."""
    if orientation not in ("files_desc_ranks_asc", "ranks_desc_files_desc"):
        raise ValueError("bad orientation")
    for v in (col, row):
        if not isinstance(v, (int, np.integer)) or v < 0 or v > 7:
            raise ValueError("col/row must be int in 0..7")
    if orientation == "files_desc_ranks_asc":
        return chr(ord('h') - int(col)) + str(int(row) + 1)
    return chr(ord('h') - int(row)) + str(8 - int(col))


def rotate_points(points, width, height, clockwise):
    """Rotate points consistent with np.rot90(image, k=-clockwise//90)."""
    if clockwise not in (0, 90, 180, 270):
        raise ValueError("clockwise must be 0,90,180,270")
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    W, H = float(width - 1), float(height - 1)
    if clockwise == 0:
        out = pts.copy()
    elif clockwise == 90:
        out = np.stack([H - pts[:, 1], pts[:, 0]], axis=1)
    elif clockwise == 180:
        out = np.stack([W - pts[:, 0], H - pts[:, 1]], axis=1)
    else:
        out = np.stack([pts[:, 1], W - pts[:, 0]], axis=1)
    return out


def _gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _quad_median(gray, cx, cy, r=2):
    h, w = gray.shape
    x0 = int(round(cx)) - r
    y0 = int(round(cy)) - r
    x1 = x0 + 2 * r + 1
    y1 = y0 + 2 * r + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    patch = gray[y0:y1, x0:x1].astype(np.float32)
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


class GridLocator:
    """Reference-assisted grid locator using sparse LK + homography."""

    def __init__(self, reference_bgr, calibration):
        if reference_bgr is None or calibration is None:
            raise ValueError("reference and calibration required")
        ref = np.array(reference_bgr, copy=True)
        self.reference = ref
        self.cal = calibration
        self.orientation = calibration.get("orientation")
        if self.orientation not in ("files_desc_ranks_asc", "ranks_desc_files_desc"):
            raise ValueError("bad orientation")
        self.width = int(calibration["width"])
        self.height = int(calibration["height"])
        nodes = np.asarray(calibration["grid_nodes"], dtype=np.float64)
        if nodes.shape != (9, 9, 2):
            raise ValueError("grid_nodes must be 9x9x2")
        self.grid_nodes = nodes
        cells = calibration.get("cells", [])
        self.cells_ref = cells
        self.ref_gray = cv2.GaussianBlur(_gray(ref), (3, 3), 0)

        ap = np.asarray(calibration.get("anchors_pixel", []), dtype=np.float64).reshape(-1, 2)
        au = np.asarray(calibration.get("anchors_uv", []), dtype=np.float64).reshape(-1, 2)
        hp = np.asarray(calibration.get("heldout_pixel", []), dtype=np.float64).reshape(-1, 2)
        hu = np.asarray(calibration.get("heldout_uv", []), dtype=np.float64).reshape(-1, 2)
        pts = np.concatenate([ap, hp], axis=0) if len(ap) or len(hp) else np.zeros((0, 2))
        uvs = np.concatenate([au, hu], axis=0) if len(au) or len(hu) else np.zeros((0, 2))
        if len(pts) != len(uvs):
            raise ValueError("anchor/uv mismatch")

        # Deduplicate exact uv, keep first occurrence.
        kept_p = []
        kept_u = []
        seen = set()
        for p, u in zip(pts, uvs):
            key = (round(float(u[0]), 6), round(float(u[1]), 6))
            if key in seen:
                continue
            if p[0] < 10 or p[1] < 10 or p[0] > self.width - 11 or p[1] > self.height - 11:
                continue
            seen.add(key)
            kept_p.append(p)
            kept_u.append(u)
        self.init_pts = np.array(kept_p, dtype=np.float32).reshape(-1, 2)
        self.init_uv = np.array(kept_u, dtype=np.float64).reshape(-1, 2)

    # ------------------------------------------------------------------
    def _invalid(self, reason, t0, matched=0, inliers=0, checker_count=0):
        return {
            "valid": False,
            "reason": reason,
            "grid_nodes": [],
            "cells": [],
            "matched": int(matched),
            "inliers": int(inliers),
            "median_fb_px": None,
            "p95_reprojection_px": None,
            "elapsed_ms": float((cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0),
            "method": "reference_grid_optical_flow",
            "reference_assisted": True,
            "checker_count": int(checker_count),
        }

    def detect(self, image_bgr):
        t0 = cv2.getTickCount()
        if image_bgr is None:
            return self._invalid("no_image", t0)
        img = np.asarray(image_bgr)
        if img.ndim < 2:
            return self._invalid("bad_image", t0)
        h, w = img.shape[:2]
        if w != self.width or h != self.height:
            return self._invalid("size_mismatch", t0)
        if len(self.init_pts) < 12:
            return self._invalid("too_few_anchors", t0)

        cur_gray = cv2.GaussianBlur(_gray(img), (3, 3), 0)
        ref_gray = self.ref_gray

        p0 = self.init_pts.copy()
        lk = dict(winSize=(31, 31), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p1, st1, err1 = cv2.calcOpticalFlowPyrLK(ref_gray, cur_gray, p0, None, **lk)
        if p1 is None:
            return self._invalid("lk_forward_failed", t0)
        p0b, st2, err2 = cv2.calcOpticalFlowPyrLK(cur_gray, ref_gray, p1, None, **lk)
        if p0b is None:
            return self._invalid("lk_backward_failed", t0)

        st1 = st1.reshape(-1).astype(bool)
        st2 = st2.reshape(-1).astype(bool)
        e1 = err1.reshape(-1)
        fb = np.linalg.norm(p0b - p0, axis=1)
        disp = np.linalg.norm(p1 - p0, axis=1)
        ok = st1 & st2 & np.isfinite(fb) & (fb <= 1.5) & (e1 <= 28.0) & (disp <= 160.0)
        # also require finite p1
        ok &= np.isfinite(p1).all(axis=1)

        idx = np.where(ok)[0]
        if len(idx) < 12:
            return self._invalid("too_few_candidates", t0, matched=len(idx))
        src = p0[idx].astype(np.float32)
        dst = p1[idx].astype(np.float32)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=1000, confidence=0.995)
        if H is None or mask is None:
            return self._invalid("homography_failed", t0, matched=len(idx))
        mask = mask.reshape(-1).astype(bool)
        inl = int(mask.sum())
        if inl < 10 or inl < 0.55 * len(idx):
            return self._invalid("insufficient_inliers", t0, matched=len(idx), inliers=inl)

        in_idx = idx[mask]
        in_uv = self.init_uv[in_idx]
        uspan = float(in_uv[:, 0].max() - in_uv[:, 0].min())
        vspan = float(in_uv[:, 1].max() - in_uv[:, 1].min())
        if uspan < 4.0 - 1e-9 or vspan < 4.0 - 1e-9:
            return self._invalid("span_too_small", t0, matched=len(idx), inliers=inl)

        fb_in = fb[in_idx]
        med_fb = float(np.median(fb_in))
        if med_fb > 1.0:
            return self._invalid("forwardback_high", t0, matched=len(idx), inliers=inl)

        # reprojection residual on inliers
        proj = cv2.perspectiveTransform(src[mask].reshape(-1, 1, 2), H).reshape(-1, 2)
        res = np.linalg.norm(proj - dst[mask], axis=1)
        p95 = float(np.percentile(res, 95))
        if p95 > 3.0:
            return self._invalid("reprojection_high", t0, matched=len(idx), inliers=inl)

        # Checker validation on internal uv integers
        verified_uv = []
        for j, ii in enumerate(in_idx):
            u, v = self.init_uv[ii]
            ui, vi = int(round(u)), int(round(v))
            if abs(u - ui) > 1e-6 or abs(v - vi) > 1e-6:
                continue
            if ui < 1 or ui > 7 or vi < 1 or vi > 7:
                continue
            # neighbor step vectors from reference grid nodes
            gr = self.grid_nodes
            du_r = (gr[vi, ui + 1] - gr[vi, ui - 1]) / 2.0
            dv_r = (gr[vi + 1, ui] - gr[vi - 1, ui]) / 2.0
            # transform through H as directions (linear part only)
            origin = self.init_pts[ii].astype(np.float64)
            local = np.array([origin, origin + du_r, origin + dv_r], np.float64)
            transformed = cv2.perspectiveTransform(local.reshape(-1, 1, 2), H).reshape(-1, 2)
            du_c = transformed[1] - transformed[0]
            dv_c = transformed[2] - transformed[0]
            ref_pt = self.init_pts[ii].astype(np.float64)
            cur_pt = p1[ii].astype(np.float64)
            if not _checker_ok(ref_gray, ref_pt, du_r, dv_r):
                continue
            if not _checker_ok(cur_gray, cur_pt, du_c, dv_c):
                continue
            verified_uv.append((ui, vi))
        checker_count = len(verified_uv)
        if checker_count < 6:
            return self._invalid("checker_guard", t0, matched=len(idx), inliers=inl,
                                 checker_count=checker_count)
        vuv = np.array(verified_uv, dtype=np.float64)
        if (vuv[:, 0].max() - vuv[:, 0].min()) < 3 or (vuv[:, 1].max() - vuv[:, 1].min()) < 3:
            return self._invalid("checker_span", t0, matched=len(idx), inliers=inl,
                                 checker_count=checker_count)

        # Map reference grid nodes with H
        flat = self.grid_nodes.reshape(-1, 1, 2).astype(np.float64)
        mapped = cv2.perspectiveTransform(flat, H).reshape(9, 9, 2)
        if not np.isfinite(mapped).all():
            return self._invalid("bad_mapped_grid", t0, matched=len(idx), inliers=inl,
                                 checker_count=checker_count)

        # Validate cell quads
        cells_out = []
        total_area = 0.0
        ref_nodes = self.grid_nodes
        for cell in self.cells_ref:
            q = np.asarray(cell["quad"], dtype=np.float64).reshape(-1, 2)
            if q.shape[0] != 4:
                return self._invalid("bad_cell_quad", t0, matched=len(idx), inliers=inl,
                                     checker_count=checker_count)
            qc = cv2.perspectiveTransform(q.reshape(-1, 1, 2), H).reshape(-1, 2)
            if not np.isfinite(qc).all():
                return self._invalid("bad_cell_map", t0, matched=len(idx), inliers=inl,
                                     checker_count=checker_count)
            area = 0.5 * abs(np.dot(qc[:, 0], np.roll(qc[:, 1], -1)) -
                             np.dot(qc[:, 1], np.roll(qc[:, 0], -1)))
            if area <= 9.0:
                return self._invalid("cell_area", t0, matched=len(idx), inliers=inl,
                                     checker_count=checker_count)
            # convexity via cross products sign consistency
            signs = []
            for k in range(4):
                a = qc[k]
                b = qc[(k + 1) % 4]
                c = qc[(k + 2) % 4]
                cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
                signs.append(cr)
            signs = np.array(signs)
            if not (np.all(signs > 0) or np.all(signs < 0)):
                return self._invalid("nonconvex", t0, matched=len(idx), inliers=inl,
                                     checker_count=checker_count)
            # orientation: positive signed area
            signed = 0.5 * (np.dot(qc[:, 0], np.roll(qc[:, 1], -1)) -
                            np.dot(qc[:, 1], np.roll(qc[:, 0], -1)))
            if signed <= 0:
                return self._invalid("orientation_flip", t0, matched=len(idx), inliers=inl,
                                     checker_count=checker_count)
            # bounds
            if (qc[:, 0].min() < -8 or qc[:, 1].min() < -8 or
                    qc[:, 0].max() > self.width + 8 or qc[:, 1].max() > self.height + 8):
                return self._invalid("cell_bounds", t0, matched=len(idx), inliers=inl,
                                     checker_count=checker_count)
            # corner displacement limit vs reference
            qr = q.reshape(-1, 2)
            if np.max(np.linalg.norm(qc - qr, axis=1)) > 160.0:
                return self._invalid("displacement", t0, matched=len(idx), inliers=inl,
                                     checker_count=checker_count)
            total_area += area
            center = qc.mean(axis=0)
            cells_out.append({
                "square": cell.get("square"),
                "col": int(cell["col"]),
                "row": int(cell["row"]),
                "quad": qc.tolist(),
                "occluded_or_estimated": bool(cell.get("occluded_or_estimated", False)),
                "center": center.tolist(),
            })

        if len(cells_out) != 64:
            return self._invalid("cell_count", t0)
        ref_area = sum(abs(cv2.contourArea(np.asarray(c["quad"], np.float32))) for c in self.cells_ref)
        ratio = total_area / ref_area if ref_area > 0 else 0.0
        if ratio < 0.5 or ratio > 2.0:
            return self._invalid("board_area_ratio", t0, matched=len(idx), inliers=inl,
                                 checker_count=checker_count)

        return {
            "valid": True,
            "reason": "ok",
            "grid_nodes": mapped.reshape(81, 2).tolist(),
            "cells": cells_out,
            "matched": int(len(idx)),
            "inliers": inl,
            "median_fb_px": med_fb,
            "p95_reprojection_px": p95,
            "elapsed_ms": float((cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0),
            "method": "reference_grid_optical_flow",
            "reference_assisted": True,
            "checker_count": checker_count,
        }


def associate(grid, detections, cam):
    """Attach bbox-base footpoint square assignment to detections."""
    out = []
    valid = bool(grid and grid.get("valid") and grid.get("cells"))
    cells = grid.get("cells", []) if valid else []
    for det in detections:
        d = dict(det)
        box = d.get("box")
        if box is None or len(box) != 4:
            d.update({"footpoint": None, "square": None,
                      "assignment_status": "invalid_box",
                      "assignment_method": "bbox_base_heuristic"})
            out.append(d)
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        fp = (0.5 * (x1 + x2), y2 - 0.12 * (y2 - y1))
        d["footpoint"] = [fp[0], fp[1]]
        d["assignment_method"] = "bbox_base_heuristic"
        if not valid:
            d["square"] = None
            d["assignment_status"] = "grid_invalid"
            out.append(d)
            continue
        best = None
        best_dist = None
        for c in cells:
            q = np.asarray(c["quad"], dtype=np.float32)
            dist = cv2.pointPolygonTest(q, (float(fp[0]), float(fp[1])), True)
            if dist >= 0 and (best_dist is None or dist < best_dist):
                best_dist = dist
                best = c
        if best is None:
            d["square"] = None
            d["assignment_status"] = "outside"
        elif best_dist is not None and best_dist <= 2.0:
            d["square"] = None
            d["assignment_status"] = "boundary_ambiguous"
        elif best.get("occluded_or_estimated"):
            d["square"] = None
            d["assignment_status"] = "occluded"
        else:
            d["square"] = best.get("square")
            d["assignment_status"] = "assigned"
        out.append(d)

    # conflict marking
    counts = {}
    for d in out:
        sq = d.get("square")
        if sq is not None and d.get("assignment_status") == "assigned":
            counts[sq] = counts.get(sq, 0) + 1
    for d in out:
        if d.get("assignment_status") == "assigned":
            sq = d.get("square")
            if sq is not None and counts.get(sq, 0) > 1:
                d["assignment_status"] = "conflict"
    return out
