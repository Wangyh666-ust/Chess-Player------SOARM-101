import argparse
import hashlib
import json
import math
import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

CENTER = [640.0, 360.0]
SCALE = 640.0
IMG_W, IMG_H = 1280, 720
SQUARE_MM = 26.0
GRID_N = 9
DICT = cv2.aruco.DICT_4X4_50 if hasattr(cv2, "aruco") else None


def _np(x):
    return np.asarray(x, dtype=np.float64)


def undistort_points(p, k, center=CENTER, scale=SCALE):
    p = _np(p)
    c = _np(center)
    q = (p - c) / scale
    s = np.sum(q * q, axis=1)
    d = 1.0 + k * s
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("invalid distortion denominator")
    out = c + scale * q / d[:, None]
    if not np.all(np.isfinite(out)):
        raise ValueError("nonfinite undistort result")
    return out


def distort_points(u, k, center=CENTER, scale=SCALE):
    u = _np(u)
    c = _np(center)
    q = (u - c) / scale
    s = np.sum(q * q, axis=1)
    disc = 1.0 - 4.0 * k * s
    if not np.all(np.isfinite(disc)) or np.any(disc < 0):
        raise ValueError("invalid distortion discriminant")
    fac = 2.0 / (1.0 + np.sqrt(disc))
    out = c + scale * q * fac[:, None]
    if not np.all(np.isfinite(out)):
        raise ValueError("nonfinite distort result")
    return out


def project(H, p):
    p = _np(p).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(p.astype(np.float64), _np(H))
    res = out.reshape(-1, 2)
    if not np.all(np.isfinite(res)):
        raise ValueError("nonfinite projection")
    return res


def homography(uv, xy):
    uv = _np(uv).reshape(-1, 1, 2)
    xy = _np(xy).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(uv, xy, 0)
    if H is None or not np.all(np.isfinite(H)):
        raise ValueError("homography failed")
    return H


def _check_noncollinear(pts):
    pts = _np(pts)
    if pts.shape[0] < 3:
        return False
    c = pts - pts.mean(axis=0)
    u, s, _ = np.linalg.svd(c, full_matrices=False)
    return s[0] > 1e-9 and s[1] > 1e-9


def _robust_loss(r):
    return np.where(r <= 2.0, 0.5 * r * r, 2.0 * (r - 1.0))


def _view_data(view):
    uv = _np(view["uv"])
    px = _np(view["pixels"])
    ti = list(view["train_indices"])
    hi = list(view.get("heldout_indices", []))
    if uv.shape[0] != px.shape[0]:
        raise ValueError("uv/pixels length mismatch")
    if len(set(ti)) != len(ti) or len(set(hi)) != len(hi) or set(ti) & set(hi):
        raise ValueError("train/heldout must be disjoint unique indices")
    if not _check_noncollinear(uv[ti]):
        raise ValueError("board train points collinear")
    if len(ti) < 8:
        raise ValueError("need at least 8 train points")
    if not _check_noncollinear(px[ti]):
        raise ValueError("train points collinear")
    return uv, px, ti, hi


def _eval_k(k, views, center, scale):
    total = 0.0
    n = 0
    for view in views:
        uv, px, ti, _ = _view_data(view)
        und = undistort_points(px[ti], k, center, scale)
        H = homography(uv[ti], und)
        uv_all = uv[ti]
        pred_und = project(H, uv_all)
        pred_raw = distort_points(pred_und, k, center, scale)
        r = np.linalg.norm(pred_raw - px[ti], axis=1)
        total += float(np.sum(_robust_loss(r)))
        n += r.shape[0]
    return total / max(n, 1)


def _view_metrics(view, k, center, scale):
    uv, px, ti, hi = _view_data(view)
    und = undistort_points(px[ti], k, center, scale)
    H = homography(uv[ti], und)
    result = {"train_homography": H.tolist()}
    for name, idxs in (("train", ti), ("heldout", hi)):
        if len(idxs) == 0:
            result[name] = {"n": 0, "mean": None, "max": None, "all_errors": []}
            continue
        uv_sel = uv[idxs]
        pred_und = project(H, uv_sel)
        pred_raw = distort_points(pred_und, k, center, scale)
        err = np.linalg.norm(pred_raw - px[idxs], axis=1)
        result[name] = {
            "n": int(err.shape[0]),
            "mean": float(np.mean(err)),
            "max": float(np.max(err)),
            "all_errors": err.tolist(),
        }
    return result


def fit_model(views, center=CENTER, scale=SCALE):
    views = list(views)
    if len(views) == 0 or len(views) > 3:
        raise ValueError("need 1 to 3 training views")
    for v in views:
        _view_data(v)

    def obj(k):
        return _eval_k(k, views, center, scale)

    res = minimize_scalar(
        obj,
        bounds=(-0.3, 0.3),
        method="bounded",
        options={"xatol": 1e-5, "maxiter": 80},
    )
    if not res.success or not math.isfinite(res.x):
        raise RuntimeError("optimization failed")
    k = float(res.x)
    loss = float(res.fun)

    before = []
    after = []
    for v in views:
        before.append(_view_metrics(v, 0.0, center, scale))
        after.append(_view_metrics(v, k, center, scale))

    return {
        "k": k,
        "loss": loss,
        "objective_evaluations": int(getattr(res, "nfev", 0)),
        "at_bound": bool(abs(k) > 0.295),
        "views": views,
        "center": list(center),
        "scale": float(scale),
        "before": before,
        "after": after,
    }


def _dict():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco unavailable")
    return cv2.aruco.getPredefinedDictionary(DICT)


def _detect_markers(gray):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(_dict(), params).detectMarkers(gray)
    out = {}
    if ids is None:
        return out
    for c, i in zip(corners, ids.ravel().tolist()):
        if int(i) in out:
            raise ValueError("duplicate marker ID")
        out[int(i)] = np.asarray(c, dtype=np.float64).reshape(4,2)
    return out

def _refine(gray, corners, radius=3):
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    corners = corners.reshape(4, 1, 2).astype(np.float32)
    cv2.cornerSubPix(gray, corners, (radius, radius), (-1, -1), crit)
    return corners.reshape(4, 2).astype(np.float64)


def _layout_points(marker):
    corners = np.asarray(marker["corners_mm"], dtype=np.float64)
    if corners.shape != (4,3) or not np.isfinite(corners).all():
        raise ValueError("actual canonical corners must be finite4x3")
    return corners[:,:2] / SQUARE_MM

def _grid_uv():
    pts = []
    for y in range(GRID_N):
        for x in range(GRID_N):
            pts.append([x, y])
    return np.array(pts, dtype=np.float64)


def _fit_test(measured_pts, measured_uv, k, center, scale):
    und = undistort_points(measured_pts, k, center, scale)
    H = homography(measured_uv, und)
    grid = _grid_uv()
    gu = project(H, grid)
    raw = distort_points(gu, k, center, scale)
    return H, raw


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _draw_overlay(img, grid_raw, corners_list, labels, title):
    out = img.copy()
    for line in grid_raw:
        pts = np.round(line).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, (0, 200, 255), 1, cv2.LINE_AA)
    for corners in corners_list:
        pts = np.round(corners).astype(np.int32)
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 1, cv2.LINE_AA)
    for (x, y), t in labels:
        cv2.putText(
            out,
            t,
            (int(round(x)), int(round(y))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        out,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _curved_grid(H, k, center, scale):
    lines=[]
    t=np.linspace(0,8,17)
    for i in range(9):
        for uv in (np.column_stack([t,np.full_like(t,i)]),np.column_stack([np.full_like(t,i),t])):
            lines.append(distort_points(project(H,uv),k,center,scale))
    return lines

def _sample_curve(pts, n):
    t = np.linspace(0.0, 1.0, n)
    # simple polyline resampling by arc length
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return np.repeat(pts[:1], n, axis=0)
    si = np.linspace(0.0, s[-1], n)
    x = np.interp(si, s, pts[:, 0])
    y = np.interp(si, s, pts[:, 1])
    return np.stack([x, y], axis=1)


def _remap_image(img, k, center, scale):
    xs, ys = np.meshgrid(np.arange(IMG_W, dtype=np.float64), np.arange(IMG_H, dtype=np.float64))
    dst = np.stack([xs.ravel(), ys.ravel()], axis=1)
    src = distort_points(dst, k, center, scale)
    map_x = src[:, 0].reshape(IMG_H, IMG_W).astype(np.float32)
    map_y = src[:, 1].reshape(IMG_H, IMG_W).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def run(config, outdir):
    if os.path.exists(outdir):
        raise FileExistsError(outdir)
    os.makedirs(outdir)

    training = list(config.get("training_views", []))
    if len(training) > 3:
        raise ValueError("at most 3 training views")
    tests = list(config.get("tests", []))
    if len(tests) > 4:
        raise ValueError("at most 4 tests")
    center = list(config.get("center", CENTER))
    scale = float(config.get("scale", SCALE))

    layout_path = config["layout"]
    with open(layout_path, "r", encoding="utf-8-sig") as f:
        layout = json.load(f)
    markers = {}
    for m in layout.get("markers", []):
        markers[int(m["id"])] = m

    fit = fit_model(training, center, scale)
    k = fit["k"]

    params = {
        "model": "pixel_division_1param",
        "lambda": k,
        "fit_at_bound": fit["at_bound"],
        "objective_evaluations": fit["objective_evaluations"],
        "center": center,
        "scale": scale,
        "image_size": [IMG_W, IMG_H],
        "source": "3 historical grid-reference views",
        "intrinsics_calibrated": False,
        "provisional": True,
    }
    with open(os.path.join(outdir, "parameters.json"), "w") as f:
        json.dump(params, f, indent=2, allow_nan=False)

    train_summary = []
    for name, gate in (("before", 0.0), ("after", k)):
        metrics = []
        for v in training:
            m = _view_metrics(v, gate, center, scale)
            metrics.append({"train": m["train"], "heldout": m["heldout"]})
        train_summary.append({"k": gate, "views": metrics})

    test_results = []
    for t in tests:
        name = t["name"]
        img_path = t["image"]
        sha = t["sha256"]
        t0 = time.time()
        actual = _hash_file(img_path)
        if actual != sha:
            raise ValueError("hash mismatch for %s" % name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cannot read %s" % name)
        if img.shape[1] != IMG_W or img.shape[0] != IMG_H:
            raise ValueError("wrong size for %s" % name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found = _detect_markers(gray)
        ids = sorted(found.keys())
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate ids")
        known = [i for i in ids if i in markers]
        if len(known) < 3:
            test_results.append({"name": name, "status": "insufficient_ids", "ids": ids})
            continue
        measured_uv = []
        measured_px = []
        outlines = []
        for i in known:
            corners = found[i]
            mpts = _layout_points(markers[i])
            measured_uv.append(mpts)
            measured_px.append(corners)
        measured_uv = np.concatenate(measured_uv, axis=0)
        measured_px = np.concatenate(measured_px, axis=0)

        H0, grid0 = _fit_test(measured_px, measured_uv, 0.0, center, scale)
        H1, grid1 = _fit_test(measured_px, measured_uv, k, center, scale)

        res0 = np.linalg.norm(distort_points(project(H0, measured_uv), 0.0, center, scale) - measured_px, axis=1)
        res1 = np.linalg.norm(distort_points(project(H1, measured_uv), k, center, scale) - measured_px, axis=1)

        ref_metrics = None
        rp = t.get("reference_points")
        if rp:
            uv = _np([r["uv"] for r in rp])
            px = _np([r["pixel"] for r in rp])
            e0 = np.linalg.norm(distort_points(project(H0, uv), 0.0, center, scale) - px, axis=1)
            e1 = np.linalg.norm(distort_points(project(H1, uv), k, center, scale) - px, axis=1)
            ref_metrics = {
                "before": {"n": int(e0.shape[0]), "mean": float(np.mean(e0)), "max": float(np.max(e0))},
                "after": {"n": int(e1.shape[0]), "mean": float(np.mean(e1)), "max": float(np.max(e1))},
            }

        lines0 = _curved_grid(H0,0.,center,scale)
        lines1 = _curved_grid(H1,k,center,scale)
        labels = []
        for y in range(8):
            for x in range(8):
                c = distort_points(project(H1,[[x+.5,y+.5]]),k,center,scale)[0]
                labels.append((tuple(c), "%s%d" % (chr(ord('a')+x),y+1)))
        outlines = [found[i] for i in known]

        before_img = _draw_overlay(img, lines0, outlines, [], "raw H")
        after_img = _draw_overlay(img, lines1, outlines, labels, "division correction (provisional)")
        cv2.imwrite(os.path.join(outdir, name + "_before.png"), before_img)
        cv2.imwrite(os.path.join(outdir, name + "_after.png"), after_img)
        und = _remap_image(img, k, center, scale)
        cv2.imwrite(os.path.join(outdir, name + "_undistorted.png"), und)

        fit_ms = float((time.time() - t0) * 1000.0)
        test_results.append(
            {
                "name": name,
                "status": "ok",
                "ids": known,
                "H_before": H0.tolist(),
                "H_after": H1.tolist(),
                "raw_grid_before": grid0.tolist(),
                "raw_grid_after": grid1.tolist(),
                "marker_residual_before": {"n": int(res0.shape[0]), "mean": float(np.mean(res0)), "max": float(np.max(res0))},
                "marker_residual_after": {"n": int(res1.shape[0]), "mean": float(np.mean(res1)), "max": float(np.max(res1))},
                "reference_metrics": ref_metrics,
                "fit_ms": fit_ms,
                "same_canvas_may_crop": True,
            }
        )

    results = {
        "parameters": params,
        "train": train_summary,
        "tests": test_results,
        "limitations": [
            "fixed principal center assumed",
            "one radial term",
            "3 views, no full intrinsics",
            "layout/image annotation uncertainty",
            "no robot",
        ],
        "provisional": True,
        "intrinsics_calibrated": False,
    }
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, allow_nan=False)

    print("k=%.6f loss=%.6f at_bound=%s evals=%d" % (k, fit["loss"], fit["at_bound"], fit["objective_evaluations"]))
    for name, gate in (("before", 0.0), ("after", k)):
        print(name)
        for i, v in enumerate(training):
            m = _view_metrics(v, gate, center, scale)
            print(
                "  view%d train mean=%.4f heldout mean=%s"
                % (i, m["train"]["mean"], "None" if m["heldout"]["mean"] is None else "%.4f" % m["heldout"]["mean"])
            )
    for tr in test_results:
        if tr.get("status") == "ok" and tr.get("reference_metrics"):
            print("%s reference before=%.4f after=%.4f" % (tr["name"], tr["reference_metrics"]["before"]["mean"], tr["reference_metrics"]["after"]["mean"]))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("outdir")
    args = ap.parse_args(argv)
    with open(args.config, "r", encoding="utf-8-sig") as f:
        config = json.load(f)
    run(config, args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
