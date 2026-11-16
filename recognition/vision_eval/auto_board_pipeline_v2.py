import hashlib
import importlib
import json
import math
import os
import sys
import time
import traceback
import uuid
from types import SimpleNamespace

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

real_camera_inference_v2 = importlib.import_module("real_camera_inference_v2")

CORNER_CLASSES = ("top_left", "top_right", "bottom_left", "bottom_right")
REQUIRED_CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")

CONFIG_KEYS = {
    "images",
    "corner_model",
    "piece_model",
    "corner_inference",
    "piece_inference",
    "crop_policy",
    "overlap_policy",
    "output_dir",
    "max_predictions",
    "max_seconds",
}

CROP_POLICY_KEYS = {
    "pad_x",
    "pad_top",
    "pad_bottom",
    "min_area_ratio",
    "max_area_ratio",
    "min_edge_ratio",
}

OVERLAP_POLICY_KEYS = {
    "min_ioa",
    "max_center_dx",
    "max_bottom_dy",
}

INFERENCE_KEYS = {"imgsz", "conf", "iou"}

SAFE_ID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


class PipelineValidationError(ValueError):
    pass


def _is_finite_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _validate_safe_id(image_id):
    if not isinstance(image_id, str) or not image_id:
        raise PipelineValidationError("image id must be a nonempty string")
    if len(image_id) > 128:
        raise PipelineValidationError("image id too long: %r" % (image_id,))
    for ch in image_id:
        if ch not in SAFE_ID_CHARS:
            raise PipelineValidationError("unsafe character in image id: %r" % (image_id,))
    if image_id in (".", ".."):
        raise PipelineValidationError("reserved image id: %r" % (image_id,))
    if not real_camera_inference_v2.ID_RE.fullmatch(image_id):
        raise PipelineValidationError("image id is incompatible with reused inference runner")


def _validate_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise PipelineValidationError("box must be a list/tuple of 4 numbers")
    vals = []
    for v in box:
        if not _is_finite_number(v):
            raise PipelineValidationError("box coordinate must be finite number: %r" % (v,))
        vals.append(float(v))
    x1, y1, x2, y2 = vals
    if not (x2 > x1 and y2 > y1):
        raise PipelineValidationError("box must satisfy x2>x1 and y2>y1")
    return vals


def _validate_detection(det):
    if not isinstance(det, dict):
        raise PipelineValidationError("detection must be a dict")
    if "box" not in det:
        raise PipelineValidationError("detection missing box")
    box = _validate_box(det["box"])
    conf = det.get("confidence")
    if not _is_finite_number(conf) or float(conf) < 0.0 or float(conf) > 1.0:
        raise PipelineValidationError("confidence must be finite in [0,1]")
    return box, float(conf)


def _validate_crop_policy(policy):
    if not isinstance(policy, dict):
        raise PipelineValidationError("crop_policy must be a dict")
    missing = CROP_POLICY_KEYS - set(policy.keys())
    if missing:
        raise PipelineValidationError("crop_policy missing keys: %s" % sorted(missing))
    for k in CROP_POLICY_KEYS:
        if not _is_finite_number(policy[k]):
            raise PipelineValidationError("crop_policy.%s must be finite number" % k)
    if policy["pad_x"] < 0 or policy["pad_top"] < 0 or policy["pad_bottom"] < 0:
        raise PipelineValidationError("crop_policy pads must be nonnegative")
    if not (0.0 <= policy["min_area_ratio"] <= policy["max_area_ratio"] <= 1.0):
        raise PipelineValidationError("crop_policy area ratios invalid")
    if not (0.0 <= policy["min_edge_ratio"] <= 1.0):
        raise PipelineValidationError("crop_policy.min_edge_ratio must be in [0,1]")


def _validate_overlap_policy(policy):
    if not isinstance(policy, dict):
        raise PipelineValidationError("overlap_policy must be a dict")
    missing = OVERLAP_POLICY_KEYS - set(policy.keys())
    if missing:
        raise PipelineValidationError("overlap_policy missing keys: %s" % sorted(missing))
    for k in OVERLAP_POLICY_KEYS:
        if not _is_finite_number(policy[k]):
            raise PipelineValidationError("overlap_policy.%s must be finite number" % k)
    if not (0.0 <= policy["min_ioa"] <= 1.0):
        raise PipelineValidationError("overlap_policy.min_ioa must be in [0,1]")
    if policy["max_center_dx"] < 0 or policy["max_bottom_dy"] < 0:
        raise PipelineValidationError("overlap_policy distances must be nonnegative")


def _polygon_area(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def select_board_crop(detections, width, height, policy):
    result = {
        "ok": False,
        "reason": None,
        "corners": None,
        "crop": None,
        "selected_indices": [],
    }
    if not isinstance(detections, list):
        return result
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        result["reason"] = "invalid_image_dimensions"
        return result
    if width <= 0 or height <= 0:
        result["reason"] = "invalid_image_dimensions"
        return result
    width = int(width)
    height = int(height)

    try:
        _validate_crop_policy(policy)
    except PipelineValidationError as e:
        result["reason"] = "invalid_policy: %s" % e
        return result

    parsed = []
    for i, det in enumerate(detections):
        try:
            box, conf = _validate_detection(det)
        except PipelineValidationError as e:
            result["reason"] = "invalid_detection_index_%d: %s" % (i, e)
            return result
        parsed.append({"index": i, "box": box, "conf": conf, "name": det.get("class_name")})

    chosen = {}
    for name in CORNER_CLASSES:
        best = None
        for p in parsed:
            if p["name"] != name:
                continue
            if best is None:
                best = p
            else:
                if p["conf"] > best["conf"]:
                    best = p
                elif p["conf"] == best["conf"] and p["index"] < best["index"]:
                    best = p
        if best is not None:
            chosen[name] = best

    missing = [n for n in CORNER_CLASSES if n not in chosen]
    if missing:
        result["reason"] = "missing_corner_classes: %s" % ",".join(missing)
        return result

    centers = {}
    for name in CORNER_CLASSES:
        p = chosen[name]
        x1, y1, x2, y2 = p["box"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if cx < 0 or cx > width or cy < 0 or cy > height:
            result["reason"] = "corner_center_outside_image: %s" % name
            return result
        centers[name] = (cx, cy, p)

    ordered = [centers[n] for n in REQUIRED_CORNER_ORDER]
    pts = [(c[0], c[1]) for c in ordered]

    crosses = []
    for i in range(4):
        o = pts[i]
        a = pts[(i + 1) % 4]
        b = pts[(i + 2) % 4]
        crosses.append(_cross(o, a, b))

    if any(abs(c) < 1e-9 for c in crosses):
        result["reason"] = "degenerate_quad"
        return result
    pos = any(c > 0 for c in crosses)
    neg = any(c < 0 for c in crosses)
    if pos and neg:
        result["reason"] = "non_convex_quad"
        return result

    quad_area = abs(_polygon_area(pts))
    img_area = float(width) * float(height)
    ratio = quad_area / img_area
    if ratio < float(policy["min_area_ratio"]) - 1e-12:
        result["reason"] = "quad_area_too_small"
        return result
    if ratio > float(policy["max_area_ratio"]) + 1e-12:
        result["reason"] = "quad_area_too_large"
        return result

    min_edge = float(policy["min_edge_ratio"]) * float(min(width, height))
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        edge = math.hypot(b[0] - a[0], b[1] - a[1])
        if edge < min_edge - 1e-9:
            result["reason"] = "quad_edge_too_short"
            return result

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    bx1 = min(xs)
    by1 = min(ys)
    bx2 = max(xs)
    by2 = max(ys)
    bw = bx2 - bx1
    bh = by2 - by1

    pad_left = float(policy["pad_x"]) * bw
    pad_right = float(policy["pad_x"]) * bw
    pad_top = float(policy["pad_top"]) * bh
    pad_bottom = float(policy["pad_bottom"]) * bh

    cx1 = int(math.floor(bx1 - pad_left))
    cy1 = int(math.floor(by1 - pad_top))
    cx2 = int(math.ceil(bx2 + pad_right))
    cy2 = int(math.ceil(by2 + pad_bottom))

    cx1 = max(0, min(cx1, width))
    cy1 = max(0, min(cy1, height))
    cx2 = max(0, min(cx2, width))
    cy2 = max(0, min(cy2, height))

    if cx2 <= cx1 or cy2 <= cy1:
        result["reason"] = "empty_crop_after_clip"
        return result

    result["ok"] = True
    result["reason"] = None
    result["corners"] = [[float(p[0]), float(p[1])] for p in pts]
    result["crop"] = [cx1, cy1, cx2, cy2]
    result["selected_indices"] = [int(ordered[i][2]["index"]) for i in range(4)]
    return result


def _cap_intersection_over_area(cand, kept):
    cx1, cy1, cx2, cy2 = cand
    kx1, ky1, kx2, ky2 = kept
    ix1 = max(cx1, kx1)
    iy1 = max(cy1, ky1)
    ix2 = min(cx2, kx2)
    iy2 = min(cy2, ky2)
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    cand_area = (cx2 - cx1) * (cy2 - cy1)
    if cand_area <= 0:
        return 0.0
    return inter / cand_area


def filter_large_boxes(detections, policy):
    kept_list = []
    kept_indices = []
    suppressed = []

    if not isinstance(detections, list):
        raise PipelineValidationError("detections must be a list")

    _validate_overlap_policy(policy)

    parsed = []
    for i, det in enumerate(detections):
        box, conf = _validate_detection(det)
        parsed.append({"index": i, "box": box, "conf": conf})

    def area(p):
        b = p["box"]
        return (b[2] - b[0]) * (b[3] - b[1])

    for p in parsed:
        p["area"] = area(p)

    order = sorted(parsed, key=lambda p: (-p["area"], -p["conf"], p["index"]))

    kept_pool = []
    suppressed_pool = []

    min_ioa = float(policy["min_ioa"])
    max_dx = float(policy["max_center_dx"])
    max_dy = float(policy["max_bottom_dy"])

    for cand in order:
        cb = cand["box"]
        cw = cb[2] - cb[0]
        ch = cb[3] - cb[1]
        ccx = (cb[0] + cb[2]) / 2.0
        cby = cb[3]
        chosen_kept = None
        chosen_metrics = None
        for kp in kept_pool:
            kb = kp["box"]
            kw = kb[2] - kb[0]
            kh = kb[3] - kb[1]
            ioa = _cap_intersection_over_area(cb, kb)
            if ioa < min_ioa:
                continue
            kcx = (kb[0] + kb[2]) / 2.0
            kb_by = kb[3]
            denom_w = min(cw, kw)
            denom_h = min(ch, kh)
            if denom_w <= 0 or denom_h <= 0:
                continue
            dx = abs(ccx - kcx) / denom_w
            dy = abs(cby - kb_by) / denom_h
            if dx > max_dx:
                continue
            if dy > max_dy:
                continue
            chosen_kept = kp
            chosen_metrics = (ioa, dx, dy)
            break
        if chosen_kept is None:
            kept_pool.append(cand)
        else:
            ioa, dx, dy = chosen_metrics
            suppressed_pool.append({
                "removed_index": int(cand["index"]),
                "kept_index": int(chosen_kept["index"]),
                "ioa_small": float(ioa),
                "center_dx_norm": float(dx),
                "bottom_dy_norm": float(dy),
            })

    kept_set = set(kp["index"] for kp in kept_pool)
    for i in range(len(detections)):
        if i in kept_set:
            kept_indices.append(i)
            kept_list.append(detections[i])

    local_pos = {i: pos for pos, i in enumerate(kept_indices)}
    for sup in suppressed_pool:
        if sup["kept_index"] in local_pos:
            pass

    return {
        "kept": kept_list,
        "kept_indices": kept_indices,
        "suppressed": suppressed_pool,
    }


def _validate_config(config):
    if not isinstance(config, dict):
        raise PipelineValidationError("config must be a dict")
    if set(config.keys()) != CONFIG_KEYS:
        missing = CONFIG_KEYS - set(config.keys())
        extra = set(config.keys()) - CONFIG_KEYS
        raise PipelineValidationError("config keys mismatch missing=%s extra=%s" % (sorted(missing), sorted(extra)))

    images = config["images"]
    if not isinstance(images, list) or not images:
        raise PipelineValidationError("images must be a nonempty list")
    if len(images) > 6:
        raise PipelineValidationError("at most 6 images allowed")

    seen_ids = set()
    for img in images:
        if not isinstance(img, dict):
            raise PipelineValidationError("each image must be a dict")
        if set(img.keys()) != {"id", "path", "sha256", "origin"}:
            raise PipelineValidationError("image keys must be exactly id,path,sha256,origin")
        _validate_safe_id(img["id"])
        if img["id"] in seen_ids:
            raise PipelineValidationError("duplicate image id: %r" % (img["id"],))
        seen_ids.add(img["id"])
        if not isinstance(img["path"], str) or not os.path.isabs(img["path"]):
            raise PipelineValidationError("image path must be absolute")
        if not isinstance(img["sha256"], str) or len(img["sha256"]) != 64:
            raise PipelineValidationError("image sha256 invalid")
        if not isinstance(img["origin"], str) or not img["origin"]:
            raise PipelineValidationError("image origin invalid")

    for key in ("corner_model", "piece_model"):
        spec = config[key]
        if not isinstance(spec, dict) or set(spec.keys()) != {"path", "sha256"}:
            raise PipelineValidationError("%s must have path and sha256" % key)
        if not isinstance(spec["path"], str) or not os.path.isabs(spec["path"]):
            raise PipelineValidationError("%s.path must be absolute" % key)
        if not isinstance(spec["sha256"], str) or len(spec["sha256"]) != 64:
            raise PipelineValidationError("%s.sha256 invalid" % key)

    for key in ("corner_inference", "piece_inference"):
        inf = config[key]
        if not isinstance(inf, dict) or set(inf.keys()) != INFERENCE_KEYS:
            raise PipelineValidationError("%s must have imgsz,conf,iou" % key)
        if not isinstance(inf["imgsz"], int) or isinstance(inf["imgsz"], bool) or inf["imgsz"] <= 0:
            raise PipelineValidationError("%s.imgsz must be positive int" % key)
        for k in ("conf", "iou"):
            if not _is_finite_number(inf[k]) or not (0.0 <= float(inf[k]) <= 1.0):
                raise PipelineValidationError("%s.%s must be finite in [0,1]" % (key, k))

    _validate_crop_policy(config["crop_policy"])
    _validate_overlap_policy(config["overlap_policy"])

    out = config["output_dir"]
    if not isinstance(out, str) or not out or not os.path.isabs(out):
        raise PipelineValidationError("output_dir must be absolute nonempty string")

    mp = config["max_predictions"]
    if not isinstance(mp, int) or isinstance(mp, bool):
        raise PipelineValidationError("max_predictions must be int")
    if mp < 2 * len(images):
        raise PipelineValidationError("max_predictions must be >= 2*len(images)")
    if mp > 12:
        raise PipelineValidationError("max_predictions must be <= 12")

    ms = config["max_seconds"]
    if not _is_finite_number(ms):
        raise PipelineValidationError("max_seconds must be finite number")
    if float(ms) <= 0.0 or float(ms) > 120.0:
        raise PipelineValidationError("max_seconds must be in (0,120]")

    for img in images:
        if not os.path.isfile(img["path"]):
            raise PipelineValidationError("image file missing: %s" % img["path"])
    for key in ("corner_model", "piece_model"):
        if not os.path.isfile(config[key]["path"]):
            raise PipelineValidationError("%s file missing: %s" % (key, config[key]["path"]))

    for img in images:
        actual = _sha256_file(img["path"])
        if actual != img["sha256"]:
            raise PipelineValidationError("image sha256 mismatch: %s" % img["id"])
    for key in ("corner_model", "piece_model"):
        actual = _sha256_file(config[key]["path"])
        if actual != config[key]["sha256"]:
            raise PipelineValidationError("%s sha256 mismatch" % key)

    if os.path.exists(out):
        raise PipelineValidationError("output_dir already exists: %s" % out)

    return True


def _monotonic_remaining(start, budget):
    return float(budget) - (time.monotonic() - start)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.flush()


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _job_write(job_path, payload):
    _write_json(job_path, payload)


def _translate_detection(det, crop):
    x1o, y1o = crop[0], crop[1]
    b = det["box"]
    new_det = dict(det)
    new_det["box"] = [float(b[0]) + x1o, float(b[1]) + y1o, float(b[2]) + x1o, float(b[3]) + y1o]
    return new_det


def _build_suite_images_from(specs, origin):
    out = []
    for s in specs:
        out.append({
            "id": s["id"],
            "path": s["path"],
            "sha256": s["sha256"],
            "origin": origin,
        })
    return out


def run_pipeline(config, model_factory=None):
    _validate_config(config)

    root = config["output_dir"]
    parent = os.path.dirname(root)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    try:
        os.mkdir(root)
    except FileExistsError:
        raise PipelineValidationError("output_dir already exists: %s" % root)

    job_path = os.path.join(root, "job.json")
    start = time.monotonic()
    total_budget = float(config["max_seconds"])
    job = {
        "job_id": uuid.uuid4().hex,
        "status": "running",
        "config": config,
        "started_at": time.time(),
        "elapsed_seconds": 0.0,
        "error": None,
        "counts": {},
    }
    _job_write(job_path, job)

    images = config["images"]
    N = len(images)

    corner_suite_dir = os.path.join(root, "corner_predictions")
    piece_suite_dir = os.path.join(root, "piece_predictions")
    crops_dir = os.path.join(root, "crops")

    localization_path = os.path.join(root, "localization.json")
    predictions_jsonl_path = os.path.join(root, "predictions.jsonl")
    summary_path = os.path.join(root, "summary.json")

    localization_records = []
    corner_calls = 0
    piece_calls = 0
    localized_count = 0
    failed_count = 0
    completed_calls = {"corner": 0, "piece": 0}

    def tracked_factory(phase):
        def factory(path):
            base = (model_factory or real_camera_inference_v2._default_model_factory)(path)
            def predict(frame, **kwargs):
                nonlocal corner_calls, piece_calls
                if _monotonic_remaining(start, total_budget) <= 0:
                    raise TimeoutError("aggregate time budget exhausted before predict")
                if corner_calls + piece_calls >= config["max_predictions"]:
                    raise RuntimeError("aggregate model call budget exhausted")
                if phase == "corner":
                    corner_calls += 1
                else:
                    piece_calls += 1
                result = base.predict(frame, **kwargs)
                completed_calls[phase] += 1
                return result
            return SimpleNamespace(names=base.names, predict=predict)
        return factory

    try:
        os.makedirs(crops_dir, exist_ok=False)
        remaining = _monotonic_remaining(start, total_budget)
        if remaining <= 0:
            raise TimeoutError("no time remaining before corner phase")

        corner_suite_config = {
            "output_dir": corner_suite_dir,
            "models": {
                "original": {"path": config["corner_model"]["path"], "sha256": config["corner_model"]["sha256"]},
                "finetuned": {"path": config["corner_model"]["path"], "sha256": config["corner_model"]["sha256"]},
            },
            "images": _build_suite_images_from(images, "original"),
            "variants": [
                {
                    "id": "corner_variant",
                    "model": "original",
                    "imgsz": int(config["corner_inference"]["imgsz"]),
                    "conf": float(config["corner_inference"]["conf"]),
                    "iou": float(config["corner_inference"]["iou"]),
                }
            ],
            "max_predictions": N,
            "max_seconds": float(remaining),
        }

        corner_summary = real_camera_inference_v2.run_suite(
            corner_suite_config,
            model_factory=tracked_factory("corner"),
        )

        records = corner_summary.get("records")
        if not isinstance(records, list) or len(records) != N:
            raise RuntimeError("corner run_suite returned unexpected record count")

        record_by_id = {}
        for r in records:
            if not isinstance(r, dict):
                raise RuntimeError("corner record is not a dict")
            rid = r.get("image_id")
            if rid in record_by_id:
                raise RuntimeError("duplicate corner record image_id: %r" % (rid,))
            record_by_id[rid] = r

        crop_specs = []
        localization_by_id = {}

        for img in images:
            remaining = _monotonic_remaining(start, total_budget)
            if remaining <= 0:
                raise TimeoutError("no time remaining during crop processing")

            rec = record_by_id.get(img["id"])
            if rec is None:
                raise RuntimeError("corner run_suite missing record for image %r" % (img["id"],))

            width = rec.get("width")
            height = rec.get("height")
            detections = rec.get("detections")
            if not isinstance(detections, list):
                raise RuntimeError("corner record detections invalid for %r" % (img["id"],))

            loc = select_board_crop(detections, width, height, config["crop_policy"])
            loc_record = {
                "image_id": img["id"],
                "ok": bool(loc["ok"]),
                "reason": loc["reason"],
                "corners": loc["corners"],
                "crop": loc["crop"],
                "selected_indices": loc["selected_indices"],
                "crop_path": None,
                "crop_sha256": None,
                "width": width,
                "height": height,
            }

            if not loc["ok"]:
                failed_count += 1
                localization_records.append(loc_record)
                localization_by_id[img["id"]] = loc_record
                _write_json(localization_path, {"records": localization_records})
                continue

            remaining = _monotonic_remaining(start, total_budget)
            if remaining <= 0:
                raise TimeoutError("no time remaining before crop write")

            frame = cv2.imread(img["path"])
            if frame is None:
                raise RuntimeError("cv2.imread failed for %s" % img["path"])
            fh, fw = int(frame.shape[0]), int(frame.shape[1])
            if fw != int(width) or fh != int(height):
                raise RuntimeError("image dimension mismatch for %r suite=(%d,%d) file=(%d,%d)"
                                   % (img["id"], int(width), int(height), fw, fh))

            cx1, cy1, cx2, cy2 = loc["crop"]
            crop_img = frame[cy1:cy2, cx1:cx2]
            if crop_img.size == 0:
                raise RuntimeError("empty crop for image %r" % (img["id"],))

            crop_path = os.path.join(crops_dir, "%s.png" % img["id"])
            ok_write = cv2.imwrite(crop_path, crop_img)
            if not ok_write:
                raise RuntimeError("failed to write crop png for %r" % (img["id"],))

            crop_sha = _sha256_file(crop_path)
            loc_record["crop_path"] = crop_path
            loc_record["crop_sha256"] = crop_sha
            localization_records.append(loc_record)
            localization_by_id[img["id"]] = loc_record
            _write_json(localization_path, {"records": localization_records})

            localized_count += 1
            crop_specs.append({
                "id": img["id"],
                "path": crop_path,
                "sha256": crop_sha,
                "origin": "corner-model crop",
            })

        piece_raw_by_id = {}

        if crop_specs:
            remaining = _monotonic_remaining(start, total_budget)
            if remaining <= 0:
                raise TimeoutError("no time remaining before piece phase")

            piece_suite_config = {
                "output_dir": piece_suite_dir,
                "models": {
                    "original": {"path": config["piece_model"]["path"], "sha256": config["piece_model"]["sha256"]},
                    "finetuned": {"path": config["piece_model"]["path"], "sha256": config["piece_model"]["sha256"]},
                },
                "images": crop_specs,
                "variants": [
                    {
                        "id": "piece_variant",
                        "model": "finetuned",
                        "imgsz": int(config["piece_inference"]["imgsz"]),
                        "conf": float(config["piece_inference"]["conf"]),
                        "iou": float(config["piece_inference"]["iou"]),
                    }
                ],
                "max_predictions": len(crop_specs),
                "max_seconds": float(remaining),
            }

            piece_summary = real_camera_inference_v2.run_suite(
                piece_suite_config,
                model_factory=tracked_factory("piece"),
            )

            precords = piece_summary.get("records")
            if not isinstance(precords, list) or len(precords) != len(crop_specs):
                raise RuntimeError("piece run_suite returned unexpected record count")

            for r in precords:
                piece_raw_by_id[r["image_id"]] = r

        final_records = []
        for img in images:
            loc_record = localization_by_id.get(img["id"])
            if loc_record is None:
                raise RuntimeError("missing localization record for %r" % (img["id"],))

            if not loc_record["ok"]:
                final_records.append({
                    "image_id": img["id"],
                    "input_path": img["path"],
                    "input_sha256": img["sha256"],
                    "width": loc_record["width"],
                    "height": loc_record["height"],
                    "localization": loc_record,
                    "raw_detections": [],
                    "detections": [],
                    "suppression": [],
                    "kept_indices": [],
                })
                continue

            prec = piece_raw_by_id.get(img["id"])
            if prec is None:
                raise RuntimeError("missing piece record for %r" % (img["id"],))

            crop = loc_record["crop"]
            crop_dets = prec.get("detections")
            if not isinstance(crop_dets, list):
                raise RuntimeError("piece detections invalid for %r" % (img["id"],))

            raw_orig = [_translate_detection(d, crop) for d in crop_dets]

            filtered = filter_large_boxes(raw_orig, config["overlap_policy"])

            remaining = _monotonic_remaining(start, total_budget)
            if remaining <= 0:
                raise TimeoutError("no time remaining during output assembly")

            final_records.append({
                "image_id": img["id"],
                "input_path": img["path"],
                "input_sha256": img["sha256"],
                "width": loc_record["width"],
                "height": loc_record["height"],
                "localization": loc_record,
                "raw_detections": raw_orig,
                "detections": filtered["kept"],
                "suppression": filtered["suppressed"],
                "kept_indices": filtered["kept_indices"],
            })

        with open(predictions_jsonl_path, "w", encoding="utf-8") as f:
            for rec in final_records:
                f.write(json.dumps(rec, allow_nan=False))
                f.write("\n")
            f.flush()

        elapsed = time.monotonic() - start
        model_calls = corner_calls + piece_calls

        summary = {
            "schema_version": 1,
            "config": config,
            "completed_calls": dict(completed_calls),
            "records": final_records,
            "corner_calls": int(corner_calls),
            "piece_calls": int(piece_calls),
            "model_calls": int(model_calls),
            "localized": int(localized_count),
            "failed_localization": int(failed_count),
            "elapsed_seconds": float(elapsed),
            "status": "completed",
        }
        _write_json(summary_path, summary)

        job["status"] = "completed"
        job["completed_calls"] = dict(completed_calls)
        job["elapsed_seconds"] = float(elapsed)
        job["counts"] = {
            "corner_calls": int(corner_calls),
            "piece_calls": int(piece_calls),
            "model_calls": int(model_calls),
            "localized": int(localized_count),
            "failed_localization": int(failed_count),
            "predictions": len(final_records),
        }
        _job_write(job_path, job)

        return summary

    except Exception as e:
        elapsed = time.monotonic() - start
        job["status"] = "error"
        job["completed_calls"] = dict(completed_calls)
        job["elapsed_seconds"] = float(elapsed)
        job["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        job["counts"] = {
            "corner_calls": int(corner_calls),
            "piece_calls": int(piece_calls),
            "model_calls": int(corner_calls + piece_calls),
            "localized": int(localized_count),
            "failed_localization": int(failed_count),
        }
        try:
            _job_write(job_path, job)
        except Exception:
            pass
        raise


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 2 or argv[0] != "--config":
        sys.stderr.write("usage: %s --config CONFIG_JSON\n" % sys.argv[0])
        return 2
    config_path = argv[1]
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        run_pipeline(config, model_factory=None)
        return 0
    except Exception as e:
        sys.stderr.write("pipeline error: %s: %s\n" % (type(e).__name__, e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
