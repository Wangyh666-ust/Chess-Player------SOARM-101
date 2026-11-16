import argparse
import json
import math
import os
import time

import cv2
import numpy as np

SQUARES = [f + str(r) for f in "abcdefgh" for r in range(1, 9)]

DEFAULT_POLICY = {
    "primary_conf": 0.8,
    "aux_conf": 0.85,
    "fill_conf": 0.9,
    "override_margin": 0.1,
    "rival_iou": 0.5,
    "rival_gap": 0.15,
    "edge_px": 3.0,
    "cell_margin_fraction": 0.08,
    "mask_ioa": 0.1,
    "max_pair_skew_ms": 350.0,
    "occluder_dy_fraction": 0.15,
}


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _iou(a, b):
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


def _ioa(box, rect):
    ix0 = max(box[0], rect[0]); iy0 = max(box[1], rect[1])
    ix1 = min(box[2], rect[2]); iy1 = min(box[3], rect[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    aa = (box[2] - box[0]) * (box[3] - box[1])
    return inter / aa if aa > 0 else 0.0


def _footpoint(box, height):
    return [0.5 * (box[0] + box[2]), box[3] - 0.12 * height]


def _cell_geometry_ok(foot, quad, frac):
    pts = np.array(quad, dtype=np.float32)
    area = abs(cv2.contourArea(pts))
    if area <= 0:
        return False
    d = cv2.pointPolygonTest(pts, (float(foot[0]), float(foot[1])), True)
    return d >= frac * math.sqrt(area)


def _touches_edge(box, roi, width, height, edge_px):
    x0, y0, x1, y1 = roi
    cands = [box[0] - x0, box[1] - y0, x1 - box[2], y1 - box[3]]
    return any(c <= edge_px for c in cands)


def _geom_reasons(view, det, square, policy):
    reasons = []
    box = det["box"]
    h = view["height"]
    foot = _footpoint(box, box[3] - box[1])
    if not view["grid_valid"]:
        reasons.append("grid_invalid")
    if det["assignment_status"] != "assigned":
        reasons.append("unassigned")
    if _touches_edge(box, view["roi"], view["width"], h, policy["edge_px"]):
        reasons.append("edge_clipped")
    cell = view["cells"].get(square) if square else None
    if cell is None:
        reasons.append("missing_cell")
    else:
        if not _cell_geometry_ok(foot, cell, policy["cell_margin_fraction"]):
            reasons.append("footpoint_outside_cell")
    for rect in view["occlusion_rects"]:
        if rect[0] <= foot[0] <= rect[2] and rect[1] <= foot[1] <= rect[3]:
            reasons.append("footpoint_in_mask")
            break
        if _ioa(box, rect) > policy["mask_ioa"]:
            reasons.append("bbox_mask_ioa")
            break
    for i, other in enumerate(view["large"]):
        if other is det:
            continue
        if other["box"][0] <= foot[0] <= other["box"][2] and other["box"][1] <= foot[1] <= other["box"][3]:
            cand_h = box[3] - box[1]
            if other["box"][3] > box[3] + policy["occluder_dy_fraction"] * cand_h:
                reasons.append("occluded_by_large")
                break
    return reasons


def _has_rival(view, det, policy):
    for other in view["raw"]:
        if other is det:
            continue
        if other["class_id"] == det["class_id"]:
            continue
        if _iou(det["box"], other["box"]) >= policy["rival_iou"] and \
                other["confidence"] >= det["confidence"] - policy["rival_gap"]:
            return True
    return False


def _count_square(view, square):
    return [d for d in view["large"] if d.get("square") == square]


def _primary_baseline(c2, square):
    if square is None:
        return None
    dets = _count_square(c2, square)
    if len(dets) != 1:
        return None
    det = dets[0]
    if det["assignment_status"] != "assigned":
        return None
    return det


def _reason_uncertainty(det, view, square, policy):
    return _geom_reasons(view, det, square, policy) + \
        (["raw_rival"] if _has_rival(view, det, policy) else [])


def _aux_candidate(c1, square, policy):
    dets = _count_square(c1, square)
    if len(dets) != 1:
        return None
    return dets[0]


def _max_c2_conf(c2, square):
    return max((d["confidence"] for d in _count_square(c2, square)), default=None)


def _eval_square(square, pair, policy):
    c2 = pair["c2"]; c1 = pair["c1"]
    reasons = []
    aux_rej = []
    prim = _primary_baseline(c2, square)
    prim_class = prim["class_id"] if prim else None
    prim_conf = prim["confidence"] if prim else None
    aux_det = None
    aux_conf = None

    uncertain = False
    if prim is None:
        uncertain = True
        reasons.append("no_primary_baseline")
    else:
        if prim_conf < policy["primary_conf"]:
            uncertain = True
            reasons.append("low_primary_conf")
        gr = _reason_uncertainty(prim, c2, square, policy)
        if gr:
            uncertain = True
            reasons.extend(gr)
        if _has_rival(c2, prim, policy):
            uncertain = True
            if "raw_rival" not in reasons:
                reasons.append("raw_rival")

    if not uncertain:
        return {
            "square": square, "primary_class": prim_class, "final_class": prim_class,
            "source": "primary_trusted", "primary_confidence": prim_conf,
            "aux_confidence": None, "uncertain_reasons": [],
            "aux_rejections": [],
        }

    # Try auxiliary
    aux_ok = True
    if not c2["grid_valid"]:
        aux_ok = False
        aux_rej.append("primary_grid_invalid")
    if not c1["grid_valid"]:
        aux_ok = False; aux_rej.append("aux_grid_invalid")
    if pair["pair_skew_ms"] > policy["max_pair_skew_ms"]:
        aux_ok = False; aux_rej.append("pair_skew_exceeded")
    if square is None:
        aux_ok = False; aux_rej.append("no_square")
    else:
        aux_det = _aux_candidate(c1, square, policy)
        if aux_det is None or len(_count_square(c1, square)) != 1:
            aux_ok = False; aux_rej.append("aux_not_unique_assigned")
        else:
            aux_conf = aux_det["confidence"]
            if aux_det["assignment_status"] != "assigned":
                aux_ok = False; aux_rej.append("aux_unassigned")
            if aux_conf < policy["aux_conf"]:
                aux_ok = False; aux_rej.append("aux_low_conf")
            agr = _reason_uncertainty(aux_det, c1, square, policy)
            if agr:
                aux_ok = False; aux_rej.extend("aux_" + r for r in agr)
            if _has_rival(c1, aux_det, policy):
                aux_ok = False
                if "aux_raw_rival" not in aux_rej:
                    aux_rej.append("aux_raw_rival")

    if not aux_ok:
        if prim is not None:
            src = "primary_fallback"
        else:
            src = "unknown"
        return {
            "square": square, "primary_class": prim_class, "final_class": prim_class,
            "source": src, "primary_confidence": prim_conf,
            "aux_confidence": aux_conf, "uncertain_reasons": reasons,
            "aux_rejections": aux_rej,
        }

    aux_class = aux_det["class_id"]
    if prim_class is not None and aux_class == prim_class:
        return {
            "square": square, "primary_class": prim_class, "final_class": prim_class,
            "source": "corroborated", "primary_confidence": prim_conf,
            "aux_confidence": aux_conf, "uncertain_reasons": reasons,
            "aux_rejections": [],
        }
    if prim_class is not None and aux_class != prim_class:
        if aux_conf >= prim_conf + policy["override_margin"]:
            return {
                "square": square, "primary_class": prim_class, "final_class": aux_class,
                "source": "auxiliary_override", "primary_confidence": prim_conf,
                "aux_confidence": aux_conf, "uncertain_reasons": reasons,
                "aux_rejections": [],
            }
        reasons.append("insufficient_margin")
        return {
            "square": square, "primary_class": prim_class, "final_class": prim_class,
            "source": "primary_fallback", "primary_confidence": prim_conf,
            "aux_confidence": aux_conf, "uncertain_reasons": reasons,
            "aux_rejections": [],
        }
    # No primary baseline
    if aux_conf < policy["fill_conf"]:
        reasons.append("fill_conf_insufficient")
        return {
            "square": square, "primary_class": None, "final_class": None,
            "source": "unknown", "primary_confidence": None,
            "aux_confidence": aux_conf, "uncertain_reasons": reasons,
            "aux_rejections": [],
        }
    maxc = _max_c2_conf(c2, square)
    if maxc is not None and aux_conf < maxc + policy["override_margin"]:
        reasons.append("fill_margin_insufficient")
        return {
            "square": square, "primary_class": None, "final_class": None,
            "source": "unknown", "primary_confidence": None,
            "aux_confidence": aux_conf, "uncertain_reasons": reasons,
            "aux_rejections": [],
        }
    return {
        "square": square, "primary_class": None, "final_class": aux_class,
        "source": "auxiliary_fill", "primary_confidence": None,
        "aux_confidence": aux_conf, "uncertain_reasons": reasons,
        "aux_rejections": [],
    }


def fuse(pair, policy):
    _validate_policy(policy)
    _validate_pair(pair)
    return {
        "group": pair["group"],
        "squares": [_eval_square(sq, pair, policy) for sq in SQUARES],
    }


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_square(v):
    return isinstance(v, str) and len(v) == 2 and v[0] in "abcdefgh" and v[1] in "12345678"


def _is_point(p):
    return isinstance(p, (list, tuple)) and len(p) == 2 and all(_finite(x) for x in p)


def _valid_square(v):
    return v is None or _is_square(v)


def _validate_det(det, name):
    if not isinstance(det, dict):
        raise ValueError(f"{name} det not object")
    if set(det) != {"box", "class_id", "confidence", "square", "assignment_status"}:
        raise ValueError(f"{name} det fields invalid")
    box = det["box"]
    if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(_finite(x) for x in box):
        raise ValueError(f"{name} det box invalid")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{name} det box area invalid")
    cid = det["class_id"]
    if not _is_int(cid) or not (0 <= cid <= 11):
        raise ValueError(f"{name} det class_id invalid")
    conf = det["confidence"]
    if not _finite(conf) or not (0 <= conf <= 1):
        raise ValueError(f"{name} det confidence invalid")
    sq = det["square"]
    if not _valid_square(sq):
        raise ValueError(f"{name} det square invalid")
    st = det["assignment_status"]
    if st not in ("assigned", "conflict", "outside", "boundary_ambiguous", "occluded", "grid_invalid", "invalid_box", "unassigned"):
        raise ValueError(f"{name} det status invalid")
    if st == "assigned" and sq is None:
        raise ValueError(f"{name} assigned without square")


def _validate_view(v, name):
    if not isinstance(v, dict):
        raise ValueError(f"{name} not object")
    expected = {"width", "height", "roi", "grid_valid", "occlusion_rects", "cells", "large", "raw"}
    if set(v) != expected:
        raise ValueError(f"{name} fields invalid")
    w = v["width"]
    h = v["height"]
    if not _is_int(w) or not _is_int(h) or w <= 0 or h <= 0:
        raise ValueError(f"{name} width/height invalid")
    roi = v["roi"]
    if not isinstance(roi, (list, tuple)) or len(roi) != 4 or not all(_finite(x) for x in roi):
        raise ValueError(f"{name}.roi invalid")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f"{name}.roi bounds invalid")
    if not isinstance(v["grid_valid"], bool):
        raise ValueError(f"{name}.grid_valid invalid")
    occ = v["occlusion_rects"]
    if not isinstance(occ, list):
        raise ValueError(f"{name}.occlusion_rects invalid")
    for i, r in enumerate(occ):
        if not isinstance(r, (list, tuple)) or len(r) != 4 or not all(_finite(x) for x in r):
            raise ValueError(f"{name} occlusion rect {i} invalid")
        if r[2] <= r[0] or r[3] <= r[1]:
            raise ValueError(f"{name} occlusion rect {i} area invalid")
    cells = v["cells"]
    if not isinstance(cells, dict):
        raise ValueError(f"{name}.cells invalid")
    for k, val in cells.items():
        if not _is_square(k):
            raise ValueError(f"{name} cell key invalid")
        if not isinstance(val, (list, tuple)) or len(val) != 4 or not all(_is_point(p) for p in val):
            raise ValueError(f"{name} cell value invalid")
    for lst in ("large", "raw"):
        if not isinstance(v[lst], list):
            raise ValueError(f"{name}.{lst} invalid")
        for det in v[lst]:
            _validate_det(det, name)


def _validate_policy(policy):
    if not isinstance(policy, dict):
        raise ValueError("policy not object")
    if set(policy) != set(DEFAULT_POLICY):
        raise ValueError("policy fields invalid")
    for k, v in policy.items():
        if not _finite(v) or v < 0:
            raise ValueError(f"policy {k} invalid")
        if k in ("primary_conf", "aux_conf", "fill_conf", "rival_iou", "rival_gap", "mask_ioa", "cell_margin_fraction", "override_margin") and v > 1:
            raise ValueError(f"policy {k} out of range")


def _validate_pair(pair):
    if not isinstance(pair, dict):
        raise ValueError("pair not object")
    if set(pair) != {"group", "pair_skew_ms", "c1", "c2"}:
        raise ValueError("pair fields invalid")
    if not _is_int(pair["group"]):
        raise ValueError("pair group invalid")
    sk = pair["pair_skew_ms"]
    if not _finite(sk) or sk < 0:
        raise ValueError("pair skew invalid")
    _validate_view(pair["c1"], "c1")
    _validate_view(pair["c2"], "c2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if os.path.exists(args.output):
        raise SystemExit(f"output already exists: {args.output}")

    t0 = time.time()
    with open(args.input, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or set(data) != {"pairs", "policy"}:
        raise ValueError("input fields invalid")
    if not isinstance(data["pairs"], list) or not 1 <= len(data["pairs"]) <= 7:
        raise ValueError("input requires 1..7 pairs")
    policy = data["policy"]
    pairs = data["pairs"]
    _validate_policy(policy)
    if len(pairs) > 7:
        raise SystemExit("too many pairs")
    for p in pairs:
        for k in ("group", "pair_skew_ms", "c1", "c2"):
            if k not in p:
                raise ValueError(f"pair missing {k}")
        _validate_view(p["c1"], "c1")
        _validate_view(p["c2"], "c2")

    results = [fuse(p, policy) for p in pairs]
    elapsed = time.time() - t0

    os.makedirs(args.output, exist_ok=False)
    out = {"policy": policy, "pairs": results, "elapsed_seconds": elapsed}
    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
