import copy
import math
from collections import deque

from selective_fusion_v2 import _geom_reasons, _ioa, SQUARES

CLASS_NAMES = [
    "white_pawn", "white_rook", "white_knight", "white_bishop",
    "white_queen", "white_king",
    "black_pawn", "black_rook", "black_knight", "black_bishop",
    "black_queen", "black_king",
]
NUM_CLASSES = 12

DEFAULT_POLICY = {
    "primary_weight": 0.7,
    "aux_weight": 0.3,
    "max_pair_skew_ms": 350.0,
    "margin": 0.08,
    "min_top": 0.4,
    "edge_px": 3.0,
    "cell_margin_fraction": 0.08,
    "mask_ioa": 0.4,
    "occluder_dy_fraction": 0.15,
}


def _policy(policy):
    p = dict(DEFAULT_POLICY)
    if policy:
        p.update(policy)
    return p


def _norm_scores(scores, where):
    if scores is None:
        raise ValueError("missing class_scores for %s" % where)
    try:
        vals = list(scores)
    except TypeError:
        raise ValueError("class_scores not a sequence for %s" % where)
    if len(vals) not in (NUM_CLASSES, NUM_CLASSES + 1):
        raise ValueError("class_scores wrong length for %s: %d" % (where, len(vals)))
    out = []
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError("non-numeric score for %s" % where)
        if not math.isfinite(f) or not 0 <= f <= 1:
            raise ValueError("invalid score for %s" % where)
        out.append(f)
    out = out[:NUM_CLASSES]
    total = sum(out)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("class_scores sum invalid for %s" % where)
    return [x / total for x in out]


def _argmax(vec):
    best = 0
    for i in range(1, len(vec)):
        if vec[i] > vec[best]:
            best = i
    return best, vec[best]


def _margin(vec, top_idx):
    second = 0.0
    for i, v in enumerate(vec):
        if i == top_idx:
            continue
        if v > second:
            second = v
    return vec[top_idx] - second


def _head_box(box):
    h = box[3] - box[1]
    return [box[0], box[1], box[2], box[1] + 0.4 * h]


def _blur_quality(det):
    bq = det.get("blur_quality")
    if bq is None:
        return 1.0
    try:
        f = float(bq)
    except (TypeError, ValueError):
        raise ValueError("blur_quality not numeric")
    if not math.isfinite(f):
        raise ValueError("blur_quality non-finite")
    if f < 0.0 or f > 1.0:
        raise ValueError("blur_quality outside 0..1: %r" % f)
    return f


def _geometry_quality(view, det, square, policy, is_primary):
    reasons = list(_geom_reasons(view, det, square, policy))
    reasons = [r for r in reasons if r != "bbox_mask_ioa" and r != "occluded_by_large"]
    hard = {
        "grid_invalid", "unassigned", "edge_clipped", "missing_cell",
        "footpoint_outside_cell", "footpoint_in_mask",
    }
    q = 1.0
    for r in reasons:
        if r in hard:
            return 0.0, reasons

    if is_primary:
        return 1.0, reasons + ["primary_priority_preserved"]

    box = det["box"]
    if box[3] <= box[1]:
        return 0.0, reasons + ["degenerate_box"]

    head = _head_box(box)
    for rect in view["occlusion_rects"]:
        ov = _ioa(box, rect)
        if ov > 0:
            q *= max(0.0, 1.0 - ov / 0.4)
            reasons.append("mask_overlap")
        if _ioa(head, rect) >= 0.2:
            return 0.0, reasons + ["head_in_mask"]
    if q <= 0:
        return 0.0, reasons + ["bbox_mask_ioa"]

    # Large occluder via tail-following boxes.
    cand_h = box[3] - box[1]
    for other in view["large"]:
        if other is det or other.get("square") == square:
            continue
        ob = other["box"]
        if ob[3] <= box[3] + policy["occluder_dy_fraction"] * cand_h:
            continue
        if ob is box or (ob[0] == box[0] and ob[1] == box[1] and ob[2] == box[2] and ob[3] == box[3]):
            continue
        head = _head_box(box)
        ov = _ioa(head, ob)
        if ov > 0:
            reasons.append("head_overlap")
            q *= max(0.0, 1.0 - ov / 0.5)

    if any(r == "occluded_by_large" for r in _geom_reasons(view, det, square, policy)):
        reasons.append("occluded_by_large")
        q *= 0.25

    q *= _blur_quality(det)
    return max(0.0, min(1.0, q)), reasons


def _assigned_candidate(view, square):
    hits = [d for d in view["large"] if d.get("square") == square and d.get("assignment_status") == "assigned"]
    if len(hits) != 1:
        return None, len(hits)
    return hits[0], 1


def _view_usable(view):
    return bool(view and view.get("grid_valid"))


def _square_row(square, c2, c1, policy):
    reasons_c2 = []
    reasons_c1 = []
    aux_rejections = []
    uncertain = []

    c2_view = c2
    c1_view = c1
    skew = c2.get("_pair_skew_ms", 0.0)
    max_skew = policy["max_pair_skew_ms"]

    q2 = 0.0
    q1 = 0.0
    probs2 = None
    probs1 = None
    det2 = None
    det1 = None

    if not _view_usable(c2_view):
        reasons_c2.append("grid_invalid")
        uncertain.append("primary_grid_invalid")

    if det2 is None and _view_usable(c2_view):
        det2, n2 = _assigned_candidate(c2_view, square)
        if det2 is None:
            if n2 == 0:
                reasons_c2.append("no_candidate")
                uncertain.append("primary_unavailable")
            else:
                reasons_c2.append("multiple_candidates")
                uncertain.append("primary_unavailable")

    if det2 is not None:
        try:
            probs2 = _norm_scores(det2.get("class_scores"), "primary:%s" % square)
        except ValueError:
            raise
        q2, geo_r = _geometry_quality(c2_view, det2, square, policy, True)
        reasons_c2.extend(geo_r)
        if det2.get("confidence", 0.0) < 0.25:
            q2 = 0.0
            reasons_c2.append("low_confidence")
        if q2 <= 0:
            uncertain.append("primary_unavailable")
            probs2_keep = probs2
        else:
            probs2_keep = probs2
    else:
        probs2_keep = None
        probs2 = None

    if not _view_usable(c1_view):
        reasons_c1.append("grid_invalid")
    elif skew > max_skew:
        reasons_c1.append("pair_skew")
        aux_rejections.append("pair_skew")
    else:
        det1, n1 = _assigned_candidate(c1_view, square)
        if det1 is None:
            if n1 == 0:
                reasons_c1.append("no_candidate")
            else:
                reasons_c1.append("multiple_candidates")
        else:
            probs1 = _norm_scores(det1.get("class_scores"), "aux:%s" % square)
            q1, geo_r1 = _geometry_quality(c1_view, det1, square, policy, False)
            reasons_c1.extend(geo_r1)
            if det1.get("confidence", 0.0) < (0.90 if q2 <= 0 else 0.25):
                q1 = 0.0
                reasons_c1.append("low_confidence")
                aux_rejections.append("confidence_below_aux_threshold")
            if q2 <= 0 and q1 < 0.5:
                aux_rejections.append("quality_below_aux_threshold")
                q1 = 0.0

    if not _view_usable(c2_view):
        q1 = 0.0
        reasons_c1.append("primary_grid_invalid")
    use2 = probs2 is not None and q2 > 0
    use1 = probs1 is not None and q1 > 0

    primary_confidence = None
    aux_confidence = None
    if probs2 is not None:
        _, primary_confidence = _argmax(probs2)
    if probs1 is not None:
        _, aux_confidence = _argmax(probs1)

    if not use2 and not use1:
        # Choose source hint.
        if not _view_usable(c2_view):
            src = "unknown"
        elif probs1 is not None and not use1:
            src = "unknown"
        else:
            src = "unknown"
        return {
            "square": square,
            "primary_class": _argmax(probs2)[0] if probs2 is not None else None,
            "primary_confidence": primary_confidence,
            "aux_confidence": aux_confidence,
            "candidate_class": None,
            "final_class": None,
            "confidence": None,
            "probs": None,
            "primary_probs": probs2_keep,
            "aux_probs": probs1,
            "weights": {"c2": 0.0, "c1": 0.0},
            "qualities": {"c2": q2, "c1": q1},
            "quality_reasons": {"c2": reasons_c2, "c1": reasons_c1},
            "source": src,
            "uncertain_reasons": uncertain,
            "aux_rejections": aux_rejections,
        }

    # Bounded weights: lower q2 cannot increase w1.
    w2 = policy["primary_weight"] * (q2 if use2 else 0.0)
    w1 = policy["aux_weight"] * (q1 if use1 else 0.0)
    if w2 <= 0 and not use1:
        return _unknown_row(square, probs2_keep, probs1, q2, q1, reasons_c2, reasons_c1, primary_confidence, aux_confidence, uncertain, aux_rejections)
    if w2 <= 0 and use1:
        source = "auxiliary_only"
        w1 = 1.0
        fused = list(probs1)
        w2_eff, w1_eff = 0.0, w1
    elif w2 > 0 and not use1:
        source = "primary_only"
        w2 = 1.0
        fused = list(probs2)
        w2_eff, w1_eff = w2, 0.0
    else:
        source = "weighted"
        tot = w2 + w1
        if tot <= 0:
            return _unknown_row(square, probs2_keep, probs1, q2, q1, reasons_c2, reasons_c1, primary_confidence, aux_confidence, uncertain, aux_rejections)
        w2_eff = w2 / tot
        w1_eff = w1 / tot
        fused = [w2_eff * a + w1_eff * b for a, b in zip(probs2, probs1)]

    if w2 > 0 and use1:
        w2_eff = w2 / (w2 + w1)
        w1_eff = w1 / (w2 + w1)
        fused = [w2_eff * a + w1_eff * b for a, b in zip(probs2, probs1)]

    # Fixed local boost, validated offline; use raw detector confidence.
    # Hard eligibility is already encoded by use1/use2. Effective weights
    # replace the soft quality multipliers only in this narrow region.
    if (use1 and use2
            and det1["confidence"] >= 0.90
            and det2["confidence"] <= 0.60
            and _argmax(probs1)[0] != _argmax(probs2)[0]):
        w1_eff, w2_eff = 0.60, 0.40
        fused = [w2_eff * a + w1_eff * b for a, b in zip(probs2, probs1)]

    cand_idx, cand_conf = _argmax(fused)
    marg = _margin(fused, cand_idx)
    final_idx = None
    final_class = None
    if cand_conf >= policy["min_top"] and marg >= policy["margin"]:
        final_idx = cand_idx
        final_class = cand_idx
    else:
        if cand_conf < policy["min_top"]:
            uncertain.append("below_min_top")
        if marg < policy["margin"]:
            uncertain.append("below_margin")

    return {
        "square": square,
        "primary_class": _argmax(probs2)[0] if probs2 is not None else None,
        "primary_confidence": primary_confidence,
        "aux_confidence": aux_confidence,
        "candidate_class": cand_idx,
        "final_class": final_class,
        "confidence": cand_conf,
        "probs": fused,
        "primary_probs": probs2_keep,
        "aux_probs": probs1,
        "weights": {"c2": w2_eff if use2 else 0.0, "c1": w1_eff if use1 else 0.0},
        "qualities": {"c2": q2, "c1": q1},
        "quality_reasons": {"c2": reasons_c2, "c1": reasons_c1},
        "source": source,
        "uncertain_reasons": uncertain,
        "aux_rejections": aux_rejections,
    }


def _unknown_row(square, probs2, probs1, q2, q1, r2, r1, pc, ac, unc, aux_rej):
    return {
        "square": square,
        "primary_class": _argmax(probs2)[0] if probs2 is not None else None,
        "primary_confidence": pc,
        "aux_confidence": ac,
        "candidate_class": None,
        "final_class": None,
        "confidence": None,
        "probs": None,
        "primary_probs": probs2,
        "aux_probs": probs1,
        "weights": {"c2": 0.0, "c1": 0.0},
        "qualities": {"c2": q2, "c1": q1},
        "quality_reasons": {"c2": r2, "c1": r1},
        "source": "unknown",
        "uncertain_reasons": unc,
        "aux_rejections": aux_rej,
    }


def fuse(pair, policy=None):
    if not isinstance(pair, dict):
        raise ValueError("pair must be dict")
    p = _policy(policy)
    c1 = pair.get("c1")
    c2 = pair.get("c2")
    if not isinstance(c2, dict):
        raise ValueError("pair missing c2")
    if c1 is None:
        c1 = {"width": 0, "height": 0, "roi": [0, 0, 0, 0], "grid_valid": False,
              "occlusion_rects": [], "cells": {}, "large": [], "raw": []}
    skew = pair.get("pair_skew_ms", 0.0)
    try:
        skew = float(skew)
    except (TypeError, ValueError):
        raise ValueError("pair_skew_ms not numeric")
    if not math.isfinite(skew):
        raise ValueError("pair_skew_ms non-finite")
    c2w = dict(c2)
    c2w["_pair_skew_ms"] = skew
    c1w = dict(c1)
    c1w["_pair_skew_ms"] = skew

    squares_order = SQUARES

    squares = {}
    for sq in squares_order:
        squares[sq] = _square_row(sq, c2w, c1w, p)

    return {
        "group": pair.get("group"),
        "squares64": squares,
    }


class _SquareHistory:
    __slots__ = ("samples",)

    def __init__(self):
        self.samples = deque(maxlen=128)


class TemporalFusion:
    def __init__(self, window=0.8, min_span=0.35, min_frames=3, margin=0.08, min_top=0.4):
        self.window = float(window)
        self.min_span = float(min_span)
        self.min_frames = int(min_frames)
        self.margin = float(margin)
        self.min_top = float(min_top)
        self._grid_token = None
        self._last_seq = None
        self._last_ts = None
        self._history = {}
        self._cache = None

    def reset(self):
        self._grid_token = None
        self._last_seq = None
        self._last_ts = None
        self._history = {}
        self._cache = None

    @staticmethod
    def _canon_seq(sequence):
        return repr(sequence)

    def update(self, result, timestamp, sequence, grid_token=None, moving=()):
        try:
            ts = float(timestamp)
        except (TypeError, ValueError):
            raise ValueError("timestamp not numeric")
        if not math.isfinite(ts):
            raise ValueError("timestamp non-finite")

        canon = self._canon_seq(sequence)

        if grid_token is not None and self._grid_token is not None and grid_token != self._grid_token:
            self.reset()
        if self._grid_token is None:
            self._grid_token = grid_token
        elif grid_token is not None:
            self._grid_token = grid_token

        moving_set = set(moving or ())

        if self._last_seq is not None and (canon == self._last_seq or ts <= self._last_ts):
            if self._cache is not None:
                return copy.deepcopy(self._cache)
            return self._render(result, moving_set)

        self._last_seq = canon
        self._last_ts = ts if self._last_ts is None else max(self._last_ts, ts)

        squares_in = result.get("squares64", {})
        for sq, row in squares_in.items():
            hist = self._history.get(sq)
            probs = row.get("probs")
            cand = row.get("candidate_class")
            if sq in moving_set:
                if hist is not None:
                    hist.samples.clear()
                continue
            if probs is None or cand is None:
                if probs is None and hist is not None:
                    hist.samples.clear()
                if hist is None:
                    hist = _SquareHistory()
                    self._history[sq] = hist
                continue
            if hist is None:
                hist = _SquareHistory()
                self._history[sq] = hist
            hist.samples.append((ts, list(probs), cand))

        out = self._render(result, moving_set, ts)
        self._cache = copy.deepcopy(out)
        return out

    def _prune(self, ts):
        lo = ts - self.window
        for sq, hist in list(self._history.items()):
            while hist.samples and hist.samples[0][0] < lo:
                hist.samples.popleft()

    def _render(self, result, moving_set, ts=None):
        out = copy.deepcopy(result)
        if ts is None and self._last_ts is not None:
            ts = self._last_ts
        if ts is not None:
            self._prune(ts)

        squares_in = out.get("squares64", {})
        for sq, row in squares_in.items():
            row["instant_class"] = row.get("final_class")
            if sq in moving_set:
                row["final_class"] = None
                row["temporal_state"] = "unknown"
                row["temporal_frames"] = 0
                row["temporal_span"] = 0.0
                row["temporal_probs"] = None
                if row.get("confidence") is not None:
                    row["confidence"] = row.get("confidence")
                continue

            hist = self._history.get(sq)
            if hist is None or not hist.samples:
                row["temporal_state"] = "unknown"
                row["temporal_frames"] = 0
                row["temporal_span"] = 0.0
                row["temporal_probs"] = None
                if row.get("probs") is None:
                    row["final_class"] = None
                continue

            samples = list(hist.samples)
            n = len(samples)
            if n == 0:
                row["temporal_state"] = "unknown"
                row["temporal_frames"] = 0
                row["temporal_span"] = 0.0
                row["temporal_probs"] = None
                continue
            span = samples[-1][0] - samples[0][0]
            mean = [0.0] * NUM_CLASSES
            for _, probs, _cand in samples:
                for i in range(NUM_CLASSES):
                    mean[i] += probs[i]
            mean = [x / n for x in mean]
            idx, top = _argmax(mean)
            marg = _margin(mean, idx)
            latest_cand = samples[-1][2]
            mean_winner = idx

            enough = n >= self.min_frames and span >= self.min_span
            confidence_ok = top >= self.min_top and marg >= self.margin
            agree = row.get("candidate_class") == mean_winner

            row["temporal_frames"] = n
            row["temporal_span"] = span
            row["temporal_probs"] = mean
            row["confidence"] = top

            if enough and confidence_ok and agree:
                row["temporal_state"] = "stable"
                row["final_class"] = mean_winner
            elif not enough:
                row["temporal_state"] = "unknown"
                row["final_class"] = None
            else:
                row["temporal_state"] = "pending"
                row["final_class"] = None

        return out
