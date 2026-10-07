"""Shared helpers for the reproducible Dubrovnik chess-board benchmark.

This module deliberately contains no model loading, no network access
and no shell execution.  It is meant to be imported by independently
generated model runner scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None


ROOT: Path = Path(__file__).resolve().parent
ALT: Path = ROOT / "alternatives"
OUT: Path = ALT / "results"

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Constants used across metrics
# ---------------------------------------------------------------------------

LEGAL_SYMBOLS = set("_PNBRQKpnbrqk")

_PIECE_LETTER = {
    "bishop": "B",
    "king": "K",
    "knight": "N",
    "pawn": "P",
    "queen": "Q",
    "rook": "R",
}


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------


def truth_board(fen: str) -> List[str]:
    """Return 64 symbols row-major a8..h8, a7..h7, ..., a1..h1.

    ``_`` for empty squares, uppercase for white, lowercase for black.
    """
    if chess is None:
        raise RuntimeError("python-chess is required for truth_board()")
    board = chess.Board(fen)
    symbols: List[str] = []
    for rank in range(7, -1, -1):
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            if piece is None:
                symbols.append("_")
            else:
                symbols.append(piece.symbol())
    return symbols


def _resolve_image_path(image_field: str) -> str:
    """Resolve possibly relative Windows-style paths against ROOT."""
    normalized = str(image_field).replace("\\", os.sep).replace("/", os.sep)
    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return str(candidate.resolve())


def load_scenes(limit: int = 0) -> List[Dict[str, Any]]:
    """Load Dubrovnik scenes from the manifest.

    Fields retained: id, position, camera_elevation_degrees, image, fen,
    corners, image_path (absolute str) and truth (list of 64 symbols).
    """
    manifest_path = ROOT / "scenes" / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        entries = data.get("scenes", data.get("images", []))
    else:
        entries = data

    scenes: List[Dict[str, Any]] = []
    for entry in entries:
        if entry.get("set") != "dubrovnik":
            continue
        scene = {
            "id": entry["id"],
            "position": entry.get("position", ""),
            "camera_elevation_degrees": entry.get("camera_elevation_degrees"),
            "image": entry.get("image"),
            "fen": entry["fen"],
            "corners": entry.get("corners"),
        }
        scene["image_path"] = _resolve_image_path(scene["image"])
        scene["truth"] = truth_board(scene["fen"])
        scenes.append(scene)

    if limit:
        scenes = scenes[:limit]

    return scenes


# ---------------------------------------------------------------------------
# Symbol / class-name mapping
# ---------------------------------------------------------------------------


def map_symbol(class_name: str) -> str:
    """Map a class name like ``white_pawn`` or ``black-knight`` to a symbol.

    Unknown class names raise ``ValueError`` instead of guessing.
    """
    if class_name is None:
        raise ValueError("class_name must not be None")

    raw = str(class_name).strip()
    if raw == "":
        raise ValueError("class_name must not be empty")

    # Direct single-symbol characters
    if len(raw) == 1 and raw in LEGAL_SYMBOLS:
        return raw

    lowered = raw.lower().replace("-", "_").replace(" ", "_")
    if lowered in ("empty", "blank", "none", "background", "_"):
        return "_"

    parts = [p for p in lowered.split("_") if p]
    color: Optional[str] = None
    piece: Optional[str] = None

    for part in parts:
        if part in ("white", "w"):
            color = "white"
        elif part in ("black", "b"):
            color = "black"
        elif part in _PIECE_LETTER:
            piece = _PIECE_LETTER[part]
        elif part in ("empty", "blank", "none", "background"):
            return "_"

    if piece is not None and color is not None:
        return piece if color == "white" else piece.lower()

    raise ValueError(f"Unknown class name: {class_name!r}")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _as_float_array(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] != 4:
        raise ValueError("quad must be a 4x2 array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("quad contains non-finite values")
    return arr


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def order_quad(points: Any) -> np.ndarray:
    """Order four finite, distinct points TL, TR, BR, BL using geometry only.

    Uses the sum/difference heuristic, suitable for the non-rotated
    benchmark imagery.  Rejects duplicates, non-convex or degenerate
    quads (area < 1).  Returns ``float32`` with shape (4, 2).
    """
    arr = _as_float_array(points)

    # Reject duplicate points
    for i in range(4):
        for j in range(i + 1, 4):
            if np.array_equal(arr[i], arr[j]):
                raise ValueError("quad contains duplicate points")

    # Compute ordering
    s = arr.sum(axis=1)
    d = np.diff(arr, axis=1).reshape(-1)

    tl = arr[np.argmin(s)]
    br = arr[np.argmax(s)]
    tr = arr[np.argmin(d)]
    bl = arr[np.argmax(d)]

    ordered = np.stack([tl, tr, br, bl], axis=0)

    # Ensure convex and non-degenerate
    area = _polygon_area(ordered)
    if area < 1.0:
        raise ValueError("quad area too small")

    if not _is_convex(ordered):
        raise ValueError("quad is not convex")

    return ordered.astype(np.float32)


def _is_convex(points: np.ndarray) -> bool:
    n = points.shape[0]
    sign = 0
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        c = points[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) < 1e-9:
            return False
        current = 1 if cross > 0 else -1
        if sign == 0:
            sign = current
        elif current != sign:
            return False
    return True


def homography(corners: Any, size: int = 800) -> np.ndarray:
    """Compute homography from a TL,TR,BR,BL quad to a square of ``size``."""
    src = order_quad(corners).astype(np.float32)
    dst = np.array(
        [[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32
    )
    H = cv2.getPerspectiveTransform(src, dst)
    return H


# ---------------------------------------------------------------------------
# Board metrics
# ---------------------------------------------------------------------------


def _quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    try:
        inter, _ = cv2.intersectConvexConvex(
            a.astype(np.float32), b.astype(np.float32)
        )
    except Exception:
        return 0.0
    area_a = _polygon_area(a)
    area_b = _polygon_area(b)
    union = area_a + area_b - float(inter)
    if union <= 0:
        return 0.0
    return float(inter) / union


def board_metrics(corners: Any, gt: Any) -> Dict[str, Any]:
    """Compare predicted board corners against ground-truth corners.

    Both quads are ordered with :func:`order_quad` first.  ``usable`` is a
    predeclared threshold: IoU >= 0.90 AND mean corner error <= 20 px AND
    max corner error <= 35 px.
    """
    result: Dict[str, Any] = {
        "found": False,
        "polygon_iou": 0.0,
        "mean_corner_error_px": None,
        "max_corner_error_px": None,
        "usable": False,
    }

    if corners is None or gt is None:
        return result

    try:
        pred = order_quad(corners)
        ref = order_quad(gt)
    except Exception:
        return result

    result["found"] = True

    iou = _quad_iou(pred, ref)
    result["polygon_iou"] = float(iou)

    diffs = np.linalg.norm(pred.astype(np.float64) - ref.astype(np.float64), axis=1)
    mean_err = float(np.mean(diffs))
    max_err = float(np.max(diffs))
    result["mean_corner_error_px"] = mean_err
    result["max_corner_error_px"] = max_err

    result["usable"] = bool(
        iou >= 0.90 and mean_err <= 20.0 and max_err <= 35.0
    )
    return result


# ---------------------------------------------------------------------------
# Piece metrics
# ---------------------------------------------------------------------------


def _validate_board(seq: Any, name: str) -> List[str]:
    if seq is None:
        raise ValueError(f"{name} must not be None")
    values = list(seq)
    if len(values) != 64:
        raise ValueError(f"{name} must contain exactly 64 entries")
    for sym in values:
        if sym not in LEGAL_SYMBOLS:
            raise ValueError(f"{name} contains illegal symbol: {sym!r}")
    return values


def piece_metrics(pred: Any, gt: Any) -> Dict[str, Any]:
    """Compute raw counts and rates for a 64-cell board prediction."""
    pred_seq = _validate_board(pred, "pred")
    gt_seq = _validate_board(gt, "gt")

    cells = 64
    occupied = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    color_correct = 0
    type_correct = 0
    joint_correct = 0
    state_correct = 0
    exact_board = False

    for p, g in zip(pred_seq, gt_seq):
        g_occ = g != "_"
        p_occ = p != "_"

        if g_occ:
            occupied += 1

        if g_occ and p_occ:
            tp += 1
            if p.isupper() == g.isupper():
                color_correct += 1
            if p.lower() == g.lower():
                type_correct += 1
            if p == g:
                joint_correct += 1
        elif g_occ and not p_occ:
            fn += 1
        elif (not g_occ) and p_occ:
            fp += 1
        else:
            tn += 1

        if p == g:
            state_correct += 1

    exact_board = state_correct == cells

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    elif occupied > 0 and tp == 0:
        f1 = 0.0
    else:
        f1 = None

    def _ratio(num: int, den: int) -> Optional[float]:
        return float(num) / float(den) if den > 0 else None

    return {
        # Raw counts
        "cells": cells,
        "occupied": occupied,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "color_correct": color_correct,
        "type_correct": type_correct,
        "joint_correct": joint_correct,
        "state_correct": state_correct,
        "exact_board": bool(exact_board),
        # Rates
        "occupancy_accuracy": float((tp + tn)) / 64.0,
        "occupancy_precision": precision,
        "occupancy_recall": recall,
        "occupancy_f1": f1,
        "color_accuracy": _ratio(color_correct, occupied),
        "type_accuracy": _ratio(type_correct, occupied),
        "joint_accuracy": _ratio(joint_correct, occupied),
        "state_accuracy": float(state_correct) / 64.0,
        "color_given_detected": _ratio(color_correct, tp),
        "type_given_detected": _ratio(type_correct, tp),
        "joint_given_detected": _ratio(joint_correct, tp),
        "detection_coverage": _ratio(tp, occupied),
    }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def timing_stats(samples: Sequence[float]) -> Dict[str, Any]:
    """Compute summary statistics for a list of millisecond samples."""
    values = [float(v) for v in samples] if samples is not None else []
    if not values:
        return {
            "n": 0,
            "median_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_builtin(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return obj


_RATE_COUNT_KEYS = (
    ("occupancy_accuracy", ("tp", "tn", "cells")),
)


def _aggregate_mode(rows: List[Dict[str, Any]], mode_name: str) -> Optional[Dict[str, Any]]:
    present = [
        r for r in rows
        if isinstance(r, dict)
        and isinstance(r.get("modes"), dict)
        and isinstance(r["modes"].get(mode_name), dict)
    ]
    if not present:
        return None

    count_keys = [
        "cells",
        "occupied",
        "tp",
        "fp",
        "fn",
        "tn",
        "color_correct",
        "type_correct",
        "joint_correct",
        "state_correct",
    ]

    totals = {k: 0 for k in count_keys}
    exact_boards = 0
    total_ms = 0.0
    have_ms = False
    errors = 0

    for row in rows:
        modes = row.get("modes")
        if not isinstance(modes, dict):
            continue
        entry = modes.get(mode_name)
        if not isinstance(entry, dict):
            continue
        scores = entry.get("scores")
        if not isinstance(scores, dict):
            continue
        for k in count_keys:
            v = scores.get(k)
            if isinstance(v, (int, float)):
                totals[k] += int(v)
        if scores.get("exact_board"):
            exact_boards += 1
        ms = entry.get("total_ms")
        if isinstance(ms, (int, float)):
            total_ms += float(ms)
            have_ms = True
        if entry.get("error"):
            errors += 1

    cells = totals["cells"] or 64 * len(rows)
    occupied = totals["occupied"]
    tp = totals["tp"]
    fp = totals["fp"]
    fn = totals["fn"]
    tn = totals["tn"]

    def _ratio(num: int, den: int) -> Optional[float]:
        return float(num) / float(den) if den > 0 else None

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    elif occupied > 0 and tp == 0:
        f1 = 0.0
    else:
        f1 = None

    agg = {
        "scenes": len(present),
        "exact_boards": int(exact_boards),
        "cells": int(cells),
        "occupied": int(occupied),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "color_correct": int(totals["color_correct"]),
        "type_correct": int(totals["type_correct"]),
        "joint_correct": int(totals["joint_correct"]),
        "state_correct": int(totals["state_correct"]),
        "occupancy_accuracy": float(tp + tn) / float(cells) if cells else None,
        "occupancy_precision": precision,
        "occupancy_recall": recall,
        "occupancy_f1": f1,
        "color_accuracy": _ratio(totals["color_correct"], occupied),
        "type_accuracy": _ratio(totals["type_correct"], occupied),
        "joint_accuracy": _ratio(totals["joint_correct"], occupied),
        "state_accuracy": float(totals["state_correct"]) / float(cells) if cells else None,
        "color_given_detected": _ratio(totals["color_correct"], tp),
        "type_given_detected": _ratio(totals["type_correct"], tp),
        "joint_given_detected": _ratio(totals["joint_correct"], tp),
        "detection_coverage": _ratio(tp, occupied),
        "errors": int(errors),
    }
    if have_ms:
        agg["total_ms"] = float(total_ms)
    return agg


def _aggregate_board(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    found = 0
    usable = 0
    iou_sum = 0.0
    err_sum = 0.0
    err_n = 0

    for row in rows:
        bl = row.get("board_location")
        if not isinstance(bl, dict):
            continue
        metrics = bl.get("metrics")
        if not isinstance(metrics, dict):
            continue
        total += 1
        if metrics.get("found"):
            found += 1
        if metrics.get("usable"):
            usable += 1
        iou = metrics.get("polygon_iou")
        if isinstance(iou, (int, float)):
            iou_sum += float(iou)
        mean_err = metrics.get("mean_corner_error_px")
        if metrics.get("found") and isinstance(mean_err, (int, float)):
            err_sum += float(mean_err)
            err_n += 1

    return {
        "total": int(total),
        "found": int(found),
        "usable": int(usable),
        "mean_iou": (float(iou_sum) / float(total)) if total > 0 else None,
        "mean_corner_error_px": (float(err_sum) / float(err_n)) if err_n > 0 else None,
    }


def _aggregate_angles(
    rows: List[Dict[str, Any]], mode_names: Iterable[str]
) -> Dict[str, Any]:
    angles: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        angle = row.get("angle")
        key = str(angle) if angle is not None else "unknown"
        angles.setdefault(key, []).append(row)

    out: Dict[str, Any] = {}
    for key, subset in angles.items():
        entry: Dict[str, Any] = {"scenes": len(subset)}
        for mode in mode_names:
            agg = _aggregate_mode(subset, mode)
            if agg is not None:
                entry[mode] = agg
        out[key] = entry
    return out


def write_result(
    name: str,
    metadata: Dict[str, Any],
    rows: List[Dict[str, Any]],
    timing: Dict[str, Any],
) -> Path:
    """Write a benchmark result JSON file and return its path."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{name}.json"
    tmp = OUT / f"{name}.json.tmp"

    scene_rows: List[Dict[str, Any]] = []
    mode_names_seen: List[str] = []
    seen_set = set()

    for row in rows or []:
        clean = dict(row)
        modes = clean.get("modes")
        if isinstance(modes, dict):
            for mode_name in modes.keys():
                if mode_name not in seen_set:
                    seen_set.add(mode_name)
                    mode_names_seen.append(mode_name)
        scene_rows.append(clean)

    aggregate: Dict[str, Any] = {
        "scenes": len(scene_rows),
        "board": _aggregate_board(scene_rows),
        "modes": {},
        "angles": _aggregate_angles(scene_rows, mode_names_seen),
    }
    for mode in mode_names_seen:
        agg = _aggregate_mode(scene_rows, mode)
        if agg is not None:
            aggregate["modes"][mode] = agg

    document = {
        "schema_version": SCHEMA_VERSION,
        "model": name,
        "metadata": metadata if metadata is not None else {},
        "timing": timing if timing is not None else {},
        "scenes": scene_rows,
        "aggregate": aggregate,
    }

    document = _to_builtin(document)

    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, target)
    return target


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------


def _self_test() -> bool:
    # 1. Perfect match
    gt = ["_"] * 64
    gt[0] = "P"
    gt[1] = "p"
    m = piece_metrics(gt, gt)
    assert m["tp"] == 2 and m["fp"] == 0 and m["fn"] == 0
    assert m["occupied"] == 2
    assert m["color_correct"] == 2
    assert m["type_correct"] == 2
    assert m["joint_correct"] == 2
    assert m["state_correct"] == 64
    assert m["occupancy_accuracy"] == 1.0
    assert m["occupancy_precision"] == 1.0
    assert m["occupancy_recall"] == 1.0
    assert m["occupancy_f1"] == 1.0
    assert m["color_accuracy"] == 1.0
    assert m["type_accuracy"] == 1.0
    assert m["joint_accuracy"] == 1.0
    assert m["state_accuracy"] == 1.0
    assert m["exact_board"] is True

    # 2. All missing on true occupied
    gt2 = ["_"] * 64
    gt2[0] = "P"
    gt2[1] = "p"
    pred2 = ["_"] * 64
    m = piece_metrics(pred2, gt2)
    assert m["tp"] == 0 and m["fn"] == 2 and m["fp"] == 0
    assert m["occupied"] == 2
    assert m["state_correct"] == 62
    assert abs(m["occupancy_accuracy"] - (62.0 / 64.0)) < 1e-12
    assert m["color_correct"] == 0
    assert m["color_accuracy"] is not None
    assert m["color_accuracy"] == 0.0
    assert m["occupancy_f1"] == 0.0

    # 3. Swapped colours: occupancy perfect, type perfect, colour wrong
    gt3 = ["_"] * 64
    gt3[0] = "P"
    gt3[1] = "p"
    pred3 = ["_"] * 64
    pred3[0] = "p"
    pred3[1] = "P"
    m = piece_metrics(pred3, gt3)
    assert m["tp"] == 2 and m["fp"] == 0 and m["fn"] == 0
    assert m["occupancy_f1"] == 1.0
    assert m["type_correct"] == 2
    assert m["color_correct"] == 0
    assert m["color_accuracy"] == 0.0
    assert m["type_accuracy"] == 1.0
    assert m["joint_correct"] == 0
    assert m["state_correct"] == 62

    # 3b. Same-colour wrong-type: colour correct, type and joint wrong
    gt3b = ["_"] * 64
    gt3b[0] = "P"
    pred3b = ["_"] * 64
    pred3b[0] = "Q"
    m = piece_metrics(pred3b, gt3b)
    assert m["tp"] == 1 and m["fp"] == 0 and m["fn"] == 0
    assert m["occupied"] == 1
    assert m["color_correct"] == 1
    assert m["type_correct"] == 0
    assert m["joint_correct"] == 0
    assert m["color_accuracy"] == 1.0
    assert m["type_accuracy"] == 0.0
    assert m["joint_accuracy"] == 0.0

    # 4. Shifted piece creates FP and FN
    gt4 = ["_"] * 64
    gt4[0] = "P"
    pred4 = ["_"] * 64
    pred4[1] = "P"
    m = piece_metrics(pred4, gt4)
    assert m["fp"] == 1 and m["fn"] == 1 and m["tp"] == 0
    assert m["occupied"] == 1
    assert m["state_correct"] == 62

    # 5. Geometry: permutation order yields same IoU
    quad = np.array([[10, 10], [110, 12], [108, 112], [12, 108]], dtype=np.float64)
    perm = np.array([quad[2], quad[0], quad[3], quad[1]], dtype=np.float64)
    m1 = board_metrics(quad, quad)
    m2 = board_metrics(perm, quad)
    assert m1["found"] and m2["found"]
    assert abs(m1["polygon_iou"] - 1.0) < 1e-6
    assert abs(m2["polygon_iou"] - 1.0) < 1e-6
    assert abs(m1["mean_corner_error_px"]) < 1e-6

    # 6. Shifted quad lowers IoU and raises error
    shifted = quad + np.array([40.0, 30.0])
    m3 = board_metrics(shifted, quad)
    assert m3["found"]
    assert m3["polygon_iou"] < 0.9
    assert m3["mean_corner_error_px"] > 20.0
    assert m3["usable"] is False

    # 7. NaN rejected
    bad = quad.copy()
    bad[0, 0] = float("nan")
    try:
        order_quad(bad)
        raise AssertionError("NaN should be rejected")
    except ValueError:
        pass

    m4 = board_metrics(bad, quad)
    assert m4["found"] is False
    assert m4["usable"] is False

    # map_symbol coverage
    assert map_symbol("white-pawn") == "P"
    assert map_symbol("white_pawn") == "P"
    assert map_symbol("black_knight") == "n"
    assert map_symbol("white-bishop") == "B"
    assert map_symbol("empty") == "_"
    try:
        map_symbol("king")
        raise AssertionError("Bare piece name should raise")
    except ValueError:
        pass
    try:
        map_symbol("dragon")
        raise AssertionError("Unknown class should raise")
    except ValueError:
        pass

    # timing
    t = timing_stats([])
    assert t["n"] == 0 and t["median_ms"] is None
    t2 = timing_stats([1.0, 2.0, 3.0, 4.0])
    assert t2["n"] == 4
    assert abs(t2["median_ms"] - 2.5) < 1e-9
    assert t2["min_ms"] == 1.0 and t2["max_ms"] == 4.0

    return True


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="bench_common utilities")
    parser.add_argument("--self-test", action="store_true", help="run self test")
    args = parser.parse_args(argv)

    if args.self_test:
        ok = _self_test()
        if ok:
            print("passed")
            return 0
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
