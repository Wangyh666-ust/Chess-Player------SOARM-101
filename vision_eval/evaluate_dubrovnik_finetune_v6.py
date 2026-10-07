"""Evaluation driver for Dubrovnik fine-tune v2 (inference only, no training)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

import bench_common as B
import chessred_adapter as A

ROOT = Path(__file__).resolve().parent

SPLIT_NAMES = ("train", "val", "test")
EXPECTED_TEST_COUNT = 456
EXPECTED_OLD9_JOINT = 116
EXPECTED_OLD9_OCCUPIED = 222

DET_CONF = 0.25
DET_IOU = 0.7
IMGSZ = 640


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_inside_root(path: Path, label: str) -> Path:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must be inside ROOT ({ROOT}): {resolved}")
    return resolved


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(path: Path):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, allow_nan=False, indent=2)


def check_names_consistency(dataset_names, original_names, trained_names):
    if list(dataset_names) != list(original_names) or list(dataset_names) != list(trained_names):
        raise ValueError(
            "class name mismatch: dataset=%s original=%s trained=%s"
            % (list(dataset_names), list(original_names), list(trained_names))
        )


def build_placement_sets(plan):
    sets = {}
    for split in SPLIT_NAMES:
        s = set()
        for group in plan["groups"]:
            if group["split"] == split:
                s.add(group["placement"])
        sets[split] = s
    excluded = set(plan.get("excluded_placements", []))
    return sets, excluded


def assert_disjoint(sets, excluded):
    keys = list(SPLIT_NAMES)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            inter = sets[keys[i]] & sets[keys[j]]
            if inter:
                raise ValueError(f"placement overlap {keys[i]}/{keys[j]}: {sorted(inter)}")
    for split in SPLIT_NAMES:
        bad = sets[split] & excluded
        if bad:
            raise ValueError(f"excluded placement appears in {split}: {sorted(bad)}")


def bbox_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def run_model_predict(model, image_path: Path, device):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = model.predict(
        source=frame,
        device=device,
        imgsz=IMGSZ,
        conf=DET_CONF,
        iou=DET_IOU,
        half=False,
        verbose=False,
        batch=1,
    )[0]
    boxes = res.boxes.xyxy.detach().cpu().numpy()
    cls = res.boxes.cls.detach().cpu().numpy()
    conf = res.boxes.conf.detach().cpu().numpy()
    dets = []
    for i in range(len(boxes)):
        cid = int(cls[i])
        name = model.names[cid]
        dets.append(
            {
                "box": [float(v) for v in boxes[i]],
                "class_name": name,
                "symbol": B.map_symbol(name),
                "confidence": float(conf[i]),
                "class_id": cid,
            }
        )
    total_ms = (time.perf_counter() - t0) * 1000.0
    return dets, total_ms


def warmup(model, image_path, device):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"failed to read warmup image: {image_path}")
    for _ in range(3):
        model.predict(
            source=frame,
            device=device,
            imgsz=IMGSZ,
            conf=DET_CONF,
            iou=DET_IOU,
            half=False,
            verbose=False,
            batch=1,
        )
    torch.cuda.synchronize()


def map_known_corners(dets, corners):
    q = np.asarray(corners, dtype=np.float32)
    assert q.shape == (4, 2), f'corners shape must be (4,2), got {q.shape}'
    assert np.all(np.isfinite(q)), 'corners contain non-finite values'
    assert cv2.isContourConvex(q), 'corners are not convex'
    assert abs(cv2.contourArea(q)) > 1, 'corners contour area too small'
    dst = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(q, dst)
    assert np.all(np.isfinite(H)), 'homography contains non-finite values'
    return A.map_board(dets, corners, native_H=H)

def evaluate_scene(model, image_path, corners, truth, scene_id, fen, elevation, group_id, device):
    dets, ms = run_model_predict(model, image_path, device)
    map_start = time.perf_counter()
    pred_board = map_known_corners(dets, corners)
    torch.cuda.synchronize()
    ms += (time.perf_counter() - map_start) * 1000.0
    scores = B.piece_metrics(pred_board, truth)
    row = {
        "id": scene_id,
        "truth": truth,
        "corners": corners,
        "image_path": str(image_path),
        "fen": fen,
        "elevation": elevation,
        "group_id": group_id,
        "raw_detections": dets,
        "modes": {
            "known_corners": {
                "board": pred_board,
                "scores": scores,
                "total_ms": ms,
            }
        },
    }
    return row


def detection_diagnostic(dets, gt_instances):
    preds = [d for d in dets if d["symbol"] != "_"]
    gts = [g for g in gt_instances if g.get("labelled")]
    gt_count = len(gts)
    matched = 0
    class_correct = 0
    pairs = []
    for gi, g in enumerate(gts):
        gbox = g["amodal_box"]
        for pi, p in enumerate(preds):
            iou = bbox_iou(gbox, p["box"])
            if iou >= 0.5:
                pairs.append((iou, gi, pi))
    pairs.sort(key=lambda x: x[0], reverse=True)
    used_gt = set()
    used_pred = set()
    for iou, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matched += 1
        if preds[pi]["class_id"] == gts[gi]["class_id"]:
            class_correct += 1
    false_pred = len(preds) - matched
    object_recall = matched / gt_count if gt_count > 0 else None
    precision = matched / (matched + false_pred) if (matched + false_pred) > 0 else None
    conditional_class_accuracy = class_correct / matched if matched > 0 else None
    joint_object_recall = class_correct / gt_count if gt_count > 0 else None
    return {
        "gt_labelled": gt_count,
        "matched": matched,
        "false_pred": false_pred,
        "class_correct": class_correct,
        "object_recall": object_recall,
        "precision": precision,
        "conditional_class_accuracy": conditional_class_accuracy,
        "joint_object_recall": joint_object_recall,
    }


def aggregate_diagnostics(diags):
    gt_labelled = 0
    matched = 0
    false_pred = 0
    class_correct = 0
    for d in diags:
        gt_labelled += d.get("gt_labelled", 0)
        matched += d.get("matched", 0)
        false_pred += d.get("false_pred", 0)
        class_correct += d.get("class_correct", 0)
    object_recall = matched / gt_labelled if gt_labelled > 0 else None
    precision = matched / (matched + false_pred) if (matched + false_pred) > 0 else None
    conditional_class_accuracy = class_correct / matched if matched > 0 else None
    joint_object_recall = class_correct / gt_labelled if gt_labelled > 0 else None
    return {
        "gt_labelled": gt_labelled,
        "matched": matched,
        "false_pred": false_pred,
        "class_correct": class_correct,
        "object_recall": object_recall,
        "precision": precision,
        "conditional_class_accuracy": conditional_class_accuracy,
        "joint_object_recall": joint_object_recall,
    }


def build_gt_rows(model, records, dataset_root, device):
    rows = []
    for rec in records:
        image_path = dataset_root / rec["image"]
        instances = rec.get("instances", [])
        dets = []
        for inst in instances:
            amodal = inst.get("amodal_box")
            if amodal is None or len(amodal) != 4:
                continue
            symbol = inst.get("symbol")
            if symbol is None:
                continue
            dets.append(
                {
                    "box": [float(v) for v in amodal],
                    "class_name": symbol,
                    "symbol": symbol,
                    "confidence": 1.0,
                    "class_id": int(inst["class_id"]),
                }
            )
        pred_board = map_known_corners(dets, rec["corners"])
        truth = B.truth_board(rec["fen"])
        scores = B.piece_metrics(pred_board, truth)
        rows.append(
            {
                "id": rec["id"],
                "truth": truth,
                "corners": rec["corners"],
                "image_path": str(image_path),
                "fen": rec["fen"],
                "elevation": rec["elevation"],
                "group_id": rec.get("group_id"),
                "raw_detections": dets,
                "modes": {
                    "known_corners": {
                        "board": pred_board,
                        "scores": scores,
                        "total_ms": 0.0,
                    }
                },
            }
        )
    return rows


def evaluate_dataset(model, scenes, device, do_diag, diag_records=None, dataset_root=None, tag=""):
    rows = []
    timings = []
    if do_diag:
        diagnostics = []
    else:
        diagnostics = None
    n = len(scenes)
    for idx, scene in enumerate(scenes):
        row = evaluate_scene(
            model,
            Path(scene["image_path"]),
            scene["corners"],
            scene["truth"],
            scene["id"],
            scene.get("fen"),
            scene.get("elevation", scene.get("camera_elevation_degrees")),
            scene.get("group_id"),
            device,
        )
        rows.append(row)
        timings.append(row["modes"]["known_corners"]["total_ms"])
        if do_diag:
            diag = detection_diagnostic(
                row["raw_detections"], scene.get("instances", [])
            )
            diag = dict(diag)
            diag["id"] = scene["id"]
            diagnostics.append(diag)
        if idx % 50 == 0:
            print(f"[{tag}] {idx + 1}/{n}", flush=True)
    agg = B._aggregate_mode(rows, "known_corners")
    timing = B.timing_stats(timings)
    return rows, agg, timing, diagnostics


def load_dataset(dataset_root, limit, weights_names, original_names):
    plan_path = dataset_root / "plan.json"
    records_path = dataset_root / "records.jsonl"
    if not plan_path.exists() or not records_path.exists():
        raise FileNotFoundError(f"missing dataset files under {dataset_root}")
    plan = load_json(plan_path)
    statistics_path = dataset_root / "statistics.json"
    statistics = load_json(statistics_path)
    if not statistics.get("complete"):
        raise ValueError("statistics.json complete flag is not True")
    planned_total = statistics.get("planned_total")
    complete_total = statistics.get("complete_total")
    if complete_total != planned_total:
        raise ValueError(
            "statistics complete_total=%s does not match planned_total=%s"
            % (complete_total, planned_total)
        )
    records = load_jsonl(records_path)
    names_config_path = dataset_root / "dataset.yaml"
    with open(names_config_path, "r", encoding="utf-8") as _fh:
        _names_cfg = yaml.safe_load(_fh)
    _names_raw = _names_cfg["names"]
    if isinstance(_names_raw, dict):
        _names_raw = {int(_k): _v for _k, _v in _names_raw.items()}
    dataset_names = [_names_raw[_k] for _k in sorted(_names_raw)]
    check_names_consistency(dataset_names, original_names, weights_names)
    scenes_by_split = {s: [] for s in SPLIT_NAMES}
    for rec in records:
        split = rec["split"]
        if split not in scenes_by_split:
            raise ValueError(f"unknown split: {split}")
        truth = rec["truth"]
        fen_truth = B.truth_board(rec["fen"])
        if truth != fen_truth:
            raise ValueError(f"truth mismatch for record {rec['id']}")
        scenes_by_split[split].append(rec)
    test_records = scenes_by_split["test"]
    if limit <= 0 and len(test_records) != EXPECTED_TEST_COUNT:
        raise ValueError(
            f"expected {EXPECTED_TEST_COUNT} test records, got {len(test_records)}"
        )
    if limit > 0:
        test_records = test_records[:limit]
    if dataset_names is not None:
        check_names_consistency(dataset_names, original_names, weights_names)
    sets, excluded = build_placement_sets(plan)
    assert_disjoint(sets, excluded)
    return plan, test_records, dataset_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", default=str(ROOT / "dubrovnik_finetune" / "dataset_v1"))
    ap.add_argument("--output", default=str(ROOT / "dubrovnik_finetune" / "evaluation" / "trial_v1"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    weights_path = require_inside_root(Path(args.weights), "--weights")
    dataset_root = require_inside_root(Path(args.dataset), "--dataset")
    output_root = require_inside_root(Path(args.output), "--output")

    if weights_path.exists() is False:
        raise FileNotFoundError(f"weights not found: {weights_path}")
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"output exists (use --overwrite): {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = "cuda:0"
    torch.set_num_threads(4)
    cv2.setNumThreads(0)

    A.load()
    old_scenes = B.load_scenes()
    if len(old_scenes) != 9:
        raise ValueError(f"expected 9 old scenes, got {len(old_scenes)}")

    original_weights = A.B.ALT / "vendor" / "chessred_yolov8" / "models" / "piece_best.pt"
    if not original_weights.exists():
        raise FileNotFoundError(f"original weights not found: {original_weights}")

    original_model = YOLO(str(original_weights))
    trained_model = YOLO(str(weights_path))
    original_names = list(original_model.names.values()) if isinstance(original_model.names, dict) else list(original_model.names)
    trained_names = list(trained_model.names.values()) if isinstance(trained_model.names, dict) else list(trained_model.names)
    check_names_consistency(original_names, original_names, trained_names)

    plan, test_records, dataset_root = load_dataset(dataset_root, args.limit, trained_names, original_names)

    test_scenes = []
    for rec in test_records:
        image_path = dataset_root / rec["image"]
        test_scenes.append(
            {
                "id": rec["id"],
                "image_path": str(image_path),
                "truth": rec["truth"],
                "corners": rec["corners"],
                "fen": rec["fen"],
                "elevation": rec["elevation"],
                "group_id": rec.get("group_id"),
                "instances": rec.get("instances", []),
            }
        )

    results = {
        "original9": {},
        "independent_test": {},
    }
    params = {
        "device": device,
        "imgsz": IMGSZ,
        "conf": DET_CONF,
        "iou": DET_IOU,
        "half": False,
        "verbose": False,
        "batch": 1,
        "torch_threads": 4,
        "cv2_threads": 0,
    }

    for model_role, model in (("original", original_model), ("finetuned", trained_model)):
        warmup(model, Path(old_scenes[0]["image_path"]), device)
        rows, agg, timing, _ = evaluate_dataset(
            model, old_scenes, device, False, tag=f"{model_role}:original9"
        )
        results["original9"][model_role] = {
            "aggregate": agg,
            "timing": timing,
            "rows": rows,
        }
        write_json(output_root / f"partial_original9_{model_role}.json", results["original9"][model_role])
        if model_role == "original":
            warnings = []
            if agg.get("joint_correct") != EXPECTED_OLD9_JOINT:
                warnings.append(
                    f"old9 original joint_correct={agg.get('joint_correct')} expected {EXPECTED_OLD9_JOINT}"
                )
            if agg.get("occupied") != EXPECTED_OLD9_OCCUPIED:
                warnings.append(
                    f"old9 original occupied={agg.get('occupied')} expected {EXPECTED_OLD9_OCCUPIED}"
                )
            for w in warnings:
                print("WARNING:", w, flush=True)

    for model_role, model in (("original", original_model), ("finetuned", trained_model)):
        warmup(model, Path(test_scenes[0]["image_path"]) if test_scenes else Path(old_scenes[0]["image_path"]), device)
        rows, agg, timing, diags = evaluate_dataset(
            model,
            test_scenes,
            device,
            True,
            tag=f"{model_role}:test",
        )
        agg_with_diag = dict(agg)
        agg_with_diag["detection_diagnostics"] = diags
        agg_with_diag["detection_diagnostic"] = aggregate_diagnostics(diags)
        results["independent_test"][model_role] = {
            "aggregate": agg_with_diag,
            "timing": timing,
            "rows": rows,
        }
        write_json(output_root / f"partial_test_{model_role}.json", results["independent_test"][model_role])

    gt_rows = build_gt_rows(original_model, test_records, dataset_root, device)
    ideal_agg = B._aggregate_mode(gt_rows, "known_corners")
    results["independent_test"]["ideal_box_mapping"] = {
        "aggregate": ideal_agg,
        "rows": gt_rows,
        "label": "hypothetical_gt_box_mapping_upper_bound",
    }

    original_sha = sha256_file(original_weights)
    trained_sha = sha256_file(weights_path)
    counts = {
        "old9": len(old_scenes),
        "new_test_full": EXPECTED_TEST_COUNT,
        "new_test_used": len(test_scenes),
    }
    comparison = {
        "metadata": {
            "roles": {"GPT": "design,review,analysis", "DeepSeek": "implementation"},
            "date": "2026-10-07",
            "original_sha": original_sha,
            "trained_sha": trained_sha,
            "parameters": params,
            "synthetic_only": True,
            "known_corners_only": True,
            "corner_model_unchanged": True,
            "known_corner_order": "semantic_TL_TR_BR_BL_from_renderer_no_sum_difference_reordering",
            "counts": counts,
            "paths": {
                "weights": str(weights_path),
                "dataset": str(dataset_root),
                "output": str(output_root),
            },
        },
        "datasets": {
            "original9": {
                "original": {
                    "aggregate": results["original9"]["original"]["aggregate"],
                    "timing": results["original9"]["original"]["timing"],
                    "rows": results["original9"]["original"]["rows"],
                },
                "finetuned": {
                    "aggregate": results["original9"]["finetuned"]["aggregate"],
                    "timing": results["original9"]["finetuned"]["timing"],
                    "rows": results["original9"]["finetuned"]["rows"],
                },
            },
            "independent_test": {
                "original": {
                    "aggregate": results["independent_test"]["original"]["aggregate"],
                    "timing": results["independent_test"]["original"]["timing"],
                    "rows": results["independent_test"]["original"]["rows"],
                },
                "finetuned": {
                    "aggregate": results["independent_test"]["finetuned"]["aggregate"],
                    "timing": results["independent_test"]["finetuned"]["timing"],
                    "rows": results["independent_test"]["finetuned"]["rows"],
                },
                "ideal_box_mapping": {
                    "aggregate": ideal_agg,
                    "rows": gt_rows,
                    "label": "hypothetical_gt_box_mapping_upper_bound",
                },
            },
        },
    }
    write_json(output_root / "comparison.json", comparison)

    csv_rows = [
        ("original9", "original", results["original9"]["original"]["aggregate"], results["original9"]["original"]["timing"]),
        ("original9", "finetuned", results["original9"]["finetuned"]["aggregate"], results["original9"]["finetuned"]["timing"]),
        ("independent_test", "original", results["independent_test"]["original"]["aggregate"], results["independent_test"]["original"]["timing"]),
        ("independent_test", "finetuned", results["independent_test"]["finetuned"]["aggregate"], results["independent_test"]["finetuned"]["timing"]),
    ]
    csv_path = output_root / "summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "dataset",
                "model",
                "N",
                "occupied",
                "joint_correct",
                "exact_boards",
                "cells",
                "tp",
                "fp",
                "fn",
                "tn",
                "color_correct",
                "type_correct",
                "state_correct",
                "occupancy_accuracy",
                "occupancy_f1",
                "occupancy_precision",
                "occupancy_recall",
                "color_accuracy",
                "type_accuracy",
                "joint_accuracy",
                "state_accuracy",
                "detection_coverage",
                "color_given_detected",
                "type_given_detected",
                "joint_given_detected",
                "median_ms",
                "p95_ms",
            ]
        )
        for ds, model_role, agg, timing in csv_rows:
            writer.writerow(
                [
                    ds,
                    model_role,
                    timing.get("n"),
                    agg.get("occupied"),
                    agg.get("joint_correct"),
                    agg.get("exact_boards"),
                    agg.get("cells"),
                    agg.get("tp"),
                    agg.get("fp"),
                    agg.get("fn"),
                    agg.get("tn"),
                    agg.get("color_correct"),
                    agg.get("type_correct"),
                    agg.get("state_correct"),
                    agg.get("occupancy_accuracy"),
                    agg.get("occupancy_f1"),
                    agg.get("occupancy_precision"),
                    agg.get("occupancy_recall"),
                    agg.get("color_accuracy"),
                    agg.get("type_accuracy"),
                    agg.get("joint_accuracy"),
                    agg.get("state_accuracy"),
                    agg.get("detection_coverage"),
                    agg.get("color_given_detected"),
                    agg.get("type_given_detected"),
                    agg.get("joint_given_detected"),
                    timing.get("median_ms"),
                    timing.get("p95_ms"),
                ]
            )

    print("origin weights:", original_weights, flush=True)
    print("trained weights:", weights_path, flush=True)
    print("old9 original aggregate:", json.dumps(results["original9"]["original"]["aggregate"], sort_keys=True), flush=True)
    print("old9 finetuned aggregate:", json.dumps(results["original9"]["finetuned"]["aggregate"], sort_keys=True), flush=True)
    print("test original aggregate (sans diag):", json.dumps({k: v for k, v in results["independent_test"]["original"]["aggregate"].items() if k != "detection_diagnostic"}, sort_keys=True), flush=True)
    print("test finetuned aggregate (sans diag):", json.dumps({k: v for k, v in results["independent_test"]["finetuned"]["aggregate"].items() if k != "detection_diagnostic"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
