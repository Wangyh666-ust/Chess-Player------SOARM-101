"""Reference-grid bridge for two-camera UI.

GridBridge connects live_grid_locator_v2 (GridLocator + associate) to an existing
two-camera UI.  No serial, device opening, network, subprocess, motor control or
model loading.  Predict is supplied by the caller and is called on demand only.
"""

import copy
import json
import math
import time
import threading
from pathlib import Path

import cv2
import numpy as np

from live_grid_locator_v5 import GridLocator, associate, rotate_points


CAMERA_CANONICAL = {0: 0, 1: 270}
CANONICAL_NAME = {0: "auxiliary", 1: "primary"}
FRESH_SECONDS = 3.0
PAIR_SKEW_SECONDS = 0.35
MAX_AUDIT_CASES = 4
AUDIT_TIMEOUT_SECONDS = 60.0


def _rot_cw_quad(pts, rotation, w, h):
    return rotate_points(pts, w, h, rotation)


def _rot_cw_size(rotation, w, h):
    return (h, w) if int(rotation) % 180 else (w, h)


class GridBridge:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        mapping_path = Path(cfg["mapping_path"])
        if not mapping_path.is_absolute():
            mapping_path = (self.config_path.parent / mapping_path).resolve()
        else:
            mapping_path = mapping_path.resolve()

        refs = cfg["reference_images"]
        if len(refs) != 2:
            raise ValueError("reference_images must list exactly two paths")
        self.reference_images = {}
        for cam in (0, 1):
            p = Path(refs[cam])
            if not p.is_absolute():
                p = (self.config_path.parent / p).resolve()
            else:
                p = p.resolve()
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cannot read reference image for camera %d" % cam)
            self.reference_images[cam] = img

        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        self.calibration = {}
        for entry in mapping.get("cameras", []):
            cid = int(entry["id"])
            if cid in (1, 2):
                cam = cid - 1
            elif cid in (0, 1) and cid not in self.calibration:
                cam = cid
            else:
                continue
            self.calibration[cam] = entry
        if 0 not in self.calibration or 1 not in self.calibration:
            raise ValueError("mapping must contain cameras id 1 and 2")

        self.locators = {}
        for cam in (0, 1):
            self.locators[cam] = GridLocator(
                self.reference_images[cam], self.calibration[cam]
            )

        self._lock = threading.RLock()
        self._latest = {0: None, 1: None}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _grid_valid(grid):
        if not isinstance(grid, dict):
            return False
        if not bool(grid.get("valid")):
            return False
        nodes = grid.get("grid_nodes")
        cells = grid.get("cells")
        if nodes is None or cells is None:
            return False
        try:
            if np.asarray(nodes).reshape(-1, 2).shape[0] != 81:
                return False
        except Exception:
            return False
        if len(cells) != 64:
            return False
        return True

    @staticmethod
    def _finite(x):
        try:
            f = float(x)
        except Exception:
            return False
        return math.isfinite(f)

    @staticmethod
    def _canonical_to_display_grid(grid, canonical_rotation, rotation, w, h):
        if not grid.get("valid"):
            return copy.deepcopy(grid)
        dw, dh = _rot_cw_size((rotation - canonical_rotation) % 360, w, h)
        delta = (rotation - canonical_rotation) % 360

        def conv_point(p):
            return _rot_cw_quad([p], delta, w, h)[0]

        nodes = np.asarray(grid.get("grid_nodes"), dtype=np.float64).reshape(81, 2)
        new_nodes = _rot_cw_quad(nodes, delta, w, h)
        cell_list = []
        for c in grid.get("cells", []):
            nc = dict(c)
            if "quad" in c and c["quad"] is not None:
                nc["quad"] = _rot_cw_quad(c["quad"], delta, w, h).tolist()
            if "center" in c and c["center"] is not None:
                nc["center"] = conv_point(c["center"]).tolist()
            cell_list.append(nc)
        out = dict(grid)
        out["grid_nodes"] = new_nodes.tolist()
        out["cells"] = cell_list
        out["display_width"] = dw
        out["display_height"] = dh
        return out

    @staticmethod
    def _det_to_display(det, delta, w, h):
        d = dict(det)
        if det.get("box") is not None:
            q = _rot_cw_quad(
                [[det["box"][0], det["box"][1]],
                 [det["box"][2], det["box"][1]],
                 [det["box"][2], det["box"][3]],
                 [det["box"][0], det["box"][3]]],
                delta, w, h,
            )
            d["box"] = [float(q[:, 0].min()), float(q[:, 1].min()),
                        float(q[:, 0].max()), float(q[:, 1].max())]
        if det.get("footpoint") is not None:
            d["footpoint"] = _rot_cw_quad([det["footpoint"]], delta, w, h)[0].tolist()
        return d

    # -------------------------------------------------------------- rendering
    @staticmethod
    def _draw_header(img, cam, grid, grid_valid, crop_box):
        if grid_valid:
            n = len(grid.get("cells", []))
            status = "GRID OK n=%d" % n
        else:
            reason = str((grid or {}).get("reason", "invalid"))
            status = "GRID LOST %s" % reason
        label = "PRIMARY C2" if cam == 1 else "AUX C1"
        header = "%s | %s | adaptive grid tracking | base estimate" % (label, status)
        cv2.putText(img, header, (6, max(14, int(0.03 * img.shape[0]) + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return header

    def _render(self, display, cam, grid_disp, dets_disp, grid_valid):
        img = display.copy()
        if grid_valid:
            for c in grid_disp.get("cells", []):
                quad = c.get("quad")
                if quad is not None:
                    q = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(img, [q], True, (255, 255, 0), 1, cv2.LINE_AA)
                center = c.get("center")
                square = c.get("square")
                if center is not None and square:
                    cv2.putText(img, str(square), (int(center[0]) - 5, int(center[1]) + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 0), 1, cv2.LINE_AA)
            nodes = np.asarray(grid_disp.get("grid_nodes"), dtype=np.float64).reshape(9, 9, 2)
            for r in range(9):
                pts = nodes[r].astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], False, (255, 255, 0), 1, cv2.LINE_AA)
            for c in range(9):
                pts = nodes[:, c].astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], False, (255, 255, 0), 1, cv2.LINE_AA)

        for det in dets_disp:
            box = det.get("box")
            status = det.get("assignment_status")
            color = (0, 255, 0) if status == "assigned" else (0, 165, 255)
            if box is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                text = "%s %.2f" % (det.get("class_name", "?"), float(det.get("confidence", 0.0)))
                sq = det.get("square")
                if sq:
                    text += "@%s" % sq
                if status != "assigned":
                    text += "?"
                if status == "conflict":
                    text += "!"
                cv2.putText(img, text, (x1, max(12, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            fp = det.get("footpoint")
            if fp is not None:
                cv2.circle(img, (int(fp[0]), int(fp[1])), 3, color, -1, cv2.LINE_AA)

        self._draw_header(img, cam, grid_disp, grid_valid, None)
        return img

    # --------------------------------------------------------------- process
    def process(self, cam, display_bgr, rotation, predict, seq=None,
                captured_at=None, remember=True):
        cam = int(cam)
        if cam not in (0, 1):
            raise ValueError("cam must be 0 or 1")
        rotation = int(rotation) % 360
        if rotation not in (0, 90, 180, 270):
            raise ValueError("rotation must be 0/90/180/270")
        if display_bgr is None or display_bgr.dtype != np.uint8:
            raise ValueError("display_bgr must be uint8 BGR image")

        t0 = time.time()
        canonical_rotation = CAMERA_CANONICAL[cam]
        delta_to_canonical = (canonical_rotation - rotation) % 360

        if delta_to_canonical:
            canonical = np.ascontiguousarray(
                np.rot90(display_bgr, k=(360 - delta_to_canonical) // 90)
            )
        else:
            canonical = display_bgr

        grid = self.locators[cam].detect(canonical)
        grid_valid = self._grid_valid(grid)
        ch, cw = canonical.shape[:2]

        crop_canonical = [0, 0, cw, ch]
        dets_canonical = []
        if predict is not None:
            if grid_valid:
                nodes = np.asarray(grid["grid_nodes"], dtype=np.float64).reshape(-1, 2)
                x1 = max(0, int(nodes[:, 0].min()))
                y1 = max(0, int(nodes[:, 1].min()))
                x2 = min(cw, int(math.ceil(nodes[:, 0].max())))
                y2 = min(ch, int(math.ceil(nodes[:, 1].max())))
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                mx = int(round(0.15 * bw))
                mt = int(round(0.25 * bh))
                mb = int(round(0.10 * bh))
                cx1 = max(0, x1 - mx)
                cy1 = max(0, y1 - mt)
                cx2 = min(cw, x2 + mx)
                cy2 = min(ch, y2 + mb)
            else:
                cx1, cy1, cx2, cy2 = 0, 0, cw, ch

            if cx2 <= cx1 or cy2 <= cy1:
                cx1, cy1, cx2, cy2 = 0, 0, cw, ch
            crop = np.ascontiguousarray(canonical[cy1:cy2, cx1:cx2])
            crop_canonical = [cx1, cy1, cx2, cy2]
            raw = predict(crop) or []
            for det in raw:
                d = copy.deepcopy(det)
                if d.get("box") is not None:
                    b = list(d["box"])
                    b[0] += cx1
                    b[1] += cy1
                    b[2] += cx1
                    b[3] += cy1
                    d["box"] = [float(v) for v in b]
                dets_canonical.append(d)

        dets_assigned = associate(grid, dets_canonical, cam)

        delta_out = (rotation - canonical_rotation) % 360
        if delta_out:
            grid_display = self._canonical_to_display_grid(
                grid,
                canonical_rotation, rotation, cw, ch)
            dets_display = [
                self._det_to_display(d, delta_out, cw, ch) for d in dets_assigned
            ]
        else:
            grid_display = copy.deepcopy(grid)
            dets_display = copy.deepcopy(dets_assigned)

        stats = {key: grid.get(key) for key in ("reason", "inliers", "checker_count", "matched", "median_fb_px", "p95_reprojection_px")}
        grid_elapsed = None
        if isinstance(grid, dict) and self._finite(grid.get("elapsed_ms")):
            grid_elapsed = float(grid["elapsed_ms"])

        annotated = self._render(display_bgr, cam, grid_display, dets_display, grid_valid)

        process_ms = (time.time() - t0) * 1000.0

        grid_meta = {
            "camera_index": cam,
            "seq": seq,
            "captured_at": captured_at,
            "rotation": rotation,
            "width": int(display_bgr.shape[1]),
            "height": int(display_bgr.shape[0]),
            "orientation_fixed": True,
            "reference_assisted": True,
            "crop_canonical": crop_canonical,
            "assignment_method": "bbox_base_heuristic",
            "valid": grid_valid,
            "grid_nodes": grid_display.get("grid_nodes", []),
            "cells": grid_display.get("cells", []),
            "stats": stats,
            "reason": stats.get("reason"),
            "inliers": stats.get("inliers"),
            "checker_count": stats.get("checker_count"),
            "elapsed_ms": grid_elapsed,
            "process_ms": process_ms,
        }

        if remember:
            self._remember(cam, seq, captured_at, grid_valid, grid_meta,
                           dets_assigned, stats, grid_elapsed, process_ms)

        return annotated, dets_display, grid_meta

    def _remember(self, cam, seq, captured_at, grid_valid, grid_meta,
                  dets_canonical, stats, grid_elapsed, process_ms):
        if not self._finite(captured_at):
            return
        with self._lock:
            prev = self._latest[cam]
            if prev is not None:
                if seq is not None and prev.get("seq") is not None:
                    try:
                        if int(seq) < int(prev["seq"]):
                            return
                    except Exception:
                        pass
                prev_t = prev.get("captured_at")
                if self._finite(prev_t) and float(captured_at) < float(prev_t):
                    return
            assigned = [d for d in dets_canonical if d.get("assignment_status") == "assigned"]
            conflict = [d for d in dets_canonical if d.get("assignment_status") == "conflict"]
            self._latest[cam] = {
                "camera_index": cam,
                "seq": seq,
                "captured_at": float(captured_at),
                "grid_valid": bool(grid_valid),
                "reason": str(stats.get("reason", "")),
                "inliers": stats.get("inliers"),
                "checker_count": stats.get("checker_count"),
                "grid_ms": grid_elapsed,
                "process_ms": process_ms,
                "detections": copy.deepcopy(dets_canonical),
                "assigned_count": len(assigned),
                "unresolved_count": len(dets_canonical) - len(assigned),
                "conflict_count": len(conflict),
            }

    # ---------------------------------------------------------------- status
    def status(self, now=None):
        if now is None:
            now = time.time()
        with self._lock:
            snaps = {cam: copy.deepcopy(self._latest[cam]) for cam in (0, 1)}

        def entry(cam):
            s = snaps[cam]
            base = {
                "camera_index": cam,
                "seq": None, "captured_at": None, "age_ms": None,
                "fresh": False, "grid_valid": False, "reason": "",
                "inliers": None, "checker_count": None,
                "grid_ms": None, "process_ms": None,
                "assigned_count": 0, "unresolved_count": 0,
            }
            if s is None:
                return base, False
            age = now - s["captured_at"]
            fresh = 0.0 <= age <= FRESH_SECONDS
            base.update({
                "seq": s["seq"],
                "captured_at": s["captured_at"],
                "age_ms": age * 1000.0,
                "fresh": bool(fresh),
                "grid_valid": bool(s["grid_valid"] and fresh),
                "reason": s["reason"],
                "inliers": s["inliers"],
                "checker_count": s["checker_count"],
                "grid_ms": s["grid_ms"],
                "process_ms": s["process_ms"],
                "assigned_count": s["assigned_count"],
                "unresolved_count": s["unresolved_count"],
            })
            return base, fresh

        cam_entries = {}
        fresh_flags = {}
        for cam in (0, 1):
            cam_entries[cam], fresh_flags[cam] = entry(cam)

        pair_skew_ms = None
        aux_usable = False
        if snaps[0] is not None and snaps[1] is not None:
            skew = abs(snaps[1]["captured_at"] - snaps[0]["captured_at"])
            pair_skew_ms = skew * 1000.0
            aux_usable = (
                fresh_flags[0] and fresh_flags[1]
                and cam_entries[0]["grid_valid"] and cam_entries[1]["grid_valid"]
                and skew <= PAIR_SKEW_SECONDS
            )

        primary_use = (
            snaps[1] is not None and fresh_flags[1] and cam_entries[1]["grid_valid"]
        )
        aux_use = aux_usable

        per_square = {}
        for i in range(64):
            per_square["abcdefgh"[i % 8] + str(8 - i // 8)] = {"primary": None, "auxiliary": None, "conflict": False}

        def group(cam, use, conflict_ok):
            if snaps[cam] is None or not use:
                return
            for d in snaps[cam]["detections"]:
                st = d.get("assignment_status")
                if st != "assigned" and not (conflict_ok and st == "conflict"):
                    continue
                sq = d.get("square")
                if sq is None:
                    continue
                key = str(sq)
                if key not in per_square:
                    continue
                rec = per_square[key]
                cand = {
                    "class_name": d.get("class_name"),
                    "confidence": float(d.get("confidence", 0.0)),
                }
                if cam == 1:
                    if st == "conflict":
                        rec["conflict"] = True
                    if rec["primary"] is None:
                        rec["primary"] = cand
                    else:
                        rec["conflict"] = True
                else:
                    if rec["auxiliary"] is None:
                        rec["auxiliary"] = cand
                    else:
                        rec["conflict"] = True

        group(1, primary_use, conflict_ok=True)
        group(0, aux_use, conflict_ok=False)

        squares = []
        for i in range(64):
            key = "abcdefgh"[i % 8] + str(8 - i // 8)
            rec = per_square[key]
            prim = rec["primary"]
            aux = rec["auxiliary"]
            conflict = rec["conflict"]
            if conflict:
                state = "conflict"
            elif prim is not None and aux is not None:
                if prim["class_name"] == aux["class_name"]:
                    state = "supported"
                    conflict = False
                else:
                    state = "conflict"
                    conflict = True
            elif prim is not None:
                state = "conflict" if conflict else "primary_only"
            elif aux is not None:
                state = "aux_only_tentative"
                conflict = False
            else:
                state = "unknown"
                conflict = False
            squares.append({
                "square": key,
                "primary": prim,
                "auxiliary": aux,
                "state": state,
                "conflict": bool(conflict),
            })

        return {
            "enabled": True,
            "mode": "tracking_with_checker_relocalization",
            "cameras": [cam_entries[0], cam_entries[1]],
            "squares": squares,
            "pair_skew_ms": pair_skew_ms,
            "aux_usable": bool(aux_usable),
        }

    # ------------------------------------------------------------ audit path
    def _audit(self, audit_cfg_path):
        t_start = time.time()
        cfg_path = Path(audit_cfg_path).resolve()
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        out_dir = Path(cfg["output_dir"])
        if not out_dir.is_absolute():
            out_dir = (cfg_path.parent / out_dir).resolve()
        if out_dir.exists():
            raise FileExistsError("output_dir already exists: %s" % out_dir)

        ref_rel = Path(cfg["reference_config"])
        if not ref_rel.is_absolute():
            ref_rel = (cfg_path.parent / ref_rel).resolve()
        bridge = GridBridge(str(ref_rel))

        inputs = cfg.get("inputs", [])
        if len(inputs) > MAX_AUDIT_CASES:
            raise ValueError("at most %d audit cases allowed" % MAX_AUDIT_CASES)

        out_dir.mkdir(parents=True, exist_ok=False)

        results = []
        origin_nodes = {}

        for idx, item in enumerate(inputs):
            if time.time() - t_start > AUDIT_TIMEOUT_SECONDS:
                raise TimeoutError("audit exceeded %ss" % AUDIT_TIMEOUT_SECONDS)
            cam = int(item["camera_index"])
            if cam not in (0, 1):
                raise ValueError("invalid camera_index in audit input")
            img_path = Path(item["image"])
            if not img_path.is_absolute():
                img_path = (cfg_path.parent / img_path).resolve()
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cannot read audit image %s" % img_path)
            dx = float(item.get("dx", 0))
            dy = float(item.get("dy", 0))
            if dx != 0.0 or dy != 0.0:
                mat = np.float32([[1, 0, dx], [0, 1, dy]])
                img = cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))

            grid = bridge.locators[cam].detect(img)
            valid = bridge._grid_valid(grid)
            stats = {key: grid.get(key) for key in ("reason", "inliers", "checker_count", "matched", "median_fb_px", "p95_reprojection_px")}
            grid_elapsed = grid.get("elapsed_ms") if isinstance(grid, dict) else None

            nodes_flat = []
            if valid:
                nodes_flat = np.asarray(grid["grid_nodes"], dtype=np.float64).reshape(81, 2).tolist()
                if cam not in origin_nodes:
                    origin_nodes[cam] = np.asarray(nodes_flat, dtype=np.float64)

            displacement = None
            if dx == 0.0 and dy == 0.0:
                origin_nodes.setdefault(cam, np.asarray(nodes_flat, dtype=np.float64)
                                        if valid else None)
            elif valid and origin_nodes.get(cam) is not None:
                cur = np.asarray(nodes_flat, dtype=np.float64)
                diff = cur - origin_nodes[cam]
                mean = diff.mean(axis=0)
                max_err = float(np.max(np.hypot(diff[:, 0] - dx, diff[:, 1] - dy)))
                displacement = {
                    "mean_dx": float(mean[0]),
                    "mean_dy": float(mean[1]),
                    "known_dx": dx,
                    "known_dy": dy,
                    "max_error_vs_known": max_err,
                }

            gdisp = grid if valid else {"grid_nodes": [], "cells": []}
            annotated = bridge._render(img, cam, gdisp, [], valid)
            case_name = "case%d.jpg" % idx
            cv2.imwrite(str(out_dir / case_name), annotated)

            results.append({
                "case": idx,
                "camera_index": cam,
                "image": str(img_path),
                "dx": dx,
                "dy": dy,
                "file": case_name,
                "valid": bool(valid),
                "reason": str(stats.get("reason", "")),
                "inliers": stats.get("inliers"),
                "checker_count": stats.get("checker_count"),
                "elapsed_ms": grid_elapsed,
                "grid_nodes": nodes_flat,
                "cells": copy.deepcopy(grid.get("cells", [])) if valid else [],
                "displacement": displacement,
            })

        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump({"cases": results}, f, indent=2)
        return results


def _main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2 or argv[0] != "--audit-config":
        print("usage: python %s --audit-config <json>" % Path(sys.argv[0]).name)
        return 2
    bridge_cfg = None
    try:
        with open(argv[1], "r", encoding="utf-8") as f:
            audit_cfg = json.load(f)
        tmp = GridBridge.__new__(GridBridge)
        tmp._audit(argv[1])
        bridge_cfg = audit_cfg
    except Exception as exc:
        print("audit failed: %s" % exc)
        return 1
    print("audit completed: %s" % bridge_cfg.get("output_dir"))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
