"""Live grid bridge v6.

Integrates the live camera bridge (v4) with offline fusion (selective_fusion_v2)
and persistent verified grid references. No robot/serial/network/subprocess/
camera/model code here.
"""
import copy
import json
import math
import os
import threading
import time

import numpy as np

import live_grid_bridge_v4 as old
from live_grid_locator_v5 import GridLocator, associate
from auto_board_pipeline_v2 import filter_large_boxes
from selective_fusion_v2 import fuse, DEFAULT_POLICY

CAMERA_CANONICAL = getattr(old, "CAMERA_CANONICAL", {0: 0, 1: 270})
C1_MASK = [270, 535, 815, 720]

LARGE_POLICY = {
    "min_ioa": 0.85,
    "max_center_dx": 0.25,
    "max_bottom_dy": 0.25,
}

EXCLUDE_STATUSES = {"outside", "invalid_box", "grid_invalid"}

DET_FIELDS = ("box", "class_id", "confidence", "square", "assignment_status", "class_scores")


def _finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _blank_grid(reason):
    return {
        "valid": False,
        "reason": reason,
        "grid_nodes": [],
        "cells": [],
        "elapsed_ms": None,
    }


def _valid_grid_dict(g):
    if not isinstance(g, dict) or not g.get("valid"):
        return False
    nodes = g.get("grid_nodes")
    cells = g.get("cells")
    if nodes is None or cells is None:
        return False
    try:
        arr = np.asarray(nodes, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return False
    if arr.shape != (81, 2) or not np.all(np.isfinite(arr)):
        return False
    if not isinstance(cells, list) or len(cells) != 64:
        return False
    return True


def _sanitize_det(d):
    if not isinstance(d, dict):
        return None
    box = d.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    box = [float(v) for v in box]
    if not all(math.isfinite(v) for v in box):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    cid = d.get("class_id")
    if not _is_int(cid) or not (0 <= cid <= 11):
        return None
    conf = d.get("confidence")
    if not _finite(conf) or not (0.0 <= float(conf) <= 1.0):
        return None
    return {
        "box": box,
        "class_id": int(cid),
        "confidence": float(conf),
        "class_scores": copy.deepcopy(d.get("class_scores")),
        "square": d.get("square"),
        "assignment_status": d.get("assignment_status"),
    }


class PersistentLocator:
    """Wraps a v5 GridLocator, retaining a verified reference on disk."""

    def __init__(self, cam, seed_reference, seed_calibration, cache_dir):
        self.cam = int(cam)
        self.seed_reference = np.array(seed_reference, copy=True)
        self.seed_calibration = copy.deepcopy(seed_calibration)
        self.cache_dir = cache_dir
        self.lock = threading.RLock()
        self.locator = GridLocator(np.array(seed_reference, copy=True),
                                   copy.deepcopy(seed_calibration))
        self.last_grid = _blank_grid("not_run")
        self.last_size = None
        self.cache_error = None
        self.cache_loaded = False
        self.cache_saved = False
        self._save_baseline = 0
        self._baseline_captured = False
        self._seed_fallback_tried = False
        self._cache_path = self._path()

    def _path(self):
        return os.path.join(self.cache_dir, "grid_cache_cam%d.npz" % self.cam)

    def start(self):
        try:
            self._try_load_cache()
        except Exception as exc:
            self.cache_error = "%s:%s" % (type(exc).__name__, exc)
            self._restore_seed()

    def _restore_seed(self):
        self.locator = GridLocator(np.array(self.seed_reference, copy=True),
                                   copy.deepcopy(self.seed_calibration))

    def _validate_cache(self, ref, cal, nodes, cells):
        if not isinstance(cal, dict):
            return "calibration_not_dict"
        ref = np.asarray(ref)
        if ref.ndim != 3 or ref.shape[2] != 3:
            return "ref_shape"
        if ref.shape[0] <= 0 or ref.shape[1] <= 0:
            return "ref_dims"
        if not (np.issubdtype(ref.dtype, np.integer) or ref.dtype == np.uint8):
            return "ref_dtype"
        w = cal.get("width")
        h = cal.get("height")
        if not _is_int(w) or not _is_int(h) or w <= 0 or h <= 0:
            return "cache_dims"
        if int(ref.shape[1]) != w or int(ref.shape[0]) != h:
            return "shape_mismatch"
        ns = cal.get("grid_nodes")
        if ns is None:
            return "missing_nodes"
        try:
            arr = np.asarray(ns, dtype=np.float64).reshape(-1, 2)
        except (TypeError, ValueError):
            return "nodes_shape"
        if arr.shape != (81, 2) or not np.all(np.isfinite(arr)):
            return "nodes_invalid"
        cl = cal.get("cells")
        if not isinstance(cl, list) or len(cl) != 64:
            return "cells_invalid"
        for c in cl:
            if not isinstance(c, dict):
                return "cell_not_dict"
            q = c.get("quad")
            if not isinstance(q, (list, tuple)) or len(q) != 4:
                return "cell_quad"
            for p in q:
                if isinstance(p, (list, tuple)):
                    if len(p) != 2 or not all(_finite(v) for v in p):
                        return "cell_point"
                else:
                    return "cell_point"
            if c.get("square") is None:
                return "cell_square"
        if (w, h) != (self.seed_calibration["width"], self.seed_calibration["height"]):
            return "seed_dimension_mismatch"
        seed_orient = self.seed_calibration.get("orientation")
        if cal.get("orientation") != seed_orient:
            return "orientation_mismatch"
        if nodes is None:
            return "missing_nodes"
        if cells is None:
            return "cells_invalid"
        return None

    def _try_load_cache(self):
        path = self._cache_path
        if not os.path.exists(path):
            return
        try:
            with np.load(path, allow_pickle=False) as data:
                ref = np.array(data["image"], copy=True)
                cal_json = str(data["calibration"])
        except Exception as exc:
            self.cache_error = "load:%s:%s" % (type(exc).__name__, exc)
            return
        try:
            cal = json.loads(cal_json)
        except Exception as exc:
            self.cache_error = "caljson:%s" % type(exc).__name__
            return
        err = self._validate_cache(ref, cal, cal.get("grid_nodes"), cal.get("cells"))
        if err is not None:
            self.cache_error = "validate:" + err
            return
        try:
            self.locator = GridLocator(ref, cal)
        except Exception as exc:
            self.cache_error = "locator_init:%s:%s" % (type(exc).__name__, exc)
            self._restore_seed()
            return
        self.cache_loaded = True

    def _maybe_save(self):
        with self.lock:
            loc = self.locator
            succ = int(getattr(loc, "reacquisition_successes", 0))
            if not self._baseline_captured:
                self._save_baseline = succ
                self._baseline_captured = True
                # initial valid seed may save once
                self._perform_save(loc)
                return
            if succ > self._save_baseline:
                self._save_baseline = succ
                self._perform_save(loc)

    def _perform_save(self, loc):
        try:
            with loc._lock:
                ref = np.array(loc.reference_bgr, copy=True)
                cal = copy.deepcopy(loc.calibration)
        except Exception as exc:
            self.cache_error = "grab:%s:%s" % (type(exc).__name__, exc)
            return
        try:
            nodes = cal.get("grid_nodes")
            cells = cal.get("cells")
            err = self._validate_cache(ref, cal, nodes, cells)
            if err is not None:
                self.cache_error = "save_validate:" + err
                return
        except Exception as exc:
            self.cache_error = "save_validate:%s" % type(exc).__name__
            return
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as exc:
            self.cache_error = "mkdir:%s" % type(exc).__name__
            return
        tmp = self._cache_path + ".tmp"
        try:
            with open(tmp, "wb") as f:
                np.savez_compressed(f, image=ref,
                                    calibration=json.dumps(cal))
            os.replace(tmp, self._cache_path)
        except Exception as exc:
            self.cache_error = "save:%s:%s" % (type(exc).__name__, exc)
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            return
        self.cache_saved = True

    def detect(self, image):
        img = np.asarray(image)
        h, w = img.shape[:2]
        with self.lock:
            self.last_size = (int(w), int(h))
            try:
                grid = self.locator.detect(img)
            except Exception as exc:
                grid = _blank_grid("detect_exception:%s" % type(exc).__name__)
            if not grid.get("valid") and self.cache_loaded and not self._seed_fallback_tried:
                self._seed_fallback_tried = True
                fallback_locator = GridLocator(self.seed_reference, self.seed_calibration)
                fallback = fallback_locator.detect(img)
                if fallback.get("valid"):
                    self.locator = fallback_locator
                    self.cache_loaded = False
                    self._baseline_captured = False
                    grid = fallback
            self.last_grid = grid
            if grid.get("valid"):
                try:
                    self._maybe_save()
                except Exception as exc:
                    self.cache_error = "auto_save:%s:%s" % (
                        type(exc).__name__, exc)
            return grid


class GridBridge(old.GridBridge):
    """Camera bridge with offline fusion and persistent grid references."""

    def __init__(self, config_path):
        super().__init__(config_path)
        self._config_path = config_path
        self._config = {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)
        except Exception as exc:
            raise ValueError("failed to load config: %s" % type(exc).__name__)
        cache_dir = self._config.get("persistent_grid_dir")
        if not cache_dir:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(config_path)),
                                     "grid_cache")
        if not os.path.isabs(cache_dir):
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(config_path)), cache_dir)
        self._cache_dir = cache_dir
        self._process_locks = {0: threading.RLock(), 1: threading.RLock()}
        self._context = {0: self._new_context(), 1: self._new_context()}
        self._views = {0: None, 1: None}
        self._class_id_to_name = {}
        cnames = self._config.get("class_names")
        if isinstance(cnames, dict):
            for k, v in cnames.items():
                try:
                    self._class_id_to_name[int(k)] = str(v)
                except (TypeError, ValueError):
                    continue
        elif isinstance(cnames, list):
            for i, v in enumerate(cnames):
                self._class_id_to_name[int(i)] = str(v)
        self._seen_names = {}
        for cam in (0, 1):
            ref = self.reference_images.get(cam)
            cal = self.calibration.get(cam)
            if ref is None or cal is None:
                raise ValueError("missing reference/calibration for cam %s" % cam)
            wrapper = PersistentLocator(cam, ref, cal, self._cache_dir)
            wrapper.start()
            self.locators[cam] = wrapper

    @staticmethod
    def _new_context():
        return {"raw": [], "suppressed": 0, "skipped": True, "large": []}

    def _reserved_name(self, cid, name):
        if name is not None:
            self._class_id_to_name[int(cid)] = str(name)

    def _wrap_predictor(self, cam, real_predict):
        bridge = self

        def wrapped(crop):
            wrapper = bridge.locators[cam]
            ctx = bridge._context[cam]
            grid = wrapper.last_grid
            if not _valid_grid_dict(grid):
                ctx["skipped"] = True
                ctx["raw"] = []
                ctx["suppressed"] = 0
                ctx["large"] = []
                return []
            ctx["skipped"] = False
            raw_out = real_predict(crop) or []
            cleaned = []
            for det in raw_out:
                if not isinstance(det, dict):
                    continue
                cid = det.get("class_id")
                if not _is_int(cid):
                    continue
                cid = int(cid)
                if not 0 <= cid <= 11:
                    continue
                box = det.get("box")
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                if not all(_finite(v) for v in box):
                    continue
                bf = [float(v) for v in box]
                if not all(math.isfinite(v) for v in bf):
                    continue
                if bf[2] <= bf[0] or bf[3] <= bf[1]:
                    continue
                conf = det.get("confidence")
                if not _finite(conf) or not 0 <= float(conf) <= 1:
                    continue
                name = det.get("class_name")
                bridge._reserved_name(cid, name)
                cleaned.append({
                    "box": bf,
                    "class_id": cid,
                    "class_name": name,
                    "confidence": float(conf),
                    "class_scores": copy.deepcopy(det.get("class_scores")),
                })
            ctx["raw"] = [copy.deepcopy(d) for d in cleaned]
            try:
                filtered = filter_large_boxes(cleaned, dict(LARGE_POLICY))
            except Exception as exc:
                bridge._context[cam]["filter_error"] = "%s" % type(exc).__name__
                raise
            ctx["suppressed"] = len(filtered.get("suppressed", []))
            ctx["large"] = [copy.deepcopy(d) for d in filtered.get("kept", [])]
            return filtered.get("kept", [])

        return wrapped

    def _remember(self, cam, seq, captured_at, grid_valid, grid_meta,
                  dets_canonical, stats, grid_elapsed, process_ms):
        if not _finite(captured_at):
            return
        cam = int(cam)
        ctx = self._context[cam]
        wrapper = self.locators[cam]
        grid = wrapper.last_grid
        size = wrapper.last_size

        crop = grid_meta.get("crop_canonical") or [0, 0, 0, 0]
        ox, oy = int(crop[0]), int(crop[1])

        rw, rh = size or (self.calibration[cam]["width"], self.calibration[cam]["height"])

        raw_canonical = []
        for d in ctx.get("raw", []):
            sd = _sanitize_det(d)
            if sd is None:
                continue
            b = sd["box"]
            sd["box"] = [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]
            raw_canonical.append(sd)

        if grid_valid and _valid_grid_dict(grid):
            associated_raw = associate(grid, raw_canonical, cam)
            associated_raw = [{k: d[k] for k in DET_FIELDS} for d in associated_raw]
        else:
            associated_raw = raw_canonical

        large_canonical = []
        for d in dets_canonical:
            sd = _sanitize_det(d)
            if sd is None:
                continue
            if sd["assignment_status"] in EXCLUDE_STATUSES:
                continue
            large_canonical.append(sd)
            name = d.get("class_name")
            if name is not None:
                self._class_id_to_name.setdefault(int(sd["class_id"]), str(name))

        with self._lock:
            prev = self._latest.get(cam)
            if prev is not None:
                if seq is not None and prev.get("seq") is not None:
                    try:
                        if int(seq) < int(prev["seq"]):
                            return
                    except (TypeError, ValueError):
                        pass
                prev_t = prev.get("captured_at")
                if _finite(prev_t) and float(captured_at) < float(prev_t):
                    return
            accepted = True
            try:
                super()._remember(cam, seq, captured_at, grid_valid, grid_meta,
                                  large_canonical, stats, grid_elapsed,
                                  process_ms)
            except Exception:
                raise
            if not accepted:
                return
            self._latest[cam]["raw_count"] = len(associated_raw)
            self._latest[cam]["large_count"] = len(large_canonical)
            self._latest[cam]["suppressed_count"] = int(ctx.get("suppressed", 0))
            self._latest[cam]["inference_skipped"] = bool(ctx.get("skipped", True))

            if not grid_valid or not _valid_grid_dict(grid):
                self._views[cam] = self._blank_view(cam)
                return

            cells = {}
            for c in grid.get("cells", []):
                sq = c.get("square")
                q = c.get("quad")
                if sq is None or q is None:
                    continue
                quad = []
                for p in q:
                    if isinstance(p, (list, tuple)) and len(p) == 2:
                        quad.append([float(p[0]), float(p[1])])
                if len(quad) == 4:
                    cells[sq] = quad

            occ = []
            if cam == 0 and rw == 1280 and rh == 720:
                occ = [list(C1_MASK)]
            elif cam == 0 and rw > 0 and rh > 0 and (rw, rh) != (1280, 720):
                self._views[cam] = self._blank_view(cam)
                return

            view = {
                "width": int(rw if rw > 0 else size[0] if size else 1),
                "height": int(rh if rh > 0 else size[1] if size else 1),
                "roi": list(crop),
                "grid_valid": True,
                "occlusion_rects": occ,
                "cells": cells,
                "large": large_canonical,
                "raw": associated_raw,
            }
            self._views[cam] = view

    def _blank_view(self, cam):
        cal = self.calibration.get(cam) or {}
        w = int(cal.get("width") or 1)
        h = int(cal.get("height") or 1)
        if w <= 0:
            w = 1
        if h <= 0:
            h = 1
        return {
            "width": w,
            "height": h,
            "roi": [0, 0, w, h],
            "grid_valid": False,
            "occlusion_rects": [],
            "cells": {},
            "large": [],
            "raw": [],
        }

    def process(self, cam, display_bgr, rotation, predict, seq=None,
                captured_at=None, remember=True):
        cam = int(cam)
        if cam not in (0, 1):
            raise ValueError("cam must be 0 or 1")
        wrapper = self.locators[cam]
        with self._process_locks[cam]:
            self._context[cam] = self._new_context()
            ctx = self._context[cam]
            ctx["filter_error"] = None
            wrapped = self._wrap_predictor(cam, predict) if predict is not None else None
            annotated, dets_display, grid_meta = super().process(
                cam, display_bgr, rotation, wrapped,
                seq=seq, captured_at=captured_at, remember=remember)
            outside_count = sum(d.get("assignment_status") == "outside" for d in dets_display)
            grid_before = wrapper.last_grid
            if not _valid_grid_dict(grid_before):
                dets_display = []
            else:
                dets_display = [d for d in dets_display
                                if isinstance(d, dict)
                                and d.get("assignment_status") not in EXCLUDE_STATUSES]
            grid_meta = dict(grid_meta)
            grid_meta["raw_count"] = len(ctx.get("raw", []))
            grid_meta["large_count"] = len(ctx.get("large", []))
            grid_meta["suppressed_count"] = int(ctx.get("suppressed", 0))
            grid_meta["outside_count"] = outside_count
            grid_meta["inference_skipped"] = bool(ctx.get("skipped", True))
            grid_meta["postprocess"] = "offline_large_v2"
            grid_meta["crop_source"] = "validated_grid"
            grid_meta["cache_loaded"] = bool(wrapper.cache_loaded)
            grid_meta["cache_saved"] = bool(wrapper.cache_saved)
            grid_meta["cache_error"] = wrapper.cache_error
            if ctx.get("filter_error"):
                grid_meta["filter_error"] = ctx["filter_error"]
            return annotated, dets_display, grid_meta

    def _signal(self, cid, conf):
        if cid is None or conf is None:
            return None
        if not _finite(conf):
            return None
        name = self._class_id_to_name[int(cid)]
        return {"class_id": int(cid), "class_name": name,
                "confidence": float(conf)}

    def _build_pair(self, views, now):
        best = {}
        for cam in (0, 1):
            v = views.get(cam)
            if v is None:
                continue
            ts = v.get("captured_at")
            if not _finite(ts):
                continue
            age = float(now) - float(ts) if now is not None else 0.0
            best[cam] = (v, age)
        skew = 1e12
        c1v = c2v = None
        if 0 in best and 1 in best:
            t0 = best[0][0].get("captured_at")
            t1 = best[1][0].get("captured_at")
            if _finite(t0) and _finite(t1):
                skew = abs(float(t0) - float(t1)) * 1000.0
        for cam in (0, 1):
            if cam in best:
                age = best[cam][1]
                if 0 <= age <= 3.0:
                    if cam == 0:
                        c1v = best[cam][0]
                    else:
                        c2v = best[cam][0]
        if c1v is None:
            c1v = self._blank_view(0)
        if c2v is None:
            c2v = self._blank_view(1)
        keys = ("width", "height", "roi", "grid_valid", "occlusion_rects", "cells", "large", "raw")
        c1v = {k: copy.deepcopy(c1v[k]) for k in keys}
        c2v = {k: copy.deepcopy(c2v[k]) for k in keys}
        for v in (c1v, c2v):
            if not v["grid_valid"]:
                v["large"], v["raw"], v["cells"] = [], [], {}
        return {"group": 0, "pair_skew_ms": float(skew), "c1": c1v, "c2": c2v}

    def status(self, now=None):
        if now is None:
            now = time.time()
        with self._lock:
            base = super().status(now)
            views = {}
            for cam in (0, 1):
                latest = self._latest.get(cam)
                raw_view = self._views.get(cam)
                if latest is None:
                    views[cam] = None
                    continue
                ts = latest.get("captured_at")
                age = (float(now) - float(ts)) if _finite(ts) else None
                if age is None or not 0 <= age <= 3.0:
                    views[cam] = None
                    continue
                if raw_view is None:
                    views[cam] = None
                    continue
                merged = dict(raw_view)
                merged["captured_at"] = ts
                merged["seq"] = latest.get("seq")
                merged["grid_valid"] = bool(latest.get("grid_valid")) and \
                    bool(raw_view.get("grid_valid"))
                views[cam] = merged
            pair = self._build_pair(views, now)

        try:
            fused = fuse(pair, dict(DEFAULT_POLICY))
        except Exception as exc:
            raise RuntimeError("fuse failed: %s" % type(exc).__name__)

        squares_out = []
        aux_usable = False
        if pair["c1"].get("grid_valid") and pair["c2"].get("grid_valid"):
            if pair["pair_skew_ms"] <= 350.0:
                aux_usable = True

        prim_by_sq = {}
        c1_view = views.get(0)
        if c1_view is not None:
            for d in c1_view.get("large", []):
                sq = d.get("square")
                if sq is None:
                    continue
                cur = prim_by_sq.get(sq)
                if cur is None or d.get("confidence", 0) > cur.get("confidence", 0):
                    prim_by_sq[sq] = d

        for row in fused.get("squares", []):
            sq = row.get("square")
            if sq is None:
                continue
            primary_sig = self._signal(row.get("primary_class"),
                                       row.get("primary_confidence"))
            rconf = None
            if primary_sig is not None:
                rconf = primary_sig["confidence"]
            final_sig = None
            source = row.get("source")
            if row.get("final_class") is not None:
                if source in ("auxiliary_override", "auxiliary_fill"):
                    fconf = row.get("aux_confidence")
                else:
                    fconf = row.get("primary_confidence")
                if fconf is not None and source != "unknown":
                    final_sig = self._signal(row.get("final_class"), fconf)
            state_map = {
                "corroborated": "supported",
                "auxiliary_override": "auxiliary_override",
                "auxiliary_fill": "auxiliary_fill",
                "primary_trusted": "primary_only",
                "primary_fallback": "primary_only",
            }
            state = state_map.get(source, "unknown")
            conflict = False
            if final_sig is None and primary_sig is not None:
                pass
            squares_out.append({
                "square": sq,
                "primary": primary_sig,
                "auxiliary": self._signal(prim_by_sq[sq]["class_id"], prim_by_sq[sq]["confidence"]) if aux_usable and sq in prim_by_sq else None,
                "final": final_sig,
                "source": source,
                "uncertain_reasons": list(row.get("uncertain_reasons") or []),
                "aux_rejections": list(row.get("aux_rejections") or []),
                "state": state,
                "conflict": conflict,
            })

        out = dict(base)
        out["squares"] = sorted(squares_out, key=lambda r: (-int(r["square"][1]), r["square"][0]))
        out["pair_skew_ms"] = float(pair["pair_skew_ms"])
        out["aux_usable"] = bool(aux_usable)
        out["group"] = int(fused.get("group", 0))
        out["mode"] = "selective_primary_uncertainty_v2"
        out["fusion_policy"] = copy.deepcopy(DEFAULT_POLICY)
        out["fusion_diagnostics"] = copy.deepcopy(fused.get("squares", []))
        cams = out.get("cameras")
        for cam in (0, 1):
            wrapper = self.locators.get(cam)
            if wrapper is None:
                continue
            entry = None
            if isinstance(cams, list) and cam < len(cams):
                entry = cams[cam]
            elif isinstance(cams, dict):
                entry = cams.get(cam)
            if isinstance(entry, dict):
                entry["cache_loaded"] = bool(wrapper.cache_loaded)
                entry["cache_saved"] = bool(wrapper.cache_saved)
                entry["cache_error"] = wrapper.cache_error
        out["pipeline_version"] = "live_alignment_v1"
        return out

    def _render(self, display, cam, grid_disp, dets_disp, grid_valid):
        allowed = [d for d in dets_disp
                   if isinstance(d, dict)
                   and d.get("assignment_status") not in EXCLUDE_STATUSES]
        out = super()._render(display, cam, grid_disp, allowed,
                              bool(grid_valid))
        if not grid_valid:
            try:
                import cv2
                cv2.putText(out, "GRID LOST - locating board; YOLO paused",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2, cv2.LINE_AA)
            except Exception:
                pass
        return out
