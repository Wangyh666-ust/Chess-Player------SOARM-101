import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import time
import copy
import sys
import pathlib
import threading

import numpy as np
import cv2

import camera1_radial_grid_v2 as radial
import guided_grid_adapter_v2 as schema


_DICT = cv2.aruco.DICT_4X4_50
_SQUARE_MM = 26.0
_C1_ORIENT = "files_desc_ranks_asc"
_C2_ORIENT = "ranks_desc_files_desc"
_C1_MASK_DEFAULT = [[270, 535, 815, 720]]
_NODES = 9
_CELLS = 8


def _bounded(msg, limit=300):
    s = "%s:%s" % (type(msg).__name__, msg) if isinstance(msg, BaseException) else str(msg)
    if len(s) > limit:
        s = s[:limit]
    return s


def _load_assets():
    root = os.environ.get("CHESS_ARUCO_ASSETS")
    if not root:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "aruco_live_assets")
    root = pathlib.Path(root)
    layout = json.loads((root / "layout.json").read_text(encoding="utf-8"))
    radial_cfg = json.loads((root / "radial.json").read_text(encoding="utf-8"))
    return layout, radial_cfg


def _invalid(reason, method, shape, elapsed, count=None):
    h, w = int(shape[0]), int(shape[1])
    d = {
        "valid": False,
        "reason": reason,
        "method": method,
        "grid_nodes": [],
        "cells": [],
        "width": w,
        "height": h,
        "elapsed_ms": float(elapsed),
    }
    if count is not None:
        d["marker_count"] = int(count)
        d["marker_ids"] = []
    return d


class GridLocator(object):
    def __init__(self, reference_bgr, calibration):
        self.calibration = copy.deepcopy(calibration)
        self.width = int(reference_bgr.shape[1]) if reference_bgr is not None else 0
        self.height = int(reference_bgr.shape[0]) if reference_bgr is not None else 0

        layout, radial_cfg = _load_assets()
        self._layout = layout
        self._radial_cfg = radial_cfg

        self.orientation = (self.calibration or {}).get("orientation")
        if self.orientation not in (_C1_ORIENT, _C2_ORIENT):
            raise ValueError("unsupported camera orientation")
        self.is_c1 = self.orientation == _C1_ORIENT

        self._marker_corners = {}
        for m in layout.get("markers", []):
            arr = np.asarray(m.get("corners_mm", []), dtype=np.float64)
            if arr.shape == (4, 3):
                self._marker_corners[int(m["id"])] = arr
        self._known_ids = set(self._marker_corners.keys())

        self._square_size_mm = float(layout.get("square_size_mm", _SQUARE_MM))

        self._mask_rects = None
        if self.is_c1:
            rects = self.calibration.get("occlusion_rects") if isinstance(self.calibration, dict) else None
            if rects:
                self._mask_rects = [list(r) for r in rects]
            else:
                self._mask_rects = [list(r) for r in _C1_MASK_DEFAULT]

        if self.is_c1:
            self._method = "measured_aruco_radial_v1"
            self._reason_tag = "radial"
        else:
            self._method = "measured_aruco_v1"
            self._reason_tag = None

        self._lam = float(radial_cfg.get("lambda", -0.19111131794683894))
        c = radial_cfg.get("center", [640.0, 360.0])
        self._center = (float(c[0]), float(c[1]))
        self._scale = float(radial_cfg.get("scale", 640.0))
        self._radial_provisional = bool(radial_cfg.get("provisional", True))

        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(_DICT), params)

        self._lock = threading.RLock()

        self.last_grid = None
        self.last_size = None

    def _undistort(self, pts):
        return radial.undistort_points(pts, self._lam, self._center, self._scale)

    def _distort(self, pts):
        return radial.distort_points(pts, self._lam, self._center, self._scale)

    def _legacy_uv(self, r, c):
        if self.is_c1:
            return [float(_CELLS - c), float(r)]
        return [float(_CELLS - r), float(_CELLS - c)]

    def _fail(self, reason, shape, t0, count=None):
        res = _invalid(reason, self._method, shape, (time.time() - t0) * 1000.0, count)
        res["radial_provisional"] = self._radial_provisional if self.is_c1 else False
        self.last_grid = res
        self.last_size = (int(shape[1]), int(shape[0]))
        return res

    def detect(self, frame_bgr):
        t0 = time.time()
        with self._lock:
            if frame_bgr is None:
                return self._fail("no_frame", (720, 1280), t0)
            shape = frame_bgr.shape[:2]
            h, w = int(shape[0]), int(shape[1])
            self.last_size = (w, h)

            if self.is_c1:
                if (w, h) != (1280, 720):
                    return self._fail("wrong_dimensions_%dx%d" % (w, h), shape, t0)
            else:
                if (w, h) != (720, 1280):
                    return self._fail("wrong_dimensions_%dx%d" % (w, h), shape, t0)

            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            except Exception as exc:
                return self._fail("gray_fail:" + _bounded(exc), shape, t0)

            corners, ids, _ = self._detector.detectMarkers(gray)

            known_raw = []
            seen = set()
            dup = False
            if ids is not None:
                for i, mid in enumerate(ids.flatten()):
                    mid = int(mid)
                    if mid not in self._known_ids:
                        continue
                    if mid in seen:
                        dup = True
                        continue
                    seen.add(mid)
                    known_raw.append((mid, np.asarray(corners[i], dtype=np.float64).reshape(4, 2)))

            marker_ids = sorted(mid for mid, _ in known_raw)
            n = len(known_raw)

            if dup:
                return self._fail("duplicate_known_marker", shape, t0, n)
            if n < 1:
                res = self._fail("no_known_markers", shape, t0, 0)
                res["marker_ids"] = []
                return res
            if n < 2:
                res = self._fail("insufficient_markers_%d_need_2" % n, shape, t0, n)
                res["marker_ids"] = marker_ids
                return res

            board_uv = []
            raw_pts = []
            for mid, raw in known_raw:
                cm = self._marker_corners[mid]
                buv = cm[:, :2] / self._square_size_mm
                board_uv.append(buv)
                raw_pts.append(raw)
            board_uv = np.vstack(board_uv)
            raw_pts = np.vstack(raw_pts)

            hull_src = board_uv.astype(np.float32)
            try:
                hull_area = float(cv2.contourArea(cv2.convexHull(hull_src)))
            except Exception as exc:
                return self._fail("hull_fail:" + _bounded(exc), shape, t0, n)
            xspan = float(np.max(board_uv[:, 0]) - np.min(board_uv[:, 0]))
            yspan = float(np.max(board_uv[:, 1]) - np.min(board_uv[:, 1]))
            if hull_area < 4.0 or xspan < 2.0 or yspan < 2.0:
                res = self._fail("board_geometry_too_small", shape, t0, n)
                res["marker_ids"] = marker_ids
                return res

            try:
                if self.is_c1:
                    und = self._undistort(raw_pts)
                    H = radial.homography(board_uv, und)
                else:
                    H = radial.homography(board_uv, raw_pts)
            except Exception as exc:
                res = self._fail("homography:" + _bounded(exc), shape, t0, n)
                res["marker_ids"] = marker_ids
                return res

            try:
                proj = radial.project(H, board_uv)
                if self.is_c1:
                    proj_raw = self._distort(proj)
                else:
                    proj_raw = proj
                err = np.linalg.norm(proj_raw - raw_pts, axis=1)
                rms = float(np.sqrt(np.mean(err * err)))
                emax = float(np.max(err))
            except Exception as exc:
                res = self._fail("reproject:" + _bounded(exc), shape, t0, n)
                res["marker_ids"] = marker_ids
                return res

            if not np.all(np.isfinite(err)):
                res = self._fail("nonfinite_reprojection", shape, t0, n)
                res["marker_ids"] = marker_ids
                return res
            if rms > 6.0 or emax > 15.0:
                res = self._fail("reprojection_out_of_bounds_rms=%.3f_max=%.3f" % (rms, emax),
                                 shape, t0, n)
                res["marker_ids"] = marker_ids
                res["reprojection_rms_px"] = rms
                return res
            p95 = float(np.percentile(err, 95)) if err.size else 0.0

            nodes_uv = []
            for r in range(_NODES):
                for c in range(_NODES):
                    nodes_uv.append(self._legacy_uv(r, c))
            nodes_uv = np.asarray(nodes_uv, dtype=np.float64)

            try:
                nodes_proj = radial.project(H, nodes_uv)
                if self.is_c1:
                    nodes_px = self._distort(nodes_proj)
                else:
                    nodes_px = nodes_proj
            except Exception as exc:
                res = self._fail("node_project:" + _bounded(exc), shape, t0, n)
                res["marker_ids"] = marker_ids
                return res

            if not np.all(np.isfinite(nodes_px)):
                res = self._fail("nonfinite_nodes", shape, t0, n)
                res["marker_ids"] = marker_ids
                return res

            grid = {"grid_nodes": nodes_px.reshape(-1, 2)}
            rects = self._mask_rects if self.is_c1 else []
            try:
                rebuilt = schema._rebuild_success(
                    grid, copy.deepcopy(self.calibration), shape, self.orientation, rects)
            except Exception as exc:
                res = self._fail("rebuild:" + _bounded(exc), shape, t0, n)
                res["marker_ids"] = marker_ids
                return res
            if rebuilt is None:
                res = self._fail("rebuild_invalid_geometry", shape, t0, n)
                res["marker_ids"] = marker_ids
                return res

            out, grid_cal = rebuilt
            elapsed = (time.time() - t0) * 1000.0

            reason = "aruco_live_%d_codes" % n
            if self._reason_tag:
                reason = reason + "_" + self._reason_tag

            out["valid"] = True
            out["reason"] = reason
            out["method"] = self._method
            out["reference_assisted"] = False
            out["matched"] = int(board_uv.shape[0])
            out["inliers"] = int(board_uv.shape[0])
            out["checker_count"] = 0
            out["median_fb_px"] = None
            out["p95_reprojection_px"] = p95
            out["elapsed_ms"] = float(elapsed)
            out["marker_ids"] = marker_ids
            out["marker_count"] = n
            out["radial_provisional"] = bool(self._radial_provisional) if self.is_c1 else False
            out["reprojection_rms_px"] = rms
            out["grid_nodes"] = out.get("grid_nodes")
            out["cells"] = out.get("cells")
            out["width"] = int(shape[1])
            out["height"] = int(shape[0])
            out["orientation"] = self.orientation

            self.last_grid = out
            self.last_size = (int(shape[1]), int(shape[0]))
            return out


_BRIDGE_CLASS = None

def install_runtime():
    global _BRIDGE_CLASS
    if _BRIDGE_CLASS is not None:
        return _BRIDGE_CLASS
    import live_weighted_grid_bridge_v2 as weighted
    from weighted_temporal_fusion_v4 import fuse as current_fuse
    weighted.fuse = current_fuse
    import occlusion_grid_preview_v2 as preview
    import score_grid_bridge_v1 as score
    import live_grid_bridge_v4 as old
    score.GridLocator = GridLocator
    old.GridLocator = GridLocator
    OriginalPersistent = score.PersistentLocator
    class NoCachePersistent(OriginalPersistent):
        def start(self):
            pass
        def _maybe_save(self):
            pass
    score.PersistentLocator = NoCachePersistent
    class LiveGridBridge(preview.GridBridge):
        @staticmethod
        def _draw_header(img, cam, grid, grid_valid, crop_box):
            role = "AUX C1 + provisional radial" if cam == 0 else "PRIMARY C2"
            text = "%s | %s | ArUco" % (role, "GRID OK" if grid_valid else "GRID LOST")
            color = (0,255,0) if grid_valid else (0,0,255)
            cv2.putText(img,text,(6,36),cv2.FONT_HERSHEY_SIMPLEX,.46,color,1,cv2.LINE_AA)
            cv2.putText(img,"Measured sticker layout | " + str(grid.get("reason","")),(6,51),cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
            return text
        def process(self,cam,*args,**kwargs):
            annotated,dets,meta = super().process(cam,*args,**kwargs)
            g = self.locators[int(cam)].last_grid
            meta.update(reference_assisted=False,grid_method=g.get("method"),marker_ids=g.get("marker_ids",[]),marker_count=g.get("marker_count",0),radial_provisional=g.get("radial_provisional",False))
            return annotated,dets,meta
        def status(self,*args,**kwargs):
            result = super().status(*args,**kwargs)
            result["grid_localization"] = "measured_aruco_live_v1"
            result["grid_pose_cache"] = False
            result["fusion_policy"] = {
                "version": "local_conf_patch_20261113_v1",
                "baseline_primary_c2": 0.7, "baseline_aux_c1": 0.3,
                "c1_min_conf": 0.9, "c2_max_conf": 0.6,
                "require_class_disagreement": True,
                "trigger_weights": {"c1": 0.6, "c2": 0.4},
                "full_class_scores": True, "learned_weight_model": False,
            }
            if hasattr(self._temporal,"summary"):
                result["voting"] = self._temporal.summary()
                result["mode"] = "weighted_7_3_local_conf_patch_v1"
                result["pipeline_version"] = "aruco_async_local_conf_patch_20261113_v1"
                result["temporal_policy"] = dict(window=self._temporal.window, min_span=self._temporal.min_span, min_frames=self._temporal.min_frames, history_weight=self._temporal.history_weight, history_half_life_s=self._temporal.history_half_life)
                result["inference_targets"] = {"primary_c2_fps":20,"auxiliary_c1_fps":float(os.environ.get("CHESS_AUX_FPS",0))}
            result["aruco_cameras"] = []
            for cam in (0,1):
                g = self.locators[cam].last_grid or {}
                result["aruco_cameras"].append(dict(camera_index=cam,**{k:g.get(k) for k in ("valid","reason","method","marker_ids","marker_count","elapsed_ms","reprojection_rms_px","radial_provisional")}))
            return result
    _BRIDGE_CLASS = LiveGridBridge
    return LiveGridBridge


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "throughput_runtime.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            os.environ.setdefault("CHESS_AUX_FPS", str(json.load(f).get("auxiliary_fps", 0)))
    cv2.setNumThreads(2)
    bridge_cls = install_runtime()

    import score_grid_bridge_v1 as score
    import live_grid_bridge_v4 as old

    if argv and argv[0] == "--audit-config":
        return old._main(argv)

    import data_collection_preview_v15 as app
    from profiled_preview_engine_v3 import CollectionEngine, wrap_bridge
    from async_capture_worker_v3 import CameraWorker
    app.base.CameraWorker = CameraWorker
    bridge_cls = wrap_bridge(bridge_cls)
    app.CollectionEngine = CollectionEngine
    if os.environ.get("CHESS_SECOND_VOTE", "1") == "1":
        import live_weighted_grid_bridge_v2 as weighted
        from voting_controls_v2 import TemporalFusion
        weighted.TemporalFusion = TemporalFusion
    from voting_controls_v2 import install_http
    from data_collection_preview_v17 import INDEX_HTML
    app.INDEX_HTML = INDEX_HTML
    install_http(app)
    app.GridBridge = bridge_cls
    app._GRID_BRIDGE_IMPORT_ERROR = None
    return app.main()


if __name__ == "__main__":
    sys.exit(main())
