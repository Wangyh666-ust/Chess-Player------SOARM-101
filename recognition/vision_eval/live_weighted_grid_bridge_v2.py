import copy
import math
import time

import score_grid_bridge_v1 as base
from weighted_temporal_fusion_v3 import fuse, TemporalFusion, DEFAULT_POLICY


_FRESH_MS = 3.0
_SKEW_MS = 350.0
_MOVE_PX = 10.0
_MOVE_FRAC = 0.35


def _finite(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


def _poly_area(pts):
    if not pts or len(pts) < 3:
        return 0.0
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i][0], pts[i][1]
        x2, y2 = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def _poly_center(pts):
    if not pts:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _median(vals):
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


class GridBridge(base.GridBridge):

    def __init__(self, config_path):
        self._temporal = TemporalFusion()
        self._stable = None
        self._primary_seq = None
        self._anchor = None
        self._feet = {}
        self._pair = None
        super().__init__(config_path)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _cells_of_view(self, view):
        if not view:
            return {}
        cells = view.get("cells") or {}
        return cells

    def _assigned_signal(self, view, square):
        if not view:
            return None
        dets = view.get("large")
        if dets is None:
            dets = view.get("raw") or []
        matches = [d for d in (dets or []) if isinstance(d, dict)
                   and d.get("assignment_status") == "assigned" and d.get("square") == square]
        if len(matches) != 1:
            return None
        for d in matches:
            if not isinstance(d, dict):
                continue
            if d.get("assignment_status") != "assigned":
                continue
            if d.get("square") != square:
                continue
            cid = d.get("class_id")
            if cid is None:
                continue
            try:
                cid = int(cid)
            except (TypeError, ValueError):
                continue
            name = self._class_id_to_name.get(cid)
            if name is None:
                name = d.get("class_name")
            conf = d.get("confidence")
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = None
            return {
                "class_id": cid,
                "class_name": name,
                "confidence": conf,
            }
        return None

    def _refresh_locked(self):
        views = {}
        latest_ts = None
        for cam in (0, 1):
            v = self._views.get(cam)
            if v is None:
                views[cam] = None
                continue
            vv = copy.deepcopy(v)
            lv = self._latest.get(cam)
            if lv is not None and vv.get("captured_at") is None:
                vv["captured_at"] = lv.get("captured_at")
            ca = vv.get("captured_at")
            if _finite(ca):
                ca = float(ca)
                if latest_ts is None or ca > latest_ts:
                    latest_ts = ca
            views[cam] = vv
        now = latest_ts if latest_ts is not None else 0.0
        return self._build_pair(views, now=now)

    def _detect_moving(self, weighted, primary_view):
        cells = self._cells_of_view(primary_view)
        feet = {}
        moving = set()
        squares = weighted.get("squares64", {}) or {}
        for sq in squares:
            poly = None
            cell = cells.get(sq)
            if isinstance(cell, dict):
                poly = cell.get("quad") or cell.get("polygon") or cell.get("points")
            elif isinstance(cell, (list, tuple)):
                poly = cell
            pts = None
            if poly:
                try:
                    pts = [(float(p[0]), float(p[1])) for p in poly]
                except (TypeError, ValueError, IndexError):
                    pts = None
            if not pts:
                continue
            area = _poly_area(pts)
            if area <= 0:
                continue
            matches = [d for d in (primary_view.get("large") or [])
                       if d.get("assignment_status") == "assigned" and d.get("square") == sq]
            if len(matches) != 1:
                continue
            box = matches[0]["box"]
            fx = (box[0] + box[2]) / 2.0
            fy = box[3] - 0.12 * (box[3] - box[1])
            feet[sq] = (fx, fy)
            prev = self._feet.get(sq)
            if prev is not None:
                thr = _MOVE_FRAC * math.sqrt(area)
                dx = fx - prev[0]
                dy = fy - prev[1]
                if math.hypot(dx, dy) > thr:
                    moving.add(sq)
        self._feet = feet
        return moving

    def _reset_locked(self):
        self._temporal.reset()
        self._stable = None
        self._primary_seq = None
        self._feet = {}
        self._anchor = None

    # ------------------------------------------------------------------
    # _remember
    # ------------------------------------------------------------------
    def _remember(self, cam, seq, captured_at, grid_valid, grid_meta,
                  dets_canonical, stats, grid_elapsed, process_ms):
        try:
            cam = int(cam)
        except (TypeError, ValueError):
            return
        with self._lock:
            super()._remember(cam, seq, captured_at, grid_valid, grid_meta,
                              dets_canonical, stats, grid_elapsed, process_ms)
            if cam != 1:
                return
            lv = self._latest.get(1)
            if lv is None:
                return
            if lv.get("captured_at") != captured_at:
                return
            if lv.get("seq") != seq:
                return
            if not _finite(captured_at):
                return
            if not grid_valid:
                self._reset_locked()
                self._pair = None
                return
            if self._primary_seq == seq:
                return
            try:
                pair = self._refresh_locked()
            except Exception:
                raise
            self._pair = pair
            weighted = fuse(pair)
            squares = weighted.get("squares64", {}) or {}
            geo_shift = None
            c2 = pair.get("c2") or {}
            cells = self._cells_of_view(c2 if c2.get("width") else None)
            views_c2 = self._views.get(1)
            cells = self._cells_of_view(views_c2)
            centers = []
            center_map = {}
            for sq in squares:
                cell = cells.get(sq)
                poly = None
                if isinstance(cell, dict):
                    poly = cell.get("quad") or cell.get("polygon") or cell.get("points")
                elif isinstance(cell, (list, tuple)):
                    poly = cell
                if not poly:
                    continue
                try:
                    pts = [(float(p[0]), float(p[1])) for p in poly]
                except (TypeError, ValueError, IndexError):
                    continue
                c = _poly_center(pts)
                if c is None:
                    continue
                center_map[sq] = c
                centers.append(c)
            reset_now = False
            if c2 and not c2.get("grid_valid", True):
                reset_now = True
            if self._anchor is not None and center_map:
                disp = []
                for sq, c in center_map.items():
                    a = self._anchor.get(sq)
                    if a is None:
                        continue
                    disp.append(math.hypot(c[0] - a[0], c[1] - a[1]))
                if disp:
                    geo_shift = _median(disp)
                    if geo_shift > _MOVE_PX:
                        reset_now = True
            if reset_now:
                self._reset_locked()
                self._anchor = copy.deepcopy(center_map)
                self._stable = None
                return
            if center_map and self._anchor is None:
                self._anchor = copy.deepcopy(center_map)
            primary_view = self._views.get(1)
            moving = self._detect_moving(weighted, primary_view)
            pseq = self._primary_seq
            if pseq is not None and pseq == seq:
                return
            ts = float(captured_at)
            stable = self._temporal.update(
                weighted, ts, [seq], grid_token="live_scene", moving=moving)
            self._primary_seq = seq
            self._stable = stable

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------
    def status(self, now=None):
        if now is None:
            now = time.time()
        out = base.old.GridBridge.status(self, now)
        out = dict(out)
        cams = out.get("cameras") or {}
        cams = copy.deepcopy(cams)

        with self._lock:
            lv1 = self._latest.get(1)
            primary_fresh = False
            if lv1 is not None and lv1.get("grid_valid"):
                ca = lv1.get("captured_at")
                if _finite(ca):
                    age = float(now) - float(ca)
                    if 0.0 <= age <= _FRESH_MS:
                        primary_fresh = True
            if not primary_fresh:
                self._reset_locked()
                self._stable = None
                self._primary_seq = None
                self._feet = {}
                self._anchor = None
                pair = None
            else:
                pair = self._pair
                if pair is None:
                    try:
                        pair = self._refresh_locked()
                    except Exception:
                        pair = None

            c1_view = self._views.get(0)
            c2_view = self._views.get(1)
            pair_ok = False
            skew = None
            if pair is not None:
                skew = pair.get("pair_skew_ms")
                if _finite(skew) and skew <= _SKEW_MS:
                    pair_ok = True
            aux_fresh = False
            if c1_view is not None:
                ca = (self._latest.get(0) or {}).get("captured_at")
                if _finite(ca):
                    age = float(now) - float(ca)
                    if 0.0 <= age <= _FRESH_MS and c1_view.get("grid_valid"):
                        aux_fresh = True

            stable = self._stable

            rows = {}
            if stable is not None:
                sq_in = stable.get("squares64", {}) or {}
            else:
                sq_in = {}

            names = self._class_id_to_name
            for sq in _ordered_squares():
                row_in = sq_in.get(sq) or {}
                p_sig = None
                a_sig = None
                if primary_fresh:
                    p_sig = self._assigned_signal(c2_view, sq)
                if pair_ok and aux_fresh:
                    a_sig = self._assigned_signal(c1_view, sq)
                final = None
                source = "unknown"
                state = "unknown"
                reasons = []
                weights = [0.0, 0.0]
                temporal_frames = None
                temporal_state = row_in.get("temporal_state")
                if stable is not None:
                    fc = row_in.get("final_class")
                    if fc is not None:
                        try:
                            fc = int(fc)
                        except (TypeError, ValueError):
                            fc = None
                    if fc is not None:
                        fconf = row_in.get("confidence")
                        try:
                            fconf = float(fconf)
                        except (TypeError, ValueError):
                            fconf = None
                        fname = names.get(fc)
                        if fname is None:
                            fname = str(fc)
                        final = {
                            "class_id": fc,
                            "class_name": fname,
                            "confidence": fconf,
                        }
                        source = row_in.get("source") or "unknown"
                        state = "supported"
                    else:
                        if temporal_state:
                            reasons.append("temporal_state:%s" % temporal_state)
                w = row_in.get("weights")
                if isinstance(w, dict):
                    try:
                        weights = [float(w.get("c2", 0)), float(w.get("c1", 0))]
                    except (TypeError, ValueError):
                        weights = [0.0, 0.0]
                tf = row_in.get("temporal_frames")
                if tf is not None:
                    temporal_frames = tf
                row = {
                    "square": sq,
                    "primary": p_sig,
                    "auxiliary": a_sig,
                    "final": final,
                    "source": source,
                    "state": state,
                    "conflict": False,
                    "uncertain_reasons": reasons,
                    "aux_rejections": list(row_in.get("aux_rejections") or []),
                    "weights": weights,
                    "temporal_frames": temporal_frames,
                    "temporal_state": temporal_state,
                }
                rows[sq] = row

            out["squares"] = list(rows.values())

            out["mode"] = "weighted_7_3_temporal_v1"
            out["pipeline_version"] = "weighted_temporal_trial_v1"
            out["fusion_policy"] = copy.deepcopy(DEFAULT_POLICY)
            out["temporal_policy"] = {
                "window": 0.8,
                "min_span": 0.35,
                "min_frames": 3,
            }
            out["experimental"] = True

            for cam in (0, 1):
                loc = self.locators.get(cam)
                if loc is None:
                    continue
                ent = cams[cam] if isinstance(cams, list) else cams.get(str(cam), cams.get(cam))
                if not isinstance(ent, dict):
                    ent = {}
                ent["cache_loaded"] = bool(getattr(loc, "cache_loaded", False))
                ent["cache_saved"] = bool(getattr(loc, "cache_saved", False))
                ent["cache_error"] = getattr(loc, "cache_error", None)
                if isinstance(cams, dict) and str(cam) in cams:
                    cams[str(cam)] = ent
                else:
                    cams[cam] = ent
            out["cameras"] = cams

        return out

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def _render(self, display, cam, grid_disp, dets_disp, grid_valid):
        copies = copy.deepcopy(dets_disp)
        try:
            cam = int(cam)
        except (TypeError, ValueError):
            cam = -1
        try:
            st = self.status()
        except Exception:
            st = None
        cells = {}
        if isinstance(st, dict):
            cells = {r["square"]: r for r in st.get("squares", [])}
        if cam == 1:
            for d in copies:
                if not isinstance(d, dict):
                    continue
                sq = d.get("square")
                row = cells.get(sq) if sq is not None else None
                if isinstance(row, dict) and row.get("final"):
                    f = row["final"]
                    d["class_name"] = f.get("class_name")
                    d["confidence"] = f.get("confidence")
                    d["label"] = f.get("class_name")
                else:
                    d["class_name"] = "pending"
                    d["confidence"] = 0.0
                    d["label"] = "pending"
        return super()._render(display, cam, grid_disp, copies, grid_valid)

    # ------------------------------------------------------------------
    # process
    # ------------------------------------------------------------------
    def process(self, cam, display_bgr, rotation, predict, seq=None,
                captured_at=None, remember=True):
        annotated, dets_display, grid_meta = super().process(
            cam, display_bgr, rotation, predict,
            seq=seq, captured_at=captured_at, remember=remember)
        try:
            cam_i = int(cam)
        except (TypeError, ValueError):
            cam_i = -1
        grid_valid = bool((grid_meta or {}).get("valid"))
        if cam_i == 1:
            grid_disp = grid_meta or {}
            display = annotated
            try:
                annotated = self._render(
                    display_bgr, 1, grid_disp, dets_display, grid_valid)
            except Exception:
                raise
        return annotated, dets_display, grid_meta


def _ordered_squares():
    ranks = "87654321"
    files = "abcdefgh"
    return ["%s%s" % (f, r) for r in ranks for f in files]
