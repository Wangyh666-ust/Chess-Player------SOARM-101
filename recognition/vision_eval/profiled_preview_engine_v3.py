import os
import time
import threading
from collections import defaultdict, deque

import parallel_preview_engine_v4 as base


# ---------------------------------------------------------------------------
# Shared, thread-local sample metrics.
# ---------------------------------------------------------------------------

_sample_tls = threading.local()


def _sample_dict():
    d = getattr(_sample_tls, "samples", None)
    if d is None:
        d = {}
        _sample_tls.samples = d
    return d


def begin_sample():
    _sample_tls.samples = {}
    _sample_tls.active = True


def add_sample(key, value_ms):
    try:
        v = float(value_ms)
    except (TypeError, ValueError):
        return
    if not (v == v) or v < 0.0:
        return
    if v == float("inf"):
        return
    d = _sample_dict()
    d[key] = d.get(key, 0.0) + v


class timed:
    def __init__(self, key):
        self.key = key
        self.t0 = None

    def __enter__(self):
        self.t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.t0 is not None:
            add_sample(self.key, (time.monotonic() - self.t0) * 1000.0)
        return False


def _canonical_snapshot(engine):
    fn = getattr(engine, "raw_snapshot", None)
    if callable(fn):
        try:
            return fn()
        except TypeError:
            pass
    fn = getattr(engine, "snapshot", None)
    if callable(fn):
        return fn()
    if callable(getattr(engine, "raw_snapshot", None)):
        return engine.raw_snapshot()
    return None


def _aux_fps_from_env():
    raw = os.environ.get("CHESS_AUX_FPS", "0")
    try:
        v = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("CHESS_AUX_FPS must be0..30") from exc
    if v != v or v < 0.0 or v > 30.0:
        raise ValueError("CHESS_AUX_FPS must be0..30")
    return v


_PHASE_KEYS = (
    "total_ms",
    "gpu_lock_wait_ms",
    "model_setup_ms",
    "predict_call_ms",
    "yolo_preprocess_ms",
    "yolo_forward_ms",
    "yolo_postprocess_ms",
    "grid_ms",
    "remember_fusion_vote_ms",
    "render_ms",
    "annotated_jpeg_ms",
)


def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def _finite_nonneg(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f < 0.0 or f == float("inf"):
        return None
    return f


class CollectionEngine(base.CollectionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auxiliary_fps = _aux_fps_from_env()
        self._sample_lock = threading.Lock()
        self._sample_history = defaultdict(lambda: deque(maxlen=240))

    # -- per-sample thread-local phase storage ------------------------------

    def _reset_timings(self, *args, **kwargs):
        res = super()._reset_timings(*args, **kwargs)
        _sample_tls.samples = {}
        return res

    # -- snapshot usability -------------------------------------------------

    def _snap_usable(self, snap):
        if isinstance(snap, dict) and snap.get("error") is None:
            bgr = snap.get("bgr")
            if bgr is not None:
                captured_at = _finite_nonneg(snap.get("captured_at"))
                age = _finite_nonneg(snap.get("age_ms"))
                if captured_at is not None and age is not None and age <= 2000.0:
                    return True
        try:
            return super()._snap_usable(snap)
        except AttributeError:
            return False

    # -- prediction ---------------------------------------------------------

    def _predict(self, img):
        from yolo_full_scores_v1 import predict_detections

        lock = self._predict_lock
        t_wait = time.monotonic()
        lock.acquire()
        add_sample("gpu_lock_wait_ms", (time.monotonic()-t_wait)*1000.0)
        try:
            t_setup = time.monotonic()
            model = self._load_model_locked()
            add_sample("model_setup_ms", (time.monotonic() - t_setup) * 1000.0)

            t_pred = time.monotonic()
            dets = predict_detections(
                model, img,
                device=self.device,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                quantize=None,
                verbose=self.verbose,
                batch=1,
            )
            pred_ms = (time.monotonic() - t_pred) * 1000.0
            add_sample("predict_call_ms", pred_ms)
            self._tl_set("predict_ms", (time.monotonic()-t_wait)*1000.0)

            try:
                speed = model.predictor.results[0].speed
            except Exception:
                speed = None
            if isinstance(speed, dict):
                for src, dst in (
                    ("preprocess", "yolo_preprocess_ms"),
                    ("inference", "yolo_forward_ms"),
                    ("postprocess", "yolo_postprocess_ms"),
                ):
                    v = _finite_nonneg(speed.get(src))
                    if v is not None:
                        add_sample(dst, v)

            return dets
        finally:
            if lock is not None:
                lock.release()

    # -- completion recording ----------------------------------------------

    def _record_completion(self, cam, seq, captured_at, entry, grid_meta,
                           success):
        super()._record_completion(
            cam, seq, captured_at, entry, grid_meta, success)

        grid_valid = False
        inference_skipped = True
        if isinstance(grid_meta, dict):
            grid_valid = bool(grid_meta.get("valid"))
            inference_skipped = grid_meta.get("inference_skipped") is not False
        if not (success and grid_valid and not inference_skipped):
            return

        try:
            total_ms = float(entry.get("inference_ms", 0.0) or 0.0)
        except (TypeError, ValueError):
            total_ms = 0.0
        grid_ms = _finite_nonneg((grid_meta or {}).get("elapsed_ms")) or 0.0
        encode_ms = _finite_nonneg(self._tl_get("encode_ms", 0.0)) or 0.0

        phases = {}
        local = dict(_sample_dict())
        local.update(grid_ms=grid_ms, annotated_jpeg_ms=encode_ms, total_ms=total_ms)
        for k in _PHASE_KEYS:
            if k in local:
                phases[k] = local[k]
            else:
                v = _finite_nonneg(self._tl_get(k, None))
                if v is not None:
                    phases[k] = v
        if "annotated_jpeg_ms" not in phases and encode_ms:
            phases["annotated_jpeg_ms"] = encode_ms
        if "grid_ms" not in phases and grid_ms:
            phases["grid_ms"] = grid_ms

        record = {
            "seq": seq,
            "complete_monotonic": time.monotonic(),
            "total_ms": total_ms,
            "grid_ms": grid_ms,
            "annotated_jpeg_ms": encode_ms,
        }
        for k in _PHASE_KEYS:
            if k in phases:
                record[k] = phases[k]

        with self._sample_lock:
            self._sample_history[cam].append(record)

    # -- status -------------------------------------------------------------

    def status(self):
        st = super().status()
        with self._sample_lock:
            snapshot = {c: list(dq) for c, dq in self._sample_history.items()}

        phase_metrics = []
        for cam in sorted(snapshot.keys()):
            samples = snapshot[cam]
            stages = {}
            keys = set()
            for s in samples:
                for k in _PHASE_KEYS:
                    if k in s:
                        keys.add(k)
            for k in sorted(keys):
                vals = sorted(s[k] for s in samples if k in s)
                if not vals:
                    stages[k] = {"mean_ms": None, "p95_ms": None, "count": 0}
                    continue
                stages[k] = {
                    "mean_ms": sum(vals) / len(vals),
                    "p95_ms": _percentile(vals, 95.0),
                    "count": len(vals),
                }
            phase_metrics.append({
                "camera_index": cam,
                "samples": len(samples),
                "stages": stages,
            })

        if not isinstance(st, dict):
            st = {"performance": {}}
        perf = st.get("performance")
        if not isinstance(perf, dict):
            perf = {}
            st["performance"] = perf
        perf["phase_metrics"] = phase_metrics
        perf["auxiliary_target_fps"] = self.auxiliary_fps
        return st



def wrap_bridge(bridge_cls):
    class _WrappedBridge(bridge_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._render_tls = threading.local()

        def _remember(self, *args, **kwargs):
            with timed("remember_fusion_vote_ms"):
                return super()._remember(*args, **kwargs)

        def process(self, cam, *args, **kwargs):
            try:
                cam_i = int(cam)
            except (TypeError, ValueError):
                cam_i = -1
            self._render_tls.cam = cam_i
            self._render_tls.render_calls = 0
            self._render_tls.in_process = True
            self._render_tls.primary_rendered = False
            try:
                result = super().process(cam, *args, **kwargs)
                expected = 2 if cam_i == 1 else 1
                if self._render_tls.render_calls != expected:
                    raise RuntimeError("Unexpected render call count")
                return result
            finally:
                self._render_tls.in_process = False
                self._render_tls.render_calls = 0
                self._render_tls.primary_rendered = False

        def _render(self, display, cam, grid_disp, dets_disp, grid_valid):
            tls = self._render_tls
            calls = getattr(tls, "render_calls", None)
            if calls is None:
                return super()._render(
                    display, cam, grid_disp, dets_disp, grid_valid)
            tls.render_calls = calls + 1
            try:
                cam_i = int(cam)
            except (TypeError, ValueError):
                cam_i = -1
            in_process = bool(getattr(tls, "in_process", False))
            if in_process and cam_i == 1:
                if not getattr(tls, "primary_rendered", False):
                    # Skip the pre-remember render; keep original frame.
                    tls.primary_rendered = True
                    return display
                with timed("render_ms"):
                    out = super()._render(
                        display, cam, grid_disp, dets_disp, grid_valid)
                tls.render_calls = getattr(tls, "render_calls", 0)
                if tls.render_calls != 2:
                    raise RuntimeError(
                        "primary bridge render interface mismatch: "
                        "expected 2 render calls, saw %d" % tls.render_calls)
                return out
            with timed("render_ms"):
                out = super()._render(
                    display, cam, grid_disp, dets_disp, grid_valid)
            if in_process:
                expected = 1 if cam_i != 1 else 2
                if tls.render_calls != expected:
                    raise RuntimeError(
                        "bridge render interface mismatch for cam %s: "
                        "expected %d render calls, saw %d"
                        % (cam_i, expected, tls.render_calls))
            return out

    _WrappedBridge.__name__ = "Wrapped%s" % getattr(bridge_cls, "__name__", "Bridge")
    return _WrappedBridge
