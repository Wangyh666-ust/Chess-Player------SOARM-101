import math
import threading
import time
from collections import deque

import data_collection_preview_v15 as app


class CollectionEngine(app.CollectionEngine):
    """Per-camera concurrent inference engine with honest effective-frame telemetry.

    Extends the base CollectionEngine (same constructor + HTTP API surface) so that
    per-camera preprocessing/postprocess/grid work runs concurrently while the
    persistent model load & predict path remains serialized behind the base's
    single GPU ``_predict_lock``.
    """

    TARGET_EFFECTIVE_FPS = 20.0
    CAPTURE_TARGET_FPS = 30.0
    ROLLING_WINDOW_SEC = 2.0
    LATENCY_WINDOW = 1200
    POST_IDLE_WAIT_SEC = 0.001
    SAME_FRAME_WAIT_SEC = 0.002
    NOT_LEASED_WAIT_SEC = 0.05
    UNUSABLE_WAIT_SEC = 0.01

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        # Per-camera worker threads.
        self._cam_threads = {}
        self._cam_threads_lock = threading.Lock()

        # Thread-local timing buckets (decode / predict / encode / grid).
        self._timing_local = threading.local()

        # Telemetry state.
        self._telemetry_lock = threading.Lock()
        self._completed = deque(maxlen=1200)
        self._accepted_seq = {}          # cam -> set of raw seqs accepted
        self._effective_total = 0
        self._processed_total = 0        # valid grid + successful inference
        self._invalid_grid_total = 0
        self._error_total = 0
        self._per_cam_effective = {}
        self._per_cam_processed = {}
        self._per_cam_error = {}
        self._per_cam_invalid = {}
        self._per_cam_last_effective_seq = {}
        self._started_at = time.monotonic()

    # -- lifecycle -------------------------------------------------------
    def _ensure_worker(self):
        with self._worker_start_lock:
            if self._closed.is_set():
                return
            indices = self._camera_indices()
            for cam in indices:
                existing = self._cam_threads.get(cam)
                if existing is not None and existing.is_alive():
                    continue
                t = threading.Thread(
                    target=self._worker_camera,
                    args=(cam,),
                    name="collection-infer-cam-%s" % (cam,),
                    daemon=True,
                )
                with self._cam_threads_lock:
                    self._cam_threads[cam] = t
                t.start()

    def _camera_indices(self):
        feeds = self.feeds
        if isinstance(feeds, dict):
            return list(feeds.keys())
        try:
            return list(range(len(feeds)))
        except TypeError:
            return []

    def close(self):
        self._closed.set()
        with self._cam_threads_lock:
            threads = list(self._cam_threads.values())
        base_worker = getattr(self, "_worker", None)
        if base_worker is not None:
            threads.append(base_worker)
        for t in threads:
            try:
                t.join(timeout=2.0)
            except Exception:
                pass

    # -- thread-local timing helpers ------------------------------------
    def _tl_get(self, name, default=0.0):
        return getattr(self._timing_local, name, default)

    def _tl_set(self, name, value):
        setattr(self._timing_local, name, value)

    def _reset_timings(self):
        self._tl_set("decode_ms", 0.0)
        self._tl_set("predict_ms", 0.0)
        self._tl_set("encode_ms", 0.0)
        self._tl_set("grid_ms", 0.0)

    # -- model (inherited GPU lock; timed) -------------------------------
    def _predict(self, img):
        t0 = time.monotonic()
        dets = super()._predict(img)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._tl_set("predict_ms", elapsed_ms)
        return dets

    # -- per-camera worker -----------------------------------------------
    def _worker_camera(self, cam):
        while not self._closed.is_set():
            try:
                leases = self._active_leases()
                if cam not in leases:
                    if self._closed.wait(self.NOT_LEASED_WAIT_SEC):
                        break
                    continue

                rotation = leases[cam]
                feed = self.feeds[cam]
                snap = feed.raw_snapshot() if hasattr(feed, "raw_snapshot") else feed.snapshot()
                if not self._snap_usable(snap):
                    if self._closed.wait(self.UNUSABLE_WAIT_SEC):
                        break
                    continue

                seq = snap.get("seq")
                if not self._accept_sequence(cam, seq):
                    # Already accepted this raw frame for this camera.
                    if self._closed.wait(self.SAME_FRAME_WAIT_SEC):
                        break
                    continue

                captured_at = snap.get("captured_at")
                bgr = snap.get("bgr")
                source_jpeg = snap.get("jpeg")

                self._reset_timings()
                t0 = time.monotonic()
                try:
                    if bgr is not None:
                        img = bgr
                    else:
                        d0 = time.monotonic()
                        img = app._decode(source_jpeg)
                        self._tl_set("decode_ms",
                                     (time.monotonic() - d0) * 1000.0)
                    up = app._rotate_np(img, rotation)
                    g0 = time.monotonic()
                    annotated, dets, grid_meta = self._process_grid(
                        cam, up, rotation, seq, captured_at)
                    self._tl_set("grid_ms",
                                 (time.monotonic() - g0) * 1000.0)
                    e0 = time.monotonic()
                    ann_bytes = app._encode(annotated)
                    self._tl_set("encode_ms",
                                 (time.monotonic() - e0) * 1000.0)
                    h, w = up.shape[:2]
                    entry = {
                        "jpeg": ann_bytes,
                        "seq": seq,
                        "camera_index": cam,
                        "rotation": rotation,
                        "captured_at": captured_at,
                        "inference_ms": (time.monotonic() - t0) * 1000.0,
                        "detections": dets,
                        "grid": grid_meta,
                        "width": int(w),
                        "height": int(h),
                        "_order": time.monotonic(),
                    }
                    self._record_completion(
                        cam, seq, captured_at, entry, grid_meta,
                        success=True)
                except Exception as exc:
                    entry = {
                        "jpeg": None,
                        "seq": seq,
                        "camera_index": cam,
                        "rotation": rotation,
                        "captured_at": captured_at,
                        "inference_ms": (time.monotonic() - t0) * 1000.0,
                        "detections": None,
                        "width": None,
                        "height": None,
                        "error": str(exc),
                        "_order": time.monotonic(),
                    }
                    self._record_completion(
                        cam, seq, captured_at, entry, None, success=False)

                self._cache_put(cam, seq, rotation, entry)
                # Auxiliary at up to 8 fresh inferences/s; primary has no extra pacing.
                rate = float(getattr(self, "auxiliary_fps", 0)) if cam == 0 else 0.0
                idle = max(self.POST_IDLE_WAIT_SEC, 1.0/rate - (time.monotonic()-t0)) if rate > 0 else self.POST_IDLE_WAIT_SEC
                if self._closed.wait(idle):
                    break
            except Exception:
                if self._closed.wait(self.POST_IDLE_WAIT_SEC):
                    break

    # -- telemetry helpers -----------------------------------------------
    def _accept_sequence(self, cam, seq):
        if not isinstance(seq,int): return False
        with self._telemetry_lock:
            previous=self._accepted_seq.get(cam)
            if previous is not None and seq <= previous: return False
            self._accepted_seq[cam]=seq
            return True

    def _record_completion(self, cam, seq, captured_at, entry, grid_meta,
                           success):
        now_mono = time.monotonic()
        elapsed_ms = entry.get("inference_ms", 0.0) or 0.0
        grid_ms = float((grid_meta or {}).get("elapsed_ms") or 0.0)
        predict_ms = self._tl_get("predict_ms", 0.0)
        decode_ms = self._tl_get("decode_ms", 0.0)
        encode_ms = self._tl_get("encode_ms", 0.0)

        grid_valid = False
        inference_skipped = False
        if isinstance(grid_meta, dict):
            grid_valid = bool(grid_meta.get("valid"))
            inference_skipped = grid_meta.get("inference_skipped") is not False

        with self._telemetry_lock:
            if success and grid_valid and not inference_skipped:
                self._effective_total += 1
                self._processed_total += 1
                self._per_cam_effective[cam] = (
                    self._per_cam_effective.get(cam, 0) + 1)
                self._per_cam_processed[cam] = (
                    self._per_cam_processed.get(cam, 0) + 1)
                self._per_cam_last_effective_seq[cam] = seq
                self._completed.append({
                    "mono": now_mono,
                    "camera_index": cam,
                    "raw_seq": seq,
                    "captured_at": captured_at,
                    "elapsed_ms": float(elapsed_ms),
                    "grid_ms": float(grid_ms),
                    "predict_ms": float(predict_ms),
                    "decode_ms": float(decode_ms),
                    "encode_ms": float(encode_ms),
                })
            elif success and grid_valid and inference_skipped:
                # Valid grid but inference intentionally skipped: processed, not effective.
                self._processed_total += 1
                self._per_cam_processed[cam] = (
                    self._per_cam_processed.get(cam, 0) + 1)
            elif success and not grid_valid:
                self._invalid_grid_total += 1
                self._per_cam_invalid[cam] = (
                    self._per_cam_invalid.get(cam, 0) + 1)
            else:
                self._error_total += 1
                self._per_cam_error[cam] = (
                    self._per_cam_error.get(cam, 0) + 1)

    # -- metrics ---------------------------------------------------------
    def _safe_float(self, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(v):
            return 0.0
        return v

    def _percentile(self, values, pct):
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return self._safe_float(ordered[0])
        rank = (pct / 100.0) * (len(ordered) - 1)
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if lo == hi:
            return self._safe_float(ordered[lo])
        frac = rank - lo
        return self._safe_float(
            ordered[lo] * (1.0 - frac) + ordered[hi] * frac)

    def _performance_snapshot(self):
        now_mono = time.monotonic()
        with self._telemetry_lock:
            completed = list(self._completed)
            effective_total = self._effective_total
            processed_total = self._processed_total
            invalid_total = self._invalid_grid_total
            error_total = self._error_total
            per_cam_effective = dict(self._per_cam_effective)
            per_cam_processed = dict(self._per_cam_processed)
            per_cam_error = dict(self._per_cam_error)
            per_cam_invalid = dict(self._per_cam_invalid)
            per_cam_last_seq = dict(self._per_cam_last_effective_seq)

        window_start = now_mono - self.ROLLING_WINDOW_SEC
        recent = [c for c in completed if c["mono"] >= window_start]
        rolling_fps = len(recent) / self.ROLLING_WINDOW_SEC

        elapsed_list = [c["elapsed_ms"] for c in completed]
        grid_list = [c["grid_ms"] for c in completed]
        predict_list = [c["predict_ms"] for c in completed]
        decode_list = [c["decode_ms"] for c in completed]
        encode_list = [c["encode_ms"] for c in completed]

        def _stage(values):
            if not values:
                return {"mean_ms": 0.0, "p95_ms": 0.0, "count": 0}
            return {
                "mean_ms": self._safe_float(sum(values) / len(values)),
                "p95_ms": self._percentile(values, 95.0),
                "count": len(values),
            }

        cameras = []
        for cam in self._camera_indices():
            cameras.append({
                "camera_index": cam,
                "effective_total": per_cam_effective.get(cam, 0),
                "processed_total": per_cam_processed.get(cam, 0),
                "error_total": per_cam_error.get(cam, 0),
                "invalid_grid_total": per_cam_invalid.get(cam, 0),
                "last_effective_seq": per_cam_last_seq.get(cam),
                "rolling_effective_fps": sum(c["camera_index"] == cam for c in recent) / self.ROLLING_WINDOW_SEC,
            })

        return {
            "target_effective_fps": self.TARGET_EFFECTIVE_FPS,
            "capture_target_fps": self.CAPTURE_TARGET_FPS,
            "mode": "parallel_per_camera_single_gpu",
            "effective_total": effective_total,
            "processed_total": processed_total,
            "invalid_grid_total": invalid_total,
            "error_total": error_total,
            "rolling_effective_fps": self._safe_float(rolling_fps),
            "rolling_window_sec": self.ROLLING_WINDOW_SEC,
            "uptime_sec": self._safe_float(now_mono - self._started_at),
            "cameras": cameras,
            "latency": {
                "total": _stage(elapsed_list),
                "grid": _stage(grid_list),
                "predict": _stage(predict_list),
                "decode": _stage(decode_list),
                "encode": _stage(encode_list),
            },
        }

    def status(self):
        base = super().status()
        base["performance"] = self._performance_snapshot()
        return base
