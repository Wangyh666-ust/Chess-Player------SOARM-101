import math
import threading
import time
from collections import deque

import dual_camera_preview_v3 as base

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
TARGET_FPS = 30
JPEG_QUALITY = 75
PERF_WINDOW = 240
FPS_WINDOW_S = 2.0


def _empty_latency():
    return {"mean_ms": None, "p95_ms": None, "count": 0}


def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = (len(sorted_vals) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


def _latency_stats(vals):
    if not vals:
        return _empty_latency()
    ordered = sorted(vals)
    return {
        "mean_ms": sum(ordered) / float(len(ordered)),
        "p95_ms": _percentile(ordered, 0.95),
        "count": len(ordered),
    }


class CameraWorker(base.CameraWorker):
    """Camera worker with decoupled capture and JPEG encode threads."""

    def __init__(self, device, label, cv2_module=None):
        super().__init__(device, label, cv2_module)
        self._cv2_impl = cv2_module
        self._raw_lock = threading.Lock()
        self._raw_bgr = None
        self._raw_seq = 0
        self._raw_width = 0
        self._raw_height = 0
        self._raw_captured_at = None
        self._raw_captured_mono = None
        self._raw_error = None

        self._enc_lock = threading.Lock()
        self._enc_jpeg = None
        self._enc_bgr = None
        self._enc_seq = 0
        self._enc_width = 0
        self._enc_height = 0
        self._enc_captured_at = None
        self._enc_captured_mono = None
        self._jpeg_error = None

        self._stop = threading.Event()
        self._wake = threading.Event()
        self._capture_thread = None
        self._encoder_thread = None

        self._metrics_lock = threading.Lock()
        self._read_ms = deque(maxlen=PERF_WINDOW)
        self._grab_ms = deque(maxlen=PERF_WINDOW)
        self._retrieve_ms = deque(maxlen=PERF_WINDOW)
        self._raw_publish_ms = deque(maxlen=PERF_WINDOW)
        self._jpeg_encode_ms = deque(maxlen=PERF_WINDOW)
        self._raw_times = deque(maxlen=PERF_WINDOW * 4)
        self._jpeg_times = deque(maxlen=PERF_WINDOW * 4)
        self._raw_count = 0
        self._jpeg_count = 0
        self._last_raw_seq = 0
        self._last_jpeg_seq = 0
        self._negotiated = {
            "fps": None,
            "width": None,
            "height": None,
            "fourcc": None,
        }
        self._capture_error = None

    def start(self):
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return
        if self._encoder_thread is not None and self._encoder_thread.is_alive():
            return
        self._stop.clear()
        self._wake.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_run,
            name="cap-{}".format(self.label),
            daemon=True,
        )
        self._encoder_thread = threading.Thread(
            target=self._encode_run,
            name="enc-{}".format(self.label),
            daemon=True,
        )
        self._capture_thread.start()
        self._encoder_thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        for t in (self._capture_thread, self._encoder_thread):
            if t is not None:
                t.join(timeout=3.0)

    def raw_snapshot(self):
        with self._raw_lock:
            bgr = self._raw_bgr
            seq = self._raw_seq
            width = self._raw_width
            height = self._raw_height
            captured_at = self._raw_captured_at
            captured_mono = self._raw_captured_mono
            error = self._capture_error
        age_ms = None
        if captured_mono is not None:
            age_ms = (time.monotonic() - captured_mono) * 1000.0
        return {
            "jpeg": None,
            "bgr": bgr,
            "seq": seq,
            "label": self.label,
            "device": self.device,
            "width": width,
            "height": height,
            "captured_at": captured_at,
            "age_ms": age_ms,
            "error": error,
            "capture_performance": self._capture_performance(),
        }

    def snapshot(self):
        with self._enc_lock:
            jpeg = self._enc_jpeg
            bgr = self._enc_bgr
            seq = self._enc_seq
            width = self._enc_width
            height = self._enc_height
            captured_at = self._enc_captured_at
            captured_mono = self._enc_captured_mono
        with self._raw_lock:
            capture_error = self._capture_error
        with self._metrics_lock:
            jpeg_error = self._jpeg_error
        age_ms = None
        if captured_mono is not None:
            age_ms = (time.monotonic() - captured_mono) * 1000.0
        error = capture_error if capture_error is not None else jpeg_error
        return {
            "jpeg": jpeg,
            "bgr": bgr,
            "seq": seq,
            "label": self.label,
            "device": self.device,
            "width": width,
            "height": height,
            "captured_at": captured_at,
            "age_ms": age_ms,
            "error": error,
            "capture_performance": self._capture_performance(),
        }

    def _capture_performance(self):
        now = time.monotonic()
        with self._metrics_lock:
            read_vals = list(self._read_ms)
            grab_vals = list(self._grab_ms)
            retrieve_vals = list(self._retrieve_ms)
            pub_vals = list(self._raw_publish_ms)
            enc_vals = list(self._jpeg_encode_ms)
            raw_count = self._raw_count
            jpeg_count = self._jpeg_count
            last_raw_seq = self._last_raw_seq
            last_jpeg_seq = self._last_jpeg_seq
            raw_times = list(self._raw_times)
            jpeg_times = list(self._jpeg_times)
            negotiated = dict(self._negotiated)
            jpeg_error = self._jpeg_error
        cutoff = now - FPS_WINDOW_S
        raw_fresh = sum(1 for t in raw_times if t >= cutoff)
        jpeg_fresh = sum(1 for t in jpeg_times if t >= cutoff)
        return {
            "raw_count": raw_count,
            "jpeg_count": jpeg_count,
            "last_raw_seq": last_raw_seq,
            "last_jpeg_seq": last_jpeg_seq,
            "raw_fps_2s": raw_fresh / FPS_WINDOW_S,
            "jpeg_fps_2s": jpeg_fresh / FPS_WINDOW_S,
            "read_ms": _latency_stats(read_vals),
            "grab_ms": _latency_stats(grab_vals),
            "retrieve_ms": _latency_stats(retrieve_vals),
            "raw_publish_ms": _latency_stats(pub_vals),
            "jpeg_encode_ms": _latency_stats(enc_vals),
            "negotiated": negotiated,
            "jpeg_error": jpeg_error,
        }

    def _set_capture_error(self, message):
        with self._raw_lock:
            self._capture_error = message

    def _resolve_cv2(self):
        cv2 = self._cv2_impl
        if cv2 is None:
            try:
                import cv2 as _cv2
                cv2 = _cv2
            except Exception as exc:
                self._set_capture_error("无法导入 cv2: {}".format(exc))
                return None
            self._cv2_impl = cv2
        return cv2

    def _get_prop(self, cv2, cap, name):
        prop = getattr(cv2, name, None)
        if prop is None:
            return None
        try:
            val = cap.get(prop)
        except Exception:
            return None
        try:
            f = float(val)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    def _negotiate(self, cv2, cap):
        fps = self._get_prop(cv2, cap, "CAP_PROP_FPS")
        width = self._get_prop(cv2, cap, "CAP_PROP_FRAME_WIDTH")
        height = self._get_prop(cv2, cap, "CAP_PROP_FRAME_HEIGHT")
        fourcc = None
        fourcc_raw = self._get_prop(cv2, cap, "CAP_PROP_FOURCC")
        if fourcc_raw is not None:
            try:
                iv = int(fourcc_raw) & 0xFFFFFFFF
                chars = [chr((iv >> (8 * i)) & 0xFF) for i in range(4)]
                fourcc = "".join(chars)
            except Exception:
                fourcc = None
        with self._metrics_lock:
            self._negotiated = {
                "auto_exposure_raw": self._get_prop(cv2, cap, "CAP_PROP_AUTO_EXPOSURE"),
                "exposure_raw": self._get_prop(cv2, cap, "CAP_PROP_EXPOSURE"),
                "fps": fps,
                "width": width,
                "height": height,
                "fourcc": fourcc,
            }

    def _capture_run(self):
        cv2 = self._resolve_cv2()
        if cv2 is None:
            self._stop.set()
            self._wake.set()
            return
        cap = None
        try:
            try:
                cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            except Exception as exc:
                self._set_capture_error("打开摄像头失败: {}".format(exc))
                return
            try:
                if not cap.isOpened():
                    self._set_capture_error("打不开摄像头")
                    return
            except Exception as exc:
                self._set_capture_error("打开摄像头失败: {}".format(exc))
                return

            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            except Exception:
                pass
            bufsize = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
            if bufsize is not None:
                try:
                    cap.set(bufsize, 1)
                except Exception:
                    pass

            self._negotiate(cv2, cap)

            while not self._stop.is_set():
                t0 = time.monotonic()
                try:
                    if hasattr(cap, "grab") and hasattr(cap, "retrieve"):
                        ok = cap.grab()
                        grabbed_at = time.monotonic()
                        ok, frame = cap.retrieve() if ok else (False, None)
                    else:
                        ok, frame = cap.read()
                        grabbed_at = None
                except Exception as exc:
                    self._set_capture_error("读取失败: {}".format(exc))
                    break
                t1 = time.monotonic()
                if not ok or frame is None:
                    self._set_capture_error("读取帧失败")
                    break
                if self._stop.is_set():
                    break

                captured_at = time.time()
                captured_mono = t1
                try:
                    bgr = frame.copy()
                except Exception as exc:
                    self._set_capture_error("读取失败: {}".format(exc))
                    break

                shape = getattr(bgr, "shape", None)
                if shape is not None and len(shape) >= 2:
                    height = int(shape[0])
                    width = int(shape[1])
                else:
                    width = DEFAULT_WIDTH
                    height = DEFAULT_HEIGHT

                with self._raw_lock:
                    self._raw_seq += 1
                    seq = self._raw_seq
                    self._raw_bgr = bgr
                    self._raw_width = width
                    self._raw_height = height
                    self._raw_captured_at = captured_at
                    self._raw_captured_mono = captured_mono
                    self._raw_error = None

                t2 = time.monotonic()

                with self._metrics_lock:
                    self._raw_count += 1
                    self._last_raw_seq = seq
                    self._read_ms.append((t1 - t0) * 1000.0)
                    if grabbed_at is not None:
                        self._grab_ms.append((grabbed_at-t0)*1000.0)
                        self._retrieve_ms.append((t1-grabbed_at)*1000.0)
                    self._raw_publish_ms.append((t2 - t1) * 1000.0)
                    self._raw_times.append(captured_mono)

                self._wake.set()
        except Exception as exc:
            self._set_capture_error("采集异常: {}".format(exc))
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self._stop.set()
            self._wake.set()

    def _encode_run(self):
        while True:
            if self._stop.is_set():
                return
            self._wake.wait(timeout=0.1)
            self._wake.clear()
            if self._stop.is_set():
                return

            cv2 = self._resolve_cv2()
            if cv2 is None:
                return

            with self._raw_lock:
                bgr = self._raw_bgr
                seq = self._raw_seq
                width = self._raw_width
                height = self._raw_height
                captured_at = self._raw_captured_at
                captured_mono = self._raw_captured_mono
                capture_error = self._capture_error

            if bgr is None or seq == 0:
                if capture_error is not None:
                    with self._metrics_lock:
                        last_jpeg_seq = self._last_jpeg_seq
                    if last_jpeg_seq >= 0 and capture_error is not None:
                        pass
                continue

            with self._metrics_lock:
                last_jpeg_seq = self._last_jpeg_seq
            if seq <= last_jpeg_seq:
                continue

            t0 = time.monotonic()
            try:
                enc_ok, encoded = cv2.imencode(
                    ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
            except Exception as exc:
                with self._metrics_lock:
                    self._jpeg_error = "JPEG 编码失败: {}".format(exc)
                return
            if not enc_ok:
                with self._metrics_lock:
                    self._jpeg_error = "JPEG 编码失败"
                return
            try:
                jpeg_bytes = encoded.tobytes()
            except Exception as exc:
                with self._metrics_lock:
                    self._jpeg_error = "JPEG 编码失败: {}".format(exc)
                return
            t1 = time.monotonic()

            if self._stop.is_set():
                return

            with self._raw_lock:
                current_seq = self._raw_seq
            with self._enc_lock:
                if seq > self._enc_seq:
                    self._enc_jpeg = jpeg_bytes
                    self._enc_bgr = bgr
                    self._enc_seq = seq
                    self._enc_width = width
                    self._enc_height = height
                    self._enc_captured_at = captured_at
                    self._enc_captured_mono = captured_mono

            with self._metrics_lock:
                self._jpeg_count += 1
                self._last_jpeg_seq = seq
                self._jpeg_encode_ms.append((t1 - t0) * 1000.0)
                self._jpeg_times.append(t1)
                if self._jpeg_error is not None:
                    self._jpeg_error = None
            _ = current_seq
            if self._stop.is_set():
                return
            if not self._wake.is_set():
                continue
