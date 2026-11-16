"""Dual-camera collection preview service (v3).

Extends the existing dual_camera_preview_v2 HTTP preview server with:

* bounded, on-demand YOLO inference leases (only when a client asks for it),
* a user-triggered paired snapshot capture (POST /capture), saving originals,
  rotated upright frames, YOLO annotations and a pair.json into a unique
  directory under a configured root, plus a ZIP download link.

No serial access, no robot control, no experiment driver, no training code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import threading

try:
    from live_weighted_grid_bridge_v2 import GridBridge
except Exception as _grid_bridge_import_error:  # pragma: no cover
    GridBridge = None
    _GRID_BRIDGE_IMPORT_ERROR = _grid_bridge_import_error
else:
    _GRID_BRIDGE_IMPORT_ERROR = None
import time
import uuid
import zipfile
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


# ---------------------------------------------------------------------------
# Base module (same directory). Imported lazily-ish at module import time since
# it is a hard dependency of this adapter, but the YOLO/ultralytics import is
# deferred until an actual model needs to be built.
# ---------------------------------------------------------------------------
import dual_camera_preview_v2 as base  # noqa: E402


CONFIG_NAME = "collection_config.json"

ROTATIONS = (0, 90, 180, 270)
STALE_MS = 2000.0
LEASE_TTL_SEC = 3.0
SKEW_WARN_MS = 250.0
FREE_SPACE_MIN = 2 * 1024 * 1024 * 1024  # 2 GB
MAX_BODY = 4096
VIEW_TIMEOUT_HEADERS = True

DEFAULT_IMGSZ = 960
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.5
DEVICE = 0
HALF = False
VERBOSE = False

JPEG_QUALITY = 92


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_rotation(value) -> int:
    if isinstance(value, bool):
        raise ValueError("rotation 必须是整数 0/90/180/270")
    if isinstance(value, float):
        if not math.isfinite(value) or value != int(value):
            raise ValueError("rotation 必须是整数 0/90/180/270")
        rot = int(value)
    elif isinstance(value, int):
        rot = value
    elif isinstance(value, str):
        try:
            rot = int(value.strip(), 10)
        except (TypeError, ValueError):
            raise ValueError("rotation 必须是整数 0/90/180/270")
    else:
        raise ValueError("rotation 必须是整数 0/90/180/270")
    if rot not in ROTATIONS:
        raise ValueError("rotation 必须是 0/90/180/270 之一")
    return rot


def _rot90_k(rotation: int) -> int:
    return (-rotation // 90) % 4


def _rotate_np(img, rotation: int):
    import numpy as np

    if rotation == 0:
        return img
    return np.rot90(img, k=_rot90_k(rotation))


def _rotate_jpeg_upright(jpeg: bytes, rotation: int) -> tuple[bytes, int, int]:
    import cv2
    import numpy as np

    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("无法解码图像")
    up = _rotate_np(img, rotation)
    ok, enc = cv2.imencode(".jpg", up, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("无法编码图像")
    h, w = up.shape[:2]
    return enc.tobytes(), int(w), int(h)


def _decode(jpeg: bytes):
    import cv2
    import numpy as np

    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("无法解码图像")
    return img


def _encode(img) -> bytes:
    import cv2

    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("无法编码图像")
    return enc.tobytes()


def _load_config(server_dir: str) -> dict:
    path = os.path.join(server_dir, CONFIG_NAME)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("collection_config.json 必须是对象")
    out = {
        "model_path": data.get("model_path"),
        "model_sha256": data.get("model_sha256"),
        "capture_root": data.get("capture_root"),
        "imgsz": int(data.get("imgsz", DEFAULT_IMGSZ)),
        "conf": float(data.get("conf", DEFAULT_CONF)),
        "iou": float(data.get("iou", DEFAULT_IOU)),
    }
    for key in ("model_path", "model_sha256", "capture_root"):
        if not out.get(key):
            raise ValueError(f"collection_config.json 缺少 {key}")
    return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def _canonicalize_capture_root(root):
    if not isinstance(root, str):
        raise ValueError("采集目录必须是非空字符串")
    if not root:
        raise ValueError("采集目录不能为空")
    if any(ord(c) < 32 or ord(c) == 127 for c in root):
        raise ValueError("采集目录包含非法控制字符")
    if "\x00" in root:
        raise ValueError("采集目录包含非法字符")
    expanded = os.path.expanduser(root)
    if not os.path.isabs(expanded):
        raise ValueError("采集目录必须是绝对路径")
    if os.name != "nt":
        drive = os.path.splitdrive(expanded)[0]
        if drive and not drive.startswith("//"):
            raise ValueError("采集目录不能使用 Windows 盘符路径")
    norm = os.path.realpath(os.path.normpath(expanded))
    if not os.path.isabs(norm):
        raise ValueError("采集目录必须是绝对路径")
    return norm


def _write_exclusive(path, data):
    # Create per-capture files with exclusive mode so an existing file with
    # the same (timestamp/prefix-unique) name is never overwritten.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    try:
        with os.fdopen(fd, "wb") as f:
            fd = None
            f.write(data)
    finally:
        if fd is not None:
            os.close(fd)


def _resolve_capture_root(requested, default_root):
    if requested is None:
        return default_root
    return _canonicalize_capture_root(requested)


class CollectionEngine:
    def __init__(self, feeds, model_path, model_sha256, capture_root,
                 model_factory=None, imgsz=DEFAULT_IMGSZ, conf=DEFAULT_CONF,
                 iou=DEFAULT_IOU):
        self.feeds = feeds
        self.model_path = model_path
        self.model_sha256 = str(model_sha256).lower()
        _default_root = _canonicalize_capture_root(capture_root)
        self.capture_root = _default_root
        self._default_capture_root = _default_root
        self._capture_roots = [self._default_capture_root]
        self._read_capture_root_registry()
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = DEVICE
        self.half = HALF
        self.verbose = VERBOSE
        self._model_factory = model_factory

        self._predict_lock = threading.Lock()  # one GPU lock: creation + predict
        self._model = None
        self._model_error = None
        self._model_loaded = False
        self._load_failed_at = None
        self._model_names = {}

        # inference cache: keyed by (camera_index, seq, rotation)
        self._cache_lock = threading.Lock()
        self._results = {}  # key -> dict(result fields)
        self._max_results = 2

        # leases: camera_index -> expiry monotonic + rotation
        self._lease_lock = threading.Lock()
        self._leases = {}  # cam -> {"expires": mono, "rotation": rot}

        # capture
        self._capture_lock = threading.Lock()

        self._closed = threading.Event()
        self._worker = None
        self._worker_start_lock = threading.Lock()
        self._grid = None
        self._grid_error = None
        if GridBridge is None:
            self._grid_error = "live_grid_bridge_v4 import failed: %s" % (
                _GRID_BRIDGE_IMPORT_ERROR,)
        else:
            _grid_config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "grid_config.json",
            )
            if not os.path.exists(_grid_config_path):
                self._grid_error = (
                    "missing grid config: %s" % _grid_config_path)
            else:
                try:
                    self._grid = GridBridge(_grid_config_path)
                except Exception as exc:
                    self._grid = None
                    self._grid_error = "grid bridge init failed: %s" % (exc,)

    # -- lifecycle -------------------------------------------------------
    def _ensure_worker(self):
        with self._worker_start_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            if self._closed.is_set():
                return
            t = threading.Thread(target=self._worker_loop, name="collection-infer", daemon=True)
            self._worker = t
            t.start()

    def close(self):
        self._closed.set()
        w = self._worker
        if w is not None:
            try:
                w.join(timeout=2.0)
            except Exception:
                pass

    # -- leases ----------------------------------------------------------
    def _renew_lease(self, cam: int, rotation: int):
        with self._lease_lock:
            self._leases[cam] = {
                "expires": time.monotonic() + LEASE_TTL_SEC,
                "rotation": rotation,
            }

    def _active_leases(self) -> dict:
        now = time.monotonic()
        out = {}
        with self._lease_lock:
            for cam, info in list(self._leases.items()):
                if info["expires"] >= now:
                    out[cam] = info["rotation"]
                else:
                    self._leases.pop(cam, None)
        return out

    # -- model -----------------------------------------------------------
    def _verify_hash(self, path: str):
        if not os.path.isfile(path):
            raise RuntimeError("模型文件不存在")
        actual = _sha256_file(path)
        if actual.lower() != self.model_sha256:
            raise RuntimeError("模型 SHA256 校验失败")

    def _build_model(self):
        if self._model_factory is not None:
            return self._model_factory(self.model_path)
        from ultralytics import YOLO  # lazy

        return YOLO(self.model_path)

    def _load_model_locked(self):
        """Caller must hold _predict_lock."""
        if self._model is not None:
            return self._model
        if self._load_failed_at is not None:
            if (time.monotonic() - self._load_failed_at) < 5.0:
                raise RuntimeError("YOLO加载失败（冷却中）: %s" % (self._model_error,))
        self._model_error = None
        try:
            self._verify_hash(self.model_path)
            model = self._build_model()
            names = getattr(model, "names", None)
            if not isinstance(names, dict):
                try:
                    names = dict(names)
                except Exception:
                    names = {}
            self._model_names = names
            self._model = model
            self._model_loaded = True
            self._load_failed_at = None
        except Exception as exc:
            self._model = None
            self._model_error = str(exc)
            self._load_failed_at = time.monotonic()
            raise RuntimeError("YOLO加载失败: %s" % (exc,))
        return self._model

    def _predict(self, img):
        from yolo_full_scores_v1 import predict_detections

        with self._predict_lock:
            model = self._load_model_locked()
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
            return dets

    def _predict_annotated(self, img, dets):
        import cv2

        out = img.copy()
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            cv2.rectangle(out, p1, p2, (0, 255, 0), 2)
            label = "%s %.2f" % (d["class_name"], d["confidence"])
            ty = max(p1[1] - 6, 12)
            cv2.putText(out, label, (p1[0], ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)
        return out

    # -- snapshot helpers ------------------------------------------------
    @staticmethod
    def _snap_usable(snap: dict):
        jpeg = snap.get("jpeg")
        error = snap.get("error")
        age_ms = snap.get("age_ms")
        if jpeg is None or error is not None:
            return False
        if age_ms is None:
            return False
        try:
            age = float(age_ms)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(age):
            return False
        if age < 0 or age > STALE_MS:
            return False
        ts = snap.get("captured_at")
        if ts is None:
            return False
        try:
            ts_f = float(ts)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(ts_f):
            return False
        return True

    def _cache_get(self, cam, seq, rotation):
        with self._cache_lock:
            return self._results.get((cam, seq, rotation))

    def _cache_latest(self, cam, rotation):
        with self._cache_lock:
            best = None
            best_order = None
            for (k_cam, _k_seq, k_rot), v in self._results.items():
                if k_cam != cam or k_rot != rotation:
                    continue
                order = v.get("_order", 0)
                if best_order is None or order > best_order:
                    best_order = order
                    best = v
            return best

    def _cache_put(self, cam, seq, rotation, entry):
        with self._cache_lock:
            # Remove all older entries for the SAME camera (any seq/rotation) first.
            stale = [k for k in self._results if k[0] == cam]
            for k in stale:
                self._results.pop(k, None)
            self._results[(cam, seq, rotation)] = entry
            while len(self._results) > self._max_results:
                oldest_key = None
                oldest_ts = None
                cur_key = (cam, seq, rotation)
                for k, v in self._results.items():
                    if k == cur_key:
                        continue
                    ts = v.get("_order", 0)
                    if oldest_ts is None or ts < oldest_ts:
                        oldest_ts = ts
                        oldest_key = k
                if oldest_key is None:
                    break
                self._results.pop(oldest_key, None)

    # -- worker ----------------------------------------------------------
    def _worker_loop(self):
        order = [0, 1]
        idx = 0
        while not self._closed.is_set():
            try:
                leases = self._active_leases()
                if not leases:
                    if self._closed.wait(0.1):
                        break
                    continue
                # alternating order
                cam = None
                for _ in range(2):
                    candidate = order[idx % 2]
                    idx += 1
                    if candidate in leases:
                        cam = candidate
                        break
                if cam is None:
                    if self._closed.wait(0.1):
                        break
                    continue
                rotation = leases[cam]
                snap = self.feeds[cam].snapshot()
                if not self._snap_usable(snap):
                    if self._closed.wait(0.1):
                        break
                    continue
                seq = snap.get("seq")
                key = (cam, seq, rotation)
                if self._cache_get(cam, seq, rotation) is not None:
                    if self._closed.wait(0.1):
                        break
                    continue
                captured_at = snap.get("captured_at")
                source_jpeg = snap["jpeg"]
                t0 = time.monotonic()
                try:
                    img = _decode(source_jpeg)
                    up = _rotate_np(img, rotation)
                    annotated, dets, grid_meta = self._process_grid(
                        cam, up, rotation, seq, captured_at)
                    ann_bytes = _encode(annotated)
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
                self._cache_put(cam, seq, rotation, entry)
                if self._closed.wait(0.1):
                    break
            except Exception:
                if self._closed.wait(0.1):
                    break

    # -- view ------------------------------------------------------------
    def view(self, camera_index: int, rotation: int, yolo) -> tuple[bytes, dict]:
        cam = int(camera_index)
        if cam not in (0, 1):
            raise ValueError("camera_index 必须是 0 或 1")
        rot = _validate_rotation(rotation)
        want_yolo = bool(yolo)
        snap = self.feeds[cam].snapshot()

        if not want_yolo:
            if not self._snap_usable(snap):
                raise RuntimeError("画面暂不可用")
            jpeg = snap["jpeg"]
            meta = {
                "seq": snap.get("seq"),
                "camera_index": cam,
                "rotation": rot,
                "captured_at": snap.get("captured_at"),
                "age_ms": snap.get("age_ms"),
                "inference_ms": None,
                "detections": None,
                "width": snap.get("width"),
                "height": snap.get("height"),
            }
            if rot != 0:
                up_bytes, w, h = _rotate_jpeg_upright(jpeg, rot)
                meta["width"] = w
                meta["height"] = h
                return up_bytes, meta
            return jpeg, meta

        # yolo path: renew lease, may need to start worker
        self._renew_lease(cam, rot)
        self._ensure_worker()

        latest = self.feeds[cam].snapshot()
        # Feed must be healthy right now even if we return an older completed result.
        if latest.get("error") is not None:
            raise RuntimeError("画面暂不可用，YOLO等待中")
        feed_ts = latest.get("captured_at")
        try:
            feed_ts_f = float(feed_ts)
        except (TypeError, ValueError):
            raise RuntimeError("画面暂不可用，YOLO等待中")
        if not math.isfinite(feed_ts_f):
            raise RuntimeError("画面暂不可用，YOLO等待中")
        if (time.time() - feed_ts_f) * 1000.0 > STALE_MS:
            raise RuntimeError("画面暂不可用，YOLO等待中")

        entry = self._cache_latest(cam, rot)
        if entry is None:
            raise RuntimeError("YOLO加载中或等待结果")
        if entry.get("error"):
            raise RuntimeError("YOLO推理失败: %s" % (entry["error"],))
        jpeg = entry.get("jpeg")
        if jpeg is None:
            raise RuntimeError("YOLO加载中或等待结果")
        captured_at = entry.get("captured_at")
        try:
            captured_at_f = float(captured_at)
        except (TypeError, ValueError):
            raise RuntimeError("YOLO结果时间戳无效")
        if not math.isfinite(captured_at_f):
            raise RuntimeError("YOLO结果时间戳无效")
        age = (time.time() - captured_at_f) * 1000.0
        if age > STALE_MS or age < -STALE_MS:
            raise RuntimeError("YOLO结果过期")
        if self._model_error:
            # model error is surfaced elsewhere; but if cached entry is good keep it
            pass
        meta = {
            "seq": entry.get("seq"),
            "camera_index": cam,
            "rotation": rot,
            "captured_at": captured_at,
            "age_ms": age,
            "inference_ms": entry.get("inference_ms"),
            "detections": entry.get("detections") or [],
            "width": entry.get("width"),
            "height": entry.get("height"),
        }
        return jpeg, meta

    # -- status ----------------------------------------------------------
    def _process_grid(self, cam, img, rotation, seq, captured_at,
                      remember=True):
        if self._grid is not None:
            return self._grid.process(
                cam, img, rotation, self._predict,
                seq=seq, captured_at=captured_at, remember=remember)
        import cv2
        out = img.copy()
        cv2.putText(out, "GRID UNAVAILABLE - YOLO paused", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out, [], {"valid": False, "reason": self._grid_error, "inference_skipped": True}

    def status(self) -> dict:
        short = self.model_sha256[:12] if self.model_sha256 else ""
        return {
            "model_loaded": self._model is not None,
            "model_error": self._model_error,
            "model_short_sha": short,
            "model_path": self.model_path,
            "capture_root": self.capture_root,
            "capture_roots": list(self._capture_roots),
            "params": {
                "imgsz": self.imgsz,
                "conf": self.conf,
                "iou": self.iou,
                "device": self.device,
                "half": self.half,
            },
            "grid_pipeline": (
                self._grid.status() if self._grid is not None
                else {"enabled": False, "error": self._grid_error}),
        }

    # -- capture root registry ------------------------------------------
    def _registry_path(self):
        return os.path.join(
            self._default_capture_root, ".capture_roots.json")

    def _read_capture_root_registry(self):
        path = self._registry_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "采集目录注册表损坏: %s" % (exc,))
        if not isinstance(data, dict):
            raise RuntimeError("采集目录注册表损坏: 顶层不是对象")
        roots = data.get("roots")
        if not isinstance(roots, list):
            raise RuntimeError("采集目录注册表损坏: roots 不是数组")
        merged = [self._default_capture_root]
        for entry in roots:
            if not isinstance(entry, str) or not entry:
                raise RuntimeError("采集目录注册表损坏: 条目无效")
            try:
                canon = _canonicalize_capture_root(entry)
            except ValueError as exc:
                raise RuntimeError("采集目录注册表损坏: %s" % (exc,))
            if canon not in merged:
                merged.append(canon)
        self._capture_roots = merged

    def _write_capture_root_registry(self):
        path = self._registry_path()
        tmp_dir = os.path.dirname(path)
        try:
            os.makedirs(tmp_dir, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=".capture_roots.", suffix=".tmp", dir=tmp_dir)
        except OSError as exc:
            raise RuntimeError("无法写入采集目录注册表: %s" % (exc,))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"roots": list(self._capture_roots)}, f,
                          ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except OSError as exc:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            raise RuntimeError("无法写入采集目录注册表: %s" % (exc,))

    def _register_capture_root(self, root):
        if root in self._capture_roots:
            return
        old_list = self._capture_roots
        new_list = list(old_list)
        new_list.append(root)
        self._capture_roots = new_list
        try:
            self._write_capture_root_registry()
        except Exception:
            # Roll back in-memory state so a retry can still persist the root
            # and previously known roots/links are preserved.
            self._capture_roots = old_list
            raise

    def download_roots(self):
        return list(self._capture_roots)

    # -- capture ---------------------------------------------------------
    def capture(self, rotations, capture_root=None) -> dict:
        if not isinstance(rotations, (list, tuple)) or len(rotations) != 2:
            raise ValueError("rotations 必须是长度为 2 的数组")
        rots = [_validate_rotation(r) for r in rotations]
        requested_root = _resolve_capture_root(capture_root, self._default_capture_root)

        if not self._capture_lock.acquire(blocking=False):
            raise RuntimeError("capture busy")
        try:
            return self._capture_locked(rots, requested_root)
        finally:
            self._capture_lock.release()

    def _capture_locked(self, rots, capture_root):
        # gather current clean snapshots FIRST
        snaps = []
        for cam in (0, 1):
            snap = self.feeds[cam].snapshot()
            if not self._snap_usable(snap):
                raise RuntimeError("camera%d 画面不可用或过期，无法拍摄" % cam)
            snaps.append(snap)

        originals = [bytes(s["jpeg"]) for s in snaps]
        times = []
        for s in snaps:
            try:
                times.append(float(s.get("captured_at")))
            except (TypeError, ValueError):
                raise RuntimeError("画面上报时间戳无效")
        skew_ms = abs(times[0] - times[1]) * 1000.0
        warning = None
        if skew_ms > SKEW_WARN_MS:
            warning = "两摄像头时间差较大 (%.0f ms)，非硬件同步" % skew_ms

        # free space check on the requested local root
        try:
            os.makedirs(capture_root, exist_ok=True)
            free = shutil.disk_usage(capture_root).free
        except OSError as exc:
            raise RuntimeError("无法访问采集目录: %s" % (exc,))
        if free < FREE_SPACE_MIN:
            raise RuntimeError("采集目录可用空间不足 2GB")

        # register chosen root under capture lock BEFORE folder creation
        self._register_capture_root(capture_root)

        cap_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
        # Flat layout: every capture file is written directly into capture_root
        # with a capture-prefixed basename. No per-capture subdirectory.
        server_dir = capture_root

        # Pre-write ALL originals and ALL rotated-upright frames before any model work.
        prepared = []
        for cam in (0, 1):
            rot = rots[cam]
            snap = snaps[cam]
            orig = originals[cam]
            upright_bytes, up_w, up_h = _rotate_jpeg_upright(orig, rot)
            camera_number = cam + 1
            orig_name = "%s_camera%d_original.jpg" % (cap_id, camera_number)
            upright_name = "%s_camera%d_upright.jpg" % (cap_id, camera_number)
            orig_path = os.path.join(server_dir, orig_name)
            upright_path = os.path.join(server_dir, upright_name)
            _write_exclusive(orig_path, orig)
            _write_exclusive(upright_path, upright_bytes)
            prepared.append({
                "cam": cam,
                "rot": rot,
                "snap": snap,
                "orig": orig,
                "upright_bytes": upright_bytes,
                "up_w": up_w,
                "up_h": up_h,
                "orig_name": orig_name,
                "upright_name": upright_name,
            })

        cameras_json = []
        infer_errors = []
        for item in prepared:
            cam = item["cam"]
            rot = item["rot"]
            snap = item["snap"]
            orig = item["orig"]
            upright_bytes = item["upright_bytes"]
            up_w = item["up_w"]
            up_h = item["up_h"]
            orig_dims = (snap.get("width"), snap.get("height"))
            camera_number = cam + 1

            dets = None
            pred_error = None
            grid_meta = None
            yolo_name = "%s_camera%d_yolo.jpg" % (cap_id, camera_number)
            yolo_path = os.path.join(server_dir, yolo_name)
            try:
                img = _decode(upright_bytes)
                annotated, dets, grid_meta = self._process_grid(
                    cam, img, rot, snap.get("seq"),
                    snap.get("captured_at"), remember=False)
                ann_bytes = _encode(annotated)
                _write_exclusive(yolo_path, ann_bytes)
                yolo_file = yolo_name
            except Exception as exc:
                pred_error = str(exc)
                infer_errors.append("camera%d: %s" % (camera_number, exc))
                yolo_file = None
                dets = None

            cam_entry = {
                "camera_index": cam,
                "camera_number": camera_number,
                "device_identity": snap.get("label"),
                "label": snap.get("label"),
                "rotation_clockwise": rot,
                "seq": snap.get("seq"),
                "captured_at": snap.get("captured_at"),
                "age_ms": snap.get("age_ms"),
                "source_sha256": _sha256_bytes(orig),
                "source_jpeg_bytes": len(orig),
                "original_file": item["orig_name"],
                "upright_file": item["upright_name"],
                "yolo_file": yolo_file,
                "dimensions_original": {
                    "width": int(orig_dims[0]) if orig_dims[0] is not None else None,
                    "height": int(orig_dims[1]) if orig_dims[1] is not None else None,
                },
                "dimensions_upright": {"width": int(up_w), "height": int(up_h)},
                "detections": dets,
                "prediction_error": pred_error,
                "grid": grid_meta,
                "context": "预测为机器自动检测，未人工确认",
            }
            cameras_json.append(cam_entry)

        if infer_errors:
            msg = "部分推理失败: " + "; ".join(infer_errors)
            warning = (warning + "；" + msg) if warning else msg

        pair = {
            "id": cap_id,
            "created_at": _now_iso(),
            "model_path": self.model_path,
            "model_sha256": self.model_sha256,
            "capture_root": capture_root,
            "params": {
                "imgsz": self.imgsz,
                "conf": self.conf,
                "iou": self.iou,
                "device": self.device,
                "half": self.half,
                "verbose": self.verbose,
            },
            "cameras": cameras_json,
            "skew_ms": skew_ms,
            "skew_warning": warning,
            "notes": {
                "labels_read_only": True,
                "labels_human_confirmed": False,
                "coordinates": "upright_image_pixels",
                "timestamps": "host_encode_time",
                "synchronized_hardware": False,
                "frames": "jpeg_not_sensor_raw",
            },
        }
        pair_name = "%s_pair.json" % cap_id
        pair_file = pair_name
        pair_path = os.path.join(server_dir, pair_name)
        pair_bytes = json.dumps(pair, ensure_ascii=False, indent=2).encode("utf-8")
        _write_exclusive(pair_path, pair_bytes)

        zip_name = "%s.zip" % cap_id
        zip_path = os.path.join(capture_root, zip_name)
        tmp_zip = zip_path + ".tmp"
        if os.path.exists(zip_path) or os.path.lexists(tmp_zip):
            raise RuntimeError("目标 ZIP 名称已存在，拒绝覆盖: %s" % zip_name)
        # Explicitly assemble members for THIS capture only. Never list the
        # shared root: no older photos, ZIPs or unrelated files can leak in.
        members = []
        for item in prepared:
            members.append((item["orig_name"], os.path.join(server_dir, item["orig_name"])))
            members.append((item["upright_name"], os.path.join(server_dir, item["upright_name"])))
        for cam_entry in cameras_json:
            yolo_file = cam_entry.get("yolo_file")
            if yolo_file:
                members.append((yolo_file, os.path.join(server_dir, yolo_file)))
        members.append((pair_name, pair_path))
        try:
            with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                # readable archive JSON name for existing ZIP readers
                for arcname, fpath in members:
                    if arcname == pair_name:
                        zf.writestr("pair.json", pair_bytes)
                    else:
                        zf.write(fpath, arcname=arcname)
            if os.path.exists(zip_path):
                raise RuntimeError("目标 ZIP 名称已存在，拒绝覆盖: %s" % zip_name)
            os.replace(tmp_zip, zip_path)
        except Exception:
            try:
                if os.path.exists(tmp_zip):
                    os.remove(tmp_zip)
            except OSError:
                pass
            raise

        return {
            "id": cap_id,
            "download_url": "/download/%s.zip" % cap_id,
            "server_directory": server_dir,
            "capture_root": capture_root,
            "pair_file": pair_file,
            "cameras": cameras_json,
            "skew_ms": skew_ms,
            "warning": warning,
        }


# ---------------------------------------------------------------------------
# HTTP handler and server
# ---------------------------------------------------------------------------
def _make_handler(feeds, engine):
    BaseHandler = base._make_handler(feeds)

    class CollectionHandler(BaseHandler):
        _engine = engine
        _feeds = feeds

        # ---- helpers ----
        def _json_send(self, code, obj, extra_headers=None):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self._send(code, body, "application/json; charset=utf-8", extra_headers)

        def _error(self, code, message):
            self._json_send(code, {"error": message})

        def _check_origin(self) -> bool:
            origin = self.headers.get("Origin")
            if origin is None:
                return True
            host = self.headers.get("Host")
            if not host:
                return False
            try:
                parts = urlsplit(origin)
            except Exception:
                return False
            if not parts.netloc:
                return False
            return parts.netloc.lower() == host.lower()

        # ---- routing ----
        def _route(self):
            try:
                parsed = urlsplit(self.path)
                path = parsed.path
            except Exception:
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return

            method = self.command

            if path == "/":
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/status":
                self._status()
                return
            if path in ("/frame/0.jpg", "/frame/1.jpg"):
                idx = 0 if path == "/frame/0.jpg" else 1
                self._frame(idx)
                return
            if path.startswith("/view/"):
                if method == "HEAD":
                    self._send(405, b"method not allowed", "text/plain; charset=utf-8")
                    return
                self._view(path, parsed.query)
                return
            if path.startswith("/download/"):
                if method == "HEAD":
                    self._send(200, b"", "text/plain; charset=utf-8")
                    return
                self._download(path)
                return
            if path == "/capture":
                if method == "POST":
                    self._capture(parsed.query)
                    return
                self._error(405, "方法不允许")
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

        def _status(self):
            cams = []
            for w in self._feeds:
                snap = w.snapshot()
                snap.pop("jpeg", None)
                cams.append(snap)
            try:
                collection = self._engine.status() if self._engine else None
                body = {
                    "cameras": cams,
                    "collection": collection,
                    "model": {k: v for k, v in collection.items() if k != "grid_pipeline"} if collection else None,
                }
                payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            except Exception:
                self._send(500, b"status error", "text/plain; charset=utf-8")
                return
            self._send(200, payload, "application/json; charset=utf-8")

        def _frame(self, idx):
            snap = self._feeds[idx].snapshot()
            jpeg = snap["jpeg"]
            error = snap["error"]
            age_ms = snap["age_ms"]
            seq = snap["seq"]
            if jpeg is None or error is not None or age_ms is None or age_ms > STALE_MS:
                self._send(503, b"frame unavailable", "text/plain; charset=utf-8")
                return
            self._send(200, jpeg, "image/jpeg", extra_headers={"X-Frame-Sequence": seq})

        def _view(self, path, query):
            tail = path[len("/view/"):]
            if not tail.endswith(".jpg"):
                self._error(400, "路径无效")
                return
            idx_str = tail[:-4]
            try:
                cam = int(idx_str)
            except ValueError:
                self._error(400, "相机编号无效")
                return
            if cam not in (0, 1):
                self._error(400, "相机编号无效")
                return
            if self._engine is None:
                self._error(503, "服务未配置")
                return
            q = parse_qs(query)
            try:
                rotation = _validate_rotation(q.get("rotation", ["0"])[0])
            except ValueError as exc:
                self._error(400, str(exc))
                return
            yolo_raw = q.get("yolo", ["0"])[0]
            if yolo_raw not in ("0", "1"):
                self._error(400, "yolo 参数必须是 0 或 1")
                return
            yolo = (yolo_raw == "1")
            try:
                jpeg, meta = self._engine.view(cam, rotation, yolo)
            except ValueError as exc:
                self._error(400, str(exc))
                return
            except RuntimeError as exc:
                msg = str(exc)
                code = 503 if ("加载" in msg or "等待" in msg or "过期" in msg or "不可用" in msg) else 500
                self._error(code, msg)
                return
            except Exception as exc:
                self._error(500, "视图错误: %s" % (exc,))
                return
            dets = meta.get("detections")
            if dets is None:
                cnt_hdr = ""
            else:
                cnt_hdr = str(len(dets))
            headers = {
                "X-Frame-Sequence": meta.get("seq") if meta.get("seq") is not None else "",
                "X-Capture-Time": meta.get("captured_at") if meta.get("captured_at") is not None else "",
                "X-Inference-Age": ("%.1f" % meta["age_ms"]) if meta.get("age_ms") is not None else "",
                "X-Inference-Ms": ("%.1f" % meta["inference_ms"]) if meta.get("inference_ms") is not None else "",
                "X-Detection-Count": cnt_hdr,
            }
            self._send(200, jpeg, "image/jpeg", extra_headers=headers)

        def _download(self, path):
            name = path[len("/download/"):]
            if not name.endswith(".zip"):
                self._error(400, "下载路径无效")
                return
            stem = name[:-4]
            # strict safe basename
            if not stem or "/" in stem or "\\" in stem or ".." in stem or "." in stem:
                self._error(400, "下载路径无效")
                return
            if not all(c.isalnum() or c in "-_" for c in stem):
                self._error(400, "下载路径无效")
                return
            if self._engine is None:
                self._error(503, "服务未配置")
                return
            roots = self._engine.download_roots()
            if not roots:
                self._error(503, "服务未配置")
                return
            real_full = None
            for root in roots:
                full = os.path.join(root, name)
                real_root = os.path.realpath(root)
                candidate = os.path.realpath(full)
                try:
                    if os.path.commonpath([real_root, candidate]) != real_root:
                        continue
                except ValueError:
                    continue
                if candidate == real_root:
                    continue
                if os.path.islink(full):
                    continue
                if os.path.isfile(candidate) and not os.path.islink(candidate):
                    real_full = candidate
                    break
            if real_full is None:
                self._error(404, "文件不存在")
                return
            try:
                with open(real_full, "rb") as f:
                    data = f.read()
            except OSError:
                self._error(500, "读取文件失败")
                return
            self._send(
                200, data, "application/zip",
                extra_headers={
                    "Content-Disposition": 'attachment; filename="%s"' % name,
                },
            )

        def _capture(self, query):
            if self._engine is None:
                self._error(503, "服务未配置")
                return
            if not self._check_origin():
                self._error(403, "跨域请求被拒绝")
                return
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                self._error(415, "需要 application/json")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                self._error(400, "Content-Length 无效")
                return
            if length <= 0 or length > MAX_BODY:
                self._error(400, "请求体大小无效")
                return
            try:
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                self._error(400, "JSON 解析失败")
                return
            if not isinstance(payload, dict):
                self._error(400, "请求体必须是对象")
                return
            rots = payload.get("rotations")
            if not isinstance(rots, list) or len(rots) != 2:
                self._error(400, "rotations 必须是长度 2 的数组")
                return
            capture_root = payload.get("capture_root")
            try:
                result = self._engine.capture(rots, capture_root)
            except ValueError as exc:
                self._error(400, str(exc))
                return
            except RuntimeError as exc:
                msg = str(exc)
                if "busy" in msg:
                    self._error(409, "已有拍摄进行中")
                else:
                    self._error(500, "拍摄失败: %s" % msg)
                return
            except Exception as exc:
                self._error(500, "拍摄失败: %s" % (exc,))
                return
            self._json_send(200, result)

        # POST routing
        def do_POST(self):
            try:
                parsed = urlsplit(self.path)
                path = parsed.path
            except Exception:
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            if path == "/capture":
                try:
                    self._capture(parsed.query)
                except Exception as exc:
                    self._error(500, "拍摄异常: %s" % (exc,))
                return
            self._error(404, "not found")

    return CollectionHandler


def create_server(host: str, port: int, feeds: list) -> ThreadingHTTPServer:
    if len(feeds) != 2:
        raise ValueError("exactly two feeds required")

    server_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = _load_config(server_dir)

    handler = _make_handler(feeds, None)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True

    engine = CollectionEngine(
        feeds,
        cfg["model_path"],
        cfg["model_sha256"],
        cfg["capture_root"],
        imgsz=cfg.get("imgsz", DEFAULT_IMGSZ),
        conf=cfg.get("conf", DEFAULT_CONF),
        iou=cfg.get("iou", DEFAULT_IOU),
    )
    server.engine = engine
    handler._engine = engine

    orig_close = server.server_close

    def _close_then_original():
        try:
            engine.close()
        except Exception:
            pass
        try:
            orig_close()
        except Exception:
            pass

    server.server_close = _close_then_original
    return server


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    base.JPEG_QUALITY = JPEG_QUALITY
    base.create_server = create_server
    return base.main(argv)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>双摄数据采集</title>
<style>
:root{
  --bg:#0d1117;--panel:#161b22;--panel2:#1c2330;--border:#30363d;
  --text:#e6edf3;--muted:#8b949e;--accent:#2f81f7;--ok:#3fb950;--warn:#d29922;--err:#f85149;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
  font-size:14px;line-height:1.5;
}
header{
  padding:16px 20px;border-bottom:1px solid var(--border);
  display:flex;flex-wrap:wrap;align-items:center;gap:12px;
}
header h1{font-size:18px;margin:0;font-weight:600}
.badge{
  font-size:12px;background:var(--panel2);border:1px solid var(--border);
  padding:2px 8px;border-radius:999px;color:var(--muted);
}
.actions{margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{
  background:var(--panel2);color:var(--text);border:1px solid var(--border);
  padding:7px 14px;border-radius:6px;cursor:pointer;font-size:13px;
}
button:hover:not(:disabled){border-color:var(--accent)}
button:disabled{opacity:.5;cursor:not-allowed}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.primary:hover:not(:disabled){filter:brightness(1.1)}
main{max-width:1600px;margin:0 auto;padding:16px 20px 40px}
.cams{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.cams{grid-template-columns:1fr}}
.card{
  background:var(--panel);border:1px solid var(--border);border-radius:10px;
  overflow:hidden;display:flex;flex-direction:column;
}
.card-head{
  padding:10px 14px;border-bottom:1px solid var(--border);
  display:flex;flex-wrap:wrap;align-items:center;gap:8px;
}
.card-title{font-weight:600;font-size:15px}
.role{font-size:12px;color:var(--muted)}
.spacer{flex:1}
.frame-wrap{
  position:relative;background:#000;height:65vh;max-height:65vh;min-height:200px;
  display:flex;align-items:center;justify-content:center;overflow:hidden;
}
.frame-wrap img{
  max-width:100%;max-height:100%;width:auto;height:auto;
  object-fit:contain;display:block;
}
.frame-empty{
  color:var(--muted);font-size:13px;text-align:center;padding:20px;
}
.paused-overlay{
  position:absolute;inset:0;background:rgba(13,17,23,.72);
  display:flex;align-items:center;justify-content:center;
  color:var(--warn);font-size:20px;font-weight:600;letter-spacing:2px;
  pointer-events:none;
}
.controls{
  padding:10px 14px;border-top:1px solid var(--border);
  display:flex;flex-wrap:wrap;align-items:center;gap:12px;
}
.controls label{display:flex;align-items:center;gap:6px;font-size:13px;color:var(--muted)}
select{
  background:var(--panel2);color:var(--text);border:1px solid var(--border);
  padding:5px 8px;border-radius:5px;font-size:13px;
}
.meta{
  padding:8px 14px;border-top:1px solid var(--border);
  display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--muted);
}
.meta span b{color:var(--text);font-weight:500}
.err{color:var(--err)}
.hint{font-size:12px;color:var(--muted)}
.capture-panel{
  margin-top:16px;background:var(--panel);border:1px solid var(--border);
  border-radius:10px;padding:14px 16px;
}
.capture-panel .row{display:flex;flex-wrap:wrap;align-items:center;gap:12px}
#cap-status{margin-top:10px;font-size:13px}
#cap-status.ok{color:var(--ok)}
#cap-status.warn{color:var(--warn)}
#cap-status.err{color:var(--err)}
#cap-status a{color:var(--accent)}
.notice{
  margin-top:14px;padding:10px 14px;border:1px solid var(--border);
  border-radius:8px;background:var(--panel2);font-size:12px;color:var(--muted);
}
.status-bar{
  margin-top:12px;display:flex;flex-wrap:wrap;gap:16px;font-size:12px;color:var(--muted);
}
.status-bar span b{color:var(--text);font-weight:500}
#grid-panel{margin:10px 0;padding:16px;border:1px solid #38414b;border-radius:12px;background:#20252b}
.grid-health{font-size:12px;color:var(--text);margin:4px 0;white-space:pre-wrap;}
.grid-board{display:grid;grid-template-columns:repeat(8,1fr);gap:2px;max-width:600px;margin:6px 0;}
.grid-sq{position:relative;aspect-ratio:1/1;display:flex;flex-direction:column;gap:3px;align-items:center;justify-content:center;border-radius:4px;font-size:11px;line-height:1.1;color:#111;padding:2px;box-sizing:border-box;overflow:hidden;}
.grid-sq.dark{background:#b9b9b9;}
.grid-sq.light{background:#e8e8e8;}
.grid-sq.supported{border:2px solid #2e9e3f;}
.grid-sq.conflict{border:2px solid #e07800;}
.grid-sq.primary_only{border:1px solid #4477cc;}
.grid-sq.aux_only_tentative{border:1px dashed #9966cc;}
.grid-sq.unknown{border:1px solid #bbbbbb;}
.grid-sq .sq-name{position:absolute;top:1px;left:2px;font-size:9px;color:#555;}
.grid-sq .sq-piece{font-weight:bold;font-size:13px;}
.grid-sq .sq-sub{font-size:9px;color:#666;}
.grid-legend{margin-top:4px;font-size:11px;color:var(--muted);}
.grid-legend .lg{margin-right:8px;padding:1px 4px;border-radius:3px;}
.lg-supported{border:2px solid #2e9e3f;}
.lg-conflict{border:2px solid #e07800;}
.lg-primary{border:1px solid #4477cc;}
.lg-aux{border:1px dashed #9966cc;}
.lg-unknown{border:1px solid #bbbbbb;}
</style>
</head>
<body>
<header>
  <h1>双摄数据采集</h1>
  <span class="badge">预览 + 手动拍摄</span>
  <div class="actions">
    <span id="model-badge" class="badge">模型: 未知</span>
  </div>
</header>
<main>
  <div class="cams">
    <section class="card" id="card-1">
      <div class="card-head">
        <span class="card-title">摄像头2</span>
        <span class="role">主视角</span>
        <span class="spacer"></span>
      </div>
      <div class="frame-wrap" id="wrap-1">
        <div class="frame-empty" id="empty-1">等待画面…</div>
        <img id="img-1" alt="摄像头2画面" style="display:none">
        <div class="paused-overlay" id="pause-1" style="display:none">已暂停</div>
      </div>
      <div class="controls">
        <label>旋转:
          <select id="rot-1">
            <option value="0">0°（顺时针 0°）</option>
            <option value="90">90°（顺时针 90°）</option>
            <option value="180">180°（顺时针 180°）</option>
            <option value="270">270°（逆时针 90°）</option>
          </select>
        </label>
        <label><input type="checkbox" id="yolo-1"> 显示 YOLO＋格线＋格子坐标</label>
        <button id="pause-btn-1">暂停</button>
      </div>
      <div class="meta">
        <span>尺寸: <b id="dim-1">--</b></span>
        <span>帧延迟: <b id="age-1">--</b></span>
        <span>推理: <b id="inf-1">--</b></span>
        <span>检测数: <b id="cnt-1">--</b></span>
        <span>序号: <b id="seq-1">--</b></span>
        <span class="err" id="err-1"></span>
      </div>
    </section>

    <section class="card" id="card-0">
      <div class="card-head">
        <span class="card-title">摄像头1</span>
        <span class="role">辅助视角</span>
        <span class="spacer"></span>
      </div>
      <div class="frame-wrap" id="wrap-0">
        <div class="frame-empty" id="empty-0">等待画面…</div>
        <img id="img-0" alt="摄像头1画面" style="display:none">
        <div class="paused-overlay" id="pause-0" style="display:none">已暂停</div>
      </div>
      <div class="controls">
        <label>旋转:
          <select id="rot-0">
            <option value="0">0°（顺时针 0°）</option>
            <option value="90">90°（顺时针 90°）</option>
            <option value="180">180°（顺时针 180°）</option>
            <option value="270">270°（逆时针 90°）</option>
          </select>
        </label>
        <label><input type="checkbox" id="yolo-0"> 显示 YOLO＋格线＋格子坐标</label>
        <button id="pause-btn-0">暂停</button>
      </div>
      <div class="meta">
        <span>尺寸: <b id="dim-0">--</b></span>
        <span>帧延迟: <b id="age-0">--</b></span>
        <span>推理: <b id="inf-0">--</b></span>
        <span>检测数: <b id="cnt-0">--</b></span>
        <span>序号: <b id="seq-0">--</b></span>
        <span class="err" id="err-0"></span>
      </div>
    </section>
  </div>

  <div class="capture-panel">
    <div class="row">
      <label for="capture-root">保存目录（Ubuntu）</label>
      <input id="capture-root" type="text" autocomplete="off" spellcheck="false" style="width:100%;max-width:640px;box-sizing:border-box;">
    </div>
    <div class="row">
      <span class="hint">只在点击拍摄时保存；直接保存到指定目录；文件名含时间戳与配对编号，不创建拍摄子文件夹。Windows 本地文件可用 ZIP 下载。</span>
    </div>
    <div class="row">
      <button id="capture-btn" class="primary">拍摄双摄图片</button>
      <span class="hint">从两个摄像头当前画面各截取一张，保存原图、旋转后图像与 YOLO 标注。暂停状态下仍按当前画面拍摄。</span>
    </div>
    <div id="cap-status"></div>
  </div>

  <div class="notice">
    本页面仅做实时预览与手动截图，不进行连续录制。勾选后显示格线与棋子坐标；相机2编号固定，参考跟踪失效时不输出格子。预测与底座归属均需人工复核。检测框坐标为旋转后图像像素；YOLO 预览较慢，显示的是模型预测结果，需人工复核。
    两摄像头时间戳为各自的主机编码时间，非硬件同步。
  </div>

  <div class="panel" id="grid-panel">
  <div class="panel-head">棋局关联 · 相机2主视角 / 相机1辅助</div>
  <div class="grid-hint">试运行：参考棋盘自动跟踪，底座位置为估计；未识别不等于空格。</div>
  <div class="grid-health" id="grid-health"></div>
  <div class="grid-board" id="grid-board"></div>
  <div class="grid-legend">
    <span class="lg lg-supported">双摄一致</span>
    <span class="lg lg-conflict">冲突</span>
    <span class="lg lg-primary">主视角</span>
    <span class="lg lg-aux">辅助待确认</span>
    <span class="lg lg-unknown">未知</span>
  </div>
</div>
<div class="status-bar">
    <span>模型版本: <b id="model-short">--</b></span>
    <span>状态: <b id="model-state">--</b></span>
    <span id="model-err" class="err"></span>
  </div>
</main>

<script>
(function(){
  "use strict";
  var LS_KEY = "dualcam.collection.v3.rotations";
  var VIEW_TIMEOUT = 3000;
  var STATUS_POLL = 1000;
  var CAM_LABEL = {0:"摄像头1", 1:"摄像头2"};

  var state = {
    0:{rotation:0, yolo:true, paused:false, gen:0, url:null, inFlight:false, timer:null, loopActive:false, ctrl:null},
    1:{rotation:270, yolo:true, paused:false, gen:0, url:null, inFlight:false, timer:null, loopActive:false, ctrl:null}
  };
  var capturing = false;
  var statusTimer = null;

  function $(id){return document.getElementById(id);}

  function loadRotations(){
    try{
      var raw = localStorage.getItem(LS_KEY);
      if(!raw) return;
      var obj = JSON.parse(raw);
      if(!obj || typeof obj !== "object") return;
      var v1 = obj["1"], v0 = obj["0"];
      if([0,90,180,270].indexOf(v1) >= 0) state[1].rotation = v1;
      if([0,90,180,270].indexOf(v0) >= 0) state[0].rotation = v0;
    }catch(e){}
  }
  function saveRotations(){
    try{
      localStorage.setItem(LS_KEY, JSON.stringify({"0":state[0].rotation, "1":state[1].rotation}));
    }catch(e){}
  }

  function setText(id, text){
    var el = $(id);
    if(el) el.textContent = text;
  }

  function clearView(cam){
    var img = $("img-"+cam);
    if(img){
      img.style.display = "none";
      img.removeAttribute("src");
    }
    var empty = $("empty-"+cam);
    if(empty) empty.style.display = "";
  }

  function setError(cam, msg){
    var el = $("err-"+cam);
    if(!el) return;
    el.textContent = msg ? msg : "";
  }

  function scheduleNext(cam){
    var st = state[cam];
    if(st.timer){ clearTimeout(st.timer); st.timer = null; }
    if(st.paused) return;
    st.timer = setTimeout(function(){ fetchView(cam); }, 200);
  }

  function fetchView(cam){
    var st = state[cam];
    if(st.paused) return;
    if(st.inFlight) { scheduleNext(cam); return; }
    st.inFlight = true;
    var myGen = st.gen;
    var rotation = st.rotation;
    var yolo = st.yolo ? "1" : "0";
    var url = "/view/" + cam + ".jpg?rotation=" + rotation + "&yolo=" + yolo;
    var ctrl = new AbortController();
    st.ctrl = ctrl;
    var to = setTimeout(function(){ ctrl.abort(); }, VIEW_TIMEOUT);

    fetch(url, {signal: ctrl.signal, cache: "no-store"})
      .then(function(resp){
        if(myGen !== st.gen) throw new Error("stale");
        if(!resp.ok){
          return resp.json().catch(function(){return {};}).then(function(body){
            throw new Error((body && body.error) || ("HTTP " + resp.status));
          });
        }
        var seq = resp.headers.get("X-Frame-Sequence");
        var ctime = resp.headers.get("X-Capture-Time");
        var age = resp.headers.get("X-Inference-Age");
        var infMs = resp.headers.get("X-Inference-Ms");
        var detCnt = resp.headers.get("X-Detection-Count");
        return resp.blob().then(function(blob){
          return {blob: blob, seq: seq, ctime: ctime, age: age, infMs: infMs, detCnt: detCnt};
        });
      })
      .then(function(payload){
        if(myGen !== st.gen) return;
        var img = $("img-"+cam);
        if(!img) return;
        var newUrl = URL.createObjectURL(payload.blob);
        var oldUrl = st.url;
        img.onload = function(){
          if(oldUrl) URL.revokeObjectURL(oldUrl);
          var w = img.naturalWidth || 0;
          var h = img.naturalHeight || 0;
          setText("dim-"+cam, w && h ? (w + "×" + h) : "--");
        };
        img.onerror = function(){
          if(oldUrl) URL.revokeObjectURL(oldUrl);
          setText("dim-"+cam, "--");
        };
        img.src = newUrl;
        st.url = newUrl;
        img.style.display = "block";
        var empty = $("empty-"+cam);
        if(empty) empty.style.display = "none";
        setError(cam, "");
        setText("seq-"+cam, payload.seq || "--");
        if(payload.infMs && !isNaN(parseFloat(payload.infMs))){
          setText("inf-"+cam, Math.round(parseFloat(payload.infMs)) + " ms");
        } else {
          setText("inf-"+cam, "--");
        }
        if(payload.detCnt !== null && payload.detCnt !== "" && !isNaN(parseInt(payload.detCnt, 10))){
          setText("cnt-"+cam, String(parseInt(payload.detCnt, 10)));
        } else {
          setText("cnt-"+cam, "--");
        }
        if(payload.age && !isNaN(parseFloat(payload.age))){
          setText("age-"+cam, Math.round(parseFloat(payload.age)) + " ms");
        } else if(payload.ctime){
          var ageMs = (Date.now() - (parseFloat(payload.ctime) * 1000));
          setText("age-"+cam, isFinite(ageMs) ? Math.round(Math.max(ageMs,0)) + " ms" : "--");
        } else {
          setText("age-"+cam, "--");
        }
      })
      .catch(function(err){
        if(myGen !== st.gen) return;
        if(err && err.name === "AbortError"){
          setError(cam, "请求超时");
        } else if(err && err.message === "stale"){
          // ignore
        } else {
          setError(cam, (err && err.message) ? err.message : "视图请求失败");
        }
        // Do not keep stale image as live: hide until next good frame
        var img = $("img-"+cam);
        if(img) img.style.display = "none";
        var empty = $("empty-"+cam);
        if(empty) empty.style.display = "";
      })
      .then(function(){
        clearTimeout(to);
        if(myGen !== st.gen) return;
        st.inFlight = false;
        st.ctrl = null;
        if(!st.paused) scheduleNext(cam);
      });
  }

  function restartLoop(cam){
    var st = state[cam];
    st.gen++;
    if(st.ctrl){ try{ st.ctrl.abort(); }catch(e){} st.ctrl = null; }
    st.inFlight = false;
    if(st.timer){ clearTimeout(st.timer); st.timer = null; }
    if(st.paused) return;
    // kick immediately
    fetchView(cam);
  }

  function bindCamera(cam){
    var st = state[cam];
    var rotSel = $("rot-"+cam);
    var yoloBox = $("yolo-"+cam);
    var pauseBtn = $("pause-btn-"+cam);
    var pauseOverlay = $("pause-"+cam);

    rotSel.value = String(st.rotation);
    yoloBox.checked = st.yolo;

    rotSel.addEventListener("change", function(){
      var v = parseInt(rotSel.value, 10);
      if([0,90,180,270].indexOf(v) < 0) return;
      st.rotation = v;
      saveRotations();
      restartLoop(cam);
    });

    yoloBox.addEventListener("change", function(){
      st.yolo = !!yoloBox.checked;
      restartLoop(cam);
    });

    pauseBtn.addEventListener("click", function(){
      st.paused = !st.paused;
      pauseBtn.textContent = st.paused ? "继续" : "暂停";
      pauseOverlay.style.display = st.paused ? "flex" : "none";
      if(st.paused){
        st.gen++;
        if(st.ctrl){ try{ st.ctrl.abort(); }catch(e){} st.ctrl = null; }
        st.inFlight = false;
        if(st.timer){ clearTimeout(st.timer); st.timer = null; }
      } else {
        restartLoop(cam);
      }
    });
  }

  var GRID_PIECE_ABBR = {WP:"WP",WN:"WN",WB:"WB",WR:"WR",WQ:"WQ",WK:"WK",BP:"BP",BN:"BN",BB:"BB",BR:"BR",BQ:"BQ",BK:"BK"};
  function gridPieceAbbr(piece){
    if(!piece) return "?";
    var cls = piece.class_name || "";
    var up = String(cls).toUpperCase();
    if(GRID_PIECE_ABBR[up]) return GRID_PIECE_ABBR[up];
    var m = up.match(/^(W|B)(P|N|B|R|Q|K)$/);
    if(m) return up;
    var words = String(cls).toLowerCase().split(/[-_ ]+/);
    var kinds = {pawn:"P",knight:"N",bishop:"B",rook:"R",queen:"Q",king:"K"};
    if(words.length === 2 && (words[0] === "white" || words[0] === "black") && kinds[words[1]]) return (words[0] === "white" ? "W" : "B") + kinds[words[1]];
    if(cls) return cls;
    return "?";
  }
  function updateGridPanel(grid){
    var healthEl = document.getElementById("grid-health");
    var boardEl = document.getElementById("grid-board");
    if(!healthEl || !boardEl) return;
    var primaryOn = state[1].yolo && !state[1].paused;
    var auxOn = state[0].yolo && !state[0].paused;
    var enabled = primaryOn || auxOn;

    function safeSignalAbbr(sig){
      if(!sig) return "?";
      try{ return gridPieceAbbr(sig); }catch(e){ return "?"; }
    }
    function safeSignalName(sig){
      if(!sig) return "未知";
      return String(sig.class_name == null ? "未知" : sig.class_name);
    }
    function safeSignalConf(sig){
      if(!sig) return "-";
      var c = sig.confidence;
      return (typeof c === "number" && isFinite(c)) ? c.toFixed(2) : "-";
    }

    if(!grid){
      healthEl.textContent = enabled
        ? "7:3＋多帧试用 | C1原始检测；C2与下方棋盘为稳定结果（分数非正确率） | 格线管道：无数据"
        : "7:3＋多帧试用 | C1原始检测；C2与下方棋盘为稳定结果（分数非正确率） | 格线管道：未启用（勾选后启用）";
      boardEl.textContent = "";
      return;
    }
    if(grid.error){
      healthEl.textContent = "7:3＋多帧试用 | C1原始检测；C2与下方棋盘为稳定结果（分数非正确率） | 格线管道：错误 " + String(grid.error);
      boardEl.textContent = "";
      return;
    }
    if(grid.enabled === false){
      healthEl.textContent = "7:3＋多帧试用 | C1原始检测；C2与下方棋盘为稳定结果（分数非正确率） | 格线管道：未启用（勾选后启用）";
      boardEl.textContent = "";
      return;
    }

    var cams = grid.cameras || [];
    var parts = [];
    for(var ci=0; ci<cams.length; ci++){
      var c = cams[ci] || {};
      var cameraIndex = c.camera_index!=null ? c.camera_index : ci;
      var camState = state[cameraIndex] || {yolo:false, paused:true};
      var camToggles = camState.paused ? " 暂停" : (!camState.yolo ? " 识别关闭" : "");
      var camLine = "C" + (cameraIndex + 1) + camToggles +
        " valid=" + (c.grid_valid?"1":"0") +
        " fresh=" + (c.fresh?"1":"0") +
        " inliers=" + (c.inliers!=null?c.inliers:"-") +
        " assigned=" + (c.assigned_count!=null?c.assigned_count:"-") +
        " 格线耗时=" + (c.grid_ms!=null?Number(c.grid_ms).toFixed(1)+"ms":"-") +
        (c.reason?(" reason=" + String(c.reason)):"");
      if(!c.grid_valid){
        camLine += " | 定位中，YOLO暂停";
      }
      if(c.cache_error) camLine += " | 缓存错误：" + String(c.cache_error);
      parts.push(camLine);
    }
    var head = parts.join(" | ");
    if(grid.pair_skew_ms!=null) head += " | 双摄时间差=" + Number(grid.pair_skew_ms).toFixed(1)+"ms";
    head += " | " + (grid.aux_usable && primaryOn && auxOn ? "辅助可用" : "辅助未参与：等待两路新鲜且时间接近的结果");
    if(grid.cache_error){
      head += " | 缓存错误：" + String(grid.cache_error);
    }
    var prefix = "7:3＋多帧试用 | C1原始检测；C2与下方棋盘为稳定结果（分数非正确率） | ";
    healthEl.textContent = enabled ? (prefix + head) : (prefix + "格线管道：未启用（勾选后启用）");

    while(boardEl.firstChild) boardEl.removeChild(boardEl.firstChild);

    var squares = grid.squares || [];
    for(var si=0; si<squares.length; si++){
      var sq = squares[si] || {};
      var primary = sq.primary || null;
      var auxiliary = sq.auxiliary || null;
      var finalSig = sq.final || null;
      var src = sq.source == null ? "unknown" : String(sq.source);
      var conflict = !!sq.conflict;
      var reasons = Array.isArray(sq.uncertain_reasons) ? sq.uncertain_reasons : [];
      var rej = Array.isArray(sq.aux_rejections) ? sq.aux_rejections : [];

      var shownSig = null;
      var shownSrc = src;
      var visibleState = "unknown";

      if(primaryOn){
        shownSig = finalSig;
        shownSrc = src;
        visibleState = shownSig ? (src === "primary_only" ? "primary_only" : "supported") : "unknown";
      } else {
        shownSig = null;
        visibleState = "unknown";
        shownSrc = "unknown";
      }

      var name = sq.square || "?";
      var fileIdx = "abcdefgh".indexOf(String(name).charAt(0));
      var rankNum = parseInt(String(name).charAt(1), 10);
      var isDark = ((fileIdx + rankNum) % 2) === 1;
      var cell = document.createElement("div");
      cell.className = "grid-sq " + (isDark?"dark":"light") + " " + visibleState;

      var titleParts = [String(name)];
      titleParts.push("原始主摄 " + (primary ? (safeSignalName(primary) + " " + safeSignalConf(primary)) : "未知"));
      titleParts.push("原始辅摄 " + (auxiliary ? (safeSignalName(auxiliary) + " " + safeSignalConf(auxiliary)) : "无"));
      titleParts.push("来源 " + shownSrc);
      titleParts.push("多帧 " + (sq.temporal_state || "pending") + " / " + (sq.temporal_frames || 0));
      if(Array.isArray(sq.weights)) titleParts.push("权重 C2:C1 " + sq.weights.map(function(x){return Number(x).toFixed(2);}).join(":"));
      if(conflict) titleParts.push("冲突");
      if(reasons.length) titleParts.push("原因 " + reasons.join(","));
      if(rej.length) titleParts.push("辅助拒绝 " + rej.join(","));
      cell.title = titleParts.join(" | ");

      var nm = document.createElement("span");
      nm.className = "sq-name";
      nm.textContent = String(name);
      cell.appendChild(nm);

      var pieceEl = document.createElement("span");
      pieceEl.className = "sq-piece";
      if(shownSig){
        pieceEl.textContent = safeSignalAbbr(shownSig);
      } else {
        pieceEl.textContent = "?";
      }
      cell.appendChild(pieceEl);

      if(visibleState === "primary_fallback"){
        var warn = document.createElement("span");
        warn.className = "sq-warn";
        warn.textContent = "!";
        cell.appendChild(warn);
      }

      var showAuxMark = (visibleState === "auxiliary_override") &&
        (shownSrc === "auxiliary_override" || shownSrc === "auxiliary_fill");
      if(showAuxMark){
        var auxMark = document.createElement("span");
        auxMark.className = "sq-sub sq-aux";
        auxMark.textContent = "+C1";
        cell.appendChild(auxMark);
      } else if(auxOn && shownSig && (shownSrc === "corroborated" || shownSrc === "auxiliary_override" || shownSrc === "auxiliary_fill")){
        var sub = document.createElement("span");
        sub.className = "sq-sub";
        sub.textContent = shownSrc === "corroborated" ? "C2+C1" : "+C1";
        cell.appendChild(sub);
      }

      boardEl.appendChild(cell);
    }
  }

  function updateModelBadge(model){
    if(!model) return;
    var short = model.model_short_sha || "--";
    setText("model-short", short ? short : "--");
    var stEl = $("model-state");
    var errEl = $("model-err");
    if(model.model_error){
      stEl.textContent = "加载失败";
      errEl.textContent = model.model_error;
    } else if(model.model_loaded){
      stEl.textContent = "已加载";
      errEl.textContent = "";
    } else {
      stEl.textContent = "未加载";
      errEl.textContent = "";
    }
    var badge = $("model-badge");
    if(badge) badge.textContent = "当前微调模型";
  }

  var CAPTURE_ROOT_KEY = "chess_capture_root_v1";
  var captureRootInitialized = false;
  var captureRootDirty = false;

  function safeStorageGet(key){
    try { return window.localStorage.getItem(key); }
    catch(e){ return null; }
  }
  function safeStorageSet(key, value){
    try { window.localStorage.setItem(key, value); }
    catch(e){ /* ignore (privacy mode) */ }
  }

  function applyDefaultCaptureRoot(serverDefault){
    var input = $("capture-root");
    if(!input || !serverDefault) return;
    input.setAttribute("placeholder", serverDefault);
    input.title = "默认: " + serverDefault;
    if(captureRootInitialized || captureRootDirty) return;
    var saved = safeStorageGet(CAPTURE_ROOT_KEY);
    if(saved && saved.length){
      input.value = saved;
    }
    captureRootInitialized = true;
  }

  function bindCaptureRoot(){
    var input = $("capture-root");
    if(!input) return;
    input.addEventListener("input", function(){
      captureRootDirty = true;
    });
    input.addEventListener("change", function(){
      captureRootDirty = true;
      safeStorageSet(CAPTURE_ROOT_KEY, input.value);
    });
  }

  function currentCaptureRoot(){
    var input = $("capture-root");
    if(!input) return "";
    return (input.value || "").trim();
  }

  function pollStatus(){
    var ctrl = new AbortController();
    var to = setTimeout(function(){ ctrl.abort(); }, 3000);
    fetch("/status", {cache:"no-store", signal: ctrl.signal})
      .then(function(r){ return r.json(); })
      .then(function(body){
        if(body && body.collection){
          updateModelBadge(body.collection);
          updateGridPanel(body.collection.grid_pipeline);
          if(body.collection.capture_root){
            applyDefaultCaptureRoot(body.collection.capture_root);
          }
        }
        else if(body && body.model) updateModelBadge(body.model);
      })
      .catch(function(){ /* ignore */ })
      .then(function(){
        clearTimeout(to);
        statusTimer = setTimeout(pollStatus, STATUS_POLL);
      });
  }

  function setCaptureStatus(kind, text, linkUrl, linkText){
    var el = $("cap-status");
    if(!el) return;
    el.className = kind || "";
    el.textContent = "";
    if(text){
      el.appendChild(document.createTextNode(text));
    }
    if(linkUrl){
      el.appendChild(document.createTextNode(" "));
      var a = document.createElement("a");
      a.href = linkUrl;
      a.textContent = linkText || "下载 ZIP";
      a.setAttribute("download", "");
      el.appendChild(a);
    }
  }

  function doCapture(){
    if(capturing) return;
    capturing = true;
    var btn = $("capture-btn");
    btn.disabled = true;
    var rootInput = $("capture-root");
    var rootValue = currentCaptureRoot();
    if(rootInput){
      rootInput.disabled = true;
      safeStorageSet(CAPTURE_ROOT_KEY, rootInput.value);
    }
    setCaptureStatus("", "拍摄中…", null, null);

    var payload = {
      rotations: [state[0].rotation, state[1].rotation]
    };
    if(rootValue.length){ payload.capture_root = rootValue; }
    var ctrl = new AbortController();
    var to = setTimeout(function(){ ctrl.abort(); }, 120000);

    fetch("/capture", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
      signal: ctrl.signal
    })
      .then(function(resp){
        return resp.json().catch(function(){return {};}).then(function(body){
          return {ok: resp.ok, status: resp.status, body: body};
        });
      })
      .then(function(result){
        if(result.ok && result.body && result.body.id){
          var body = result.body;
          var text = "拍摄成功。保存目录: " + (body.server_directory || "--");
          if(body.warning) text += " 警告: " + body.warning;
          setCaptureStatus(body.warning ? "warn" : "ok", text, body.download_url, "下载配对 ZIP");
        } else {
          var errMsg = (result.body && result.body.error) ? result.body.error : ("HTTP " + result.status);
          setCaptureStatus("err", "拍摄失败: " + errMsg, null, null);
        }
      })
      .catch(function(err){
        if(err && err.name === "AbortError"){
          setCaptureStatus("err", "拍摄请求超时（未自动重试）", null, null);
        } else {
          setCaptureStatus("err", "拍摄失败: " + ((err && err.message) || "网络错误"), null, null);
        }
      })
      .then(function(){
        clearTimeout(to);
        capturing = false;
        btn.disabled = false;
        if(rootInput) rootInput.disabled = false;
      });
  }

  function init(){
    loadRotations();
    bindCamera(0);
    bindCamera(1);
    bindCaptureRoot();
    var btn = $("capture-btn");
    btn.addEventListener("click", doCapture);
    // start both view loops
    restartLoop(0);
    restartLoop(1);
    pollStatus();
  }

  if(document.readyState === "loading"){
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
