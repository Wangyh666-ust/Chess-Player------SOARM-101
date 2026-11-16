#!/usr/bin/env python3
"""Two-camera live preview utility.

Standalone preview adapter: captures MJPG 1280x720 from two V4L2 devices
independently, encodes JPEG at up to 8 fps, and serves a self-contained
browser preview (HTML + /status + /frame/N.jpg). No inference, no recording,
no filesystem serving.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

TARGET_FPS = 8.0
MIN_INTERVAL = 1.0 / TARGET_FPS
STALE_MS = 2000.0
JPEG_QUALITY = 75
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


class CameraWorker:
    """Independent capture thread for one camera."""

    def __init__(self, device: str, label: str, cv2_module=None):
        self.device = device
        self.label = label
        self._cv2 = cv2_module
        self._lock = threading.Lock()
        self._jpeg = None
        self._seq = 0
        self._width = 0
        self._height = 0
        self._captured_at = None
        self._captured_mono = None
        self._error = None
        self._stop_event = threading.Event()
        self._thread = None
        self._cap = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"cam-{self.label}", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        t = self._thread
        if t is not None:
            t.join(timeout=3.0)
        # Do not release cap from this (foreign) thread: release is owned
        # exclusively by _run's finally block. A truly stuck read keeps the
        # fd until daemon process exit closes it.

    def snapshot(self) -> dict:
        with self._lock:
            age_ms = None
            if self._captured_mono is not None:
                age_ms = (time.monotonic() - self._captured_mono) * 1000.0
            return {
                "jpeg": self._jpeg,
                "seq": self._seq,
                "label": self.label,
                "device": self.device,
                "width": self._width,
                "height": self._height,
                "captured_at": self._captured_at,
                "age_ms": age_ms,
                "error": self._error,
            }

    def _set_error(self, message: str):
        with self._lock:
            self._error = message

    def _run(self):
        cv2 = self._cv2
        if cv2 is None:
            try:
                import cv2 as _cv2  # lazy import
                cv2 = _cv2
            except Exception as exc:
                self._set_error(f"无法导入 cv2: {exc}")
                return

        try:
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        except Exception as exc:
            self._set_error(f"打开摄像头失败: {exc}")
            return

        self._cap = cap
        try:
            if not cap.isOpened():
                self._set_error("打不开摄像头")
                return

            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
            except Exception:
                pass
            buffersize = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
            if buffersize is not None:
                try:
                    cap.set(buffersize, 1)
                except Exception:
                    pass

            last_encode_mono = None

            while not self._stop_event.is_set():
                try:
                    ok, frame = cap.read()
                except Exception as exc:
                    self._set_error(f"读取失败: {exc}")
                    break
                if not ok or frame is None:
                    self._set_error("读取帧失败")
                    break

                now = time.monotonic()
                if last_encode_mono is not None and (now - last_encode_mono) < MIN_INTERVAL:
                    continue

                try:
                    enc_ok, encoded = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                    )
                except Exception as exc:
                    self._set_error(f"JPEG 编码失败: {exc}")
                    break
                if not enc_ok:
                    self._set_error("JPEG 编码失败")
                    break

                try:
                    jpeg_bytes = encoded.tobytes()
                except Exception as exc:
                    self._set_error(f"JPEG 编码失败: {exc}")
                    break

                shape = getattr(frame, "shape", None)
                if shape is not None and len(shape) >= 2:
                    height, width = int(shape[0]), int(shape[1])
                else:
                    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT

                last_encode_mono = now
                with self._lock:
                    self._jpeg = jpeg_bytes
                    self._seq += 1
                    self._width = width
                    self._height = height
                    self._captured_at = time.time()
                    self._captured_mono = time.monotonic()
                    self._error = None
        except Exception as exc:
            self._set_error(f"采集异常: {exc}")
        finally:
            try:
                cap.release()
            except Exception:
                pass
            self._cap = None


INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>双摄像头实时预览</title>
<style>
  :root {
    --bg: #14171c;
    --card: #1c2128;
    --border: #2d333b;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #3b82f6;
    --danger: #f85149;
    --ok: #3fb950;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 16px;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
  }
  h1 { font-size: 20px; font-weight: 600; text-align: center; margin: 8px 0 16px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px; display: flex; flex-direction: column;
  }
  .head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
  .title { font-weight: 600; }
  .device { color: var(--muted); font-size: 11px; word-break: break-all; }
  .frame-wrap {
    position: relative; margin-top: 10px; width: 100%;
    aspect-ratio: 16 / 9; background: #0d1117;
    border-radius: 6px; overflow: hidden;
  }
  .frame-wrap img {
    width: 100%; height: 100%; object-fit: contain; display: block;
    transition: opacity 0.2s;
  }
  .frame-wrap img.dim { opacity: 0.3; }
  .overlay {
    position: absolute; inset: 0; display: none;
    align-items: center; justify-content: center;
    font-size: 14px; font-weight: 600; color: var(--muted);
    background: rgba(13,17,23,0.55); text-align: center; padding: 8px;
  }
  .overlay.show { display: flex; }
  .controls { display: flex; gap: 8px; margin-top: 10px; }
  button {
    flex: 1; cursor: pointer; border: 1px solid var(--border);
    background: #21262d; color: var(--text); padding: 7px 8px;
    border-radius: 6px; font-size: 13px;
  }
  button:hover { border-color: var(--accent); }
  .meta {
    display: flex; flex-wrap: wrap; gap: 6px 14px;
    margin-top: 10px; font-size: 12px; color: var(--muted);
  }
  .meta span { white-space: nowrap; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
  .dot.live { background: var(--ok); }
  .dot.stale { background: #d29922; }
  .dot.err { background: var(--danger); }
  .foot {
    margin-top: 18px; text-align: center; font-size: 12px;
    color: var(--muted); line-height: 1.7;
  }
  .err-text { color: var(--danger); }
</style>
</head>
<body>
<h1>双摄像头实时预览</h1>
<div class="grid">
  <div class="card" id="card-0">
    <div class="head"><div class="title">摄像头1</div><div class="device" id="dev-0"></div></div>
    <div class="frame-wrap">
      <img id="img-0" alt="摄像头1">
      <div class="overlay" id="ov-0">等待画面…</div>
    </div>
    <div class="controls">
      <button id="rot-0">旋转180°</button>
      <button id="pause-0">暂停</button>
    </div>
    <div class="meta">
      <span><span class="dot" id="dot-0"></span><span id="state-0">连接中</span></span>
      <span id="seq-0">帧号 -</span>
      <span id="res-0">分辨率 -</span>
      <span id="age-0">时效 -</span>
    </div>
    <div class="meta err-text" id="err-0"></div>
  </div>
  <div class="card" id="card-1">
    <div class="head"><div class="title">摄像头2</div><div class="device" id="dev-1"></div></div>
    <div class="frame-wrap">
      <img id="img-1" alt="摄像头2">
      <div class="overlay" id="ov-1">等待画面…</div>
    </div>
    <div class="controls">
      <button id="rot-1">旋转180°</button>
      <button id="pause-1">暂停</button>
    </div>
    <div class="meta">
      <span><span class="dot" id="dot-1"></span><span id="state-1">连接中</span></span>
      <span id="seq-1">帧号 -</span>
      <span id="res-1">分辨率 -</span>
      <span id="age-1">时效 -</span>
    </div>
    <div class="meta err-text" id="err-1"></div>
  </div>
</div>
<div class="foot">
  两路独立采集，画面不保证同步。暂停仅暂停显示；本次服务最长运行2小时。
</div>
<script>
(function () {
  "use strict";
  var CAMERAS = [0, 1];
  var TARGET_INTERVAL = 1000 / 8;
  var FRAME_TIMEOUT = 3000;
  var STATUS_TIMEOUT = 3000;
  var STATUS_INTERVAL = 1000;
  var STALE_MS = 2000;

  function el(id) { return document.getElementById(id); }

  function setState(i, text, cls) {
    el("state-" + i).textContent = text;
    var dot = el("dot-" + i);
    dot.className = "dot" + (cls ? " " + cls : "");
  }

  function setOverlay(i, text) {
    var ov = el("ov-" + i);
    if (text) { ov.textContent = text; ov.classList.add("show"); }
    else { ov.classList.remove("show"); }
  }

  function formatAge(ms) {
    if (ms === null || ms === undefined) return "-";
    if (ms < 1000) return Math.round(ms) + " 毫秒前";
    return (ms / 1000).toFixed(1) + " 秒前";
  }

  // ---- lifecycle guards -------------------------------------------------
  var closed = false;
  var sleepers = new Set();
  var controllers = new Set();

  function later(fn, ms) {
    if (closed) return null;
    var id = setTimeout(function () {
      sleepers.delete(handle);
      if (!closed) { try { fn(); } catch (e) {} }
    }, ms);
    var handle = { id: id, resolve: null };
    sleepers.add(handle);
    return handle;
  }

  function sleep(ms) {
    return new Promise(function (resolve) {
      if (closed) { resolve(); return; }
      var handle = { id: null, resolve: resolve };
      handle.id = setTimeout(function () {
        sleepers.delete(handle);
        resolve();
      }, ms);
      sleepers.add(handle);
    });
  }

  function makeController() {
    var controller = new AbortController();
    controllers.add(controller);
    return controller;
  }

  function dropController(controller) {
    controllers.delete(controller);
  }

  function abortController(controller) {
    if (!controller) return;
    try { controller.abort(); } catch (e) {}
    controllers.delete(controller);
  }

  function revokeUrl(url) {
    if (!url) return;
    try { URL.revokeObjectURL(url); } catch (e) {}
  }

  // ---- per-camera state -------------------------------------------------
  var state = CAMERAS.map(function (i) {
    return {
      idx: i,
      paused: false,
      rotated: false,
      generation: 0,
      currentUrl: null,
      controller: null
    };
  });

  function decodeImage(url) {
    return new Promise(function (resolve, reject) {
      var probe = new Image();
      probe.onload = function () { resolve(url); };
      probe.onerror = function () { reject(new Error("image decode failed")); };
      probe.src = url;
    });
  }

  async function cameraLoop(st) {
    var i = st.idx;
    while (!closed) {
      if (!st.paused) {
        var gen = st.generation;
        var controller = makeController();
        st.controller = controller;
        var timeout = setTimeout(function () {
          try { controller.abort(); } catch (e) {}
        }, FRAME_TIMEOUT);
        var transientUrl = null;
        try {
          var resp = await fetch("/frame/" + i + ".jpg?t=" + Date.now(), {
            cache: "no-store",
            signal: controller.signal
          });
          if (!resp.ok) throw new Error("HTTP " + resp.status);
          var seqHeader = resp.headers.get("X-Frame-Sequence");
          var blob = await resp.blob();
          if (closed) throw new Error("closed");
          if (gen !== st.generation || st.paused) {
            throw new Error("superseded");
          }
          transientUrl = URL.createObjectURL(blob);
          await decodeImage(transientUrl);
          if (closed) throw new Error("closed");
          if (gen !== st.generation || st.paused) {
            throw new Error("superseded");
          }
          var img = el("img-" + i);
          var oldUrl = st.currentUrl;
          img.src = transientUrl;
          img.classList.remove("dim");
          st.currentUrl = transientUrl;
          transientUrl = null;
          revokeUrl(oldUrl);
          if (seqHeader !== null && seqHeader !== undefined && seqHeader !== "") {
            el("seq-" + i).textContent = "帧号 " + seqHeader;
          }
          setOverlay(i, null);
          setState(i, "在线", "live");
        } catch (err) {
          revokeUrl(transientUrl);
          transientUrl = null;
          if (!closed && gen === st.generation && !st.paused) {
            var im = el("img-" + i);
            im.classList.add("dim");
            setOverlay(i, "连接中 / 画面中断");
            setState(i, "画面中断", "err");
          }
        } finally {
          clearTimeout(timeout);
          dropController(controller);
          if (st.controller === controller) st.controller = null;
        }
      }
      await sleep(TARGET_INTERVAL);
    }
  }

  // ---- status polling ---------------------------------------------------
  var statusTimer = null;

  async function statusLoop() {
    while (!closed) {
      var controller = makeController();
      var timeout = setTimeout(function () {
        try { controller.abort(); } catch (e) {}
      }, STATUS_TIMEOUT);
      try {
        var resp = await fetch("/status", {
          cache: "no-store",
          signal: controller.signal
        });
        if (!resp.ok) throw new Error("status");
        var data = await resp.json();
        if (closed) break;
        applyStatus(data);
      } catch (err) {
        if (closed) break;
        applyStatusFailure();
      } finally {
        clearTimeout(timeout);
        dropController(controller);
      }
      await sleep(STATUS_INTERVAL);
    }
  }

  function applyStatus(data) {
    if (!data || !Array.isArray(data.cameras)) return;
    data.cameras.forEach(function (cam, i) {
      if (i > 1 || !cam) return;
      var st = state[i];
      el("dev-" + i).textContent = cam.device || "";
      if (cam.width && cam.height) {
        el("res-" + i).textContent = cam.width + "×" + cam.height;
      }
      el("age-" + i).textContent = "时效 " + formatAge(cam.age_ms);
      if (st.paused) return;
      var hasError = cam.error !== null && cam.error !== undefined && cam.error !== "";
      if (hasError) {
        el("err-" + i).textContent = cam.error;
        var im = el("img-" + i);
        im.classList.add("dim");
        setOverlay(i, "连接中 / 画面中断");
        setState(i, "错误", "err");
      } else if (cam.age_ms === null || cam.age_ms === undefined) {
        el("err-" + i).textContent = "";
        var im2 = el("img-" + i);
        im2.classList.add("dim");
        setOverlay(i, "等待画面…");
        setState(i, "连接中", "stale");
      } else if (cam.age_ms > STALE_MS) {
        el("err-" + i).textContent = "";
        var im3 = el("img-" + i);
        im3.classList.add("dim");
        setOverlay(i, "连接中 / 画面中断");
        setState(i, "画面中断", "err");
      } else {
        el("err-" + i).textContent = "";
      }
    });
  }

  function applyStatusFailure() {
    state.forEach(function (st) {
      if (st.paused) return;
      var im = el("img-" + st.idx);
      im.classList.add("dim");
      setOverlay(st.idx, "连接中 / 画面中断");
      setState(st.idx, "画面中断", "err");
    });
  }

  // ---- controls ---------------------------------------------------------
  CAMERAS.forEach(function (i) {
    var st = state[i];
    el("rot-" + i).addEventListener("click", function () {
      st.rotated = !st.rotated;
      el("img-" + i).style.transform = st.rotated ? "rotate(180deg)" : "";
    });
    el("pause-" + i).addEventListener("click", function () {
      st.paused = !st.paused;
      st.generation += 1;
      var btn = el("pause-" + i);
      if (st.paused) {
        btn.textContent = "继续";
        abortController(st.controller);
        st.controller = null;
        var img = el("img-" + i);
        img.classList.add("dim");
        setOverlay(i, "已暂停显示");
        setState(i, "暂停", "stale");
      } else {
        btn.textContent = "暂停";
        setOverlay(i, "连接中");
        setState(i, "连接中", "stale");
      }
    });
  });

  function cleanup() {
    if (closed) return;
    closed = true;
    if (statusTimer) { clearTimeout(statusTimer); statusTimer = null; }
    controllers.forEach(function (c) {
      try { c.abort(); } catch (e) {}
    });
    controllers.clear();
    sleepers.forEach(function (h) {
      try { clearTimeout(h.id); } catch (e) {}
      if (h.resolve) { try { h.resolve(); } catch (e) {} }
    });
    sleepers.clear();
    state.forEach(function (st) {
      revokeUrl(st.currentUrl);
      st.currentUrl = null;
      st.controller = null;
    });
  }

  window.addEventListener("pagehide", function () {
    cleanup();
  });
  window.addEventListener("pageshow", function (ev) {
    if (ev && ev.persisted) {
      try { window.location.reload(); } catch (e) {}
    }
  });

  CAMERAS.forEach(function (i) {
    cameraLoop(state[i]);
  });
  statusLoop();
})();
</script>
</body>
</html>
"""


def _make_handler(feeds):
    class PreviewHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def log_error(self, fmt, *args):
            pass

        def _send(self, code, body: bytes, content_type: str, extra_headers=None):
            try:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                if extra_headers:
                    for k, v in extra_headers.items():
                        self.send_header(k, str(v))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            self._route()

        def do_HEAD(self):
            self._route()

        def _route(self):
            try:
                path = urlsplit(self.path).path
            except Exception:
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return

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
            self._send(404, b"not found", "text/plain; charset=utf-8")

        def _status(self):
            cams = []
            for w in feeds:
                snap = w.snapshot()
                snap.pop("jpeg", None)
                cams.append(snap)
            try:
                body = json.dumps({"cameras": cams}, ensure_ascii=False).encode("utf-8")
            except Exception:
                self._send(500, b"status error", "text/plain; charset=utf-8")
                return
            self._send(200, body, "application/json; charset=utf-8")

        def _frame(self, idx):
            snap = feeds[idx].snapshot()
            jpeg = snap["jpeg"]
            error = snap["error"]
            age_ms = snap["age_ms"]
            seq = snap["seq"]
            if jpeg is None or error is not None or age_ms is None or age_ms > STALE_MS:
                self._send(503, b"frame unavailable", "text/plain; charset=utf-8")
                return
            self._send(
                200, jpeg, "image/jpeg", extra_headers={"X-Frame-Sequence": seq}
            )

    return PreviewHandler


def create_server(host: str, port: int, feeds: list) -> ThreadingHTTPServer:
    if len(feeds) != 2:
        raise ValueError("exactly two feeds required")
    handler = _make_handler(feeds)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def _validate_cli(args) -> tuple[bool, str]:
    if not (1 <= args.port <= 65535):
        return False, "端口必须在 1..65535 之间"
    try:
        max_seconds = float(args.max_seconds)
    except (TypeError, ValueError):
        return False, "--max-seconds 必须是数字"
    if not math.isfinite(max_seconds):
        return False, "--max-seconds 必须是有限数字"
    if max_seconds <= 0:
        return False, "--max-seconds 必须为正数"
    if not os.path.exists(args.camera_a) or not os.path.exists(args.camera_b):
        return False, "摄像头设备路径不存在"
    if os.path.realpath(args.camera_a) == os.path.realpath(args.camera_b):
        return False, "两个摄像头设备路径必须不同"
    return True, ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="双摄像头实时预览")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--camera-a", required=True)
    parser.add_argument("--camera-b", required=True)
    parser.add_argument("--max-seconds", default=7200)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 2

    ok, msg = _validate_cli(args)
    if not ok:
        print(f"参数错误: {msg}", file=sys.stderr)
        return 2

    try:
        max_seconds = float(args.max_seconds)
    except (TypeError, ValueError):
        print("参数错误: --max-seconds 必须是数字", file=sys.stderr)
        return 2

    worker_a = CameraWorker(args.camera_a, "摄像头1")
    worker_b = CameraWorker(args.camera_b, "摄像头2")
    feeds = [worker_a, worker_b]

    try:
        server = create_server(args.host, args.port, feeds)
    except OSError as exc:
        print(f"无法启动 HTTP 服务: {exc}", file=sys.stderr)
        return 1

    server.timeout = 0.5
    stop_event = threading.Event()

    def _on_signal(signum, frame):
        stop_event.set()

    old_int = signal.signal(signal.SIGINT, _on_signal)
    old_term = signal.signal(signal.SIGTERM, _on_signal)

    print(
        f"预览服务已启动: http://{args.host}:{args.port}/ "
        f"(最长运行 {int(max_seconds)} 秒)",
        file=sys.stderr,
    )

    started = time.monotonic()
    try:
        worker_a.start()
        worker_b.start()
        while not stop_event.is_set():
            if (time.monotonic() - started) >= max_seconds:
                print("已达到最长运行时间，退出。", file=sys.stderr)
                break
            try:
                server.handle_request()
            except Exception as exc:
                print(f"请求处理异常: {exc}", file=sys.stderr)
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"服务异常: {exc}", file=sys.stderr)
        return 1
    finally:
        stop_event.set()
        try:
            server.server_close()
        except Exception:
            pass
        worker_a.stop()
        worker_b.stop()
        try:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
        except Exception:
            pass

    print("预览服务已停止。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
