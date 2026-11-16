"""Local capture bridge v1.

A stdlib-only Windows loopback server / launcher adapter that:

* reuses ``local_capture_storage_v2`` for safe ZIP extraction to a LOCAL folder
* reuses ``dual_camera_launcher_v2`` for connecting/starting the Ubuntu preview
* serves the same dual-camera preview UI from ``data_collection_preview_v9.py``
  (extracted via ``ast``, never imported) and rewrites it so that capture
  results are auto-saved onto this Windows machine.

Run ``python local_capture_bridge_v1.py --serve ...`` to serve.
Run ``python local_capture_bridge_v1.py --launch ...`` to launch (default).
"""

from __future__ import annotations

import ast
import hashlib
import hmac
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_HERE = Path(__file__).resolve()
_VISION_EVAL_DIR = _HERE.parent
_PROJECT_ROOT = _VISION_EVAL_DIR.parent
_RESULTS_DIR = _VISION_EVAL_DIR / "results" / "local_capture_bridge"
_LOG_PATH = _RESULTS_DIR / "server.log"
_LAST_RUN_PATH = _RESULTS_DIR / "last_run.json"

_DEFAULT_TEMPLATE = _VISION_EVAL_DIR / "data_collection_preview_v17.py"
_DEFAULT_LOCAL_ROOT = _PROJECT_ROOT / "chessreal_dual"
_DEFAULT_UPSTREAM = "http://127.0.0.1:8766/"

_CAPTURE_TOKEN_HEADER = "X-Local-Capture-Token"
_MARKER_SERVICE = "chess_local_capture_v1"

_MAX_STATUS_BODY = 1 * 1024 * 1024
_MAX_CAPTURE_BODY = 1 * 1024 * 1024
_MAX_FRAME_BODY = 8 * 1024 * 1024
_MAX_ZIP_BODY = 64 * 1024 * 1024
_TIMEOUT_FRAME = 5.0
_TIMEOUT_STATUS = 5.0
_TIMEOUT_CAPTURE = 60.0
_TIMEOUT_ZIP = 30.0
_MAX_VIEW_QUERY = 4096

_EXTRA_FORWARD_HEADERS = (
    "X-Frame-Sequence",
    "X-Capture-Time",
    "X-Frame-Age",
    "X-Inference-Age",
    "X-Inference-Ms",
    "X-Detection-Count",
    "X-Image-Width",
    "X-Image-Height",
    "X-Rotation",
    "X-View-Mode",
)

_ALLOWED_HOSTS = ("127.0.0.1", "localhost")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _source_sha256() -> str:
    try:
        data = _HERE.read_bytes()
    except Exception:
        return ""
    return hashlib.sha256(data).hexdigest()


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _ensure_results_dir() -> None:
    try:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _sanitize_error(msg) -> str:
    text = str(msg or "")
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace(str(_HERE), "<bridge>")
    if "token" in text.lower():
        text = "error (details hidden)"
    return text[:2000]


def _write_last_run(payload: dict) -> None:
    _ensure_results_dir()
    try:
        tmp = _LAST_RUN_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(_LAST_RUN_PATH))
    except Exception:
        pass


def _validate_upstream_url(url: str) -> str:
    if not isinstance(url, str) or not url:
        raise ValueError("upstream URL must be a non-empty string")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "http":
        raise ValueError("upstream URL must use http scheme")
    if not parts.netloc:
        raise ValueError("upstream URL must have a host")
    if parts.username or parts.password:
        raise ValueError("upstream URL must not contain credentials")
    if parts.query or parts.fragment:
        raise ValueError("upstream URL must not contain query or fragment")
    path = parts.path or "/"
    if not path.endswith("/"):
        path = path + "/"
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, "", ""))


# ---------------------------------------------------------------------------
# module imports (stdlib-only, lazily loaded from siblings)
# ---------------------------------------------------------------------------


def _load_storage():
    if str(_VISION_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(_VISION_EVAL_DIR))
    import local_capture_storage_v2 as storage  # type: ignore

    return storage


def _load_launcher():
    if str(_VISION_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(_VISION_EVAL_DIR))
    import dual_camera_launcher_v2 as launcher  # type: ignore

    return launcher


# ---------------------------------------------------------------------------
# template extraction / rewriting
# ---------------------------------------------------------------------------


def _extract_index_html(template_path: Path) -> str:
    try:
        text = Path(template_path).read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError("cannot read template %s: %s" % (template_path, exc))
    try:
        tree = ast.parse(text, filename=str(template_path))
    except SyntaxError as exc:
        raise RuntimeError("cannot parse template %s: %s" % (template_path, exc))

    found = []
    for node in tree.body:
        targets = []
        value = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        if not targets or value is None:
            continue
        for tgt in targets:
            if isinstance(tgt, ast.Name) and tgt.id == "INDEX_HTML":
                found.append(value)
                break
    if len(found) != 1:
        raise RuntimeError(
            "template %s must define INDEX_HTML exactly once (found %d)"
            % (template_path, len(found))
        )
    try:
        html = ast.literal_eval(found[0])
    except Exception as exc:
        raise RuntimeError("INDEX_HTML is not a literal: %s" % exc)
    if not isinstance(html, str):
        raise RuntimeError("INDEX_HTML must be a string")
    return html


def _replace_once(html: str, old: str, new: str, label: str) -> str:
    count = html.count(old)
    if count != 1:
        raise RuntimeError(
            "template replacement failed for %s: expected exactly 1 occurrence, found %d"
            % (label, count)
        )
    return html.replace(old, new)


_OLD_FETCH_HEADERS = (
    'headers: {"Content-Type": "application/json"},'
)
_NEW_FETCH_HEADERS = (
    'headers: JSON.parse(__b64Decode("__LOCAL_HEADERS_B64__")),'
)


def _rewrite_ui(html: str, token: str) -> str:
    # Localized labels / storage keys / messages.
    html = _replace_once(
        html,
        "保存目录（Ubuntu）",
        "保存目录（本地 Windows）",
        "save-dir-label",
    )
    html = _replace_once(
        html,
        "chess_capture_root_v1",
        "chess_local_capture_root_v1",
        "storage-key",
    )
    html = _replace_once(
        html,
        "Windows 本地文件可用 ZIP 下载。",
        "拍摄后自动保存到本机，无需手动下载。",
        "hint-text",
    )
    html = _replace_once(
        html,
        "拍摄成功。保存目录: ",
        "已保存到本地：",
        "success-text",
    )
    html = _replace_once(
        html,
        'var rootValue = currentCaptureRoot();',
        'var rootValue = currentCaptureRoot();',
        "current-capture-root-anchor",
    )

    # Browsers cannot set arbitrary headers via a JSON.parse of an object;
    # however fetch() headers option accepts an object. Use a small helper
    # that builds the object at runtime so the token bytes stay identical.
    headers_js = json.dumps(
        {"Content-Type": "application/json", _CAPTURE_TOKEN_HEADER: token},
        ensure_ascii=False,
    )
    encoded = _b64encode(headers_js)

    # Handle the two variants in the template: one with payload containing
    # capture_root and one without.
    html = _replace_once(html, _OLD_FETCH_HEADERS, _NEW_FETCH_HEADERS, "fetch-headers-anchor")

    # Ensure helper function exists before its use.
    helper = (
        '<script>\n'
        'function __b64Decode(s){\n'
        '  try {\n'
        '    var bin = atob(s);\n'
        '    var bytes = new Uint8Array(bin.length);\n'
        '    for (var i = 0; i < bin.length; i++) { bytes[i] = bin.charCodeAt(i); }\n'
        '    return new TextDecoder("utf-8").decode(bytes);\n'
        '  } catch(e) { return "{}"; }\n'
        '}\n'
        '</script>\n'
    )
    html = html.replace("</head>", helper + "</head>", 1)
    html = html.replace("__LOCAL_HEADERS_B64__", encoded, 1)
    html = html.replace("</head>", '<script>window.__votingHeaders=JSON.parse(__b64Decode("'+encoded+'"));</script></head>', 1)

    # Concise notice: current page saves to this Windows computer.
    notice = (
        '<div id="local-capture-notice" style="margin:8px 0;padding:8px 12px;'
        'border-radius:6px;background:rgba(60,160,120,0.18);color:#cfe;'
        'font-size:13px;line-height:1.5;">'
        '照片自动保存到当前 Windows 本机；Ubuntu 也保留拍摄备份。'
        '</div>'
    )
    if "</body>" in html:
        html = html.replace("</body>", notice + "</body>", 1)
    else:
        html = html + notice
    return html


def _b64encode(text: str) -> str:
    import base64

    return base64.b64encode(text.encode("utf-8")).decode("ascii")


# ---------------------------------------------------------------------------
# upstream client
# ---------------------------------------------------------------------------


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def http_error_301(self, req, fp, code, msg, headers):
        return fp

    def http_error_302(self, req, fp, code, msg, headers):
        return fp

    def http_error_303(self, req, fp, code, msg, headers):
        return fp

    def http_error_307(self, req, fp, code, msg, headers):
        return fp

    def http_error_308(self, req, fp, code, msg, headers):
        return fp

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _build_opener():
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirectHandler()
    )


def _read_bounded(resp, limit: int) -> bytes:
    chunks = []
    total = 0
    while True:
        try:
            chunk = resp.read(min(65536, limit - total + 1))
        except Exception as exc:
            raise _UpstreamError("上游响应读取中断: %s" % exc)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise _UpstreamTooLarge("upstream body exceeds limit")
        chunks.append(chunk)
    return b"".join(chunks)


class _UpstreamTooLarge(Exception):
    pass


class _UpstreamResult:
    __slots__ = ("status", "headers", "body")

    def __init__(self, status, headers, body):
        self.status = status
        self.headers = headers
        self.body = body


def _http_request(method, url, limit, timeout, data=None, extra_headers=None):
    opener = _build_opener()
    headers = dict(extra_headers or {})
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with opener.open(req, timeout=timeout) as resp:
            body = _read_bounded(resp, limit)
            return _UpstreamResult(resp.status, dict(resp.headers.items()), body)
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read(limit + 1)
        except Exception:
            raw = b""
        if len(raw) > limit:
            raw = raw[:limit]
        return _UpstreamResult(exc.code, dict(exc.headers.items()) if exc.headers else {}, raw)
    except _UpstreamTooLarge:
        raise
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise _UpstreamError("无法连接上游预览服务: %s" % reason)
    except socket.timeout:
        raise _UpstreamError("上游预览服务响应超时")
    except Exception as exc:
        raise _UpstreamError("上游预览服务请求失败: %s" % exc)


class _UpstreamError(Exception):
    pass


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class _CaptureState:
    def __init__(self, upstream_url, default_root, token):
        self.upstream_url = upstream_url
        self.default_root = default_root
        self.token = token
        self.lock = threading.Lock()
        self.source_sha256 = _source_sha256()


class _BridgeHandler(BaseHTTPRequestHandler):
    server_version = "LocalCaptureBridge/1.0"
    sys_version = ""

    # silence per-frame request logs
    def log_message(self, fmt, *args):
        return

    # ---- helpers -------------------------------------------------------

    def _send_json(self, status, obj, extra_headers=None):
        body = _json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _send_error_json(self, status, message, extra_headers=None):
        self._send_json(status, {"error": message}, extra_headers)

    def _validate_host(self):
        host_header = self.headers.get("Host", "")
        if not host_header:
            return False
        actual_port = self.server.server_address[1]
        allowed = {
            "127.0.0.1:%d" % actual_port,
            "localhost:%d" % actual_port,
        }
        return host_header.strip().lower() in {a.lower() for a in allowed}

    def _sec_fetch_ok(self):
        sfs = self.headers.get("Sec-Fetch-Site")
        if sfs is None:
            return True
        return sfs.strip().lower() in ("same-origin", "none")

    def _origin_ok(self):
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        actual_port = self.server.server_address[1]
        allowed = {
            "http://127.0.0.1:%d" % actual_port,
            "http://localhost:%d" % actual_port,
        }
        return origin.strip().lower() in {a.lower() for a in allowed}

    def _constant_time_token_ok(self):
        provided = self.headers.get(_CAPTURE_TOKEN_HEADER)
        if not provided:
            return False
        expected = self.server.capture_state.token
        try:
            return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))
        except Exception:
            return False

    def _common_guard(self):
        if not self._validate_host():
            self._send_error_json(403, "invalid Host header")
            return False
        if not self._sec_fetch_ok():
            self._send_error_json(403, "cross-site request rejected")
            return False
        return True

    def do_OPTIONS(self):
        self._send_error_json(403, "cross-origin access disabled")

    # ---- GET -----------------------------------------------------------

    def do_GET(self):
        if not self._common_guard():
            return
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        query = parsed.query

        if path == "/":
            return self._serve_index()
        if path == "/local-health":
            return self._serve_health()

        relay_map = {
            "/status": ("/status", _MAX_STATUS_BODY, _TIMEOUT_STATUS, False),
            "/frame/0.jpg": ("/frame/0.jpg", _MAX_FRAME_BODY, _TIMEOUT_FRAME, False),
            "/frame/1.jpg": ("/frame/1.jpg", _MAX_FRAME_BODY, _TIMEOUT_FRAME, False),
            "/view/0.jpg": ("/view/0.jpg", _MAX_FRAME_BODY, _TIMEOUT_FRAME, True),
            "/view/1.jpg": ("/view/1.jpg", _MAX_FRAME_BODY, _TIMEOUT_FRAME, True),
        }
        if path in relay_map:
            upstream_path, limit, timeout, allow_query = relay_map[path]
            return self._relay_get(upstream_path, limit, timeout, query if allow_query else "")

        if path.startswith("/download/") and path.endswith(".zip"):
            remainder = path[len("/download/"):-len(".zip")]
            if not _is_safe_capture_id(remainder):
                return self._send_error_json(400, "invalid capture id")
            return self._relay_get(
                "/download/%s.zip" % remainder, _MAX_ZIP_BODY, _TIMEOUT_ZIP, ""
            )

        return self._send_error_json(404, "not found")

    def _serve_index(self):
        html = getattr(self.server, "index_html", "")
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _serve_health(self):
        self._send_json(
            200,
            {
                "service": _MARKER_SERVICE,
                "source_sha256": self.server.capture_state.source_sha256,
                "ready": True,
            },
        )

    def _relay_get(self, upstream_path, limit, timeout, query):
        state = self.server.capture_state
        url = urllib.parse.urljoin(state.upstream_url, upstream_path.lstrip("/"))
        if query:
            if len(query) > _MAX_VIEW_QUERY:
                return self._send_error_json(400, "query too long")
            url = url + "?" + query
        try:
            result = _http_request("GET", url, limit, timeout)
        except _UpstreamTooLarge:
            return self._send_error_json(502, "上游响应过大")
        except _UpstreamError as exc:
            return self._send_error_json(502, "上游请求失败: %s" % exc)

        if upstream_path == "/status" and result.status == 200:
            try:
                status = json.loads(result.body.decode("utf-8"))
                collection = status["collection"]
                if not isinstance(collection, dict):
                    raise ValueError("missing collection")
                collection["remote_capture_root"] = collection.get("capture_root")
                collection["capture_root"] = state.default_root
                collection["local_capture"] = True
                result.body = _json_bytes(status)
            except (ValueError, KeyError, TypeError):
                return self._send_error_json(502, "上游状态格式无效")
        self.send_response(result.status)
        self._forward_headers(result.headers)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(result.body)))
        self.end_headers()
        try:
            self.wfile.write(result.body)
        except Exception:
            pass

    def _forward_headers(self, headers):
        ct = headers.get("Content-Type")
        if ct:
            self.send_header("Content-Type", ct)
        for name in _EXTRA_FORWARD_HEADERS:
            val = headers.get(name)
            if val is not None:
                self.send_header(name, val)

    # ---- POST ----------------------------------------------------------

    def do_POST(self):
        if not self._common_guard():
            return
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path not in ("/capture", "/voting/settings"):
            return self._send_error_json(404, "not found")
        if not self._origin_ok():
            return self._send_error_json(403, "invalid Origin header")
        if not self._constant_time_token_ok():
            return self._send_error_json(403, "invalid capture token")

        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return self._send_error_json(415, "Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            return self._send_error_json(400, "invalid Content-Length")
        if length < 0 or length > 8192:
            return self._send_error_json(413, "request body too large")
        try:
            raw = self.rfile.read(length) if length else b""
        except Exception:
            return self._send_error_json(400, "cannot read request body")
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            return self._send_error_json(400, "invalid JSON body")
        if not isinstance(body, dict):
            return self._send_error_json(400, "invalid JSON body")

        if parsed.path == "/voting/settings":
            if set(body) != {"window_s"} or type(body.get("window_s")) is not int or body["window_s"] not in (1,2,3,4):
                return self._send_error_json(400, "window_s must be 1,2,3,4")
            try:
                up = _http_request("POST", urllib.parse.urljoin(self.server.capture_state.upstream_url, "voting/settings"), 8192, 5.0,
                                   data=_json_bytes(body), extra_headers={"Content-Type":"application/json"})
                result = json.loads(up.body.decode("utf-8"))
                if not isinstance(result, dict): raise ValueError("invalid reply")
                return self._send_json(up.status, result)
            except Exception:
                return self._send_error_json(502, "upstream voting settings unavailable")

        rotations = body.get("rotations")
        if not isinstance(rotations, list) or len(rotations) != 2:
            return self._send_error_json(400, "rotations must be a list of 2 integers")
        clean_rot = []
        for r in rotations:
            if not isinstance(r, int) or isinstance(r, bool) or r not in (0, 90, 180, 270):
                return self._send_error_json(400, "rotations must be 0/90/180/270")
            clean_rot.append(int(r))

        storage = _load_storage()
        raw_root = body.get("capture_root")
        if raw_root is None or (isinstance(raw_root, str) and raw_root.strip() == ""):
            raw_root = str(self.server.capture_state.default_root)
        if not isinstance(raw_root, str):
            return self._send_error_json(400, "capture_root must be a string")
        try:
            canon_root = storage.validate_root(raw_root)
        except Exception as exc:
            return self._send_error_json(400, "invalid local root: %s" % exc)

        lock = self.server.capture_state.lock
        if not lock.acquire(blocking=False):
            return self._send_error_json(409, "capture busy")
        try:
            return self._handle_capture(clean_rot, canon_root, storage)
        finally:
            lock.release()

    def _handle_capture(self, rotations, canon_root, storage):
        state = self.server.capture_state
        upstream_capture = urllib.parse.urljoin(state.upstream_url, "capture")
        payload = _json_bytes({"rotations": rotations})

        try:
            up = _http_request(
                "POST",
                upstream_capture,
                _MAX_CAPTURE_BODY,
                _TIMEOUT_CAPTURE,
                data=payload,
                extra_headers={"Content-Type": "application/json"},
            )
        except _UpstreamTooLarge:
            return self._send_error_json(502, "上游拍摄响应过大")
        except _UpstreamError as exc:
            return self._send_error_json(502, "上游拍摄失败: %s" % exc)

        if up.status != 200:
            msg = _extract_upstream_error(up.body, up.status)
            return self._send_error_json(up.status, msg)

        try:
            up_json = json.loads(up.body.decode("utf-8"))
        except Exception:
            return self._send_error_json(502, "上游返回了无效的 JSON")
        if not isinstance(up_json, dict):
            return self._send_error_json(502, "上游返回了无效的 JSON")
        capture_id = up_json.get("id")
        if not _is_safe_capture_id(capture_id):
            return self._send_error_json(502, "上游返回了无效的拍摄 ID")

        zip_path = urllib.parse.urljoin(
            state.upstream_url, "download/%s.zip" % capture_id
        )
        try:
            zres = _http_request("GET", zip_path, _MAX_ZIP_BODY, _TIMEOUT_ZIP)
        except _UpstreamTooLarge:
            return self._send_error_json(502, "上游 ZIP 过大")
        except _UpstreamError as exc:
            return self._send_error_json(
                502,
                "拍摄已在上游保留 (group id: %s)，但下载失败: %s" % (capture_id, exc),
            )
        if zres.status != 200:
            return self._send_error_json(
                zres.status,
                "拍摄已在上游保留 (group id: %s)，但下载失败: HTTP %d"
                % (capture_id, zres.status),
            )

        try:
            save = storage.save_capture(zres.body, canon_root, capture_id)
        except Exception as exc:
            return self._send_error_json(
                500,
                "拍摄已在上游保留 (group id: %s)，但本地保存失败: %s"
                % (capture_id, exc),
            )

        upstream_warning = up_json.get("warning")
        remote_dir = up_json.get("server_directory")
        result = {
            "id": capture_id,
            "server_directory": save.get("local_directory"),
            "local_directory": save.get("local_directory"),
            "local_saved": True,
            "saved_files": save.get("files", []),
            "pair_file": save.get("pair_file"),
            "download_url": None,
            "warning": upstream_warning or None,
            "remote_directory": remote_dir,
        }
        return self._send_json(200, result)


def _extract_upstream_error(body, status):
    try:
        obj = json.loads(body.decode("utf-8"))
        if isinstance(obj, dict) and obj.get("error"):
            return str(obj["error"])
    except Exception:
        pass
    try:
        text = body.decode("utf-8", "replace").strip()
    except Exception:
        text = ""
    if text:
        return text[:500]
    return "上游返回 HTTP %d" % status


def _is_safe_capture_id(cid):
    if not isinstance(cid, str):
        return False
    if not cid or len(cid) > 128:
        return False
    for ch in cid:
        if not (ch.isalnum() or ch in ("-", "_")):
            return False
    return True


# ---------------------------------------------------------------------------
# server construction
# ---------------------------------------------------------------------------


def create_server(
    upstream_url: str,
    default_root: str,
    host: str = "127.0.0.1",
    port: int = 8766,
    template_path=None,
) -> ThreadingHTTPServer:
    if host not in _ALLOWED_HOSTS:
        raise ValueError("host must be 127.0.0.1 or localhost")

    upstream_url = _validate_upstream_url(upstream_url)
    default_root_str = _load_storage().validate_root(str(default_root))

    tmpl = Path(template_path) if template_path else _DEFAULT_TEMPLATE
    raw_html = _extract_index_html(tmpl)

    token = secrets.token_urlsafe(32)
    html = _rewrite_ui(raw_html, token)

    capture_state = _CaptureState(upstream_url, default_root_str, token)

    server = ThreadingHTTPServer((host, int(port)), _BridgeHandler)
    server.daemon_threads = True
    server.capture_state = capture_state
    server.capture_token = token
    server.capture_lock = capture_state.lock
    server.source_sha256 = capture_state.source_sha256
    server.index_html = html
    return server


# ---------------------------------------------------------------------------
# launcher side
# ---------------------------------------------------------------------------


def _probe_health_direct(host, port, timeout=1.0):
    url = "http://%s:%d/local-health" % (host, port)
    try:
        opener = _build_opener()
        with opener.open(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            raw = resp.read(65536)
        obj = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("service") != _MARKER_SERVICE:
        return None
    return obj


def _port_in_use(host, port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def _python_for_child():
    if os.name == "nt":
        exe = Path(sys.executable)
        pythonw = exe.with_name("pythonw.exe")
        if pythonw.exists():
            return str(pythonw)
        return sys.executable
    return sys.executable


def _launcher_messagebox(no_ui, title, message):
    if no_ui:
        return
    try:
        launcher = _load_launcher()
        mb = getattr(launcher, "_messagebox", None)
        if callable(mb):
            mb(title + "：" + message)
    except Exception:
        pass


def launch_bridge(config_path=None, port: int = 8766, open_browser: bool = True, default_root=None) -> dict:
    no_ui = not open_browser
    started = False
    try:
        launcher = _load_launcher()
        config = launcher.load_config(config_path)
        try:
            ensure_result = launcher.ensure_preview(config, open_browser=False)
        except Exception as exc:
            raise RuntimeError("无法连接/启动 Ubuntu 预览服务: %s" % exc)

        if not isinstance(ensure_result, dict) or not ensure_result.get("ready"):
            warning = ""
            if isinstance(ensure_result, dict):
                warning = ensure_result.get("warning") or ""
            message = "Ubuntu 预览服务未就绪。%s" % (warning or "")
            _write_last_run(
                {"ok": False, "error": _sanitize_error(message), "ts": time.time()}
            )
            _launcher_messagebox(no_ui, "本地拍摄桥", message)
            return {
                "url": "http://127.0.0.1:%d/" % port,
                "ready": False,
                "started": False,
                "error": message,
                "upstream": ensure_result,
            }

        upstream_url = _derive_upstream_url(config)
        default_root = _load_storage().validate_root(str(default_root or _derive_default_root(config)))

        existing = _probe_health_direct("127.0.0.1", port)
        if existing is not None:
            current_sha = _source_sha256()
            if existing.get("source_sha256") == current_sha:
                url = "http://127.0.0.1:%d/" % port
                if open_browser:
                    _open_browser(url)
                _write_last_run(
                    {
                        "ok": True,
                        "url": url,
                        "started": False,
                        "ready": True,
                        "reused": True,
                        "ts": time.time(),
                    }
                )
                return {"url": url, "ready": True, "started": False, "reused": True}
            message = "端口 %d 已被不同版本的本地拍摄桥占用，请先关闭它。" % port
            _write_last_run(
                {"ok": False, "error": _sanitize_error(message), "ts": time.time()}
            )
            _launcher_messagebox(no_ui, "本地拍摄桥", message)
            raise RuntimeError(message)

        if _port_in_use("127.0.0.1", port):
            message = "端口 %d 已被其他程序占用，无法启动本地拍摄桥。" % port
            _write_last_run(
                {"ok": False, "error": _sanitize_error(message), "ts": time.time()}
            )
            _launcher_messagebox(no_ui, "本地拍摄桥", message)
            raise RuntimeError(message)

        _ensure_results_dir()
        child_argv = [
            _python_for_child(),
            str(_HERE),
            "--serve",
            "--port",
            str(int(port)),
            "--default-root",
            default_root,
        ]
        if config_path:
            child_argv.extend(["--config", str(config_path)])

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP

        env = dict(os.environ)
        # pass upstream url via env to avoid argv confusion, no secrets copied
        env["LOCAL_CAPTURE_BRIDGE_UPSTREAM"] = upstream_url

        log_fh = open(str(_LOG_PATH), "ab", buffering=0)
        try:
            proc = subprocess.Popen(
                child_argv,
                cwd=str(_PROJECT_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=log_fh,
                close_fds=True,
                creationflags=creationflags,
                env=env,
            )
        finally:
            try:
                log_fh.close()
            except Exception:
                pass
        started = True

        healthy = _wait_for_health(port, timeout=10.0, interval=0.2)
        if not healthy:
            message = "本地拍摄桥未能在限定时间内就绪，请查看日志：%s" % _LOG_PATH
            _write_last_run(
                {"ok": False, "error": _sanitize_error(message), "ts": time.time()}
            )
            _launcher_messagebox(no_ui, "本地拍摄桥", message)
            return {
                "url": "http://127.0.0.1:%d/" % port,
                "ready": False,
                "started": True,
                "pid": proc.pid,
                "error": message,
            }

        url = "http://127.0.0.1:%d/" % port
        if open_browser:
            _open_browser(url)
        result = {
            "url": url,
            "ready": True,
            "started": True,
            "pid": proc.pid,
        }
        _write_last_run({"ok": True, **result, "ts": time.time()})
        return result
    except Exception as exc:
        message = _sanitize_error(exc)
        _write_last_run({"ok": False, "error": message, "ts": time.time()})
        _launcher_messagebox(no_ui, "本地拍摄桥", "启动失败: %s" % message)
        return {
            "url": "http://127.0.0.1:%d/" % port,
            "ready": False,
            "started": started,
            "error": message,
        }


def _wait_for_health(port, timeout, interval):
    deadline = time.monotonic() + timeout
    expected_sha = _source_sha256()
    while time.monotonic() < deadline:
        info = _probe_health_direct("127.0.0.1", port)
        if info is not None and info.get("source_sha256") == expected_sha:
            return True
        time.sleep(interval)
    return False


def _derive_upstream_url(config):
    if isinstance(config, dict):
        for key in ("upstream_url", "local_upstream", "bridge_upstream"):
            val = config.get(key)
            if isinstance(val, str) and val:
                return _validate_upstream_url(val)
        url = config.get("url")
        if isinstance(url, str) and url:
            return _validate_upstream_url(url)
    env = os.environ.get("LOCAL_CAPTURE_BRIDGE_UPSTREAM")
    if env:
        return _validate_upstream_url(env)
    return _validate_upstream_url(_DEFAULT_UPSTREAM)


def _derive_default_root(config):
    if isinstance(config, dict):
        val = config.get("capture_root") or config.get("default_root")
        if isinstance(val, str) and val:
            return str(val)
    return str(_DEFAULT_LOCAL_ROOT)


def _open_browser(url):
    try:
        import webbrowser

        webbrowser.open(url, new=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def _config_upstream_and_root(config_path):
    config = _load_launcher().load_config(config_path)
    return _validate_upstream_url(config["url"]), _derive_default_root(config)

def _serve(config_path=None, port=8766, default_root=None):
    upstream_url, configured_root = _config_upstream_and_root(config_path)
    root = default_root if default_root else configured_root
    try:
        server = create_server(upstream_url, root, host="127.0.0.1", port=port)
    except OSError as exc:
        if sys.stderr is not None:
            sys.stderr.write("无法绑定端口 %d: %s\n" % (port, exc))
        return 1
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.server_close()
        except Exception:
            pass
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parse_args(argv):
    mode = "launch"
    config_path = None
    port = 8766
    default_root = None
    open_browser = True
    no_ui = False

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--serve":
            mode = "serve"
        elif arg == "--launch":
            mode = "launch"
        elif arg == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
            i += 1
        elif arg == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
            i += 1
        elif arg == "--default-root" and i + 1 < len(argv):
            default_root = argv[i + 1]
            i += 1
        elif arg == "--no-browser":
            open_browser = False
        elif arg == "--no-ui":
            no_ui = True
            open_browser = False
        i += 1
    return {
        "mode": mode,
        "config_path": config_path,
        "port": port,
        "default_root": default_root,
        "open_browser": open_browser,
        "no_ui": no_ui,
    }


def main(argv=None):
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))

    if args["mode"] == "serve":
        return _serve(
            config_path=args["config_path"],
            port=args["port"],
            default_root=args["default_root"],
        )

    result = launch_bridge(
        config_path=args["config_path"],
        port=args["port"],
        open_browser=args["open_browser"] and not args["no_ui"],
        default_root=args["default_root"],
    )
    if result.get("ready"):
        return 0
    if sys.stderr is not None:
        sys.stderr.write(_sanitize_error(result.get("error", "")) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
