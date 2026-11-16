import json
import os
import secrets
import re
import sys
import threading
from time import monotonic
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import webbrowser

from live_capture import collect, SQUARES
from qwen_live import infer as infer_qwen
from qwen_trial import MODEL as REVIEW_MODEL
from evidence_v2 import Cropper
from review import LABELS

BASE = Path(__file__).resolve().parent
CONFIG_PATH = BASE.parent / "dual_camera_launcher_config.json"
SQUARE_SET = set(SQUARES)
VALID_LABELS = set(LABELS) | {"empty", "unsure"}


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{secrets.token_hex(8)}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _json_bytes(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_msg(exc):
    try:
        t = type(exc).__name__
    except Exception:
        t = "Exception"
    return "internal error: " + t


class Runtime:
    def __init__(self, upstream, runs_dir, collector, inferer, cropper, reference_bytes):
        self.upstream = upstream
        self.runs_dir = Path(runs_dir) if runs_dir is not None else BASE / "runs"
        self.collector = collector
        self.inferer = inferer
        self._cropper = cropper
        self._cropper_ready = cropper is not None
        self._reference_bytes = reference_bytes
        self.lock = threading.Lock()
        self.phase = "idle"
        self.seconds = 0.0
        self.review_started = None
        self.message = "idle"
        self.draft = None
        self.confirmed = None
        self.token = self._new_token()
        self._worker = None
        self._reference_loaded = reference_bytes is not None
        self._load_latest()

    def _new_token(self):
        return secrets.token_hex(24)

    def get_reference(self):
        if not self._reference_loaded:
            try:
                self._reference_bytes = (BASE / "field_references/reference.jpg").read_bytes()
            except Exception:
                self._reference_bytes = b""
            self._reference_loaded = True
        return self._reference_bytes or b""

    def get_cropper(self):
        with self.lock:
            if not self._cropper_ready:
                self._cropper = Cropper()
                self._cropper_ready = True
            return self._cropper

    def _load_latest(self):
        p = self.runs_dir / "latest_confirmed.json"
        try:
            if not p.exists():
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            cid = data.get("id")
            board = data.get("board")
            accepted = data.get("accepted")
            confirmed_at = data.get("confirmed_at")
            if not isinstance(cid, str) or not re.fullmatch("[a-f0-9]{16}", cid) or not isinstance(board, dict) or not isinstance(accepted, list) or not isinstance(confirmed_at, str):
                return
            if len(board) != 64:
                return
            clean = {}
            for sq in SQUARES:
                lab = board.get(sq)
                if lab not in VALID_LABELS:
                    return
                clean[sq] = lab
            acc = [s for s in accepted if isinstance(s, str) and s in SQUARE_SET]
            self.confirmed = {"id": cid, "board": clean, "accepted": acc, "confirmed_at": confirmed_at}
            self.phase = "confirmed"
            self.message = "confirmed"
        except (ValueError, TypeError, OSError):
            return

    def snapshot(self):
        with self.lock:
            reviewing = self.phase == "reviewing" and self.review_started is not None
            return {
                "phase": self.phase,
                "seconds": self.seconds,
                "review_seconds": round(monotonic() - self.review_started, 1) if reviewing else None,
                "review_model": REVIEW_MODEL,
                "message": self.message,
                "draft": self.draft,
                "confirmed": self.confirmed,
                "token": self.token,
            }

    def confirmed_board(self):
        with self.lock:
            return dict(self.confirmed["board"]) if self.confirmed else None

    def set_progress(self, phase, seconds, message):
        with self.lock:
            self.review_started = monotonic() if phase == "reviewing" else None
            self.phase = phase
            self.seconds = float(seconds) if isinstance(seconds, (int, float)) else 0.0
            self.message = str(message)

    def start(self):
        with self.lock:
            if self.phase not in ("idle", "ready", "confirmed", "error"):
                return False, "busy"
            self.phase = "warmup"
            self.seconds = 0.0
            self.review_started = None
            self.message = "starting"
            self.draft = None
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()
            return True, "started"

    def _run(self):
        def progress(phase, seconds):
            self.set_progress(phase, seconds, phase)

        try:
            capture = self.collector(self.upstream, self.get_cropper(), progress,
                                     previous=self.confirmed_board())
            local = capture["local"]
            if local.keys() != SQUARE_SET or any(label not in VALID_LABELS for label in local.values()):
                raise RuntimeError("采集结果缺少有效的 64 格棋盘。")
            requests = capture["requests"]
            run_id = secrets.token_hex(8)
            run_dir = self.runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            centers = {}
            reference_bytes = self.get_reference() if requests else b""
            if reference_bytes:
                _atomic_write(run_dir / "reference.jpg", reference_bytes)
            for square, pair in requests.items():
                centers[square] = {}
                for camera in ("C1", "C2"):
                    jpg, center = pair[camera]
                    _atomic_write(run_dir / f"crop_{square}_{camera}.jpg", jpg)
                    centers[square][camera] = center
            metadata = {key: value for key, value in capture.items() if key != "requests"}
            metadata.update(id=run_id, evidence_centers=centers)
            _atomic_write(run_dir / "capture.json", _json_bytes(metadata))
            opinions, review_error, review_seconds = {}, None, None
            if requests:
                prerequisite = (self.inferer({}, b"", preflight=True) if reference_bytes else
                                {"error": "reference_unavailable"})
                if prerequisite.get("preflight_ok") is True:
                    self.set_progress("reviewing", 6.0, "reviewing")
                    started = monotonic()
                    reply = self.inferer(requests, reference_bytes)
                    review_seconds = round(monotonic() - started, 1)
                else:
                    reply = {"error": prerequisite.get("error", "preflight_failed")}
                _atomic_write(run_dir / "response.json", _json_bytes(reply))
                result = reply.get("result")
                if isinstance(result, dict) and result.keys() == requests.keys():
                    opinions = result
                else:
                    review_error = {
                        "preflight_failed": "本地接口预检失败",
                        "reference_unavailable": "棋子参考照不可用",
                        "credential_unavailable": "本机的阿里云凭据缺失或格式有误",
                        "authentication_failed": "Qwen 未接受当前凭据或该模型权限不足",
                        "timeout": "Qwen 响应超时",
                        "network_error": "暂时无法连接 Qwen 接口",
                        "schema_error": "Qwen 回复格式未通过校验",
                        "unfinished_response": "Qwen 未生成完整回复",
                    }.get(reply.get("error"), "接口未返回有效结果")
            with self.lock:
                self.draft = {
                    "id": run_id, "local": local, "selected": capture["selected"],
                    "skipped": capture["skipped"], "opinions": opinions,
                    "window_frames": [win["frames"] for win in capture["windows"]],
                    "review_error": review_error,
                    "review_seconds": review_seconds,
                }
                self.phase = "ready"
                self.review_started = None
                self.seconds = capture["observed_seconds"]
                self.message = "ready"
        except (RuntimeError, ValueError, KeyError, OSError) as exc:
            self.set_progress("error", 0.0, str(exc) if type(exc) is RuntimeError else _safe_msg(exc))
        except Exception as exc:
            self.set_progress("error", 0.0, _safe_msg(exc))

    def actionable(self):
        with self.lock:
            if self.phase != "ready" or not self.draft:
                return None
            draft = self.draft
        out = {}
        for sq in SQUARES:
            local = draft["local"].get(sq)
            op = draft.get("opinions", {}).get(sq)
            if not isinstance(op, dict):
                continue
            label = op.get("label")
            if not isinstance(label, str):
                continue
            if label != "empty" and label not in LABELS:
                continue
            if label == local:
                continue
            out[sq] = label
        return out

    def confirm(self, run_id, accept):
        act = self.actionable()
        if act is None:
            return 409, {"error": "not_ready"}
        with self.lock:
            if not isinstance(run_id, str) or not self.draft or self.draft["id"] != run_id:
                return 409, {"error": "stale_id"}
            draft = self.draft
        seen = set()
        clean_accept = []
        for sq in accept:
            if sq in seen:
                return 400, {"error": "duplicate"}
            seen.add(sq)
            if sq not in SQUARE_SET:
                return 400, {"error": "unknown_square"}
            if sq not in act:
                return 400, {"error": "not_actionable"}
            clean_accept.append(sq)
        with self.lock:
            if self.phase != "ready" or not self.draft or self.draft["id"] != run_id:
                return 409, {"error": "not_ready"}
            board = dict(draft["local"])
            for sq in clean_accept:
                board[sq] = act[sq]
            confirmed = {
                "id": run_id,
                "board": board,
                "accepted": clean_accept,
                "confirmed_at": _now_iso(),
            }
            run_dir = self.runs_dir / run_id
            payload = _json_bytes(confirmed)
            try:
                _atomic_write(run_dir / "confirmed.json", payload)
                _atomic_write(self.runs_dir / "latest_confirmed.json", payload)
            except OSError:
                return 500, {"error": "save_failed"}
            self.confirmed = confirmed
            self.phase = "confirmed"
            self.message = "confirmed"
            self.draft = None
            self.seconds = 0.0
        return 200, {"ok": True, "confirmed": confirmed}


class Handler(BaseHTTPRequestHandler):
    server_version = "BoardServer/1.0"
    protocol_version = "HTTP/1.0"
    timeout = 4

    def log_message(self, fmt, *args):
        pass

    def _host_ok(self):
        expected = f"127.0.0.1:{self.server.server_address[1]}"
        host = self.headers.get("Host", "")
        return host == expected

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _json(self, code, obj):
        self._send(code, _json_bytes(obj))

    def _origin_ok(self):
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin", "")
        return origin == f"http://{host}"

    def _path_parts(self):
        p = urlparse(self.path).path
        return p

    def do_GET(self):
        try:
            if not self._host_ok():
                self._json(403, {"error": "forbidden"})
                return
            p = self._path_parts()
            rt = self.server.runtime
            if p in ("/", "/index.html"):
                idx = BASE / "index.html"
                try:
                    data = idx.read_bytes()
                except Exception:
                    self._json(404, {"error": "not_found"})
                    return
                self._send(200, data, "text/html; charset=utf-8")
                return
            if p == "/api/state":
                self._json(200, rt.snapshot())
                return
            if p == "/reference.jpg":
                data = rt.get_reference()
                if not data:
                    self._json(404, {"error": "not_found"})
                    return
                self._send(200, bytes(data), "image/jpeg")
                return
            m = re.match(r"^/api/crop/([A-Za-z0-9_-]+)/([a-h][1-8])/(C1|C2)$", p)
            if m:
                cid, sq, cam = m.group(1), m.group(2), m.group(3)
                with rt.lock:
                    run_dir = rt.runs_dir / cid
                    valid = rt.draft is not None and rt.draft.get("id") == cid
                if not valid:
                    self._json(404, {"error": "not_found"})
                    return
                fp = run_dir / f"crop_{sq}_{cam}.jpg"
                if not fp.exists() or fp.parent != run_dir:
                    self._json(404, {"error": "not_found"})
                    return
                try:
                    data = fp.read_bytes()
                except Exception:
                    self._json(404, {"error": "not_found"})
                    return
                self._send(200, data, "image/jpeg")
                return
            self._json(404, {"error": "not_found"})
        except Exception as e:
            try:
                self._json(500, {"error": "internal", "message": _safe_msg(e)})
            except Exception:
                pass

    def do_POST(self):
        try:
            if not self._host_ok():
                self._json(403, {"error": "forbidden"})
                return
            if not self._origin_ok():
                self._json(403, {"error": "forbidden"})
                return
            p = self._path_parts()
            rt = self.server.runtime
            tok = self.headers.get("X-Board-Token", "")
            with rt.lock:
                current_token = rt.token
            if not tok or tok != current_token:
                self._json(403, {"error": "bad_token"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except Exception:
                length = -1
            if length < 0 or length > 2048:
                self._json(413, {"error": "too_large"})
                return
            if self.headers.get_content_type() != "application/json":
                self._json(415, {"error": "bad_content_type"})
                return
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                self._json(400, {"error": "bad_json"})
                return
            if not isinstance(body, dict):
                self._json(400, {"error": "bad_json"})
                return
            if p == "/api/start":
                if set(body.keys()) != set():
                    self._json(400, {"error": "bad_fields"})
                    return
                ok, why = rt.start()
                if not ok:
                    self._json(409, {"error": "busy"})
                    return
                self._json(202, {"ok": True, "message": "started"})
                return
            if p == "/api/confirm":
                if set(body.keys()) != {"id", "accept"}:
                    self._json(400, {"error": "bad_fields"})
                    return
                cid = body.get("id")
                accept = body.get("accept")
                if not isinstance(cid, str) or not isinstance(accept, list):
                    self._json(400, {"error": "bad_fields"})
                    return
                for sq in accept:
                    if not isinstance(sq, str):
                        self._json(400, {"error": "bad_fields"})
                        return
                code, out = rt.confirm(cid, accept)
                self._json(code, out)
                return
            self._json(404, {"error": "not_found"})
        except Exception as e:
            try:
                self._json(500, {"error": "internal", "message": _safe_msg(e)})
            except Exception:
                pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, runtime):
        self.runtime = runtime
        super().__init__(addr, Handler)


def _resolve_upstream(upstream):
    if upstream:
        return upstream
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        u = cfg.get("url") if isinstance(cfg, dict) else None
        if isinstance(u, str) and u:
            return u
    except Exception:
        pass
    return ""


def create_server(upstream, port=8767, *, runs_dir=BASE / "runs", collector=collect, inferer=infer_qwen, cropper=None, reference_bytes=None):
    up = _resolve_upstream(upstream)
    rt = Runtime(up, runs_dir, collector, inferer, cropper, reference_bytes)
    srv = Server(("127.0.0.1", port), rt)
    return srv


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=None)
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--open", action="store_true", help="Open the Windows browser")
    args = ap.parse_args()
    try:
        srv = create_server(args.upstream, args.port)
    except OSError as error:
        if not args.open or getattr(error, "winerror", None) != 10048:
            raise
        webbrowser.open(f"http://127.0.0.1:{args.port}/")
        return
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    if args.open:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    if sys.stdout is not None:
        print(url, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
