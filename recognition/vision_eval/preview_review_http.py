"""Board-review routes integrated into the Windows preview ThreadingHTTPServer.

install(bridge) must be called exactly once, from main, after the board_confirmation
package (and its dependencies) have been added to sys.path.  The original camera,
capture, voting and health routes are left untouched; only the original handler
class is subclassed and the original create_server is wrapped to construct one
board Runtime per server instance.
"""

import json
import re
import threading

_JSON_RE = re.compile(r"^/board-review/(start|confirm)$")
_STATE_RE = re.compile(r"^/board-review/state$")
_CROP_RE = re.compile(r"^/board-review/crop/([a-f0-9]{16})/([a-h][1-8])/(C1|C2)$")

_installed = False
_install_lock = threading.Lock()


def install(bridge):
    """Install board-review routes onto ``bridge`` exactly once."""
    global _installed
    with _install_lock:
        if _installed:
            return
        _installed = True

    original_create_server = bridge.create_server
    original_handler = bridge._BridgeHandler

    class _BoardReviewHandler(original_handler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if _STATE_RE.match(path):
                return self._board_review_state()
            m = _CROP_RE.match(path)
            if m:
                return self._board_review_crop(m.group(1), m.group(2), m.group(3))
            return super().do_GET()

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if _JSON_RE.match(path):
                return self._board_review_post(path)
            return super().do_POST()

        # ---- helpers -------------------------------------------------------

        def _board_review_runtime(self):
            return getattr(self.server, "board_review_runtime", None)

        def _board_review_state(self):
            if not self._common_guard():
                return
            rt = self._board_review_runtime()
            if rt is None:
                return self._send_json(503, {"error": "runtime_unavailable"})
            try:
                snap = rt.snapshot()
            except Exception:
                return self._send_json(500, {"error": "snapshot_failed"})
            if isinstance(snap, dict):
                snap = dict(snap)
                snap.pop("token", None)
            return self._send_json(200, snap)

        def _board_review_crop(self, run_id, square, angle):
            if not self._common_guard():
                return
            rt = self._board_review_runtime()
            if rt is None:
                return self._send_json(503, {"error": "runtime_unavailable"})
            with rt.lock:
                active = rt.draft and rt.draft["id"] == run_id and square in rt.draft["selected"]
            if not active:
                return self._send_json(404, {"error": "not_found"})
            path = rt.runs_dir / run_id / f"crop_{square}_{angle}.jpg"
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
            except OSError:
                return self._send_json(404, {"error": "not_found"})
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(data)
            except Exception:
                pass
            return None

        def _board_review_post(self, path):
            if not self.headers.get("Origin") or not self._origin_ok():
                return self._send_json(403, {"error": "forbidden_origin"})
            if not self._constant_time_token_ok():
                return self._send_json(403, {"error": "forbidden_token"})
            if not self._common_guard():
                return
            body = self._read_json_body()
            if body is None:
                return

            rt = self._board_review_runtime()
            if rt is None:
                return self._send_json(503, {"error": "runtime_unavailable"})

            if path == "/board-review/start":
                if set(body) - {"use_qwen"} or type(body.get("use_qwen", False)) is not bool:
                    return self._send_json(400, {"error": "invalid_body"})
                try:
                    ok, message = rt.start(use_qwen=body.get("use_qwen", False))
                except Exception:
                    return self._send_json(500, {"error": "start_failed"})
                if ok:
                    return self._send_json(202, {"status": "accepted", "message": message})
                return self._send_json(409, {"status": "conflict", "message": message})

            # /board-review/confirm
            if set(body.keys()) != {"id", "manual"}:
                return self._send_json(400, {"error": "invalid_body"})
            run_id = body.get("id")
            manual = body.get("manual")
            if not isinstance(run_id, str) or not isinstance(manual, dict):
                return self._send_json(400, {"error": "invalid_body"})
            try:
                act = rt.actionable()
            except Exception:
                return self._send_json(500, {"error": "confirm_failed"})
            if act is None:
                return self._send_json(409, {"error": "not_ready"})
            if not isinstance(act, dict):
                act = {}
            try:
                status, obj = rt.confirm(run_id, list(act), manual)
            except Exception:
                return self._send_json(500, {"error": "confirm_failed"})
            return self._send_json(int(status), obj)

        def _read_json_body(self):
            if self.headers.get_content_type() != "application/json":
                self._send_json(415, {"error": "invalid_content_type"})
                return None
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                length = 0
            if length <= 0 or length > 2048:
                self._send_json(400, {"error": "invalid_body"})
                return None
            try:
                raw = self.rfile.read(length)
            except Exception:
                self._send_json(400, {"error": "invalid_body"})
                return None
            try:
                body = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeError):
                self._send_json(400, {"error": "invalid_body"})
                return None
            if not isinstance(body, dict):
                self._send_json(400, {"error": "invalid_body"})
                return None
            return body

    def create_server(*args, **kwargs):
        review_factory = kwargs.pop("review_factory", None)
        from server_budget import Runtime, collect, infer_qwen

        server = original_create_server(*args, **kwargs)
        upstream = server.capture_state.upstream_url
        try:
            server.board_review_runtime = (review_factory(upstream) if review_factory else
                                           Runtime(upstream, None, collect, infer_qwen, None, None))
        except Exception:
            server.server_close()
            raise
        return server

    bridge._BridgeHandler = _BoardReviewHandler
    bridge.create_server = create_server
