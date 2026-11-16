import copy
import json
import os
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlsplit

import second_window_vote_v2 as old


SETTINGS_PATH = Path(__file__).with_name("voting_settings.json")
ALLOWED_PERIODS = (1, 2, 3, 4)
HISTORY_WEIGHT = 0.4


def _default_settings():
    return {"window_s": 1}


def validate_period(value):
    if type(value) is not int:
        raise ValueError("window_s must be an integer in [1,2,3,4]")
    if value not in ALLOWED_PERIODS:
        raise ValueError("window_s must be one of [1,2,3,4]")
    return value


def _load_settings():
    if not SETTINGS_PATH.exists():
        return _default_settings()
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        raise ValueError("invalid voting settings file: %s" % (exc,))
    if not isinstance(data, dict):
        raise ValueError("invalid voting settings file: not an object")
    if set(data.keys()) != {"window_s"}:
        raise ValueError("invalid voting settings file: unexpected fields")
    return {"window_s": validate_period(data.get("window_s"))}


def read_period():
    return _load_settings()["window_s"]


def _atomic_save(payload):
    data = {"window_s": validate_period(payload.get("window_s"))}
    txt = json.dumps(data, separators=(",", ":"), sort_keys=True)
    directory = str(SETTINGS_PATH.parent)
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=SETTINGS_PATH.name + ".", suffix=".tmp", dir=directory
        )
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fd = None
            fh.write(txt)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, str(SETTINGS_PATH))
        tmp_path = None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


class TemporalFusion(old.TemporalFusion):
    def __init__(self, window=None):
        if window is None:
            window = read_period()
        period = validate_period(window)
        self._period_s = period
        self._target_frames = 20 * period
        super().__init__(
            window=float(period),
            min_span=0.25 * period,
            min_frames=3 * period,
            history_weight=HISTORY_WEIGHT,
            history_half_life=0.5,
        )

    def _empty_output(self, result):
        out = super()._empty_output(result)
        voting = out.get("voting")
        if not isinstance(voting, dict):
            voting = {}
            out["voting"] = voting
        voting["target_frames"] = self._target_frames
        voting["period_s"] = self._period_s
        frames = voting.get("frames")
        if type(frames) is not int:
            frames = 0
        voting["target_met"] = frames >= self._target_frames
        return out

    def _publish(self, window, result, moving_set, closing_ts):
        with self._lock:
            out = super()._publish(window, result, moving_set, closing_ts)
            frames = len(window.seqs)
            voting = out.get("voting")
            if not isinstance(voting, dict):
                voting = {}
                out["voting"] = voting
            voting["target_frames"] = self._target_frames
            voting["target_met"] = frames >= self._target_frames
            voting["period_s"] = self._period_s

            last_completed = getattr(self, "_last_completed", None)
            if isinstance(last_completed, dict):
                last_completed["target_frames"] = self._target_frames
                last_completed["target_met"] = frames >= self._target_frames
                last_completed["period_s"] = self._period_s

            published = getattr(self, "_published", None)
            if isinstance(published, dict):
                pub_voting = published.get("voting")
                if not isinstance(pub_voting, dict):
                    pub_voting = {}
                    published["voting"] = pub_voting
                pub_voting["target_frames"] = self._target_frames
                pub_voting["target_met"] = frames >= self._target_frames
                pub_voting["period_s"] = self._period_s
            return out

    def summary(self):
        data = super().summary()
        if not isinstance(data, dict):
            data = {}
        data["available_periods"] = list(ALLOWED_PERIODS)
        data["target_frames"] = self._target_frames
        return data


def _validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    if set(payload.keys()) != {"window_s"}:
        raise ValueError("payload must contain exactly window_s")
    return validate_period(payload.get("window_s"))


def apply_settings(engine, payload):
    period = _validate_payload(payload)
    grid = getattr(engine, "_grid", None)
    if grid is None or not hasattr(grid, "_lock"):
        raise RuntimeError("engine grid unavailable")
    lock = grid._lock
    with lock:
        existing = getattr(grid, "_temporal", None)
        same = (
            isinstance(existing, TemporalFusion)
            and getattr(existing, "_period_s", None) == period
            and float(getattr(existing, "history_weight", -1)) == HISTORY_WEIGHT
        )
        if same:
            return {
                "ok": True,
                "window_s": period,
                "history_weight": HISTORY_WEIGHT,
                "reset": False,
            }
        replacement = TemporalFusion(window=period)
        _atomic_save({"window_s": period})
        grid._temporal = replacement
        grid._stable = None
        grid._primary_seq = None
        return {
            "ok": True,
            "window_s": period,
            "history_weight": HISTORY_WEIGHT,
            "reset": True,
        }


def _wrap_factory(app, factory):
    def wrapped(feeds, engine):
        base = factory(feeds, engine)

        class VotingSettingsHandler(base):
            def do_POST(self):
                try:
                    parsed = urlsplit(self.path)
                    path = parsed.path
                except Exception:
                    self._error(404, "not found")
                    return
                if path != "/voting/settings":
                    return super().do_POST()
                try:
                    self._handle_voting_settings()
                except Exception:
                    try:
                        self._error(500, "internal error")
                    except Exception:
                        pass

            def _handle_voting_settings(self):
                try:
                    ok_origin = self._check_origin()
                except Exception:
                    ok_origin = False
                if not ok_origin:
                    self._error(403, "forbidden")
                    return

                ctype = self.headers.get("Content-Type", "")
                if not isinstance(ctype, str):
                    ctype = ""
                media = ctype.split(";", 1)[0].strip().lower()
                if media != "application/json":
                    self._error(400, "Content-Type must be application/json")
                    return

                raw_len = self.headers.get("Content-Length")
                try:
                    length = int(raw_len)
                except Exception:
                    self._error(400, "invalid Content-Length")
                    return
                if length < 1 or length > 1024:
                    self._error(400, "invalid Content-Length")
                    return
                try:
                    body = self.rfile.read(length)
                except Exception:
                    self._error(400, "invalid body")
                    return
                try:
                    payload = json.loads(body.decode("utf-8"))
                except Exception:
                    self._error(400, "invalid JSON")
                    return
                try:
                    _validate_payload(payload)
                except Exception:
                    self._error(400, "invalid payload")
                    return

                eng = getattr(self, "_engine", None)
                if eng is None or getattr(eng, "_grid", None) is None:
                    self._error(503, "engine unavailable")
                    return
                try:
                    result = apply_settings(eng, payload)
                except Exception:
                    self._error(500, "settings update failed")
                    return
                self._json_send(200, result)

        return VotingSettingsHandler

    return wrapped


def install_http(app):
    marker = "_voting_controls_installed"
    if getattr(app, marker, False):
        return app
    factory = getattr(app, "_make_handler")
    setattr(app, "_make_handler", _wrap_factory(app, factory))
    setattr(app, marker, True)
    return app
