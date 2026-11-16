import copy
import json
import math
import threading

NUM_CLASSES = 12


def _argmax(vec):
    best = 0
    bestv = vec[0]
    for i in range(1, len(vec)):
        if vec[i] > bestv:
            bestv = vec[i]
            best = i
    return best, bestv


def _margin(vec, idx):
    top = vec[idx]
    second = None
    for i, v in enumerate(vec):
        if i == idx:
            continue
        if second is None or v > second:
            second = v
    if second is None:
        return top
    return top - second


def _validate_probs(probs):
    if probs is None:
        return None
    if isinstance(probs, (str, bytes)):
        raise ValueError("probs not numeric sequence")
    try:
        vals = list(probs)
    except TypeError:
        raise ValueError("probs not iterable")
    if len(vals) != NUM_CLASSES:
        raise ValueError("probs wrong length")
    out = []
    total = 0.0
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError("prob not numeric")
        if not math.isfinite(f) or f < 0.0:
            raise ValueError("prob invalid")
        out.append(f)
        total += f
    if total <= 0.0:
        return None
    return [v / total for v in out]


class _SquareHistory:
    __slots__ = ("samples",)

    def __init__(self):
        self.samples = []


class _Window:
    __slots__ = (
        "start",
        "end",
        "anchor_ts",
        "samples",
        "seqs",
        "last_ts",
        "last_candidates",
        "closed",
        "boundary_ts",
    )

    def __init__(self, start, end, anchor_ts):
        self.start = start
        self.end = end
        self.anchor_ts = anchor_ts
        self.samples = {}
        self.seqs = set()
        self.last_ts = None
        self.last_candidates = {}
        self.closed = False
        self.boundary_ts = None


class TemporalFusion:
    def __init__(
        self,
        window=1.0,
        min_span=0.25,
        min_frames=3,
        margin=0.08,
        min_top=0.4,
        history_weight=0.15,
        history_half_life=0.5,
    ):
        self.window = float(window)
        self.min_span = float(min_span)
        self.min_frames = int(min_frames)
        self.margin = float(margin)
        self.min_top = float(min_top)
        self.history_weight = float(history_weight)
        self.history_half_life = float(history_half_life)
        self._lock = threading.RLock()
        self._init_state()

    def _init_state(self):
        self._grid_token = None
        self._active = None
        self._published = None
        self._published_end = None
        self._published_ts = None
        self._publish_count = 0
        self._prev_publish_end = None
        self._last_ts = None
        self._last_seq = None
        self._total_unique_frames = 0
        self._duplicate_frames = 0
        self._last_completed = None
        self._last_candidates = {}
        self._prev_posterior = {}
        self._history_ends = {}
        self._have_prev_posterior = False
        self._last_published_end_for_history = None

    def reset(self):
        with self._lock:
            self._init_state()

    def _canon_seq(self, sequence):
        try:
            return repr(sequence)
        except Exception:
            return str(id(sequence))

    def _make_active(self, anchor):
        start = anchor
        end = anchor + self.window
        return _Window(start, end, anchor)

    def _empty_output(self, result):
        out = {
            "squares64": {},
            "voting": {
                "window_start": None,
                "window_end": None,
                "published_at": None,
                "frames": 0,
                "target_frames": 20,
                "target_met": False,
                "accepted_current_frames": 0,
                "publish_count": self._publish_count,
            },
        }
        meta = {}
        if isinstance(result, dict):
            for k, v in result.items():
                if k != "squares64":
                    meta[k] = copy.deepcopy(v)
        out.update(meta)
        squares_in = result.get("squares64", {}) if isinstance(result, dict) else {}
        for sq, row in squares_in.items():
            newrow = copy.deepcopy(row) if isinstance(row, dict) else {}
            newrow["instant_class"] = None
            newrow["final_class"] = None
            newrow["confidence"] = None
            newrow["temporal_state"] = "unknown"
            newrow["temporal_frames"] = 0
            newrow["temporal_span"] = 0.0
            newrow["temporal_probs"] = None
            newrow["vote_probs"] = None
            newrow["history_alpha"] = 0.0
            out["squares64"][sq] = newrow
        return out

    def _alpha_for(self, window_end, square):
        if not self._have_prev_posterior:
            return 0.0
        if self._last_published_end_for_history is None:
            return 0.0
        if square not in self._history_ends:
            return 0.0
        delta = window_end - self._history_ends[square] - self.window
        delta = max(0.0, delta)
        return self.history_weight * (2.0 ** (-delta / self.history_half_life))

    def _append_sample(self, window, ts, result, moving_set):
        sq_in = result.get("squares64", {})
        if not isinstance(sq_in, dict):
            raise ValueError("squares64 not dict")
        for sq, row in sq_in.items():
            if row is None:
                continue
            if not isinstance(row, dict):
                raise ValueError("row not dict")
            if sq in moving_set:
                continue
            probs = row.get("probs")
            vals = _validate_probs(probs)
            cand = row.get("candidate_class")
            if cand is not None:
                try:
                    cand = int(cand)
                except (TypeError, ValueError):
                    raise ValueError("candidate_class invalid")
                if cand < 0 or cand >= NUM_CLASSES:
                    raise ValueError("candidate_class out of range")
            if vals is None:
                continue
            weights = row.get("weights")
            w = 1.0
            if isinstance(weights, dict):
                try:
                    c1 = float(weights.get("c1", 0.0))
                    c2 = float(weights.get("c2", 0.0))
                    w = c1 + c2
                    if not math.isfinite(w):
                        raise ValueError("weights nonfinite")
                except (TypeError, ValueError):
                    raise ValueError("weights invalid")
                w = max(0.0, min(1.0, w))
            if w <= 0:
                continue
            hist = window.samples.get(sq)
            if hist is None:
                hist = _SquareHistory()
                window.samples[sq] = hist
            hist.samples.append((ts, vals, cand, w))
            if len(hist.samples) > 256:
                del hist.samples[0 : len(hist.samples) - 256]

    def _finalize(self, window, published_at):
        squares_out = {}
        all_squares = set(window.samples.keys())
        for sq in all_squares:
            hist = window.samples.get(sq)
            samples = hist.samples if hist else []
            n = len(samples)
            if n == 0:
                continue
            sq_ts = [s[0] for s in samples]
            span = max(sq_ts) - min(sq_ts)
            mean = [0.0] * NUM_CLASSES
            wsum = 0.0
            for _ts, vals, _cand, w in samples:
                for i in range(NUM_CLASSES):
                    mean[i] += w * vals[i]
                wsum += w
            if wsum <= 0.0:
                continue
            mean = [x / wsum for x in mean]
            idx, top = _argmax(mean)
            marg = _margin(mean, idx)

            strong = top >= 0.65 and marg >= 0.25
            alpha = 0.0 if strong else self._alpha_for(window.end, sq)
            final_probs = list(mean)
            if alpha > 0.0 and self._have_prev_posterior:
                prev = self._prev_posterior.get(sq)
                if prev is not None:
                    for i in range(NUM_CLASSES):
                        final_probs[i] = (1.0 - alpha) * mean[i] + alpha * prev[i]
                    s = sum(final_probs)
                    if s > 0:
                        final_probs = [x / s for x in final_probs]

            pidx, ptop = _argmax(final_probs)
            pmarg = _margin(final_probs, pidx)

            enough = n >= self.min_frames and span >= self.min_span
            strong = top >= 0.65 and marg >= 0.25

            if not enough:
                state = "unknown"
                final_class = None
                confidence = None
            elif strong:
                state = "stable"
                final_class = idx
                confidence = top
            elif ptop >= self.min_top and pmarg >= self.margin:
                state = "stable"
                final_class = pidx
                confidence = ptop
            else:
                state = "pending"
                final_class = None
                confidence = ptop

            latest_cand = samples[-1][2]
            squares_out[sq] = {
                "instant_class": latest_cand,
                "final_class": final_class,
                "confidence": confidence,
                "temporal_state": state,
                "temporal_frames": n,
                "temporal_span": span,
                "temporal_probs": final_probs,
                "vote_probs": list(mean),
                "history_alpha": alpha,
            }

        return squares_out

    def _publish(self, window, result, moving_set, closing_ts):
        sq_out = self._finalize(window, closing_ts)

        out = {}
        if isinstance(result, dict):
            for k, v in result.items():
                if k != "squares64":
                    out[k] = copy.deepcopy(v)
        out["squares64"] = {}

        sq_in = result.get("squares64", {}) if isinstance(result, dict) else {}
        for sq, row in sq_in.items():
            newrow = copy.deepcopy(row) if isinstance(row, dict) else {}
            if sq in moving_set:
                newrow["instant_class"] = None
                newrow["final_class"] = None
                newrow["confidence"] = None
                newrow["temporal_state"] = "unknown"
                newrow["temporal_frames"] = 0
                newrow["temporal_span"] = 0.0
                newrow["temporal_probs"] = None
                newrow["vote_probs"] = None
                newrow["history_alpha"] = 0.0
            elif sq in sq_out:
                info = sq_out[sq]
                newrow["instant_class"] = info["instant_class"]
                newrow["final_class"] = info["final_class"]
                newrow["confidence"] = info["confidence"]
                newrow["temporal_state"] = info["temporal_state"]
                newrow["temporal_frames"] = info["temporal_frames"]
                newrow["temporal_span"] = info["temporal_span"]
                newrow["temporal_probs"] = info["temporal_probs"]
                newrow["vote_probs"] = info["vote_probs"]
                newrow["history_alpha"] = info["history_alpha"]
            else:
                newrow["instant_class"] = row.get("candidate_class")
                newrow["final_class"] = None
                newrow["confidence"] = None
                newrow["temporal_state"] = "unknown"
                newrow["temporal_frames"] = 0
                newrow["temporal_span"] = 0.0
                newrow["temporal_probs"] = None
                newrow["vote_probs"] = None
                newrow["history_alpha"] = 0.0
            out["squares64"][sq] = newrow

        for sq, info in sq_out.items():
            if sq not in out["squares64"]:
                newrow = {
                    "instant_class": info["instant_class"],
                    "final_class": info["final_class"],
                    "confidence": info["confidence"],
                    "temporal_state": info["temporal_state"],
                    "temporal_frames": info["temporal_frames"],
                    "temporal_span": info["temporal_span"],
                    "temporal_probs": info["temporal_probs"],
                    "vote_probs": info["vote_probs"],
                    "history_alpha": info["history_alpha"],
                }
                out["squares64"][sq] = newrow

        self._publish_count += 1

        out["voting"] = {
            "window_start": window.start,
            "window_end": window.end,
            "published_at": closing_ts,
            "frames": len(window.seqs),
            "target_frames": 20,
            "target_met": len(window.seqs) >= 20,
            "accepted_current_frames": len(window.seqs),
            "publish_count": self._publish_count,
        }

        for sq, info in sq_out.items():
            if info["temporal_probs"] is not None and info["temporal_state"] == "stable":
                self._prev_posterior[sq] = list(info["temporal_probs"])
                self._history_ends[sq] = window.end
        for sq in list(self._prev_posterior):
            if window.end - self._history_ends.get(sq, -1e30) > 2.0:
                self._prev_posterior.pop(sq, None)
                self._history_ends.pop(sq, None)
        self._have_prev_posterior = bool(self._prev_posterior)
        self._last_published_end_for_history = max(self._history_ends.values(), default=None)
        self._prev_publish_end = window.end
        self._last_completed = {
            "window_start": window.start,
            "window_end": window.end,
            "published_at": closing_ts,
            "frames": len(window.seqs),
            "target_frames": 20,
            "target_met": len(window.seqs) >= 20,
            "accepted_current_frames": len(window.seqs),
            "publish_count": self._publish_count,
        }
        self._published = copy.deepcopy(out)
        self._published_ts = closing_ts
        return copy.deepcopy(out)

    def update(self, result, timestamp, sequence, grid_token=None, moving=()):
        with self._lock:
            ts = float(timestamp)
            if not math.isfinite(ts) or not isinstance(result, dict):
                raise ValueError("invalid input")
            if not isinstance(result.get("squares64", {}), dict):
                raise ValueError("squares64 not dict")
            for row in result.get("squares64", {}).values():
                if row is not None:
                    if not isinstance(row, dict): raise ValueError("row not dict")
                    _validate_probs(row.get("probs"))
            if grid_token is not None and self._grid_token is not None and grid_token != self._grid_token:
                self._init_state()
            self._grid_token = grid_token
            canon = self._canon_seq(sequence)
            w = self._active
            if ((self._last_ts is not None and ts <= self._last_ts) or canon == self._last_seq or (w is not None and canon in w.seqs)):
                self._duplicate_frames += 1
                return copy.deepcopy(self._published) if self._published is not None else self._empty_output(result)
            if w is None:
                w = self._make_active(ts)
                self._active = w
            elif ts >= w.end:
                if self._last_ts is not None and ts - self._last_ts > 2.0:
                    self._prev_posterior.clear(); self._history_ends.clear()
                    self._have_prev_posterior = False
                    self._published = None
                else:
                    self._publish(w, self._last_result, set(), ts)
                start = w.end + math.floor((ts-w.end)/self.window)*self.window
                w = self._make_active(start)
                self._active = w
            if len(w.seqs) >= 256:
                return copy.deepcopy(self._published) if self._published is not None else self._empty_output(result)
            self._append_sample(w,ts,result,set(moving or ()))
            w.seqs.add(canon);w.last_ts=ts
            self._last_ts=ts;self._last_seq=canon
            self._total_unique_frames += 1
            self._last_result=copy.deepcopy(result)
            self._last_candidates=self._extract_candidates(result,set(moving or ()))
            return copy.deepcopy(self._published) if self._published is not None else self._empty_output(result)

    def _extract_candidates(self, result, moving_set):
        out = {}
        sqin = result.get("squares64", {}) if isinstance(result, dict) else {}
        if not isinstance(sqin, dict):
            return out
        for sq, row in sqin.items():
            if sq in moving_set:
                continue
            if not isinstance(row, dict):
                continue
            vals = _validate_probs(row.get("probs"))
            cand = row.get("candidate_class")
            if cand is not None:
                try:
                    cand = int(cand)
                except (TypeError, ValueError):
                    raise ValueError("candidate_class invalid")
            out[sq] = {"probs": vals, "candidate_class": cand}
        return out

    def summary(self):
        with self._lock:
            active = self._active
            current_unique = len(active.seqs) if active is not None else 0
            current_ts = active.last_ts if active is not None else None
            return json.loads(
                json.dumps(
                    {
                        "last_completed_window": copy.deepcopy(self._last_completed),
                        "current_unique_frames": current_unique,
                        "total_unique_frames": self._total_unique_frames,
                        "publish_count": self._publish_count,
                        "duplicate_frames": self._duplicate_frames,
                        "history_weight": self.history_weight,
                        "history_half_life": self.history_half_life,
                        "publish_period_s": self.window,
                    }
                )
            )
