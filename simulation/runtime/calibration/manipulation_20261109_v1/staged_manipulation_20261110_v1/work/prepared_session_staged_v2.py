import json
import os
import sys
import time
import re
import hashlib
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_WORK_FAR = ROOT / "calibration" / "manipulation_20261109_v1" / "far_types_random3x10_20261110_v1" / "work"
_WORK_GRASP = ROOT / "calibration" / "manipulation_20261109_v1" / "far_grasp_fast_20261110_v1" / "work"
for _p in (str(_WORK_FAR), str(_WORK_GRASP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import resident_planner_staged_v2 as M
import sequence_coordinator_staged_v1 as C
import staged_runner_v2 as E


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(o):
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
    raise TypeError("unserializable: %r" % (type(o),))


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with open(path, "w") as f:
        json.dump(obj, f, default=_json_default, indent=2)


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _safe_id(s):
    if not isinstance(s, str) or not s:
        raise ValueError("bad session id")
    if re.fullmatch(r"[A-Za-z0-9_-]+", s) is None:
        raise ValueError("bad session id %r" % s)
    if "/" in s or "\\" in s or "\x00" in s:
        raise ValueError("bad session id %r" % s)
    return s


def _dispatch_factory(rp, session_dir, deadline):
    def _dispatch(script_path, args_list, timeout_s, env_pp):
        base = os.path.basename(str(script_path))
        if base not in ("center_snapshot_v1.py", "route_snapshot_v1.py"):
            raise RuntimeError("unsupported script basename: %r" % (base,))
        pairs = {}
        i = 0
        while i < len(args_list):
            a = args_list[i]
            if not isinstance(a, str) or not a.startswith("--"):
                raise ValueError("bad arg %r" % (a,))
            if i + 1 >= len(args_list):
                raise ValueError("missing value for %r" % (a,))
            pairs[a[2:]] = args_list[i + 1]
            i += 2
        t0 = time.time()
        timeout = timeout_s
        if deadline:
            remaining = deadline - t0
            if timeout is None or remaining < timeout:
                timeout = remaining
        if timeout is not None and timeout <= 0:
            return {"returncode": 1, "timeout": True, "stdout": "", "stderr": "deadline"}
        if base == "center_snapshot_v1.py":
            out = pairs.get("output")
            if not out:
                raise ValueError("center missing --output")
            q = Path(out)
            if q.exists():
                raise RuntimeError("refuse overwrite %s" % out)
            try:
                res = _do_center(rp, pairs, q, timeout, deadline)
            except Exception:
                raise
            rc = res.get("returncode", 0)
            so = json.dumps({k: v for k, v in res.items() if k != "returncode"}, default=_json_default)
            return {"returncode": rc, "timeout": bool(res.get("timeout", False)), "stdout": so, "stderr": ""}
        elif base == "route_snapshot_v1.py":
            out = pairs.get("out") or pairs.get("output")
            if not out:
                raise ValueError("route missing --out")
            try:
                res = _do_route(rp, pairs, Path(out) / "result.json", timeout)
            except Exception:
                raise
            rc = res.get("returncode", 0)
            so = json.dumps({k: v for k, v in res.items() if k != "returncode"}, default=_json_default)
            return {"returncode": rc, "timeout": bool(res.get("timeout", False)), "stdout": so, "stderr": ""}
        raise RuntimeError("unreachable")
    return _dispatch


_CENTER_CACHE = {}


def _do_center(rp, pairs, out_path, timeout, deadline):
    query_path = pairs.get("config")
    ik_path = pairs.get("ik")
    if not query_path or not ik_path:
        raise ValueError("center requires --config and --ik")
    query_path = str(Path(query_path).resolve())
    ik = _load_json(ik_path)
    query = _load_json(query_path)
    cfg = _load_json(query_path)
    catalog_candidates = list(cfg.get("catalog_candidates", []))
    hover_options = list(cfg.get("hover_options_m", []))
    if not 1 <= len(catalog_candidates) <= 28 or not 1 <= len(hover_options) <= 4:
        raise ValueError("catalog/hover size overflow")
    t_start = time.time()
    step_dir = Path(query_path).parent
    family_path = step_dir / "family_search.json"
    tried = []
    elapsed_flag = False
    selected = None
    from adaptive_selector_bound_v1 import choose
    selection = choose(rp, query_path, ik, catalog_candidates, hover_options, timeout, deadline)
    tried = selection["tried"]
    elapsed_flag = selection["deadline_flag"]
    selected = selection["selected"]
    elapsed_total = time.time() - t_start
    elapsed_flag = elapsed_flag or (timeout is not None and elapsed_total > timeout) or (deadline is not None and time.time() > deadline)
    if selected is not None:
        selected_center = selected["center"]
        selected_center["selected_hover_m"] = selected["hover"]
        selected_center["selected_catalog_path"] = selected["catalog"]
        selected_center["selected_placement_yaw_deg"] = selected["yaw"]
        _write_json(step_dir / "query_before_selection.json", query)
        query["placement_yaw_deg"] = selected["yaw"]
        query["keypoint_catalog_path"] = selected["catalog"]
        Path(query_path).write_text(json.dumps(query, default=_json_default, indent=2))
        _write_json(out_path, selected_center)
        center_sha = _sha256(out_path)
        _CENTER_CACHE[query_path] = {
            "canonicalquerypath": query_path,
            "query_sha256": _sha256(query_path),
            "canonicalcenterpath": str(out_path.resolve()),
            "center_sha256": center_sha,
            "plan": selected["plan"],
            "hover": selected["hover"],
        }
        _write_json(family_path, {"tried": tried, "elapsed_s": elapsed_total,
                                  "deadline_flag": elapsed_flag, "counts": selection["counts"], "selected": True})
        _write_json(step_dir / "selected_plan.json", selected["plan"])
        return {"returncode": 0, "timeout": elapsed_flag}
    reason = "family_budget" if elapsed_flag else "boundedfamily_no_full_plan"
    blocked = {"blocked": True, "reason": reason, "family_search_path": str(family_path)}
    _write_json(out_path, blocked)
    _write_json(family_path, {"tried": tried, "elapsed_s": elapsed_total,
                              "deadline_flag": elapsed_flag, "counts": selection["counts"], "selected": False})
    return {"returncode": 0, "timeout": elapsed_flag}


def _do_route(rp, pairs, out_path, timeout):
    action_cfg_path = pairs.get("config")
    if not action_cfg_path:
        raise ValueError("route requires --config")
    action_cfg = _load_json(action_cfg_path)
    query_path = str(Path(action_cfg.get("query")).resolve())
    center_path = str(Path(action_cfg.get("center")).resolve())
    hover = action_cfg.get("hover_m")
    cached = _CENTER_CACHE.get(query_path)
    if cached is None:
        raise RuntimeError("no cache for query %s" % query_path)
    if cached["canonicalquerypath"] != query_path:
        raise RuntimeError("canonical query mismatch")
    if cached["canonicalcenterpath"] != center_path:
        raise RuntimeError("canonical center mismatch")
    if _sha256(query_path) != cached["query_sha256"]:
        raise RuntimeError("query sha mismatch")
    if _sha256(center_path) != cached["center_sha256"]:
        raise RuntimeError("center sha mismatch")
    if hover != cached["hover"]:
        raise RuntimeError("hover mismatch")
    if out_path.exists():
        raise RuntimeError("refuse overwrite %s" % out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out_path, cached["plan"])
    del _CENTER_CACHE[query_path]
    return {"returncode": 0, "timeout": False}


def run_prepared_session(cfg, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    session_cfg = cfg
    expected = session_cfg.get("expected_steps")
    steps = session_cfg.get("steps")
    if not isinstance(steps, list) or len(steps) != expected:
        raise ValueError("steps length mismatch")
    if expected not in (1, 2, 10):
        raise ValueError("unexpected expected_steps %r" % (expected,))
    if len(steps) > 10:
        raise ValueError("steps too long")
    if not 1 <= int(session_cfg["max_steps_per_action"]) <= 24000:
        raise ValueError("per-action max too high")
    if (outdir / "result.json").exists() or (outdir / "planner_stats.json").exists():
        raise FileExistsError(outdir)
    _CENTER_CACHE.clear()
    budget = session_cfg.get("deadline_s")
    deadline = time.time() + budget if budget is not None else None
    rp = M.ResidentPlanner()
    saved = {}
    saved["E"] = C.E
    saved["_run_cli"] = getattr(C, "_run_cli", None)
    saved["_build_bridge"] = getattr(C, "_build_bridge", None)
    saved["_planning_context"] = getattr(C, "_planning_context", None)
    stats = None
    t0 = time.time()
    err = None
    try:
        dispatch = _dispatch_factory(rp, outdir, deadline)
        C.E = E
        C._run_cli = dispatch
        C._build_bridge = rp.bridge
        C._planning_context = lambda actual, querycfg, case, snapshot: setattr(rp, "_start_context", (actual, querycfg, case, snapshot))
        if hasattr(rp, "_dispatch"):
            rp._dispatch = dispatch
        rc = C.run_session(session_cfg, outdir)
        stats = {
            "wall_s": time.time() - t0,
            "stages": getattr(rp, "stats", None),
            "query_worlds_created": getattr(rp, "worlds_created", None),
            "guard_calls": getattr(rp, "guard_calls", None),
            "returncode": rc,
        }
    except Exception as exc:
        err = exc
        stats = {
            "wall_s": time.time() - t0,
            "stages": getattr(rp, "stats", None),
            "query_worlds_created": getattr(rp, "worlds_created", None),
            "guard_calls": getattr(rp, "guard_calls", None),
            "error": "".join(traceback.format_exception_only(type(exc), exc)),
        }
    finally:
        _CENTER_CACHE.clear()
        C.E = saved["E"]
        if saved["_run_cli"] is not None:
            C._run_cli = saved["_run_cli"]
        elif hasattr(C, "_run_cli"):
            delattr(C, "_run_cli")
        if saved["_build_bridge"] is not None:
            C._build_bridge = saved["_build_bridge"]
        elif hasattr(C, "_build_bridge"):
            delattr(C, "_build_bridge")
        if saved["_planning_context"] is not None:
            C._planning_context = saved["_planning_context"]
        elif hasattr(C, "_planning_context"):
            delattr(C, "_planning_context")
        try:
            rp.close()
        except Exception as close_exc:
            if err is None:
                err = close_exc
            else:
                err = Exception(str(err) + " | close: " + str(close_exc))
    try:
        _write_json(outdir / "planner_stats.json", stats)
    except Exception as write_exc:
        if err is None:
            err = write_exc
    if err is not None:
        raise err
    return int(stats.get("returncode", 1))


