#!/usr/bin/env python3
"""Three-action persistent-world smoke-test coordinator."""
import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from types import MethodType

WORKSPACE = Path(__file__).resolve().parents[4]
for _rel in (
    "calibration/manipulation_20261109_v1/smoke_grasps_20261110_v1/work",
    "calibration/manipulation_20261109_v1/keypoint_grasps_20261110_v1/work",
    "reachability_eval",
    "calibration/manipulation_20261109_v1/work",
    "experiment_review/tools",
):
    _p = str(WORKSPACE / _rel)
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import numpy as np
    import pybullet as p
except Exception:
    np = None
    p = None

try:
    import persistent_execute_support_v1 as E
    from query_world_snapshot_v1 import QueryWorld
    from keypoint_geometry_v3 import Geometry
    import rook_route_v5 as routebase
    from ompl_continuous_helpers_v4 import protect
except Exception:
    E = None
    QueryWorld = None
    Geometry = None
    routebase = None
    protect = None


MAX_ACTIONS = 10
AGGREGATE_DEADLINE_S = 900.0
MAX_CLI_TIMEOUT_S = 65.0
CONTACT_FORCE_GUARD_N = 0.01
GRIPPER_STEP_DEG = 0.5


class TaskStop(Exception):
    def __init__(self, reason, classification):
        super().__init__(reason)
        self.reason = reason
        self.classification = classification


def _json_default(o):
    try:
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _snapshot_copy(snap):
    if snap is None:
        return None
    out = {}
    for k, v in snap.items():
        out[k] = {
            "pos": [float(x) for x in v["pos"]],
            "quat": [float(x) for x in v["quat"]],
        }
    return out


def _make_pythonpath(cfg_paths):
    parts = []
    for rel in (
        "calibration/manipulation_20261109_v1/smoke_grasps_20261110_v1/work",
        "calibration/manipulation_20261109_v1/keypoint_grasps_20261110_v1/work",
        "reachability_eval",
        "calibration/manipulation_20261109_v1/work",
        "experiment_review/tools",
    ):
        parts.append(str(WORKSPACE / rel))
    for extra in cfg_paths:
        if extra:
            parts.append(str(extra))
    import os
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    sep = ";" if sys.platform.startswith("win") else ":"
    return sep.join(parts)


def _run_cli(script_path, args_list, timeout_s, env_pp):
    import subprocess
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = env_pp
    cmd = [sys.executable, "-X", "utf8", str(script_path)] + list(args_list)
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": None,
            "timeout": True,
            "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
            "stderr": exc.stderr if isinstance(exc.stderr, str) else "",
        }
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "timeout": False,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _traceback_has_error(text):
    if not text:
        return False
    return "Traceback (most recent call last)" in text


def _read_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _current_meta_pieces(actual):
    out = []
    for pc in actual.case["pieces"]:
        out.append({
            "key": pc["key"],
            "square": pc["square"],
            "type": pc["type"],
            "color": pc.get("color", True) if isinstance(pc, dict) else True,
        })
    return out


def _read_actual_q_g(actual):
    q = []
    for j in actual.arm_joints:
        q.append(float(p.getJointState(actual.robot, j, physicsClientId=actual.client)[0]))
    g = float(p.getJointState(actual.robot, actual.gripper_joint, physicsClientId=actual.client)[0])
    return q, g


def _build_bridge(actual, query_base_cfg, current_case, initial_snapshot, plan):
    home_q = [float(x) for x in plan["home_q"]]
    g_open = float(plan["g_open"])
    qact, gact = _read_actual_q_g(actual)
    w = None
    try:
        qcfg = json.loads(json.dumps(query_base_cfg))
        w = QueryWorld(qcfg, current_case, qact, gact)
        geo = Geometry(w)
        geo.reset(current_case, initial_snapshot, qact, gact)
    except Exception:
        if w is not None:
            try:
                p.disconnect(w.client)
            except Exception:
                pass
        raise

    try:
        qa = np.asarray(qact, dtype=float)
        qb = np.asarray(home_q, dtype=float)
        dq_deg = np.degrees(np.abs(qb - qa))
        jump = float(np.max(dq_deg)) if dq_deg.size else 0.0
        n_q = max(1, int(math.ceil(jump / routebase.MAX_EDGE_JOINT_STEP_DEG)))
        g_delta_deg = abs(math.degrees(g_open - gact))
        n_g = max(1, int(math.ceil(g_delta_deg / GRIPPER_STEP_DEG)))
        n = max(n_q, n_g)
        ts = np.linspace(0.0, 1.0, n + 1)
        nodes = []
        for t in ts:
            q = ((1.0 - t) * qa + t * qb).tolist()
            g = float((1.0 - t) * gact + t * g_open)
            target_key = current_case["target_key"]
            chk = geo.check(q, g, "retreat", initial_snapshot[target_key])
            if not (isinstance(chk, dict) and chk.get("ok") is True):
                raise TaskStop("bridge_collision:%r" % chk, "preparation_blocked")
            try:
                routebase._table_check(w, q)
            except Exception as exc:
                raise TaskStop("bridge_table:%s" % exc, "preparation_blocked")
            nodes.append({"q": [float(x) for x in q], "g": g, "phase": "reposition"})
        return nodes
    finally:
        try:
            p.disconnect(w.client)
        except Exception:
            pass


def _wrap_runner(actual, runner):
    orig_record, orig_guard = runner._record, runner._guard
    def wrapped_record(obs, *args, **kwargs):
        obs["target_neighbor_contacts"] = [
            {"key": other, "points": [list(cp) for cp in p.getContactPoints(
                bodyA=actual.bodies[runner.target_key], bodyB=actual.bodies[other], physicsClientId=actual.client)]}
            for other in actual.bodies if other != runner.target_key]
        return orig_record(obs, *args, **kwargs)
    def wrapped_guard(obs):
        result = orig_guard(obs)
        for entry in obs.get("target_neighbor_contacts", []):
            for cp in entry["points"]:
                if float(cp[9]) > CONTACT_FORCE_GUARD_N:
                    raise E.TaskStop("target_neighbor_contact", "physical_task_failed")
        return result
    runner._record, runner._guard = wrapped_record, wrapped_guard
    return orig_record, orig_guard


def run_session(cfg, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / "result.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite existing %s" % result_path)

    t_start = time.time()
    session = {
        "attempted_actions": [],
        "completed_steps": [],
        "planned_steps": [],
        "total_physics_steps": 0,
        "total_wall_s": 0.0,
        "per_step": [],
        "before_snapshot": None,
        "after_snapshot": None,
        "client_id": None,
        "body_ids": None,
        "outer_guard_calls": [],
        "error": None,
        "classification": None,
        "ok": False,
    }
    actual = None
    restore_outer = None
    outer_calls = []
    disconnected = False

    def _cleanup():
        nonlocal disconnected
        for name, fn in [("restore_outer", restore_outer), ("disconnect_actual", (lambda: p.disconnect(actual.client)) if actual is not None and not disconnected else None)]:
            if fn is not None:
                try:
                    fn()
                except Exception as exc:
                    session.setdefault("cleanup_errors", []).append(name + ":" + str(exc))
                    session["ok"] = False
                    session["classification"] = "program_error"
        disconnected = actual is not None

    def _remaining():
        return min(AGGREGATE_DEADLINE_S, float(cfg.get("deadline_s", 360))) - (time.time() - t_start)

    try:
        base_query_path = Path(cfg["base_query"])
        home_plan_path = Path(cfg["initial_home_plan"])
        steps = cfg["steps"]
        pieces_cfg = cfg["pieces"]
        if len(steps) != int(cfg.get("expected_steps", 10)) or not 1 <= len(steps) <= MAX_ACTIONS:
            raise RuntimeError("step count does not match declared1..10 budget")
        session["planned_steps"] = ["step%02d_%s" % (i + 1, st["id"]) for i, st in enumerate(steps)]
        if E is None or QueryWorld is None or Geometry is None or protect is None:
            raise RuntimeError("required modules unavailable")

        with open(base_query_path, "r") as f:
            base_query = json.load(f)
        with open(home_plan_path, "r") as f:
            home_plan = json.load(f)

        home_q = [float(x) for x in home_plan["home_q"]]
        g_open = float(home_plan["g_open"])

        initial_case = {
            "target_key": steps[0]["key"],
            "id": "initial_" + steps[0]["id"],
            "type": steps[0]["type"],
            "source": steps[0]["source"],
            "dest": steps[0]["dest"],
            "pieces": [
                {"key": pc["key"], "square": pc["square"],
                 "type": pc["type"], "color": pc.get("color", True)}
                for pc in pieces_cfg
            ],
        }

        actual = QueryWorld(base_query, initial_case, home_q, g_open)
        actual.step = MethodType(E.PhysicsWorld.step, actual)
        restore_outer, outer_calls = protect(actual)
        session["client_id"] = int(actual.client)
        session["body_ids"] = {k: int(v) for k, v in actual.bodies.items()}
        session["before_snapshot"] = _snapshot_copy(actual.snapshot())

        run_failed = False
        for idx, step in enumerate(steps):
            step_id = step["id"]
            step_name = "step%02d_%s" % (idx + 1, step_id)
            stepdir = outdir / step_name
            stepdir.mkdir(parents=True, exist_ok=False)
            step_result = {
                "id": step_id,
                "name": step_name,
                "status": "pending",
                "path": str(stepdir),
                "results": None,
                "before_snapshot": None,
                "after_snapshot": None,
                "client_id": None,
                "body_ids": None,
                "world_steps_start": None,
                "world_steps_end": None,
                "physics_steps_delta": 0,
                "error": None,
                "classification": None,
            }
            session["per_step"].append(step_result)

            try:
                rem = _remaining()
                if rem <= 1.0:
                    raise TaskStop("aggregate_deadline", "preparation_blocked")

                current_meta = _current_meta_pieces(actual)
                cur_keys = sorted(pc["key"] for pc in current_meta)
                cur_types = sorted(pc["key"] + ":" + pc["type"] for pc in current_meta)
                case = {
                    "id": step_id,
                    "type": step["type"],
                    "target_key": step["key"],
                    "source": step["source"],
                    "dest": step["dest"],
                    "pieces": current_meta,
                }
                assert step["key"] in cur_keys, "target key missing"
                assert step["type"] == dict(
                    (pc["key"], pc["type"]) for pc in current_meta
                ).get(step["key"]), "target type mismatch"

                snap_before = actual.snapshot()
                step_result["before_snapshot"] = _snapshot_copy(snap_before)
                step_result["client_id"] = int(actual.client)
                step_result["body_ids"] = {k: int(v) for k, v in actual.bodies.items()}
                try:
                    step_result["world_steps_start"] = int(actual.steps)
                except Exception:
                    step_result["world_steps_start"] = None

                query_cfg = json.loads(json.dumps(base_query))
                query_cfg["case"] = case
                query_cfg["safe_home_q"] = home_q
                query_cfg["initial_snapshot"] = snap_before
                query_cfg["keypoint_catalog_path"] = str(step["catalog"])
                query_cfg["catalog_candidates"] = list(step["catalog_candidates"])
                query_cfg["hover_options_m"] = list(cfg["hover_options_m"])
                query_cfg["gripper_reference_path"] = str(step["reference"])
                _planning_context(actual, query_cfg, case, snap_before)
                query_path = stepdir / "query.json"
                _write_json(query_path, query_cfg)

                env_pp = _make_pythonpath([])

                center_script = WORKSPACE / "calibration/manipulation_20261109_v1/smoke_grasps_20261110_v1/work/center_snapshot_v1.py"
                center_out = stepdir / "center.json"
                to_center = min(MAX_CLI_TIMEOUT_S, _remaining())
                center_run = _run_cli(
                    center_script,
                    ["--config", str(query_path),
                     "--ik", str(step["ik_seed"]),
                     "--output", str(center_out)],
                    to_center, env_pp,
                )
                (stepdir / "center.stdout").write_text(center_run.get("stdout", "") or "")
                (stepdir / "center.stderr").write_text(center_run.get("stderr", "") or "")
                if center_run.get("timeout"):
                    raise TaskStop("center_timeout", "preparation_blocked")
                rc = center_run.get("returncode")
                if _traceback_has_error(center_run.get("stderr", "")):
                    raise TaskStop("center_traceback", "program_error")
                center_json = _read_json(center_out) if center_out.exists() else None
                if isinstance(center_json, dict) and center_json.get("blocked") is True and rc in (0, 1):
                    raise TaskStop("center_blocked", "preparation_blocked")
                if rc != 0:
                    raise TaskStop("center_rc_%s" % rc, "program_error")
                if not (isinstance(center_json, dict) and center_json.get("blocked") is False
                        and isinstance(center_json.get("check_close"), dict) and center_json["check_close"].get("ok") is True
                        and isinstance(center_json.get("check_descend"), dict) and center_json["check_descend"].get("ok") is True):
                    raise TaskStop("center_malformed", "program_error")

                query_cfg = _read_json(query_path)
                if float(query_cfg.get("placement_yaw_deg", 0.0)) != float(center_json["selected_placement_yaw_deg"]):
                    raise RuntimeError("selected yaw not persisted into actual query")
                plan_dir = stepdir / "plan"
                plan_result_path = plan_dir / "result.json"
                action_cfg = {
                    "query": str(query_path),
                    "center": str(center_out),
                    "hover_m": float(center_json["selected_hover_m"]),
                    "target_force_N": 0.25,
                    "max_steps": int(cfg.get("max_steps_per_action", 24000)),
                    "wall_s": float(cfg.get("wall_per_action_s", 120)),
                    "plan": str(plan_result_path),
                }
                action_path = stepdir / "action.json"
                _write_json(action_path, action_cfg)

                route_script = WORKSPACE / "calibration/manipulation_20261109_v1/smoke_grasps_20261110_v1/work/route_snapshot_v1.py"
                to_route = min(MAX_CLI_TIMEOUT_S, _remaining())
                route_run = _run_cli(
                    route_script,
                    ["--config", str(action_path), "--out", str(plan_dir)],
                    to_route, env_pp,
                )
                (stepdir / "route.stdout").write_text(route_run.get("stdout", "") or "")
                (stepdir / "route.stderr").write_text(route_run.get("stderr", "") or "")
                if route_run.get("timeout"):
                    raise TaskStop("route_timeout", "preparation_blocked")
                rrc = route_run.get("returncode")
                if _traceback_has_error(route_run.get("stderr", "")):
                    raise TaskStop("route_traceback", "program_error")
                plan_json = _read_json(plan_result_path) if plan_result_path.exists() else None
                if rrc == 1:
                    if not isinstance(plan_json, dict):
                        raise TaskStop("route_malformed", "program_error")
                    if plan_json.get("ok") is not False:
                        raise TaskStop("route_malformed", "program_error")
                    raise TaskStop("route_blocked", "preparation_blocked")
                if rrc != 0:
                    raise TaskStop("route_rc_%s" % rrc, "program_error")
                if not isinstance(plan_json, dict) or plan_json.get("ok") is not True:
                    raise TaskStop("route_malformed", "program_error")

                try:
                    bridge_nodes = _build_bridge(
                        actual, query_cfg, case, snap_before, plan_json
                    )
                except TaskStop:
                    raise
                except Exception as exc:
                    raise TaskStop("bridge_error:%s" % exc, "preparation_blocked")

                if _remaining() <= 0:
                    raise TaskStop("aggregate_deadline", "preparation_blocked")
                assert {pc["key"]: pc["type"] for pc in actual.case["pieces"]} == {pc["key"]: pc["type"] for pc in case["pieces"]}
                actual.case, actual.cfg = case, query_cfg
                action_cfg["wall_s"] = min(float(cfg.get("wall_per_action_s", 120)), _remaining())
                _write_json(action_path, action_cfg)
                trial_dir = stepdir / "trial"
                trial_dir.mkdir(parents=True, exist_ok=False)
                runner = E.Runner(action_cfg, plan_json, trial_dir,
                                  shared_actual=actual,
                                  initial_bridge=bridge_nodes)
                _wrap_runner(actual, runner)
                session["attempted_actions"].append(step_id)
                rres = runner.run()

                step_result["results"] = rres if isinstance(rres, dict) else None
                delta = 0
                if isinstance(rres, dict):
                    try:
                        wse = rres.get("world_steps_end")
                        wss = rres.get("world_steps_start")
                        if wse is not None and wss is not None:
                            delta = int(wse) - int(wss)
                        else:
                            delta = int(rres.get("physics_steps") or 0)
                    except Exception:
                        delta = 0
                step_result["physics_steps_delta"] = delta
                session["total_physics_steps"] += max(0, delta)
                try:
                    step_result["world_steps_end"] = int(actual.steps)
                except Exception:
                    step_result["world_steps_end"] = None
                step_result["after_snapshot"] = _snapshot_copy(actual.snapshot())

                ok = bool(isinstance(rres, dict) and rres.get("ok") is True
                          and rres.get("completed") is True
                          and rres.get("classification") == "physical_task_success")
                if not ok:
                    step_result["status"] = "failed"
                    step_result["classification"] = (
                        rres.get("classification") if isinstance(rres, dict) else "physical_task_failed"
                    )
                    session["classification"] = step_result["classification"]
                    session["ok"] = False
                    run_failed = True
                    break

                step_result["status"] = "completed"
                for pc in actual.case["pieces"]:
                    if pc["key"] == step["key"]:
                        pc["square"] = step["dest"]
                        break
                session["completed_steps"].append(step_id)

            except TaskStop as exc:
                session["classification"] = exc.classification
                step_result["status"] = "failed"
                step_result["classification"] = exc.classification
                step_result["error"] = exc.reason
                session["error"] = exc.reason
                session["ok"] = False
                run_failed = True
                break
            except Exception as exc:
                session["classification"] = "program_error"
                session["error"] = "%s: %s" % (type(exc).__name__, exc)
                session["error_traceback"] = traceback.format_exc()
                step_result["status"] = "failed"
                step_result["classification"] = "program_error"
                step_result["error"] = str(exc)
                session["ok"] = False
                run_failed = True
                break

        if not run_failed and len(session["completed_steps"]) == int(cfg.get("expected_steps", 10)):
            session["ok"] = True
            session["classification"] = "physical_task_success"
        try:
            session["after_snapshot"] = _snapshot_copy(actual.snapshot())
        except Exception:
            session["after_snapshot"] = None

    except Exception as exc:
        session["ok"] = False
        session["classification"] = "program_error"
        session["error"] = "%s: %s" % (type(exc).__name__, exc)
        session["error_traceback"] = traceback.format_exc()
    finally:
        if actual is not None:
            session["total_physics_steps"] = int(actual.steps)
            session["total_sim_time_s"] = float(actual.time)
        _cleanup()
        session["outer_guard_calls"] = list(outer_calls)
        session["total_wall_s"] = float(time.time() - t_start)
        session["planned_expected"] = len(session["planned_steps"]) == int(cfg.get("expected_steps", 10))
        _write_json(result_path, session)

    if session.get("ok") is True and len(session.get("completed_steps", [])) == int(cfg.get("expected_steps", 10)):
        return 0
    if session.get("classification") == "program_error":
        return 2
    if session.get("classification") in ("preparation_blocked", "physical_task_failed"):
        return 1
    return 2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    with open(args.config, "r") as f:
        cfg = json.load(f)
    try:
        return run_session(cfg, Path(args.out))
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
