import contextlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

_root = Path(__file__).resolve().parents[4]


def _bootstrap_paths(root):
    root = Path(root)
    folders = [
        "calibration/manipulation_20261109_v1/far_grasp_fast_20261110_v1/work",
        "calibration/manipulation_20261109_v1/knight_support_20261110_v1/work",
        "calibration/manipulation_20261109_v1/smoke_grasps_20261110_v1/work",
        "calibration/manipulation_20261109_v1/keypoint_grasps_20261110_v1/work",
        "reachability_eval",
        "calibration/manipulation_20261109_v1/work",
        "experiment_review/tools",
    ]
    for f in folders:
        p = str(root / f)
        if p not in sys.path:
            sys.path.insert(0, p)


_bootstrap_paths(_root)

import numpy as np
from scipy.spatial.transform import Rotation

import pybullet as p

from query_world_snapshot_v1 import QueryWorld
from core import Robot
from query_safety_bound_v1 import Geometry
from tilt_ik_v1 import solve_pose, pose_fk
import stage_routes_bound_v3 as routes
from start_bridge_filter_v1 import filter_plan
import tilt_route_base_v1 as base


class _StepGuard:
    def __init__(self, planner):
        self.planner = planner
        self.orig = None

    def __enter__(self):
        self.orig = self.planner.p.stepSimulation

        def _raise(*args, **kwargs):
            self.planner.guard_calls += 1
            raise RuntimeError("stepSimulation called during static query")

        self.planner.p.stepSimulation = _raise
        return self

    def __exit__(self, exc_type, exc, tb):
        self.planner.p.stepSimulation = self.orig
        return False


class ResidentPlanner:
    _LAYOUT_KEYS = (
        "asset_dir",
        "geometry",
        "square_m",
        "yaw_rad",
        "center_m",
        "board_top_m",
        "board_bottom_m",
        "board_half_xy_m",
        "native_collision_floor_offset_m",
        "dt_s",
    )

    _KNOWN_ROUTE_REASONS = (
        "solve_pose fail",
        "joint jump",
        "geo.check failed",
        "route: first node jump",
        "table collision",
        "rebuild: open check rejected",
        "home/opening rejection",
        "staged_no_plan:",
    )

    def __init__(self, deps=None, root=None):
        self.w = None
        self.geo = None
        self.stats = []
        self.worlds_created = 0
        self.guard_calls = 0

        deps = deps or {}
        self._world_cls = deps.get("world_cls", QueryWorld)
        self._geometry_cls = deps.get("geometry_cls", Geometry)
        # deps['kin'] is a callable class receiving the URDF path
        self._kin_cls = deps.get("kin", Robot)
        self._solve = deps.get("solve", solve_pose)
        self._pose_fk = deps.get("pose_fk", pose_fk)
        self._routes = deps.get("routes", routes)
        self._base = deps.get("base", base)
        self.p = deps.get("p", p)

        if root is None:
            root = Path(__file__).resolve().parents[4]
        self.root = Path(root)
        self.urdf = (
            self.root / "reachability_eval" / "model" / "so101_new_calib.urdf"
        )
        self.kin = self._kin_cls(str(self.urdf))
        self._layout = None
        self._closed = False

    def _guard(self):
        return _StepGuard(self)

    def _stat(self, **kw):
        self.stats.append(kw)

    @staticmethod
    def _key_set(cfg):
        pieces = cfg.get("case", {}).get("pieces")
        if pieces is None:
            return frozenset()
        out = set()
        for pc in pieces:
            out.add((pc.get("type"), pc.get("key")))
        return frozenset(out)

    def _layout_vals(self, cfg):
        vals = {}
        for k in self._LAYOUT_KEYS:
            if k in cfg:
                v = cfg[k]
                try:
                    hash(v)
                    vals[k] = v
                except TypeError:
                    vals[k] = json.dumps(v, sort_keys=True)
        return vals

    def _check_compat(self, cfg):
        keys = self._key_set(cfg)
        if self._layout is None:
            self._layout = self._layout_vals(cfg)
            self._layout_keys = keys
            return
        if keys != self._layout_keys:
            raise ValueError("piece key:type set changed; new world refused")
        vals = self._layout_vals(cfg)
        a = self._layout
        b = vals
        diff = set(a) | set(b)
        for k in diff:
            if a.get(k) != b.get(k):
                raise ValueError("immutable layout changed: %s" % k)

    def _acquire(self, cfg, q, g):
        if self.w is None:
            w = self._world_cls(cfg, cfg["case"], q, g)
            self.w = w
            self.worlds_created += 1
        else:
            self.w.cfg = cfg
            self.w.set_case(
                cfg["case"], q, g, poses=cfg.get("initial_snapshot")
            )
        return self.w

    def center(self, querycfg, ik):
        t0 = time.time()
        created_before = self.worlds_created
        with self._guard():
            res = self._center_impl(querycfg, ik, created_before)
        elapsed = time.time() - t0
        self._stat(
            stage="center",
            wall_s=elapsed,
            candidate_count=len(res.get("candidate_attempts") or []),
            selected_catalog=res.get("selected_catalog_path"),
        )
        return res

    def _center_impl(self, querycfg, ik, created_before):
        records = ik.get("records") or []
        seed_rec = None
        for rec in records:
            if not rec.get("accepted"):
                continue
            tgt = rec.get("target") or {}
            if tgt.get("kind") == "head_grip_current" and tgt.get("phase") == "source":
                seed_rec = rec
                break
        if seed_rec is None:
            return {
                "blocked": True,
                "reason": "no_seed",
                "physics_steps": 0,
                "query_worlds_created": 0,
            }

        seed = list(seed_rec["solution"]["q"])
        target = seed_rec["target"]
        pos = list(target["pos"])
        pos[2] = pos[2] + float(querycfg.get("native_collision_floor_offset_m", 0.0))
        quat = target["quat"]

        sol = self._solve(self.kin, pos, quat, seed)
        if not sol.get("pass"):
            return {
                "blocked": True,
                "reason": "seed_solve_failed",
                "seed_solve": sol,
                "physics_steps": 0,
                "query_worlds_created": self.worlds_created - created_before,
            }

        cfg = dict(querycfg)
        self._check_compat(cfg)
        q_init = list(sol["q"])
        self._acquire(cfg, q_init, 0.7)
        snapshot = self.w.snapshot()

        if self.geo is None:
            self.geo = self._geometry_cls(self.w)
        self.geo.reset(cfg["case"], snapshot, q_init, 0.7)

        if "catalog_candidates" in cfg and cfg["catalog_candidates"]:
            catalog_paths = list(cfg["catalog_candidates"])
        else:
            catalog_paths = [cfg["keypoint_catalog_path"]]

        attempts = []
        selected_path = None
        chosen = None
        T = self._pose_fk(self.kin, q_init)
        fk_pos = np.asarray(T)[:3, 3].tolist()
        fk_quat = Rotation.from_matrix(np.asarray(T)[:3, :3]).as_quat().tolist()

        for path in catalog_paths[:7]:
            self.w.cfg["keypoint_catalog_path"] = path
            self.geo.reset(cfg["case"], snapshot, q_init, 0.7)
            cand = self.geo.contact_grasp(
                q_init, fk_pos, fk_quat, q_init, self.kin, 0.7
            )
            attempts.append(cand)
            if cand is not None and cand.get("ok") is True:
                chosen = cand
                selected_path = path
                break

        delta = self.worlds_created - created_before

        if chosen is None:
            return {
                "blocked": True,
                "reason": "no_contact",
                "source_snapshot": snapshot,
                "candidate_attempts": attempts,
                "physics_steps": 0,
                "query_worlds_created": delta,
            }

        q = list(chosen["q"])
        gc = float(chosen["gcontact"])
        self.geo.reset(cfg["case"], snapshot, q, gc)
        go = self.geo.opening(q, gc, 3.0)
        lo = self.geo.lo
        hi = self.geo.hi
        if go is None or not (lo <= gc < go <= hi):
            return {
                "blocked": True,
                "reason": "opening_bounds",
                "source_snapshot": snapshot,
                "q_contact": q,
                "gcontact_rad": gc,
                "gopen_rad": go,
                "urdf_gripper_lo": lo,
                "urdf_gripper_hi": hi,
                "contact_result": chosen,
                "candidate_attempts": attempts,
                "selected_catalog_path": selected_path,
                "physics_steps": 0,
                "query_worlds_created": delta,
            }

        tk = cfg["case"]["target_key"]
        chk_close = self.geo.check(q, gc, "close", snapshot[tk])
        chk_desc = self.geo.check(q, go, "descend", snapshot[tk])
        blocked = (chk_close.get("ok") is not True) or (chk_desc.get("ok") is not True)
        out = {
            "blocked": blocked,
            "reason": "" if not blocked else "geo_check_failed",
            "source_snapshot": snapshot,
            "q_contact": q,
            "gcontact_rad": gc,
            "gopen_rad": go,
            "urdf_gripper_lo": lo,
            "urdf_gripper_hi": hi,
            "check_close": chk_close,
            "check_descend": chk_desc,
            "contact_result": chosen,
            "candidate_attempts": attempts,
            "selected_catalog_path": selected_path,
            "physics_steps": 0,
            "query_worlds_created": delta,
        }
        return out

    def plan(self, actioncfg, center):
        t0 = time.time()
        created_before = self.worlds_created
        with self._guard():
            res = filter_plan(self, self._plan_impl(actioncfg, center, created_before))
        self._stat(stage="plan", wall_s=time.time() - t0)
        return res

    def _route_reason(self, exc):
        msg = str(exc)
        for m in self._KNOWN_ROUTE_REASONS:
            if m in msg:
                return msg
        return None

    def _plan_impl(self, actioncfg, center, created_before):
        if center.get("blocked") is not False:
            return {"ok": False, "reason": "center_blocked", "traceback": None}
        if center.get("check_close", {}).get("ok") is not True:
            return {"ok": False, "reason": "check_close_failed", "traceback": None}
        if center.get("check_descend", {}).get("ok") is not True:
            return {"ok": False, "reason": "check_descend_failed", "traceback": None}

        cfg = actioncfg
        querycfg_path = cfg["query"]
        with open(querycfg_path) as f:
            querycfg = json.load(f)

        self._check_compat(querycfg)
        snapshot = center["source_snapshot"]
        q = list(center["q_contact"])
        gc = float(center["gcontact_rad"])
        go = float(center["gopen_rad"])

        qcfg = dict(querycfg)
        qcfg["placement_yaw_deg"] = float(cfg.get("placement_yaw_deg", querycfg.get("placement_yaw_deg", 0.0)))
        qcfg["case"] = qcfg.get("case", querycfg["case"])
        self._acquire(qcfg, q, gc)
        self.w.cfg = qcfg
        self.w.set_case(qcfg["case"], q, gc, poses=snapshot)
        self.geo.reset(qcfg["case"], snapshot, q, gc)

        hover_m = float(cfg["hover_m"])

        case = qcfg["case"]
        snap = snapshot[case["target_key"]]

        t0 = time.time()
        Tcontact = self._pose_fk(self.kin, q)
        Tcontact = np.asarray(Tcontact, dtype=float)
        Thover = Tcontact.copy()
        Thover[2, 3] += hover_m
        hover_pos = Thover[:3, 3].tolist()
        hover_quat = Rotation.from_matrix(Thover[:3, :3]).as_quat().tolist()

        home_sol = self._solve(self.kin, hover_pos, hover_quat, q)
        if not home_sol.get("pass"):
            return {"ok": False, "reason": "hover_solve_failed", "traceback": None}
        homeq = list(home_sol["q"])
        self._stat(stage="plan_hover", wall_s=time.time() - t0)

        t0 = time.time()
        try:
            r = self._routes.route(
                self.w, self.geo, self.kin, Thover, Tcontact,
                homeq, go, "descend", stationary=snap,
            )
        except RuntimeError as exc:
            reason = self._route_reason(exc)
            if reason is None:
                raise
            return {"ok": False, "reason": reason, "traceback": None}
        self._stat(stage="plan_source_route", wall_s=time.time() - t0)

        t0 = time.time()
        gl = max(self.geo.lo, gc - math.radians(5.0))
        steps = int(math.ceil((go - gl) / math.radians(0.5)))
        if steps < 1:
            steps = 1
        for i in range(steps + 1):
            gi = go + (gl - go) * (i / float(steps))
            if gi < gl:
                gi = gl
            chk = self.geo.check(q, gi, "close", snap)
            if chk.get("ok") is not True:
                return {
                    "ok": False,
                    "reason": "close sweep failed: %s" % chk.get("reason"),
                    "traceback": None,
                }
            try:
                self._base._table_check(self.w, q)
            except RuntimeError as exc:
                reason = self._route_reason(exc)
                if reason is None:
                    raise
                return {"ok": False, "reason": reason, "traceback": None}
        self._stat(stage="plan_closure", wall_s=time.time() - t0)

        t0 = time.time()
        try:
            nominal = self._routes.rebuild(
                self.w, self.geo, self.kin, snapshot, q, gc,
                homeq, go, hover_m, True,
            )
        except RuntimeError as exc:
            reason = self._route_reason(exc)
            if reason is None:
                raise
            return {"ok": False, "reason": reason, "traceback": None}
        self._stat(stage="plan_nominal_suffix", wall_s=time.time() - t0)

        return {
            "ok": True,
            "case": case,
            "home_q": homeq,
            "contact_q": q,
            "g_open": go,
            "g_contact": gc,
            "g_limit": gl,
            "source_snapshot": snapshot,
            "prefix": r,
            "nominal": nominal,
            "physics_steps": 0,
            "query_worlds_created": self.worlds_created - created_before,
        }

    def bridge(self, actual, querycfg, case, snapshot, plan):
        t0 = time.time()
        with self._guard():
            res = self._bridge_impl(actual, querycfg, case, snapshot, plan)
        self._stat(stage="bridge", wall_s=time.time() - t0)
        return res

    def _bridge_impl(self, actual, querycfg, case, snapshot, plan):
        client = actual.client
        q_meas = []
        for j in actual.arm_joints:
            st = self.p.getJointState(actual.robot, j, physicsClientId=client)
            q_meas.append(st[0])
        g_meas = self.p.getJointState(
            actual.robot, actual.gripper_joint, physicsClientId=client
        )[0]

        cfg = dict(querycfg)
        cfg["case"] = case
        cfg["initial_snapshot"] = snapshot
        self._check_compat(cfg)
        q0 = list(plan["home_q"])
        g0 = float(plan["g_open"])

        max_jd = max(abs(a - b) for a, b in zip(q_meas, q0))
        max_gd = abs(g_meas - g0)
        n = max(
            1,
            int(math.ceil(max_jd / math.radians(0.25))),
            int(math.ceil(max_gd / math.radians(0.5))),
        )

        self._acquire(cfg, q_meas, g_meas)
        self.w.cfg = cfg
        self.w.set_case(case, q_meas, g_meas, poses=snapshot)
        self.geo.reset(case, snapshot, q_meas, g_meas)

        tk = case["target_key"]
        ref = snapshot[tk]
        nodes = []
        for i in range(n + 1):
            t = i / float(n)
            qi = [a + (b - a) * t for a, b in zip(q_meas, q0)]
            gi = g_meas + (g0 - g_meas) * t
            chk = self.geo.check(qi, gi, "retreat", ref)
            if chk.get("ok") is not True:
                raise RuntimeError(
                    "bridge geo.check failed at node %d: %s"
                    % (i, chk.get("reason"))
                )
            try:
                self._base._table_check(self.w, qi)
            except RuntimeError as exc:
                reason = self._route_reason(exc)
                if reason is None:
                    raise
                raise RuntimeError(
                    "bridge table collision at node %d: %s" % (i, reason)
                )
            nodes.append({"q": qi, "g": gi, "phase": "reposition"})
        return nodes

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.w is None:
                return
            steps = getattr(self.w, "steps", 0)
            tm = getattr(self.w, "time", 0)
            if steps != 0 or tm != 0:
                raise RuntimeError(
                    "query world was stepped: steps=%r time=%r" % (steps, tm)
                )
        finally:
            if self.w is not None:
                client = getattr(self.w, "client", None)
                if client is not None:
                    self.p.disconnect(client)


def run_session(cfg, outdir):
    import continuous_support_v1 as C
    import fast_runner_v1 as E

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    orig_E = C.E
    orig_run_cli = C._run_cli
    orig_build_bridge = C._build_bridge
    rp = ResidentPlanner()
    t0 = time.time()
    session_int = None
    close_exc = None

    def _run_cli(script_path, args_list, timeout_s, env_pp):
        name = Path(script_path).name
        start = time.time()
        if name == "center_snapshot_v1.py":
            def _arg(flag):
                if flag in args_list:
                    return args_list[args_list.index(flag) + 1]
                return None

            qcfg_p = _arg("--config")
            ik_p = _arg("--ik")
            out_p = _arg("--output")
            if out_p is not None and Path(out_p).exists():
                raise RuntimeError("refusing to overwrite existing center output: %s" % out_p)
            with open(qcfg_p) as f:
                querycfg = json.load(f)
            with open(ik_p) as f:
                ik = json.load(f)
            res = rp.center(querycfg, ik)
            with open(out_p, "w") as f:
                json.dump(res, f)
            elapsed = time.time() - start
            if elapsed > timeout_s:
                return {"returncode": 0, "timeout": True,
                        "stdout": "", "stderr": "", "output_preserved": True}
            return {"returncode": 0, "timeout": False, "stdout": "", "stderr": ""}
        elif name == "route_snapshot_v1.py":
            def _arg(flag):
                if flag in args_list:
                    return args_list[args_list.index(flag) + 1]
                return None

            action_p = _arg("--config")
            out_p = _arg("--out")
            outdir_p = Path(out_p)
            result_path = outdir_p / "result.json"
            if result_path.exists():
                raise RuntimeError("refusing to overwrite existing route output: %s" % result_path)
            outdir_p.mkdir(parents=True, exist_ok=True)
            with open(action_p) as f:
                actioncfg = json.load(f)
            with open(actioncfg["center"]) as f:
                center = json.load(f)
            res = rp.plan(actioncfg, center)
            with open(result_path, "w") as f:
                json.dump(res, f)
            rc = 0 if res.get("ok") else 1
            elapsed = time.time() - start
            if elapsed > timeout_s:
                return {"returncode": rc, "timeout": True,
                        "stdout": "", "stderr": "", "output_preserved": True}
            return {"returncode": rc, "timeout": False, "stdout": "", "stderr": ""}
        else:
            raise ValueError("unsupported script: %s" % name)

    def _build_bridge(actual, querycfg, case, snapshot, plan):
        return rp.bridge(actual, querycfg, case, snapshot, plan)

    try:
        C.E = E
        C._run_cli = _run_cli
        C._build_bridge = _build_bridge
        session_int = C.run_session(cfg, outdir)
    finally:
        try:
            rp.close()
        except Exception as e:
            close_exc = e
        C.E = orig_E
        C._run_cli = orig_run_cli
        C._build_bridge = orig_build_bridge
        total = time.time() - t0
        stats_path = outdir / "planner_stats.json"
        with open(stats_path, "w") as f:
            json.dump({
                "stages": rp.stats,
                "query_worlds_created": rp.worlds_created,
                "guard_calls": rp.guard_calls,
                "ready_pose_policy": "checked destination hover; no return to old source",
                "total_wall_s": total,
            }, f)
    if close_exc is not None:
        raise close_exc
    return session_int


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[4]
    _bootstrap_paths(root)

    with open(args.config) as f:
        cfg = json.load(f)
    rc = run_session(cfg, args.out)
    sys.exit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
