import gzip
from support_feedback_v2 import confirm_support
import json
import math
import sys
import time
import traceback
from pathlib import Path
from types import MethodType

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

_WORKDIR = Path(__file__).resolve().parent
_WS = Path(__file__).resolve().parents[4]
for _extra in (
    _WORKDIR,
    _WS / "reachability_eval",
    _WS / "calibration" / "manipulation_20261109_v1" / "work",
    _WS / "experiment_review" / "tools",
):
    _s = str(_extra)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from query_world_snapshot_v1 import QueryWorld  # noqa: E402
from physics_world import PhysicsWorld  # noqa: E402
from core import Robot  # noqa: E402
from tilt_ik_v1 import pose_fk  # noqa: E402
from query_safety_bound_v1 import Geometry, target_clearance  # noqa: E402
from recontact_hold_control_v1 import PressureGrip, apply_motor  # noqa: E402
from ompl_continuous_helpers_v4 import protect  # noqa: E402
from stage_routes_bound_v3 import route, rebuild, StagePlanRejected  # noqa: E402
from release_planner_bound_v1 import plan_exit

workspace = _WS


def _quat_to_rot(q):
    return Rotation.from_quat(list(q))


def _rot_to_quat(r):
    return list(r.as_quat())


class TaskStop(Exception):
    def __init__(self, reason, classification="physical_task_failed"):
        super().__init__(reason)
        self.reason = reason
        self.classification = classification


class Runner:
    def __init__(self, cfg, plan, outdir, shared_actual=None, initial_bridge=None):
        self.cfg = cfg
        self.plan = plan
        self.outdir = Path(outdir)
        self.hover_m = float(cfg.get("hover_m", 0.04))
        self.target_force_N = float(cfg.get("target_force_N", 0.25))
        self.max_steps = int(cfg.get("max_steps", 60000))
        self.wall_s = float(cfg.get("wall_s", 180.0))

        with open(cfg["query"], "r") as f:
            self.query_cfg = json.load(f)
        self.case = self.query_cfg["case"] if "case" in self.query_cfg else self.query_cfg

        self.home_q = list(plan["home_q"])
        self.contact_q = list(plan["contact_q"])
        self.g_open = float(plan["g_open"])
        self.g_contact = float(plan["g_contact"])
        self.g_limit = float(plan["g_limit"])
        self.prefix = list(plan["prefix"])
        self.source_snapshot = plan.get("source_snapshot")
        self.target_key = self.case["target_key"]
        self.dest = self.case["dest"]

        urdf = str(workspace / "reachability_eval/model/so101_new_calib.urdf")
        self.kin = Robot(urdf)

        self.shared_actual = shared_actual
        self.initial_bridge = list(initial_bridge or [])
        self.actual = shared_actual if shared_actual is not None else QueryWorld(self.query_cfg, self.case, self.home_q, self.g_open)
        self.start_steps = int(self.actual.steps)
        self.start_time = float(self.actual.time)
        self.wq = QueryWorld(self.query_cfg, self.case, self.contact_q, self.g_contact)
        self.geo = Geometry(self.wq)
        self.wq.cfg["_runtime_reselect"] = True
        self.wq.plan_audits = []

        self.fixed_finger_idx = self.actual.link_index["fixed_jaw_collision"]
        self.moving_finger_idx = self.actual.link_index["moving_jaw_so101_v1_link"]
        self.finger_link_indices = {self.fixed_finger_idx, self.moving_finger_idx, self.actual.link_index["gripper_frame_link"]}

        info = p.getDynamicsInfo(self.actual.robot, 0, physicsClientId=self.actual.client)
        piece_body = self.actual.bodies[self.target_key]
        piece_info = p.getDynamicsInfo(piece_body, -1, physicsClientId=self.actual.client)
        lat = math.sqrt(0.4)
        p.changeDynamics(piece_body, -1, lateralFriction=lat, physicsClientId=self.actual.client)
        check_lat = p.getDynamicsInfo(piece_body, -1, physicsClientId=self.actual.client)[1]
        if abs(check_lat - lat) > 1e-9:
            raise RuntimeError("lateralFriction bind failed: %.6f" % check_lat)

        self.actual.step = MethodType(PhysicsWorld.step, self.actual)

        self.restore, self.guard_calls = protect(self.actual)

        self.trace_path = self.outdir / "trace.jsonl.gz"
        self.trace_fh = gzip.open(self.trace_path, "wt")
        self.initial_obs = self.actual.observe("init", self.home_q, self.g_open)
        self.last = self.initial_obs
        self.initial_target_z = float(self.initial_obs["target_pos"][2])
        self.qcmd = list(self.initial_obs["actual_q5"])
        self.gcmd = float(self.initial_obs["actual_g"])
        self.phase_counts = {}
        self.released = False
        self.supported = False
        self.drop = False
        self.max_rise = 0.0
        self.press_ctrl = None
        self.t0 = time.time()
        self._record(self.initial_obs, step=self.start_steps, phase="init", mode="position",
                     command_q=self.qcmd, command_g=self.gcmd,
                     control_decision=None, gaps_m=None, boardforceN=0.0,
                     gripperstate=tuple(p.getJointState(self.actual.robot, self.actual.gripper_joint, physicsClientId=self.actual.client)))

    def _check_caps(self):
        if self.actual.steps - self.start_steps >= self.max_steps:
            raise TaskStop("max_steps", "resource_block")
        if (time.time() - self.t0) > self.wall_s:
            raise TaskStop("wall_clock", "resource_block")

    def _gap_query(self):
        gaps = []
        any_null = False
        target_body = self.actual.bodies[self.target_key]
        for finger in (self.fixed_finger_idx, self.moving_finger_idx):
            cp = p.getClosestPoints(self.actual.robot, target_body, 0.05,
                                    linkIndexA=finger, physicsClientId=self.actual.client)
            if not cp:
                any_null = True
            else:
                gaps.append(min(c[8] for c in cp))
        if any_null or len(gaps) < 2:
            return None
        return gaps

    def _record(self, obs, step, phase, mode, command_q, command_g,
                control_decision, gaps_m, boardforceN, gripperstate):
        row = {
            "step": int(step - self.start_steps),
            "world_step": int(step),
            "world_time": float(obs["time"]),
            "phase": phase,
            "mode": mode,
            "command_q": [float(x) for x in command_q],
            "command_g": float(command_g),
            "control_decision": control_decision,
            "boardforceN": float(boardforceN),
            "gripperstate": list(gripperstate),
            "gaps_m": list(gaps_m) if gaps_m is not None else None,
        }
        for k, v in obs.items():
            if k == "time":
                row["time"] = v - self.start_time
            elif k in ("actual_q5", "target_pos", "target_quat", "tcp_pos", "tcp_quat"):
                row[k] = list(v) if hasattr(v, "__iter__") else v
            elif k == "actual_g":
                row[k] = float(v)
            else:
                row[k] = v
        self.trace_fh.write(json.dumps(row, default=str) + "\n")

    def _guard(self, obs):
        rb = obs.get("robot_board_contacts") or []
        for k, c in enumerate(rb):
            if isinstance(c, dict) and c.get("normalForce", 0.0) > 0.01:
                raise TaskStop("guard:robot_board %s" % k, "physical_task_failed")
        sc = obs.get("self_contacts") or []
        for k, c in enumerate(sc):
            if isinstance(c, dict) and c.get("normalForce", 0.0) > 0.01:
                if tuple(sorted((int(c["linkA"]), int(c["linkB"])))) not in self.actual.excluded_pairs:
                    raise TaskStop("guard:self %s" % (k,), "physical_task_failed")
        nc = obs.get("neighbor_contacts") or []
        for k, c in enumerate(nc):
            if isinstance(c, dict) and c.get("normalForce", 0.0) > 0.01:
                raise TaskStop("guard:neighbor %s" % k, "physical_task_failed")
        tc = obs.get("contacts") or []
        for k, c in enumerate(tc):
            if not isinstance(c, dict):
                continue
            if c.get("normalForce", 0.0) > 0.01:
                rl = c.get("robot_link")
                if rl is None:
                    rl = c.get("linkA")
                if rl is not None and rl not in ("fixed_jaw_collision", "moving_jaw_so101_v1_link", "gripper_frame_link"):
                    raise TaskStop("guard:target %s" % (k,), "physical_task_failed")
        cp_all = p.getContactPoints(self.actual.robot, self.actual.table,
                                    physicsClientId=self.actual.client)
        for cp in cp_all:
            if cp[3] != -1 and cp[9] > 0.01:
                raise TaskStop("guard:robot_table", "physical_task_failed")

    def tick(self, phase, q, g, mode):
        self._check_caps()
        js = p.getJointState(self.actual.robot, self.actual.gripper_joint,
                             physicsClientId=self.actual.client)
        before = self.last
        fixed_f = float(before["fixed_jaw_collision_sum"])
        moving_f = float(before["moving_jaw_collision_sum"])
        forces = [fixed_f, moving_f]
        gaps = self._gap_query() if mode == "hold" and min(forces) <= 0.01 else None

        decision = None
        if mode == "close":
            d = self.press_ctrl.close(js[0], js[1], forces, self.actual.dt, 0.0, 0.0)
            if not d.get("ok"):
                raise TaskStop("close_not_ok", "physical_task_failed")
            g = float(d["command"])
            decision = {k: d[k] for k in d if k != "command"}
        elif mode == "hold":
            self.press_ctrl.velocity = js[1]
            self.press_ctrl.measured_torque = js[3]
            if gaps is not None and len(gaps) >= 2:
                self.press_ctrl.jaw_gaps_mm = [1000.0 * x for x in gaps]
            else:
                self.press_ctrl.jaw_gaps_mm = None
            board_sum = 0.0
            for c in (before.get("board_contacts") or []):
                if isinstance(c, dict):
                    board_sum += float(c.get("normalForce", 0.0))
            supported = board_sum > 0.005
            d = self.press_ctrl.hold(js[0], forces, self.actual.dt, supported=supported)
            decision = {k: d[k] for k in d if k != "command"}
        elif mode != "position":
            raise RuntimeError("bad mode %r" % mode)

        self.actual.command(q, g)
        if mode == "hold":
            apply_motor(p, self.actual, decision if "target_velocity_rad_s" in decision else d)
        self.actual.step()

        obs = self.actual.observe(phase, q, g)
        if obs["target_pos"][2] - self.initial_target_z > self.max_rise:
            self.max_rise = float(obs["target_pos"][2] - self.initial_target_z)
        self.last = obs
        self.last_decision = decision
        self.qcmd = list(q)
        self.gcmd = float(g)
        self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1

        boardforce = 0.0
        for c in (obs.get("board_contacts") or []):
            if isinstance(c, dict):
                boardforce += float(c.get("normalForce", 0.0))
        target_body = self.actual.bodies[self.target_key]
        raw_cps = list(p.getContactPoints(self.actual.robot, target_body,
                                          physicsClientId=self.actual.client))
        fresh_gaps = self._gap_query() if not self.released and self.max_rise > 0.003 and obs["fixed_jaw_collision_sum"] < 0.001 and obs["moving_jaw_collision_sum"] < 0.001 else None
        gstate = tuple(p.getJointState(self.actual.robot, self.actual.gripper_joint,
                                       physicsClientId=self.actual.client))

        obs["raw_contact_points"] = [list(cp) for cp in raw_cps]
        self._record(obs, self.actual.steps, phase, mode, q, g, decision,
                     fresh_gaps, boardforce, gstate)

        if mode == "hold":
            inside_ok_phases = phase in ("lower", "support", "open", "settle_board")
            if self.max_rise > 0.003 and not inside_ok_phases and boardforce > 0.005:
                self.drop = True
                raise TaskStop("drop:airborne_boardforce", "physical_task_failed")

        if not self.released and self.max_rise > 0.003:
            post_fixed = float(obs["fixed_jaw_collision_sum"])
            post_moving = float(obs["moving_jaw_collision_sum"])
            post_gaps = fresh_gaps
            if post_gaps is not None and len(post_gaps) >= 2:
                native_open = post_fixed < 0.001 and post_moving < 0.001 and \
                    post_gaps[0] > 0.0002 and post_gaps[1] > 0.0002
                if native_open:
                    rel_z = float(obs["target_pos"][2]) - float(obs["tcp_pos"][2])
                    rel_z_prev = float(before["target_pos"][2]) - float(before["tcp_pos"][2])
                    dz_dt = (rel_z - rel_z_prev) / self.actual.dt
                    if dz_dt < -0.02:
                        self.drop = True
                        raise TaskStop("drop:pre_release", "physical_task_failed")

        self._guard(obs)
        return obs

    def hold(self, seconds, phase, mode):
        n = max(1, int(math.ceil(seconds / self.actual.dt)))
        for _ in range(n):
            self.tick(phase, list(self.qcmd), float(self.gcmd), mode)

    def move_nodes(self, nodes, mode):
        for node in nodes:
            q0 = list(self.qcmd)
            q1 = list(node["q"])
            g_target = float(node["g"])
            g0 = float(self.gcmd)
            pos0 = pose_fk(self.kin, q0)[:3, 3]
            pos1 = pose_fk(self.kin, q1)[:3, 3]
            dpos = float(np.linalg.norm(np.array(pos1) - np.array(pos0)))
            dq = float(np.max(np.abs(np.array(q1) - np.array(q0)))) if len(q0) else 0.0
            dur = 1.5 * max(dpos / 0.015, dq / 0.3, self.actual.dt)
            n = max(1, int(math.ceil(dur / self.actual.dt)))
            for i in range(1, n + 1):
                alpha = i / n
                s = alpha * alpha * (3.0 - 2.0 * alpha)
                qi = [q0[k] + s * (q1[k] - q0[k]) for k in range(len(q0))]
                gi = g0 + s * (g_target - g0)
                self.tick(node["phase"], qi, gi, mode)
        last_phase = nodes[-1]["phase"] if nodes else "move"
        self.hold(0.2, last_phase, mode)

    def run(self):
        result = {
            "ok": False,
            "completed": False,
            "classification": None,
            "reason": None,
            "physics_steps": 0,
            "time": 0.0,
            "phase_counts": {},
            "release_started": False,
            "support_confirmed": False,
            "dropflag": False,
            "guard_calls": [],
            "initial_obs": None,
            "final_obs": None,
            "tracepath": str(self.trace_path),
        }
        try:
            if self.initial_bridge:
                self.move_nodes(self.initial_bridge, "position")
            self.hold(0.4, "home", "position")
            self.move_nodes(self.prefix, "position")
            self.hold(0.2, "source", "position")
            contact_command_q = list(self.qcmd)
            contact_command_g = float(self.gcmd)

            self.press_ctrl = PressureGrip(self.g_open, self.g_limit, self.target_force_N)
            t_close = 0.0
            ready = False
            while t_close < 8.5:
                self.tick("close", contact_command_q, contact_command_g, "close")
                t_close += self.actual.dt
                f_fixed = float(self.last["fixed_jaw_collision_sum"])
                f_moving = float(self.last["moving_jaw_collision_sum"])
                if f_fixed > 0.01 and f_moving > 0.01 and self.last_decision.get("ready") is True:
                    ready = True
                    break
            if not ready:
                raise TaskStop("close_no_contact", "physical_task_failed")
            self.hold(0.3, "hold", "hold")

            snap = self.actual.snapshot()
            checked = rebuild(self.wq, self.geo, self.kin, snap,
                              list(self.last["actual_q5"]), float(self.last["actual_g"]),
                              self.home_q, self.g_open, self.hover_m, True)
            self.hover_m = checked["choice"]["hover_m"]
            self.wq.cfg["placement_yaw_deg"] = checked["choice"]["yaw_deg"]
            lift_nodes = [n for n in checked["suffix"] if n["phase"] == "lift"]
            if not lift_nodes:
                raise TaskStop("no_lift", "physical_task_failed")
            self.move_nodes(lift_nodes, "hold")
            rise = float(self.last["target_pos"][2]) - self.initial_target_z
            if rise <= 0.003:
                raise TaskStop("lift_failed rise=%.5f" % rise, "physical_task_failed")

            snap = self.actual.snapshot()
            checked = rebuild(self.wq, self.geo, self.kin, snap,
                              list(self.last["actual_q5"]), float(self.last["actual_g"]),
                              self.home_q, self.g_open, self.hover_m, False)
            trans_nodes = [n for n in checked["suffix"] if n["phase"] in ("transfer", "lower")]
            if not trans_nodes:
                raise TaskStop("no_transfer", "physical_task_failed")
            self.move_nodes(trans_nodes, "hold")

            self.support_feedback = confirm_support(self, route, pose_fk)
            result["support_feedback"] = self.support_feedback
            if self.support_feedback["ok"] is not True:
                raise TaskStop(self.support_feedback["reason"], "physical_task_failed")
            self.supported = True
            result["preopen_exit_plan"] = plan_exit(self.wq, self.geo, self.kin, self.actual.snapshot(), list(self.last["actual_q5"]), float(self.last["actual_g"]), self.g_open, self.query_cfg["safe_home_q"], route)

            self.released = True
            result["release_started"] = True
            g_target_open = float(self.g_open)
            g_start = float(self.gcmd)
            sweep = max(1, int(math.ceil(abs(g_target_open - g_start) / (0.05 * self.actual.dt))))
            for i in range(1, sweep + 1):
                alpha = i / sweep
                g_i = g_start + alpha * (g_target_open - g_start)
                self.tick("open", list(self.qcmd), g_i, "position")
            self.hold(0.6, "settle_board", "position")

            dest_xy = self.dest
            if dest_xy is not None:
                square_xy = self.actual.square_xyz(dest_xy) if hasattr(self.actual, "square_xyz") else None
                if square_xy is not None:
                    aim = np.array(square_xy)
                else:
                    aim = np.array([float(dest_xy[0]), float(dest_xy[1]), self.last["target_pos"][2]])
            else:
                aim = np.array(self.last["target_pos"])
            tcp_xy = np.array(self.last["target_pos"])
            xy_err = float(np.linalg.norm(tcp_xy[:2] - np.array(aim)[:2]))
            R_tcp = _quat_to_rot(self.last["target_quat"])
            tilt = math.acos(max(-1.0, min(1.0, R_tcp.as_matrix()[2, 2])))
            boardforce = 0.0
            for c in (self.last.get("board_contacts") or []):
                if isinstance(c, dict):
                    boardforce += float(c.get("normalForce", 0.0))
            if xy_err > 0.003 or tilt > math.radians(10.0) or boardforce <= 0.005:
                raise TaskStop("final_check xy=%.4f tilt=%.4f bf=%.4f" % (xy_err, tilt, boardforce),
                               "physical_task_failed")

            snap = self.actual.snapshot()
            qact = list(self.last["actual_q5"])
            exit_plan = plan_exit(self.wq, self.geo, self.kin, snap, qact,
                                  float(self.last["actual_g"]), self.g_open,
                                  self.query_cfg["safe_home_q"], route)
            result["actual_exit_plan"] = exit_plan
            self.move_nodes(exit_plan["exit"], "position")
            self.move_nodes(exit_plan["home"], "position")
            self.hold(0.2, "ready", "position")
            home_err = float(np.degrees(np.max(np.abs(np.asarray(self.last["actual_q5"])-self.query_cfg["safe_home_q"]))))
            actual_snap = self.actual.snapshot()
            home_gap = target_clearance(self.wq, self.last["actual_q5"], self.last["actual_g"], actual_snap[self.target_key])
            result["checked_home"] = {"joint_error_deg":home_err, "target_gap_m":home_gap}
            if home_err > 2 or home_gap < .0005:
                raise TaskStop("home_arrival_or_clearance", "physical_task_failed")

            final_support = False
            for c in (self.last.get("board_contacts") or []):
                if isinstance(c, dict) and float(c.get("normalForce", 0.0)) > 0.005:
                    final_support = True
                    break
            if final_support and self.supported and not self.drop and not self.guard_calls and float(np.linalg.norm(np.asarray(self.last["target_pos"])[:2] - np.asarray(aim)[:2])) <= 0.003 and float(Rotation.from_quat(self.last["target_quat"]).as_matrix()[2, 2]) >= math.cos(math.radians(10)):
                result["completed"] = True
                result["ok"] = True
                result["classification"] = "physical_task_success"
                result["reason"] = "completed" if final_support else "completed_no_final_support"
            result["support_confirmed"] = self.supported and final_support
        except StagePlanRejected as exc:
            result["classification"] = "post_grasp_planning_block"
            result["reason"] = str(exc)
        except TaskStop as ts:
            result["classification"] = ts.classification
            result["reason"] = ts.reason
        except Exception as e:
            result["classification"] = "program_error"
            result["reason"] = "%s: %s" % (type(e).__name__, str(e))
            result["traceback"] = traceback.format_exc()
        finally:
            try:
                self.trace_fh.close()
            except Exception:
                pass
            try:
                self.restore()
            except Exception:
                pass
            try:
                if self.shared_actual is None:
                    p.disconnect(self.actual.client)
            except Exception:
                pass
            try:
                p.disconnect(self.wq.client)
            except Exception:
                pass
            result["physics_steps"] = int(self.actual.steps - self.start_steps)
            result["world_steps_end"] = int(self.actual.steps)
            result["world_steps_start"] = self.start_steps
            result["wall_s"] = float(time.time() - self.t0)
            result["time"] = float(self.actual.time - self.start_time)
            result["stage_plan_audits"] = self.wq.plan_audits
            result["phase_counts"] = dict(self.phase_counts)
            result["release_started"] = bool(self.released)
            result["dropflag"] = bool(self.drop)
            result["guard_calls"] = list(self.guard_calls)
            result["initial_obs"] = self.initial_obs
            result["final_obs"] = self.last
            try:
                with open(self.outdir / "result.json", "w") as f:
                    json.dump(result, f, default=str, indent=2)
            except Exception:
                pass
        return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    trace = outdir / "trace.jsonl.gz"
    result_path = outdir / "result.json"
    if trace.exists() or result_path.exists():
        print("REFUSE: existing result/trace in %s" % outdir, file=sys.stderr)
        return 2

    with open(args.config, "r") as f:
        cfg = json.load(f)
    with open(cfg["plan"], "r") as f:
        plan = json.load(f)
    if plan.get("ok") is not True:
        print("REFUSE: plan.ok is not true", file=sys.stderr)
        return 2

    runner = Runner(cfg, plan, outdir)
    result = runner.run()
    print(json.dumps({k: result.get(k) for k in
                      ("ok", "completed", "classification", "reason",
                       "physics_steps", "time", "phase_counts")}, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
