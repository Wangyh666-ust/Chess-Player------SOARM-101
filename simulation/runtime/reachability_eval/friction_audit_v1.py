import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pybullet as p

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dynamics_cli import _load_source
from physics_world import PhysicsWorld
from dynamics_drive import run
from dynamics_evaluate import evaluate

SOURCE_LINK = (
    "https://raw.githubusercontent.com/bulletphysics/bullet3/master/"
    "src/BulletCollision/CollisionDispatch/btManifoldResult.cpp"
)

DEFAULT_DT = 1.0 / 240.0


def _is_finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


class AuditedWorld(PhysicsWorld):
    def __init__(self, case, q, g, *, dt=DEFAULT_DT, grip_force=0.2,
                 board_mu=0.3, jaw_mu=0.3, piece_friction=0.6,
                 rolling=0.001, spinning=0.001, legacy=False):
        if not _is_finite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive number")
        if not _is_finite(grip_force) or grip_force <= 0:
            raise ValueError("grip_force must be a finite positive number")
        for name, val in (("board_mu", board_mu), ("jaw_mu", jaw_mu),
                          ("piece_friction", piece_friction),
                          ("rolling", rolling), ("spinning", spinning)):
            if not _is_finite(val):
                raise ValueError("%s must be finite" % name)
            if val < 0:
                raise ValueError("%s must be non-negative" % name)
        if piece_friction <= 0:
            raise ValueError("piece_friction must be > 0")
        if board_mu > 10 or jaw_mu > 10:
            raise ValueError("board_mu/jaw_mu must be <= 10")

        super().__init__(case, q, g, friction=piece_friction, dt=dt,
                         grip_force=grip_force)

        self._dt = dt
        self._legacy = legacy
        self._board_mu = board_mu
        self._jaw_mu = jaw_mu
        self._piece_friction = piece_friction
        self._rolling = rolling
        self._spinning = spinning
        self._client = self.client

        self._requested = {
            "board_mu": board_mu,
            "jaw_mu": jaw_mu,
            "piece_friction": piece_friction,
            "rolling": rolling,
            "spinning": spinning,
            "legacy": bool(legacy),
            "dt": dt,
            "grip_force": grip_force,
        }

        if not legacy:
            try:
                self._apply_materials()
                self._validate_readback()
            except Exception:
                self.close()
                raise

    # ------------------------------------------------------------------
    def _apply_materials(self):
        cid = self._client
        pf = self._piece_friction
        board_lat = self._board_mu / pf
        jaw_lat = self._jaw_mu / pf

        def set_one(body_id, link, lateral=None, rolling=None, spinning=None):
            if lateral is not None:
                p.changeDynamics(body_id, link, lateralFriction=float(lateral),
                                 physicsClientId=cid)
            if rolling is not None:
                p.changeDynamics(body_id, link, rollingFriction=float(rolling),
                                 physicsClientId=cid)
            if spinning is not None:
                p.changeDynamics(body_id, link, spinningFriction=float(spinning),
                                 physicsClientId=cid)

        if self.board is not None:
            set_one(self.board, -1, lateral=board_lat)

        link_index = getattr(self, "link_index", {}) or {}
        robot = getattr(self, "robot", None)
        if robot is not None:
            num_joints = p.getNumJoints(robot, physicsClientId=cid)
            for link in range(-1, num_joints):
                set_one(robot, link, lateral=jaw_lat)

        for key, body_id in (self.bodies or {}).items():
            set_one(body_id, -1, lateral=pf, rolling=self._rolling,
                    spinning=self._spinning)

    # ------------------------------------------------------------------
    def _read_material(self, body_id, link):
        cid = self._client
        info = p.getDynamicsInfo(body_id, link, physicsClientId=cid)
        return {
            "mass": info[0],
            "lateral": info[1],
            "rolling": info[6],
            "spinning": info[7],
        }

    def runtime_materials(self):
        cid = self._client
        result = {
            "requested": dict(self._requested),
            "actual": {"board": None, "robot": {}, "pieces": {}},
            "effective": {},
            "notes": [
                "default Bullet material combiner uses multiplication",
                "source link: " + SOURCE_LINK,
            ],
        }

        if self.board is not None:
            result["actual"]["board"] = self._read_material(self.board, -1)

        robot = getattr(self, "robot", None)
        if robot is not None:
            num_joints = p.getNumJoints(robot, physicsClientId=cid)
            for link in range(-1, num_joints):
                result["actual"]["robot"][str(link)] = self._read_material(robot, link)

        for key, body_id in (self.bodies or {}).items():
            result["actual"]["pieces"][key] = self._read_material(body_id, -1)

        def clamp(x):
            return max(-10.0, min(10.0, float(x)))

        board_lat = None
        if result["actual"]["board"] is not None:
            board_lat = result["actual"]["board"]["lateral"]
        for key, mat in result["actual"]["pieces"].items():
            if board_lat is not None:
                result["effective"]["board_piece_%s" % key] = clamp(
                    board_lat * mat["lateral"])
        fix_link = (self.link_index or {}).get("fixed_jaw_collision")
        mov_link = (self.link_index or {}).get("moving_jaw_so101_v1_link")
        if robot is not None:
            fix_mat = result["actual"]["robot"].get(str(fix_link))
            mov_mat = result["actual"]["robot"].get(str(mov_link))
            for key, mat in result["actual"]["pieces"].items():
                if fix_mat is not None:
                    result["effective"]["fixed_jaw_piece_%s" % key] = clamp(
                        fix_mat["lateral"] * mat["lateral"])
                if mov_mat is not None:
                    result["effective"]["moving_jaw_piece_%s" % key] = clamp(
                        mov_mat["lateral"] * mat["lateral"])

        return result

    # ------------------------------------------------------------------
    def _validate_readback(self):
        cid = self._client
        pf = self._piece_friction
        board_lat = self._board_mu / pf
        jaw_lat = self._jaw_mu / pf

        def chk(body_id, link, expected_lat, expected_roll, expected_spin):
            info = p.getDynamicsInfo(body_id, link, physicsClientId=cid)
            got = {"lateral": info[1], "rolling": info[6], "spinning": info[7]}
            exp = {"lateral": expected_lat, "rolling": expected_roll,
                   "spinning": expected_spin}
            for k in exp:
                if exp[k] is None:
                    continue
                if not math.isclose(got[k], exp[k], abs_tol=1e-10):
                    raise RuntimeError(
                        "material readback mismatch %s link=%s field=%s got=%r exp=%r"
                        % (body_id, link, k, got[k], exp[k]))

        if self.board is not None:
            chk(self.board, -1, board_lat, None, None)

        robot = getattr(self, "robot", None)
        if robot is not None:
            num_joints = p.getNumJoints(robot, physicsClientId=cid)
            for link in range(-1, num_joints):
                chk(robot, link, jaw_lat, None, None)

        for key, body_id in (self.bodies or {}).items():
            chk(body_id, -1, pf, self._rolling, self._spinning)

    # ------------------------------------------------------------------
    @staticmethod
    def _decode_contact(c):
        return {
            "body_a": c[1], "body_b": c[2],
            "link_a": c[3], "link_b": c[4],
            "position_on_a": list(c[5]), "position_on_b": list(c[6]),
            "contact_normal_on_b": list(c[7]),
            "distance": c[8], "normal_force": c[9],
            "lateral_friction_1": c[10], "lateral_friction_dir_1": list(c[11]),
            "lateral_friction_2": c[12], "lateral_friction_dir_2": list(c[13]),
        }

    def _classify(self, c):
        body_a = c[1]
        body_b = c[2]
        board_id = self.board
        robot_id = getattr(self, "robot", None)
        piece_ids = {v: k for k, v in (self.bodies or {}).items()}

        def kind(bid):
            if bid == board_id:
                return "board"
            if bid == robot_id:
                return "robot"
            if bid in piece_ids:
                return "piece"
            return "other"

        ka, kb = kind(body_a), kind(body_b)
        key_a = piece_ids.get(body_a)
        key_b = piece_ids.get(body_b)
        return ka, kb, (key_a, key_b)

    def _extract_contacts(self, contacts):
        all_pb = []
        piece_pairs = []
        robot_piece = []
        for c in contacts:
            ka, kb, keys = self._classify(c)
            decoded = self._decode_contact(c)
            decoded["kind_a"] = ka
            decoded["kind_b"] = kb
            decoded["piece_a"], decoded["piece_b"] = keys
            pair = {ka, kb}
            if pair == {"piece", "board"}:
                all_pb.append(decoded)
            elif ka == "piece" and kb == "piece":
                key_a, key_b = keys
                decoded["pieces"] = tuple(sorted([str(key_a), str(key_b)]))
                piece_pairs.append(decoded)
            elif pair == {"robot", "piece"}:
                robot_piece.append(decoded)
        return all_pb, piece_pairs, robot_piece

    def observe(self, phase, q, g):
        obs = super().observe(phase, q, g)
        raw = p.getContactPoints(physicsClientId=self._client)
        all_pb, piece_pairs, robot_piece = self._extract_contacts(raw)
        obs["all_piece_board_contacts"] = all_pb
        obs["piece_pair_contacts"] = piece_pairs
        obs["robot_piece_contact_vectors"] = robot_piece
        return obs


# ----------------------------------------------------------------------
def _sha256(path):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_hashes(original_path=None):
    files = [
        os.path.join(HERE, "friction_audit_v1.py"),
        os.path.join(HERE, "physics_world.py"),
        os.path.join(HERE, "full_piece_v2.py"),
        os.path.join(HERE, "dynamics_drive.py"),
        os.path.join(HERE, "dynamics_evaluate.py"),
        os.path.join(HERE, "dynamics_cli.py"),
    ]
    out = {}
    for f in files:
        out[os.path.basename(f)] = _sha256(f)
    if original_path:
        out["original_input"] = _sha256(original_path)
    return out


def _env_info():
    return {
        "python": sys.version,
        "pybullet": importlib.metadata.version("pybullet"),
        "numpy": np.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _label_ok(label):
    return bool(re.fullmatch(r"[A-Za-z0-9_\-]+", label or ""))


def _out_dir():
    return os.path.join(HERE, "results", "friction_audit_v1")


def _save(label, payload, summary_obj):
    outdir = _out_dir()
    os.makedirs(outdir, exist_ok=True)
    gz_path = os.path.join(outdir, label + ".json.gz")
    sm_path = os.path.join(outdir, label + "_summary.json")
    if os.path.exists(gz_path):
        raise SystemExit("refusing to overwrite existing %s" % gz_path)
    if os.path.exists(sm_path):
        raise SystemExit("refusing to overwrite existing %s" % sm_path)
    with gzip.open(gz_path, "wt") as fh:
        json.dump(payload, fh, indent=2, default=str)
    with open(sm_path, "w") as fh:
        json.dump(summary_obj, fh, indent=2, default=str)
    return gz_path, sm_path


# ----------------------------------------------------------------------
def _joint_limits(world):
    cid = world.client
    out = {}
    try:
        gj = getattr(world, "gripper_joint", None)
        if gj is not None:
            info = p.getJointInfo(world.robot, gj, physicsClientId=cid)
            out["gripper_joint_index"] = gj
            out["lower_deg"] = math.degrees(info[8])
            out["upper_deg"] = math.degrees(info[9])
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _quat_tilt_deg(quat):
    up_z = p.getMatrixFromQuaternion(quat)[8]
    return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))


def _kinetic_energy(world, cid):
    total = 0.0
    for key, body_id in (world.bodies or {}).items():
        pos, orn = world.bottom_pose(key)
        lin, ang = p.getBaseVelocity(body_id, physicsClientId=cid)
        info = p.getDynamicsInfo(body_id, -1, physicsClientId=cid)
        mass = info[0]
        inertia = info[2]
        r = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        omega_world = np.array(ang)
        omega_local = r.T.dot(omega_world)
        v = np.array(lin)
        ke = 0.5 * mass * float(v.dot(v))
        ke += 0.5 * float(np.dot(omega_local, np.array(inertia) * omega_local))
        total += ke
    return total


def _snapshot_poses(world, cid):
    out = {}
    for key, body_id in (world.bodies or {}).items():
        pos, orn = world.bottom_pose(key)
        vel, ang = p.getBaseVelocity(body_id, physicsClientId=cid)
        out[key] = {
            "pos": [float(x) for x in pos],
            "quat": [float(x) for x in orn],
            "lin_vel": [float(x) for x in vel],
            "ang_vel": [float(x) for x in ang],
        }
    return out


def _contact_detail(world, cid):
    raw = p.getContactPoints(physicsClientId=cid)
    all_pb, piece_pairs, robot_piece = world._extract_contacts(raw)
    return all_pb, piece_pairs, robot_piece


# ----------------------------------------------------------------------
def mode_probe(args):
    case_id = args.case
    _, case, q5, initial_g, frames = _load_source(case_id, "paths")
    world = AuditedWorld(case, q5, initial_g, dt=args.dt,
                         grip_force=args.grip_force, board_mu=args.board_mu,
                         jaw_mu=args.jaw_mu, piece_friction=args.piece_friction,
                         rolling=args.rolling, spinning=args.spinning,
                         legacy=args.legacy)
    try:
        if not args.legacy:
            world._validate_readback()
        materials = world.runtime_materials()
        limits = _joint_limits(world)
        summary = {
            "mode": "probe",
            "case": case_id,
            "legacy": bool(args.legacy),
            "gripper_limits": limits,
            "material_requested": materials["requested"],
            "robot_removed": False,
            "steps": 0,
        }
        payload = {
            "configuration": _config_from_args(args),
            "runtime_materials": materials,
            "case": case,
            "summary": summary,
            "frames": [],
        }
    finally:
        world.close()
    return payload, summary


def _config_from_args(args):
    return {
        "case": args.case,
        "label": args.label,
        "mode": args.mode,
        "legacy": bool(args.legacy),
        "board_mu": args.board_mu,
        "jaw_mu": args.jaw_mu,
        "piece_friction": args.piece_friction,
        "rolling": args.rolling,
        "spinning": args.spinning,
        "dt": args.dt,
        "seconds": args.seconds,
        "preload_deg": args.preload_deg,
        "grip_force": args.grip_force,
    }


# ----------------------------------------------------------------------
def mode_static(args):
    case_id = args.case
    _, case, q5, initial_g, frames = _load_source(case_id, "paths")
    world = AuditedWorld(case, q5, initial_g, dt=args.dt,
                         grip_force=args.grip_force, board_mu=args.board_mu,
                         jaw_mu=args.jaw_mu, piece_friction=args.piece_friction,
                         rolling=args.rolling, spinning=args.spinning,
                         legacy=args.legacy)
    try:
        if not args.legacy:
            world._validate_readback()
        materials_before = world.runtime_materials()
        cid = world.client

        p.removeBody(world.robot, physicsClientId=cid)
        world.robot = None

        piece_keys = list((world.bodies or {}).keys())
        initial_positions = {}
        for key, body_id in (world.bodies or {}).items():
            pos, orn = world.bottom_pose(key)
            initial_positions[key] = {
                "pos": [float(x) for x in pos],
                "quat": [float(x) for x in orn],
            }

        start_time = 0.0
        initial_xy = {k: np.array(v["pos"][:2]) for k, v in initial_positions.items()}

        reference_xy = None
        reference_time = None
        first_exceed = None
        per_piece_max = {k: 0.0 for k in piece_keys}
        per_piece_max_from_ref = {k: 0.0 for k in piece_keys}
        per_piece_tilt = {k: 0.0 for k in piece_keys}
        overall_max = 0.0
        overall_max_from_ref = 0.0
        initialization_max = 0.0
        peak_ke = 0.0
        samples = []
        samples.append({"step": 0, "time": 0.0,
                        "poses": _snapshot_poses(world, cid),
                        "piece_board_contacts": [], "piece_pair_contacts": [],
                        "robot_piece_contacts": [], "kinetic_energy": 0.0})

        total_steps = int(math.ceil(max(0.0, args.seconds) / args.dt))
        sample_stride = 24
        last_time = 0.0

        for step_i in range(total_steps):
            world.step()
            now = world.time

            for key, body_id in (world.bodies or {}).items():
                pos, orn = world.bottom_pose(key)
                dxy = float(np.linalg.norm(np.array(pos[:2]) - initial_xy[key]))
                if dxy > per_piece_max[key]:
                    per_piece_max[key] = dxy
                if dxy > overall_max:
                    overall_max = dxy
                if now <= .5 + 1e-10:
                    initialization_max = max(initialization_max, dxy)

                if reference_xy is not None:
                    dxy_ref = float(np.linalg.norm(
                        np.array(pos[:2]) - reference_xy[key]))
                    if dxy_ref > per_piece_max_from_ref[key]:
                        per_piece_max_from_ref[key] = dxy_ref
                    if dxy_ref > overall_max_from_ref:
                        overall_max_from_ref = dxy_ref
                    if (first_exceed is None and overall_max_from_ref > 0.002):
                        first_exceed = {
                            "time": float(now - reference_time),
                            "abs_time": float(now),
                        }

                tilt = _quat_tilt_deg(orn)
                if tilt > per_piece_tilt[key]:
                    per_piece_tilt[key] = tilt

            if (reference_xy is None) and (now - start_time) >= 0.5:
                reference_xy = {
                    k: np.array(world.bottom_pose(k)[0][:2])
                    for k, bid in (world.bodies or {}).items()
                }
                reference_time = now

            if step_i % sample_stride == 0 or step_i == total_steps - 1:
                ke = _kinetic_energy(world, cid)
                peak_ke = max(peak_ke, ke)
                poses = _snapshot_poses(world, cid)
                apb, ppair, rpc = _contact_detail(world, cid)
                samples.append({
                    "step": step_i + 1,
                    "time": float(now),
                    "poses": poses,
                    "piece_board_contacts": apb,
                    "piece_pair_contacts": ppair,
                    "robot_piece_contacts": rpc,
                    "kinetic_energy": ke,
                })
            last_time = now

        summary = {
            "mode": "static",
            "case": case_id,
            "legacy": bool(args.legacy),
            "robot_removed": True,
            "seconds": float(args.seconds),
            "total_steps": total_steps,
            "dt": args.dt,
            "start_time": float(start_time),
            "end_time": float(last_time),
            "num_pieces": len(piece_keys),
            "initial_positions": initial_positions,
            "overall_max_xy_disp": float(overall_max),
            "initialization_max_xy_disp": float(initialization_max),
            "overall_max_xy_disp_from_reference": float(overall_max_from_ref),
            "reference_time": float(reference_time) if reference_time else None,
            "first_exceed_2mm_after_reference": first_exceed,
            "per_piece_max_xy_disp": {k: float(v) for k, v in per_piece_max.items()},
            "per_piece_max_xy_disp_from_reference": {
                k: float(v) for k, v in per_piece_max_from_ref.items()},
            "per_piece_max_tilt_deg": {k: float(v) for k, v in per_piece_tilt.items()},
            "sampled_peak_kinetic_energy": float(peak_ke),
            "sampling": {
                "stride_steps": sample_stride,
                "state": "sampled",
                "note": "only recorded samples are complete; no continuous certificate",
            },
            "disclaimer": "No causal conclusion is inferred by this module.",
        }

        payload = {
            "configuration": _config_from_args(args),
            "runtime_materials": materials_before,
            "case": case,
            "summary": summary,
            "frames": [],
            "samples": samples,
        }
    finally:
        world.close()
    return payload, summary


# ----------------------------------------------------------------------
def _phase_diag(observations, world):
    phases = ["close_hold", "lift", "lift_hold", "transfer"]
    out = {}
    limits = _joint_limits(world)
    for phase in phases:
        obs_list = [o for o in observations if o.get("phase") == phase]
        if not obs_list:
            continue
        both_ok = 0
        fix_forces = []
        mov_forces = []
        errs = []
        acts = []
        cmds = []
        for o in obs_list:
            fix_sum = float(o['fixed_jaw_collision_sum'])
            mov_sum = float(o['moving_jaw_collision_sum'])
            if fix_sum > 1e-4 and mov_sum > 1e-4:
                both_ok += 1
            fix_forces.append(fix_sum)
            mov_forces.append(mov_sum)

            cmds.append(math.degrees(float(o['commanded_g'])))
            acts.append(math.degrees(float(o['actual_g'])))

        if len(acts) == len(cmds) and acts:
            for a, c in zip(acts, cmds):
                errs.append(abs(a - c))

        out[phase] = {
            "samples": len(obs_list),
            "two_jaw_contact_fraction": both_ok / float(len(obs_list)),
            "mean_fixed_normal_force": float(np.mean(fix_forces)) if fix_forces else None,
            "peak_fixed_normal_force": float(np.max(fix_forces)) if fix_forces else None,
            "mean_moving_normal_force": float(np.mean(mov_forces)) if mov_forces else None,
            "peak_moving_normal_force": float(np.max(mov_forces)) if mov_forces else None,
            "max_abs_grip_error_deg": float(np.max(errs)) if errs else None,
            "actual_g_range_deg": ([float(np.min(acts)), float(np.max(acts))]
                                   if acts else None),
            "commanded_g_range_deg": ([float(np.min(cmds)), float(np.max(cmds))]
                                      if cmds else None),
            "gripper_limits": limits,
            "note": "force cap is not asserted to equal actual contact force",
        }
    out["gripper_limits"] = limits
    out["note"] = "force cap is not asserted to equal actual contact force"
    return out


def mode_motion(args):
    case_id = args.case
    _, case, q5, initial_g, frames = _load_source(case_id, "paths")
    world = AuditedWorld(case, q5, initial_g, dt=args.dt,
                         grip_force=args.grip_force, board_mu=args.board_mu,
                         jaw_mu=args.jaw_mu, piece_friction=args.piece_friction,
                         rolling=args.rolling, spinning=args.spinning,
                         legacy=args.legacy)
    try:
        if not args.legacy:
            world._validate_readback()
        materials = world.runtime_materials()
        observations = run(world, frames, preload_deg=args.preload_deg,
                           speed=0.025, no_grip=False)
        summary_eval = evaluate(case, observations, world)
        phase_diag = _phase_diag(observations, world)

        summary = {
            "mode": "motion",
            "case": case_id,
            "legacy": bool(args.legacy),
            "evaluate": summary_eval,
            "phase_diagnostics": phase_diag,
            "gripper_limits": _joint_limits(world),
            "robot_removed": False,
            "note": "force cap is not asserted to equal actual contact force",
        }

        payload = {
            "configuration": _config_from_args(args),
            "runtime_materials": materials,
            "case": case,
            "summary": summary,
            "frames": observations,
        }
    finally:
        world.close()
    return payload, summary


# ----------------------------------------------------------------------
def _parse_args(argv):
    parser = argparse.ArgumentParser(prog="friction_audit_v1")
    parser.add_argument("mode", choices=["probe", "static", "motion"])
    parser.add_argument("--case", default="opening_e2e3")
    parser.add_argument("--label", required=True)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--board-mu", type=float, default=0.3)
    parser.add_argument("--jaw-mu", type=float, default=0.3)
    parser.add_argument("--piece-friction", type=float, default=0.6)
    parser.add_argument("--rolling", type=float, default=0.001)
    parser.add_argument("--spinning", type=float, default=0.001)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--preload-deg", type=float, default=2.0)
    parser.add_argument("--grip-force", type=float, default=0.2)

    args = parser.parse_args(argv)

    if not _label_ok(args.label):
        parser.error("label must match [A-Za-z0-9_-]+")

    for name in ("board_mu", "jaw_mu", "piece_friction", "rolling",
                 "spinning", "grip_force"):
        v = getattr(args, name)
        if not _is_finite(v):
            parser.error("%s must be finite" % name)
        if v < 0:
            parser.error("%s must be non-negative" % name)
    if args.piece_friction <= 0:
        parser.error("piece_friction must be > 0")
    if args.board_mu > 10 or args.jaw_mu > 10:
        parser.error("board_mu/jaw_mu must be <= 10")
    if not _is_finite(args.dt) or args.dt <= 0:
        parser.error("dt must be positive")
    if not _is_finite(args.seconds) or args.seconds <= 0:
        parser.error("seconds must be > 0")
    if not _is_finite(args.grip_force) or args.grip_force <= 0:
        parser.error("grip_force must be > 0")
    if not _is_finite(args.preload_deg):
        parser.error("preload_deg must be finite")

    return args


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    for ext in ['.json.gz', '_summary.json']:
        if os.path.exists(os.path.join(_out_dir(), args.label + ext)):
            raise SystemExit('refusing to overwrite label ' + args.label)
    env = _env_info()

    if args.mode == "probe":
        payload, summary = mode_probe(args)
    elif args.mode == "static":
        payload, summary = mode_static(args)
    else:
        payload, summary = mode_motion(args)

    payload["env"] = env
    input_path, _, _, _, _ = _load_source(args.case, 'paths')
    payload["source_hashes"] = _source_hashes(input_path)

    summary_obj = {
        "summary": summary,
        "configuration": payload.get("configuration"),
        "runtime_materials": payload.get("runtime_materials"),
        "env": env,
        "source_hashes": payload["source_hashes"],
    }

    gz_path, sm_path = _save(args.label, payload, summary_obj)

    compact = {
        "mode": args.mode,
        "label": args.label,
        "case": args.case,
        "legacy": bool(args.legacy),
        "saved_gz": os.path.basename(gz_path),
        "saved_summary": os.path.basename(sm_path),
        "summary": {
            k: v for k, v in summary.items()
            if k not in ("initial_positions",)
        },
    }
    print(json.dumps(compact, default=str))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
