from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from core import Robot
from tilt_ik_v1 import solve_pose, pose_fk
from query_world_reuse_v2 import QueryWorld
from pilot_geometry_v2 import Geometry

try:
    import pybullet as p
except Exception:
    p = None

WORKSPACE = Path(__file__).resolve().parents[4]
WORKDIR = WORKSPACE / "reachability_eval" / "calibration" / "manipulation_20261109_v1" / "work"

MAX_TRANSLATION_STEP_M = 0.001
MAX_ORIENTATION_STEP_DEG = 1.0
MAX_JOINT_JUMP_DEG = 2.0
MAX_EDGE_JOINT_STEP_DEG = 0.25
CLOSE_ENVELOPE_STEP_DEG = 0.5
TABLE_DISTANCE_M = 0.00005
TABLE_PENETRATION_M = -0.00005
DEST_CLEARANCE_M = 0.00003


def _as_pos_list(x):
    return [float(x[0]), float(x[1]), float(x[2])]


def _as_quat_xyzw(x):
    return [float(x[0]), float(x[1]), float(x[2]), float(x[3])]


def _mat_to_pose_dict(T):
    T = np.asarray(T, dtype=float)
    pos = _as_pos_list(T[:3, 3])
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    return {"pos": pos, "quat": _as_quat_xyzw(quat)}


def _pose_dict_to_mat(d):
    if not isinstance(d, dict):
        raise TypeError("target pose must be dict with pos/quat")
    if "pos" not in d or "quat" not in d:
        raise ValueError("target pose dict requires pos and quat")
    pos = np.asarray(d["pos"], dtype=float)
    quat = np.asarray(d["quat"], dtype=float)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat).as_matrix()
    T[:3, 3] = pos
    return T


def _pose_dict(T):
    return _mat_to_pose_dict(T)


def _interp_T(Ta, Tb, n_steps):
    Ta = np.asarray(Ta, dtype=float)
    Tb = np.asarray(Tb, dtype=float)
    pa = Ta[:3, 3]
    pb = Tb[:3, 3]
    dist = float(np.linalg.norm(pb - pa))
    trans_steps = max(1, int(math.ceil(dist / MAX_TRANSLATION_STEP_M)))
    Ra = Rotation.from_matrix(Ta[:3, :3])
    Rb = Rotation.from_matrix(Tb[:3, :3])
    rel = (Ra.inv() * Rb).magnitude()
    rel_deg = float(math.degrees(rel))
    rot_steps = max(1, int(math.ceil(rel_deg / MAX_ORIENTATION_STEP_DEG)))
    n = max(n_steps, trans_steps, rot_steps)
    ts = np.linspace(0.0, 1.0, n + 1)
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([Ra, Rb]))
    out = []
    for t in ts:
        T = np.eye(4)
        T[:3, :3] = slerp(float(t)).as_matrix()
        T[:3, 3] = (1.0 - t) * pa + t * pb
        out.append(T)
    return out


def _joint_max_jump_deg(qa, qb):
    qa = np.asarray(qa, dtype=float)
    qb = np.asarray(qb, dtype=float)
    return float(np.degrees(np.max(np.abs(qb - qa)))) if qa.size else 0.0


def _joint_interp(qa, qb):
    qa = np.asarray(qa, dtype=float)
    qb = np.asarray(qb, dtype=float)
    jump_deg = _joint_max_jump_deg(qa, qb)
    n = max(1, int(math.ceil(jump_deg / MAX_EDGE_JOINT_STEP_DEG)))
    ts = np.linspace(0.0, 1.0, n + 1)
    return [(1.0 - t) * qa + t * qb for t in ts]


def _table_check(w, q):
    if p is None:
        raise RuntimeError("pybullet unavailable for table check")
    pts = p.getClosestPoints(w.robot, w.table, distance=TABLE_DISTANCE_M,
                            physicsClientId=w.client)
    for cp in pts:
        if cp[3] == -1:
            continue
        if cp[8] < TABLE_PENETRATION_M:
            raise RuntimeError(
                f"table collision at q={list(q)} distance={cp[8]}"
            )


def _check_edge(w, geo, kin, qa, qb, g, phase, stationary, H):
    qs = _joint_interp(qa, qb)
    last = qa
    for q in qs[1:]:
        qlist = [float(x) for x in q]
        if H is not None:
            Tfk = pose_fk(kin, qlist)
            target = _pose_dict(Tfk @ np.asarray(H, dtype=float))
        else:
            target = stationary
            if target is not None and not isinstance(target, dict):
                raise TypeError("stationary target must be dict or None")
        res = geo.check(qlist, g, phase, target)
        if not isinstance(res, dict) or "ok" not in res:
            raise RuntimeError(f"geo.check invalid result at phase {phase}: {res!r}")
        if not res.get("ok"):
            raise RuntimeError(
                f"geo.check failed at phase={phase} q={qlist} "
                f"reason={res.get('reason')} clearance={res.get('clearance_mm')}"
            )
        _table_check(w, qlist)
        last = q
    return last


def _segment_route(w, geo, kin, Ta, Tb, qa, g, phase, stationary, H):
    nodes = []
    frames = _interp_T(Ta, Tb, 1)
    prev_q = list(qa)
    for i, T in enumerate(frames):
        if i == 0 and nodes:
            continue
        pos = _as_pos_list(T[:3, 3])
        quat = _as_quat_xyzw(Rotation.from_matrix(T[:3, :3]).as_quat())
        sol = solve_pose(kin, pos, quat, prev_q)
        if not isinstance(sol, dict) or "q" not in sol:
            raise RuntimeError(f"solve_pose invalid at frame {i}: {sol!r}")
        if not sol.get("pass"):
            raise RuntimeError(
                f"solve_pose fail at phase={phase} frame={i} pos={pos} quat={quat} sol={sol!r}"
            )
        q = [float(x) for x in sol["q"]]
        jump = _joint_max_jump_deg(prev_q, q)
        if jump > MAX_JOINT_JUMP_DEG:
            raise RuntimeError(
                f"joint jump {jump} deg exceeds {MAX_JOINT_JUMP_DEG} at phase={phase} frame={i}"
            )
        _check_edge(w, geo, kin, prev_q, q, g, phase, stationary, H)
        nodes.append({"phase": phase, "q": q, "g": float(g)})
        prev_q = q
    return nodes


def route(w, geo, kin, A, B, seed, g, phase, H=None, stationary=None):
    if stationary is not None and not isinstance(stationary, dict):
        raise TypeError("stationary must be pos/quat dict or None")
    Ta = _pose_dict_to_mat(A) if isinstance(A, dict) else np.asarray(A, dtype=float)
    Tb = _pose_dict_to_mat(B) if isinstance(B, dict) else np.asarray(B, dtype=float)
    Hmat = None
    if H is not None:
        Hmat = np.asarray(H, dtype=float)

    frames = _interp_T(Ta, Tb, 1)
    nodes = []
    prev_q = [float(x) for x in seed]
    for i, T in enumerate(frames):
        pos = _as_pos_list(T[:3, 3])
        quat = _as_quat_xyzw(Rotation.from_matrix(T[:3, :3]).as_quat())
        sol = solve_pose(kin, pos, quat, prev_q)
        if not isinstance(sol, dict) or "q" not in sol:
            raise RuntimeError(f"solve_pose invalid at frame {i}: {sol!r}")
        if not sol.get("pass"):
            raise RuntimeError(
                f"solve_pose fail at phase={phase} frame={i} pos={pos} quat={quat} sol={sol!r}"
            )
        q = [float(x) for x in sol["q"]]
        jump = _joint_max_jump_deg(prev_q, q)
        if jump > MAX_JOINT_JUMP_DEG and i > 0:
            raise RuntimeError(
                f"joint jump {jump} deg exceeds {MAX_JOINT_JUMP_DEG} at phase={phase} frame={i}"
            )
        if i > 0:
            _check_edge(w, geo, kin, prev_q, q, g, phase, stationary, Hmat)
        nodes.append({"phase": phase, "q": q, "g": float(g)})
        prev_q = q
    return nodes


def _T_from_xyz_z(xyz, z):
    T = np.eye(4)
    T[:3, :3] = np.eye(3)
    T[0, 3] = float(xyz[0])
    T[1, 3] = float(xyz[1])
    T[2, 3] = float(z)
    return T


def rebuild(w, geo, kin, snapshot, q, g, home_q, gopen, hover_m, do_lift):
    if hover_m is None:
        hover_m = 0.04
    hover_m = float(hover_m)
    geo.reset(w.case, snapshot, q, g)

    q_cur = [float(x) for x in q]
    Ttcp = pose_fk(kin, q_cur)
    target_key = w.case["target_key"]
    target_snap = snapshot[target_key]
    Ttarget = _pose_dict_to_mat(target_snap)
    H = np.linalg.inv(Ttcp) @ Ttarget

    dest_key = w.case["dest"]
    dest_xyz = w.square_xyz(dest_key)
    board_top = float(dest_xyz[2])
    TpDest = _T_from_xyz_z(dest_xyz, board_top - DEST_CLEARANCE_M)
    Tdest = TpDest @ np.linalg.inv(H)

    suffix = []
    current_q = q_cur
    current_g = float(g)

    lift_z = hover_m
    Tlift = Tdest.copy()
    Tlift[2, 3] += lift_z

    if do_lift:
        src_xyz = w.square_xyz(w.case["source"]) if "source" in w.case else dest_xyz
        _ = src_xyz
        Tcur = Tlift
        Tstart = Tcur.copy()
    else:
        Tstart = Tdest.copy()

    Ttransfer = Tlift.copy()
    transfer_nodes = route(
        w, geo, kin,
        _pose_dict(Tstart), _pose_dict(Ttransfer),
        current_q, current_g, "transfer",
        H=H, stationary=None,
    )
    suffix.extend(transfer_nodes)
    if transfer_nodes:
        current_q = transfer_nodes[-1]["q"]

    lower_nodes = route(
        w, geo, kin,
        _pose_dict(Ttransfer), _pose_dict(Tdest),
        current_q, current_g, "lower",
        H=H, stationary=None,
    )
    suffix.extend(lower_nodes)
    if lower_nodes:
        current_q = lower_nodes[-1]["q"]

    dest_piece_upright = _pose_dict(TpDest)
    dest_stationary = dest_piece_upright

    currents = np.linspace(current_g, float(gopen), max(1, int(math.ceil(abs(float(gopen) - current_g) * 180.0 / math.pi / CLOSE_ENVELOPE_STEP_DEG)) + 1))
    for gg in currents[1:]:
        res = geo.check([float(x) for x in current_q], float(gg), "open", dest_stationary)
        if not isinstance(res, dict) or not res.get("ok"):
            raise RuntimeError(
                f"open envelope failed g={gg} reason={res.get('reason') if isinstance(res, dict) else res!r}"
            )

    suffix.append({"phase": "open", "q": [float(x) for x in current_q], "g": float(gopen)})
    current_g = float(gopen)

    Tdest_hover = Tdest.copy()
    Tdest_hover[2, 3] += hover_m
    retreat1 = route(
        w, geo, kin,
        _pose_dict(Tdest), _pose_dict(Tdest_hover),
        current_q, current_g, "retreat", H=None, stationary=dest_stationary,
    )
    retreat = list(retreat1)
    if retreat1:
        current_q = retreat1[-1]["q"]

    Thome = pose_fk(kin, [float(x) for x in home_q])
    home_nodes = route(
        w, geo, kin,
        _pose_dict(Tdest_hover), _pose_dict(Thome),
        current_q, current_g, "retreat", H=None, stationary=dest_stationary,
    )
    retreat.extend(home_nodes)

    return {
        "suffix": suffix,
        "retreat": retreat,
        "H": np.asarray(H, dtype=float).tolist(),
        "place": dest_piece_upright,
    }


def _max_gripper_limits(kin):
    lo = None
    hi = None
    for j in getattr(kin, "joints", []):
        if "lower" in j and (lo is None or j["lower"] < lo):
            pass
    try:
        if hasattr(kin, "gripper_limits"):
            lo, hi = kin.gripper_limits
    except Exception:
        pass
    return lo


def _precheck_center(w, center, geo):
    qc = center["q_contact"]
    gc = center["gcontact_rad"]
    res_c = geo.check(qc, gc, "center", None)
    if not isinstance(res_c, dict) or not res_c.get("ok"):
        raise RuntimeError(f"center check_close failed: {res_c!r}")
    if "check_close" in center:
        if not center["check_close"].get("ok"):
            raise RuntimeError(f"center check_close failed: {center['check_close']!r}")
    if "check_descend" in center:
        if not center["check_descend"].get("ok"):
            raise RuntimeError(f"center check_descend failed: {center['check_descend']!r}")


def plan(cfg):
    global_guard = None
    w = None
    result = {
        "ok": False,
        "physics_steps": 0,
        "query_worlds": 0,
    }
    try:
        if p is not None:
            global_guard = p.stepSimulation
            def _raise_step(*a, **k):
                raise RuntimeError("global p.stepSimulation disabled in plan")
            p.stepSimulation = _raise_step

        from core import Robot as _Robot
        urdf_path = str(WORKSPACE / "reachability_eval" / "model" / "so101_new_calib.urdf")
        robot_model = _Robot(urdf_path)
        kin = robot_model

        query_dir = Path(cfg["query"])
        center_path = Path(cfg["center"])
        with open(query_dir, "r") as f:
            querycfg = json.load(f)
        with open(center_path, "r") as f:
            center = json.load(f)

        if center.get("blocked") is not False:
            raise RuntimeError(f"center blocked flag not False: {center.get('blocked')!r}")

        case = querycfg.get("case") if isinstance(querycfg, dict) else None
        if case is None:
            case = center.get("case")
        if case is None:
            raise RuntimeError("no case found in query/center config")

        q_contact = center["q_contact"]
        g_contact = float(center["gcontact_rad"])
        source_snapshot = center["source_snapshot"]

        w = QueryWorld(querycfg, case, q_contact, g_contact)
        result["query_worlds"] = 1

        geo = Geometry(w)
        _precheck_center(w, center, geo)

        snapshot = center.get("source_snapshot")
        geo.reset(w.case, snapshot, q_contact, g_contact)

        lo_hi = _max_gripper_limits(kin)
        if isinstance(lo_hi, tuple) and len(lo_hi) == 2:
            lo = float(lo_hi[0])
        else:
            lo = -2.0 * math.pi

        g_open = float(center.get("gopen_rad", center.get("g_open", center.get("gopen", 1.0))))

        qc = [float(x) for x in q_contact]
        Tcontact = pose_fk(kin, qc)
        Tlift = Tcontact.copy()
        Tlift[2, 3] += float(cfg.get("hover_m", 0.04))

        home_q = center.get("home_q", qc)
        Thome = pose_fk(kin, [float(x) for x in home_q])

        source_pose = _pose_dict(Tcontact)

        hover_m = float(cfg.get("hover_m", 0.04))

        Tstart = Thome.copy()
        Tstart[2, 3] = Tlift[2, 3]
        Tstart[:3, :3] = Thome[:3, :3]

        pos = _as_pos_list(Tlift[:3, 3])
        quat = _as_quat_xyzw(Rotation.from_matrix(Tlift[:3, :3]).as_quat())
        sol_home = solve_pose(kin, pos, quat, qc)
        if not isinstance(sol_home, dict) or not sol_home.get("pass"):
            raise RuntimeError(f"home ik failed: {sol_home!r}")
        home_q_final = [float(x) for x in sol_home["q"]]

        prefix = route(
            w, geo, kin,
            _pose_dict(Thome), _pose_dict(Tlift),
            home_q_final, g_open, "descend",
            H=None, stationary=source_pose,
        )

        g_limit = max(lo, g_contact - math.radians(5))
        gs = np.linspace(g_open, g_limit,
                         max(1, int(math.ceil(abs(g_open - g_limit) * 180.0 / math.pi / CLOSE_ENVELOPE_STEP_DEG)) + 1))
        for gg in gs:
            res = geo.check(qc, float(gg), "close", source_pose)
            if not isinstance(res, dict) or not res.get("ok"):
                raise RuntimeError(f"close envelope failed g={gg} res={res!r}")

        nominal = rebuild(
            w, geo, kin, snapshot, qc, g_contact,
            home_q_final, g_open, hover_m, True,
        )

        result.update({
            "ok": True,
            "case": case,
            "home_q": [float(x) for x in home_q_final],
            "contact_q": qc,
            "g_open": float(g_open),
            "g_contact": float(g_contact),
            "g_limit": float(g_limit),
            "source_snapshot": snapshot,
            "prefix": prefix,
            "nominal": nominal,
            "physics_steps": 0,
            "query_worlds": result["query_worlds"],
        })
        return result

    except Exception as exc:
        result["ok"] = False
        result["reason"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        return result

    finally:
        if p is not None and global_guard is not None:
            try:
                p.stepSimulation = global_guard
            except Exception:
                pass
        if w is not None:
            try:
                if getattr(w, "client", 0):
                    w.disconnect()
            except Exception:
                pass


def _serialize(obj):
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main(argv=None):
    ap = argparse.ArgumentParser(prog="staticplan")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    result_path = out_dir / "result.json"
    if result_path.exists():
        raise SystemExit(f"refusing to overwrite existing {result_path}")
    if out_dir.exists() and not out_dir.is_dir():
        raise SystemExit(f"out path exists and is not a directory: {out_dir}")

    with open(args.config, "r") as f:
        cfg = json.load(f)

    res = plan(cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(_serialize(res), f, indent=2, sort_keys=True)
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
