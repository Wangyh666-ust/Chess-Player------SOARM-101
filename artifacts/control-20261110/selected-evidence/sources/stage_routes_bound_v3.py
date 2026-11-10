import math
import time

import numpy as np
from numpy.linalg import inv

from scipy.spatial.transform import Rotation
from tilt_ik_v1 import pose_fk, solve_pose
import tilt_route_v1 as legacy
from query_safety_bound_v1 import pose_dict, pose_matrix, tilt_deg, board_gap, no_step
from release_planner_bound_v1 import preview_support, plan_exit

from transfer_bridge_v2 import transfer_route
route = legacy.route


class StagePlanRejected(RuntimeError):
    pass


def _table_check(w):
    for name in ("cfg", "case"):
        if not hasattr(w, name):
            raise TypeError("world missing attribute: " + name)
    if "safe_home_q" not in w.cfg:
        raise TypeError("cfg missing safe_home_q")
    safe = w.cfg["safe_home_q"]
    if not isinstance(safe, (list, tuple)) or len(safe) != 5:
        raise TypeError("safe_home_q must be list5")
    key = w.case["target_key"]
    if key is None:
        raise TypeError("target_key is None")
    return safe, key


def _audit(w):
    if not hasattr(w, "plan_audits") or w.plan_audits is None:
        w.plan_audits = []
    return w.plan_audits


def _route_list(res):
    if not isinstance(res, (list, tuple)) or len(res) < 1:
        raise TypeError("route must return node list")
    for n in res:
        if not isinstance(n, dict):
            raise TypeError("route node not dict")
        if "q" not in n or "g" not in n or "phase" not in n:
            raise TypeError("route node malformed")
        if len(n["q"]) != 5:
            raise TypeError("route node q not list5")
    return res


def rebuild(w, geo, kin, snapshot, q, g, home_q, gopen, hover_m, do_lift):
    t_start = time.time()
    safe_home, target_key = _table_check(w)
    q = [float(x) for x in q]
    if len(q) != 5:
        raise TypeError("q must be list5")

    if target_key not in snapshot:
        raise KeyError("target_key missing in snapshot")
    tgt = snapshot[target_key]
    tgt_pos = [float(x) for x in tgt["pos"]]
    tgt_quat = [float(x) for x in tgt["quat"]]
    if len(tgt_pos) != 3 or len(tgt_quat) != 4:
        raise TypeError("snapshot target malformed")

    with no_step():
        dest = w.square_xyz(w.case["dest"])
        dp = [float(dest[0]), float(dest[1]), float(dest[2])]

        geo.reset(w.case, snapshot, q, g)
        A = pose_fk(kin, q)
        if A.shape != (4, 4):
            raise TypeError("pose_fk not 4x4")
        H = inv(A) @ pose_matrix({"pos": tgt_pos, "quat": tgt_quat})
        Hinv = inv(H)

        floor = float(w.cfg.get("native_collision_floor_offset_m", 0.0))
        runtime = bool(w.cfg.get("_runtime_reselect", False))
        yaw_sel = float(w.cfg.get("placement_yaw_deg", 0.0))

        hover_opts = [float(hover_m)]
        if runtime:
            hcur = float(hover_m)
            hover_opts = [hcur]
            for d in (0.01, 0.02, 0.03, 0.04):
                hcand = d
                if all(abs(hcand - h) > 1e-9 for h in hover_opts):
                    hover_opts.append(hcand)

        yaws = [yaw_sel]
        if runtime:
            for d in (15.0, -15.0, 30.0, -30.0, 90.0, -90.0, 180.0):
                yaws.append(yaw_sel + d)

        candidates = []
        for h in hover_opts:
            for yaw in yaws:
                candidates.append((h, yaw))
        if len(candidates) > 12:
            candidates = candidates[:12]

        attempts = []
        success = None
        for h, yaw in candidates:
            if time.time() - t_start > 45.0:
                break
            attempt_start = time.time()
            rec = {"yaw_deg": float(yaw), "hover_m": float(h),
                   "ok": False, "reason": "", "elapsed_s": 0.0}
            try:
                geo.reset(w.case, snapshot, q, g)
                yaw_rad = math.radians(yaw)
                c = math.cos(yaw_rad)
                s = math.sin(yaw_rad)
                Rz = np.array([[c, -s, 0.0, 0.0],
                               [s, c, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]])
                Tpiece = np.eye(4)
                Tpiece[:3, :3] = Rz[:3, :3]
                Tpiece[0, 3] = dp[0]
                Tpiece[1, 3] = dp[1]
                Tpiece[2, 3] = dp[2] + floor + 0.002

                seed = list(q)
                solved = None
                gap = None
                converged = False
                reason = "solve_fail"
                for _ in range(3):
                    tcp_goal = Tpiece @ Hinv
                    res = solve_pose(kin, tcp_goal[:3,3].tolist(), Rotation.from_matrix(tcp_goal[:3,:3]).as_quat().tolist(), seed)
                    if not isinstance(res, dict) or res.get("pass") is not True:
                        reason = "solve_no_pass"
                        break
                    solq = [float(x) for x in res["q"]]
                    if len(solq) != 5:
                        reason = "solve_q_bad"
                        break
                    actual_piece = pose_fk(kin, solq) @ H
                    gap = float(board_gap(w, actual_piece))
                    tilt = float(tilt_deg(actual_piece))
                    if 0.0015 <= gap <= 0.0025 and tilt <= 10.0 + 1e-7:
                        converged = True
                        solved = solq
                        break
                    correction = 0.002 - gap
                    if abs(correction) > 0.005:
                        reason = "correction_too_large"
                        break
                    Tpiece = Tpiece.copy()
                    Tpiece[2, 3] += correction
                    seed = solq
                if not converged:
                    rec["reason"] = reason
                    rec["elapsed_s"] = time.time() - attempt_start
                    attempts.append(rec)
                    continue

                CURRENT = A
                seed = list(q)
                suffix = []

                if do_lift:
                    lift_goal = CURRENT.copy()
                    lift_goal[2, 3] += float(h)
                    ns = _route_list(route(w, geo, kin, CURRENT, lift_goal,
                                           seed, g, "lift", H=H))
                    suffix.extend(ns)
                    seed = list(ns[-1]["q"])
                    CURRENT = pose_fk(kin, seed)

                transfer_goal = (Tpiece @ Hinv).copy()
                transfer_goal[2, 3] += float(h)
                ns = _route_list(transfer_route(w, geo, kin, CURRENT, transfer_goal,
                                       seed, g, H))
                suffix.extend(ns)
                seed = list(ns[-1]["q"])
                CURRENT = pose_fk(kin, seed)

                ns = _route_list(route(w, geo, kin, CURRENT, Tpiece @ Hinv,
                                       seed, g, "lower", H=H))
                suffix.extend(ns)
                last_q = list(ns[-1]["q"])
                real_pred = pose_fk(kin, last_q) @ H
                real_gap = float(board_gap(w, real_pred))
                real_tilt = float(tilt_deg(real_pred))
                if not (0.001 <= real_gap <= 0.003 and real_tilt <= 10.0 + 1e-7):
                    rec["reason"] = "final_gap_tilt"
                    rec["elapsed_s"] = time.time() - attempt_start
                    attempts.append(rec)
                    continue

                preview = preview_support(w, geo, kin, last_q, g, H, route)
                if not isinstance(preview, dict) or "piece" not in preview or "q" not in preview:
                    raise TypeError("preview_support malformed")
                predicted_snapshot = dict(snapshot)
                predicted_snapshot[target_key] = preview["piece"]
                exit_info = plan_exit(w, geo, kin, predicted_snapshot, preview["q"],
                                     g, gopen, safe_home, route)
                if not isinstance(exit_info, dict) or "exit" not in exit_info or "home" not in exit_info:
                    raise TypeError("plan_exit malformed")

                success = {
                    "suffix": suffix,
                    "retreat": exit_info["exit"],
                    "home": exit_info["home"],
                    "H": H.tolist(),
                    "place": pose_dict(real_pred),
                    "support_preview": preview,
                    "empty_preview": exit_info,
                    "choice": {"yaw_deg": float(yaw), "hover_m": float(h),
                               "preland_gap_m": real_gap},
                }
                rec["ok"] = True
                rec["reason"] = "ok"
                rec["elapsed_s"] = time.time() - attempt_start
                attempts.append(rec)
                break
            except StagePlanRejected:
                raise
            except (TypeError, ValueError, KeyError):
                raise
            except RuntimeError as e:
                rec["reason"] = "runtime:" + str(e)
                rec["elapsed_s"] = time.time() - attempt_start
                attempts.append(rec)
                continue

        audit = {
            "stage": "measured" if runtime else "nominal",
            "do_lift": bool(do_lift),
            "attempts": attempts,
            "choice": success["choice"] if success is not None else None,
            "actualH": H.tolist(),
            "transfer_audits": list(getattr(w, "transfer_audits", [])),
            "wall_s": time.time() - t_start,
        }
        if success is None:
            _audit(w).append(audit)
            raise StagePlanRejected("staged_no_plan:" + str(attempts))

        audit["search_audit"] = {"attempts_count": len(attempts)}
        _audit(w).append(audit)

        success["search_audit"] = audit
        return success
