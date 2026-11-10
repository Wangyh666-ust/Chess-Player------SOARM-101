import math
import time

import numpy as np

from tilt_ik_v1 import pose_fk
import tilt_route_base_v1 as base
from query_safety_bound_v1 import (
    pose_dict,
    pose_matrix,
    tilt_deg,
    board_gap,
    target_clearance,
    no_step,
)


def _q_list(q):
    return [float(x) for x in q]


def _fk_pose(kin, q):
    return np.asarray(pose_fk(kin, q), dtype=float)


def _with_world_dz(A, dz):
    B = np.array(A, dtype=float, copy=True)
    B[2, 3] = B[2, 3] + float(dz)
    return B


def _piece_from_q(w, q, H):
    return _fk_pose(w.kin if hasattr(w, "kin") else None, q)


def _copy_pose_dict(pd):
    return pose_dict(pose_matrix(pd))


@no_step()
def plan_exit(w, geo, kin, snapshot, q, g_start, gopen, home_q, route_fn):
    geo.reset(w.case, snapshot, q, g_start)
    targetpose = snapshot[w.case["target_key"]]

    q = _q_list(q)
    home_q = _q_list(home_q)
    g_start = float(g_start)
    gopen = float(gopen)

    tried = []

    # Opening sweep: g_start -> gopen at <= 0.5 deg increments.
    sweep = []
    g = g_start
    if g_start <= gopen:
        while g <= gopen + 1e-12:
            sweep.append(g)
            g += math.radians(0.5)
        if not sweep or sweep[-1] < gopen - 1e-12:
            sweep.append(gopen)
        else:
            sweep[-1] = gopen
    else:
        while g >= gopen - 1e-12:
            sweep.append(g)
            g -= math.radians(0.5)
        if not sweep or sweep[-1] > gopen + 1e-12:
            sweep.append(gopen)
        else:
            sweep[-1] = gopen

    for gg in sweep:
        ok = geo.check(q, gg, "open", targetpose)
        if not isinstance(ok, dict) or not isinstance(ok.get("ok"), bool):
            raise TypeError("invalid open verdict")
        if not ok["ok"]:
            raise RuntimeError("open_sweep_failed:"+str(ok))
        base._table_check(w, q)

    A = _fk_pose(kin, q)

    heights = [0.02, 0.03, 0.04, 0.015]
    exit_nodes = None
    endpoint = None
    end_q = None
    end_gap = None
    chosen_height = None

    for dz in heights:
        try:
            B = _with_world_dz(A, dz)
            nodes = route_fn(
                w,
                geo,
                kin,
                A,
                B,
                q,
                gopen,
                "empty_exit",
                stationary=targetpose,
            )
            nodes = list(nodes) if nodes is not None else []
            if not nodes:
                raise RuntimeError("empty_exit_empty_route")

            last = nodes[-1]
            new_q = _q_list(last["q"])
            new_g = float(last.get("g", gopen))

            endpoint = _fk_pose(kin, new_q)
            gap = float(target_clearance(w, new_q, new_g, targetpose))
            if gap < 0.0005:
                raise RuntimeError("empty_exit_target_clearance")

            ok = geo.check(new_q, new_g, "empty_home", targetpose)
            if not isinstance(ok, dict) or not isinstance(ok.get("ok"), bool):
                raise TypeError("invalid home verdict")
            if not ok["ok"]:
                raise RuntimeError("empty_home_check_failed:"+str(ok))
            base._table_check(w, new_q)

            base._check_edge(
                w,
                geo,
                kin,
                new_q,
                home_q,
                new_g,
                "empty_home",
                targetpose,
                None,
            )

            home_nodes = []
            for hq in base._joint_interp(new_q, home_q):
                home_nodes.append(
                    {
                        "phase": "empty_home",
                        "q": _q_list(hq),
                        "g": new_g,
                    }
                )

            home_endpoint = _fk_pose(kin, home_q)
            home_gap = float(
                target_clearance(w, home_q, new_g, targetpose)
            )
            if home_gap < 0.0005:
                raise RuntimeError("empty_home_target_clearance")

            exit_nodes = nodes
            end_q = new_q
            end_gap = gap
            chosen_height = float(dz)
            tried.append({"height_m": float(dz), "ok": True})
            return {
                "exit": list(exit_nodes),
                "home": home_nodes,
                "exit_height_m": chosen_height,
                "end_gap_m": float(end_gap),
                "home_gap_m": float(home_gap),
                "tried": tried,
                "piece": targetpose,
            }
        except RuntimeError as exc:
            tried.append(
                {
                    "height_m": float(dz),
                    "ok": False,
                    "reason": str(exc),
                }
            )
            continue

    raise RuntimeError("empty_return_no_plan:" + str(tried))


@no_step()
def preview_support(w, geo, kin, q, g, H, route_fn):
    q = _q_list(q)
    g = float(g)
    H = np.asarray(H, dtype=float)

    if H.shape != (4, 4):
        raise ValueError("H must be a 4x4 matrix")

    A0 = _fk_pose(kin, q)
    piece0 = A0 @ H

    # Reset target through board_gap prediction.
    gap0 = float(board_gap(w, piece0))
    piece_target = pose_dict(piece0)

    tilt0 = float(tilt_deg(piece0))
    if tilt0 > 10.0 + 1e-7:
        raise RuntimeError("support_preview_tilt_exceeded")

    if -0.001 <= gap0 <= 0.0:
        return {
            "q": q,
            "piece": piece_target,
            "gap_m": gap0,
            "probes": [],
        }

    qcur = list(q)
    probes = []

    for i in range(1, 7):
        A = _fk_pose(kin, qcur)
        old_z = float(A[2, 3])
        B = np.array(A, dtype=float, copy=True)
        B[2, 3] = B[2, 3] - 0.0005

        nodes = route_fn(
            w,
            geo,
            kin,
            A,
            B,
            qcur,
            g,
            "lower",
            H=H,
            stationary=None,
        )
        nodes = list(nodes) if nodes is not None else []
        if not nodes:
            raise RuntimeError("support_preview_no_route")

        last = nodes[-1]
        new_q = _q_list(last["q"])

        Anew = _fk_pose(kin, new_q)
        new_z = float(Anew[2, 3])
        if not (new_z < old_z):
            raise RuntimeError("support_preview_no_descent")

        piecen = Anew @ H
        signedgap = float(board_gap(w, piecen))
        tiltn = float(tilt_deg(piecen))
        if tiltn > 10.0 + 1e-7:
            raise RuntimeError("support_preview_tilt_exceeded")
        if signedgap < -0.001:
            raise RuntimeError("support_preview_penetration")

        probes.append(
            {
                "probe": i,
                "requested_dz_m": -0.0005,
                "gap_m": signedgap,
                "q": _q_list(new_q),
            }
        )

        qcur = new_q
        if signedgap <= 0.0:
            piece_final = _fk_pose(kin, qcur) @ H
            return {
                "q": _q_list(qcur),
                "piece": pose_dict(piece_final),
                "gap_m": signedgap,
                "probes": probes,
            }

    raise RuntimeError("support_preview_no_contact")
