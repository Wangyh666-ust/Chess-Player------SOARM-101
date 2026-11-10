import sys
import math
import json
import traceback
import argparse
from pathlib import Path

workspace = Path(__file__).resolve().parents[4]
for _p in (
    str(Path(__file__).resolve().parent),
    str(workspace / "reachability_eval"),
    str(workspace / "calibration" / "manipulation_20261109_v1" / "work"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as p

import rook_route_v5 as old
from core import Robot
from rook_path_ik_v5 import solve_pose, pose_fk
from query_world_snapshot_v1 import QueryWorld
from keypoint_geometry_v3 import Geometry


def _identity4_translation(x, y, z):
    T = np.eye(4)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def _ang_deg(qa, qb):
    a = np.asarray(qa, dtype=float)
    b = np.asarray(qb, dtype=float)
    return float(np.degrees(np.max(np.abs(a - b))))


def route(w, geo, kin, A, B, seed, g, phase, H=None, stationary=None):
    nodes = old.route(w, geo, kin, A, B, seed, g, phase, H=H, stationary=stationary)
    first_q = np.asarray(nodes[0]["q"], dtype=float)
    seed_q = np.asarray(seed, dtype=float)
    if _ang_deg(first_q, seed_q) > 2.0:
        raise RuntimeError(
            "route: first node jump from seed exceeds 2deg (%.6f)"
            % _ang_deg(first_q, seed_q)
        )
    last_q = old._check_edge(
        w, geo, kin, seed_q, first_q, g, phase, stationary, H
    )
    return nodes


def rebuild(w, geo, kin, snapshot, q, g, home_q, gopen, hover_m, do_lift):
    geo.reset(w.case, snapshot, q, g)

    Tstart = pose_fk(kin, q)
    H = np.linalg.inv(Tstart) @ old._pose_dict_to_mat(snapshot[w.case["target_key"]])

    dp = w.square_xyz(w.case["dest"])
    TpDest = _identity4_translation(dp[0], dp[1], dp[2] + w.cfg["native_collision_floor_offset_m"] - 0.00003)

    Tdest = TpDest @ np.linalg.inv(H)
    Tdh = Tdest.copy()
    Tdh[2, 3] += hover_m

    suffix = []
    seed = q
    CURRENT = Tstart

    if do_lift:
        Tsh = CURRENT.copy()
        Tsh[2, 3] += hover_m
        lift_nodes = route(
            w, geo, kin, CURRENT, Tsh, seed, g, "lift", H=H
        )
        suffix.extend(lift_nodes)
        old._table_check(w, np.asarray(lift_nodes[-1]["q"], dtype=float))
        CURRENT = Tsh
        seed = lift_nodes[-1]["q"]

    transfer_nodes = route(
        w, geo, kin, CURRENT, Tdh, seed, g, "transfer", H=H
    )
    suffix.extend(transfer_nodes)
    old._table_check(w, np.asarray(transfer_nodes[-1]["q"], dtype=float))
    seed = transfer_nodes[-1]["q"]

    lower_nodes = route(
        w, geo, kin, Tdh, Tdest, seed, g, "lower", H=H
    )
    suffix.extend(lower_nodes)
    old._table_check(w, np.asarray(lower_nodes[-1]["q"], dtype=float))
    final_q = lower_nodes[-1]["q"]

    for gg in np.linspace(g, gopen, max(2, int(math.ceil(abs(gopen-g)/math.radians(0.5)))+1)):
        open_verdict = geo.check(final_q, float(gg), "open", old._pose_dict(TpDest))
        if not isinstance(open_verdict, dict) or open_verdict.get("ok") is not True:
            raise RuntimeError("rebuild: open check rejected: %r" % (open_verdict,))
        old._table_check(w, np.asarray(final_q, dtype=float))

    stationary = old._pose_dict(TpDest)
    retreat1 = route(
        w, geo, kin, Tdest, Tdh, final_q, gopen, "retreat",
        stationary=stationary
    )
    prev_last = retreat1[-1]["q"]
    old._table_check(w, np.asarray(prev_last, dtype=float))
    retreat2 = route(
        w, geo, kin, Tdh, pose_fk(kin, home_q), prev_last, gopen, "retreat",
        stationary=stationary
    )
    old._table_check(w, np.asarray(retreat2[-1]["q"], dtype=float))
    retreat = list(retreat1) + list(retreat2)

    return {
        "suffix": suffix,
        "retreat": retreat,
        "H": H.tolist(),
        "place": old._pose_dict(TpDest),
    }


def plan(cfg):
    out = {
        "ok": False,
        "physics_steps": 0,
        "query_worlds": 0,
        "reason": None,
        "traceback": None,
    }
    orig_step = p.stepSimulation
    w = None
    disconnected = False

    def _guarded_step(*args, **kwargs):
        raise RuntimeError("physics stepSimulation must not be called during plan")

    try:
        with open(cfg["query"], "r") as f:
            querycfg = json.load(f)
        with open(cfg["center"], "r") as f:
            center = json.load(f)

        if center.get("blocked", None) is not False:
            raise RuntimeError("plan: center.blocked is not False")
        if center["check_close"]["ok"] is not True:
            raise RuntimeError("plan: center.check_close.ok is not True")
        if center["check_descend"]["ok"] is not True:
            raise RuntimeError("plan: center.check_descend.ok is not True")

        p.stepSimulation = _guarded_step

        kin = Robot(str(workspace / "reachability_eval/model/so101_new_calib.urdf"))

        w = QueryWorld(
            querycfg, querycfg["case"], center["q_contact"], center["gcontact_rad"]
        )
        out["query_worlds"] = 1
        geo = Geometry(w)

        snapshot = center["source_snapshot"]
        qc = center["q_contact"]
        gc = center["gcontact_rad"]
        go = center["gopen_rad"]
        lo = center["urdf_gripper_lo"]
        gl = max(lo, gc - math.radians(5))

        geo.reset(w.case, snapshot, qc, gc)

        Tcontact = pose_fk(kin, qc)
        Thover = Tcontact.copy()
        Thover[2, 3] += cfg["hover_m"]

        homeSol = solve_pose(
            kin,
            Thover[:3, 3],
            Rotation.from_matrix(Thover[:3, :3]).as_quat(),
            qc,
        )
        if not homeSol.get("pass"):
            raise RuntimeError("plan: solve_pose for home failed: %r" % (homeSol,))
        home_q = homeSol["q"]

        station = snapshot[w.case["target_key"]]
        prefix = route(
            w, geo, kin, Thover, Tcontact, home_q, go, "descend",
            stationary=station
        )
        old._table_check(w, np.asarray(prefix[-1]["q"], dtype=float))

        for gg in np.linspace(go, gl, max(2, int(math.ceil(abs(go-gl)/math.radians(0.5)))+1)):
            close_verdict = geo.check(qc, float(gg), "close", station)
            if not isinstance(close_verdict, dict) or close_verdict.get("ok") is not True:
                raise RuntimeError("plan: close check rejected: %r" % (close_verdict,))
            old._table_check(w, np.asarray(qc, dtype=float))

        nominal = rebuild(
            w, geo, kin, snapshot, qc, gc, home_q, go, cfg["hover_m"], True
        )

        out.update({
            "ok": True,
            "case": w.case,
            "home_q": home_q,
            "contact_q": qc,
            "g_open": go,
            "g_contact": gc,
            "g_limit": gl,
            "source_snapshot": snapshot,
            "prefix": prefix,
            "nominal": nominal,
            "physics_steps": 0,
            "query_worlds": 1,
        })
    except Exception as e:
        out["ok"] = False
        out["reason"] = "%s: %s" % (type(e).__name__, e)
        out["traceback"] = traceback.format_exc()
    finally:
        p.stepSimulation = orig_step
        if w is not None and not disconnected:
            p.disconnect(physicsClientId=w.client)
            disconnected = True
        p.stepSimulation = orig_step

    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    with open(args.config, "r") as f:
        cfg = json.load(f)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "result.json"
    if outpath.exists():
        raise SystemExit("refusing to overwrite existing %s" % outpath)

    result = plan(cfg)
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
