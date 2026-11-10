import json
import numpy as np
import pybullet as p
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import pilot_geometry_contact1mm_v1 as base
from rook_path_ik_v5 import pose_fk, solve_pose

CONFIG = base.CONFIG
LINKS = ("fixed_jaw_collision", "moving_jaw_so101_v1_link")
DEG = 180.0 / np.pi


def _quat_xyzw_to_rot(q):
    return Rotation.from_quat(np.asarray(q, dtype=float))


def _rot_to_matrix(r):
    return r.as_matrix()


def _read_q5(w):
    return [float(p.getJointState(w.robot, j, physicsClientId=w.client)[0])
            for j in w.arm_joints]


def _read_g(w):
    return float(p.getJointState(w.robot, w.gripper_joint,
                                 physicsClientId=w.client)[0])


def read_robot(w):
    return _read_q5(w), _read_g(w)


def link_pose(w, name):
    idx = w.link_index[name]
    st = p.getLinkState(w.robot, idx, computeForwardKinematics=True,
                        physicsClientId=w.client)
    pos = np.asarray(st[4], dtype=float)
    rot = _quat_xyzw_to_rot(st[5])
    return pos, rot


def derive_refs(w, reference):
    q0, g0 = read_robot(w)
    acc = {name: [] for name in LINKS}
    wsum = {name: 0.0 for name in LINKS}
    try:
        w.set_query_robot(list(reference["q5"]), float(reference["g"]))
        for c in reference["contacts"]:
            name = c["link"]
            if name not in acc:
                continue
            wt = float(c["weight_N"])
            if wt <= 0.0:
                continue
            pos, rot = link_pose(w, name)
            local = rot.inv().apply(np.asarray(c["point_on_robot_world_m"],
                                               dtype=float) - pos)
            acc[name].append(wt * local)
            wsum[name] += wt
    finally:
        w.set_query_robot(q0, g0)
    refs = {}
    for name in LINKS:
        if wsum[name] <= 0.0:
            raise RuntimeError("no contact weight for link %s" % name)
        refs[name] = np.sum(acc[name], axis=0) / wsum[name]
    return refs


def _target_world(profile, piece_pos, piece_rot):
    pts = np.asarray(profile["points_m"], dtype=float)
    nrm = np.asarray(profile["normals_local"], dtype=float)
    local = pts + nrm * (profile["collision_envelope_m"] + profile["desired_gap_m"])
    world = piece_rot.apply(local) + piece_pos
    return local, world


def _jaw_world_targets(w, refs):
    out = []
    for name in LINKS:
        pos, rot = link_pose(w, name)
        out.append(pos + rot.apply(refs[name]))
    return out


def fit_pair(w, kin, profile, refs, seed, gripper_lo, gripper_hi):
    q0, g0 = read_robot(w)
    try:
        snap = w.snapshot()[w.case["target_key"]]
        piece_pos = np.asarray(snap["pos"], dtype=float)
        piece_rot = _quat_xyzw_to_rot(snap["quat"])
        _, targets_world = _target_world(profile, piece_pos, piece_rot)
        r_goal = piece_rot * _quat_xyzw_to_rot(profile["tool_quat_local"])

        lo = np.asarray(kin.lo, dtype=float) + (5.01 / DEG)
        hi = np.asarray(kin.hi, dtype=float) - (5.01 / DEG)
        arm_seed = np.asarray(seed, dtype=float)
        arm_seed = np.clip(arm_seed, np.asarray(kin.lo) + (5.01 / DEG),
                           np.asarray(kin.hi) - (5.01 / DEG))
        gseed = float(np.clip(profile["seed_g"], gripper_lo, gripper_hi))
        x_seed = np.concatenate([arm_seed, [gseed]])
        lb = np.concatenate([lo, [gripper_lo]])
        ub = np.concatenate([hi, [gripper_hi]])

        def resid(x):
            q = x[:5]
            g = float(x[5])
            w.set_query_robot(list(q), g)
            jaws = _jaw_world_targets(w, refs)
            r = [(jaws[i] - targets_world[i]) * 1000.0 for i in range(2)]
            fk = pose_fk(kin, q)
            rot_cur = Rotation.from_matrix(np.asarray(fk)[:3, :3])
            r.append(60.0 * (r_goal.inv() * rot_cur).as_rotvec())
            r.append(0.005 * (np.asarray(q) - arm_seed))
            return np.concatenate(r)

        sol = least_squares(resid, x_seed, bounds=(lb, ub), max_nfev=80, jac="3-point", diff_step=0.01)
        qfit = np.asarray(sol.x[:5], dtype=float)
        gfit = float(sol.x[5])

        fk = np.asarray(pose_fk(kin, qfit))
        tcp_pos = fk[:3, 3].copy()
        tcp_rot = Rotation.from_matrix(fk[:3, :3])
        tcp_quat = tcp_rot.as_quat().tolist()

        w.set_query_robot(qfit.tolist(), gfit)
        point_errors = [float(np.linalg.norm(j - t) * 1000.0)
                        for j, t in zip(_jaw_world_targets(w, refs), targets_world)]
        pt_err = max(point_errors)
        down_axis = tcp_rot.apply([0.0, 0.0, 1.0])
        tilt = float(np.degrees(np.arccos(np.clip(-down_axis[2], -1.0, 1.0))))
        rot_err = float(np.degrees((r_goal.inv() * tcp_rot).magnitude()))
        margin = float(kin.margin(qfit))
        ok = (pt_err <= 1.0 and tilt <= 10.0 and rot_err <= 10.0
              and margin >= 5.0)
        return {"ok": ok, "q": qfit.tolist(), "g": gfit,
                "pos": tcp_pos.tolist(), "quat": tcp_quat,
                "point_error_mm": point_errors, "tilt_deg": tilt,
                "rotation_error_deg": rot_err, "margin_deg": margin}
    finally:
        w.set_query_robot(q0, g0)


def verify_native(w, profile, refs, parent):
    q0, g0 = read_robot(w)
    try:
        w.set_query_robot(list(parent["q"]), float(parent["gcontact"]))
        snap = w.snapshot()[w.case["target_key"]]
        piece_pos = np.asarray(snap["pos"], dtype=float)
        piece_rot = _quat_xyzw_to_rot(snap["quat"])
        envelope = profile["collision_envelope_m"]
        desired = profile["desired_gap_m"]
        pts = np.asarray(profile["points_m"], dtype=float)
        nrm = np.asarray(profile["normals_local"], dtype=float)

        region_errs = []
        patch_errs = []
        gaps = []
        failures = []
        for i, name in enumerate(LINKS):
            lpos, lrot = link_pose(w, name)
            idx = w.link_index[name]
            cps = p.getClosestPoints(w.robot, w.bodies[w.case["target_key"]],
                                     0.05, linkIndexA=idx,
                                     physicsClientId=w.client)
            if not cps:
                failures.append("no_contact:%s" % name)
                continue
            cp = min(cps, key=lambda c: c[8])
            piece_pt = np.asarray(cp[6], dtype=float)
            robot_pt = np.asarray(cp[5], dtype=float)
            gap = float(cp[8])
            gaps.append(abs(gap - desired))
            piece_local = piece_rot.inv().apply(piece_pt - piece_pos)
            exp_local = pts[i] + nrm[i] * envelope
            region_errs.append(float(np.linalg.norm(piece_local - exp_local) * 1000.0))
            robot_local = lrot.inv().apply(robot_pt - lpos)
            patch_errs.append(float(np.linalg.norm(robot_local - refs[name]) * 1000.0))

        region_err = max(region_errs) if region_errs else float("inf")
        patch_err = max(patch_errs) if patch_errs else float("inf")
        gap_err = max(gaps) if gaps else float("inf")
        actual = None
        if not failures:
            actual = []
            for i, name in enumerate(LINKS):
                lpos, lrot = link_pose(w, name)
                idx = w.link_index[name]
                cps = p.getClosestPoints(w.robot, w.bodies[w.case["target_key"]],
                                         0.05, linkIndexA=idx,
                                         physicsClientId=w.client)
                cp = min(cps, key=lambda c: c[8])
                actual.append(piece_rot.inv().apply(
                    np.asarray(cp[6], dtype=float) - piece_pos).tolist())
            if region_err <= profile["region_radius_m"] * 1000.0:
                pass
            else:
                failures.append("region_radius")
            if patch_err <= profile["jaw_patch_radius_m"] * 1000.0:
                pass
            else:
                failures.append("jaw_patch_radius")
            if gap_err <= 0.0003:
                pass
            else:
                failures.append("gap")
        else:
            if region_err > profile["region_radius_m"] * 1000.0:
                failures.append("region_radius")
            if patch_err > profile["jaw_patch_radius_m"] * 1000.0:
                failures.append("jaw_patch_radius")

        return {"ok": not failures,
                "reason": "ok" if not failures else ";".join(failures),
                "actual_contact_local_m": actual,
                "region_error_mm": region_err,
                "jaw_patch_error_mm": patch_err}
    finally:
        w.set_query_robot(q0, g0)


class Geometry(base.Geometry):
    def contact_grasp(self, q, pos, quat, seed, kin, nominal_g):
        w = self.w
        q0, g0 = read_robot(w)
        records = []
        old_config = base.CONFIG
        old_solve = base.solve_pose
        try:
            with open(w.cfg["keypoint_catalog_path"]) as f:
                catalog = json.load(f)
            with open(w.cfg["gripper_reference_path"]) as f:
                reference = json.load(f)

            candidates = catalog["pieces"][w.case["type"]]["candidates"][:4]
            refs = derive_refs(w, reference)

            for profile in candidates:
                fit = fit_pair(w, kin, profile, refs, seed,
                               float(self.lo), float(self.hi))
                if not fit["ok"]:
                    records.append({"profile": profile.get("id"),
                                    "stage": "fit", "fit": fit})
                    continue
                try:
                    base.solve_pose = solve_pose
                    base.CONFIG = dict(old_config, max_center_mm=12)
                    parent = super().contact_grasp(
                        fit["q"], fit["pos"], fit["quat"],
                        fit["q"], kin, fit["g"])
                finally:
                    base.solve_pose = old_solve
                    base.CONFIG = old_config
                if not parent or not parent.get("ok"):
                    records.append({"profile": profile.get("id"),
                                    "stage": "parent", "fit": fit,
                                    "parent": parent})
                    continue
                ver = verify_native(w, profile, refs, parent)
                if not ver["ok"]:
                    records.append({"profile": profile.get("id"),
                                    "stage": "verify", "fit": fit,
                                    "verify": ver})
                    continue
                records.append({"profile": profile["id"], "stage": "selected", "fit": fit, "verify": ver})
                out = dict(parent)
                out.update({
                    "keypoint_profile": profile,
                    "reference_points_link_m": {k: v.tolist()
                                                for k, v in refs.items()},
                    "actual_contact_local_m": ver["actual_contact_local_m"],
                    "region_error_mm": ver["region_error_mm"],
                    "jaw_patch_error_mm": ver["jaw_patch_error_mm"],
                    "fit": fit,
                    "candidate_records": records,
                })
                return out

            return {"ok": False, "reason": "no_valid_keypoint_contact",
                    "candidate_records": records}
        finally:
            w.set_query_robot(q0, g0)
