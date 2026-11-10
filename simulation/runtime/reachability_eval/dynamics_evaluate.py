"""Dynamics evaluation for reachability/placement cases.

All physical parameters (masses, thresholds, contact forces) are uncalibrated
assumptions for relative comparison only. No claims about VLA or hardware success.
"""

import math
from typing import Any, Dict, List, Optional

import pybullet


_TILT_THRESHOLD_NOTE = (
    "All physical parameters and thresholds are uncalibrated assumptions; "
    "results do not constitute VLA or hardware success claims."
)

_LIFTED_PHASES = ("lift", "lift_hold", "transfer", "transfer_via")
_TRANSFER_PHASES = ("transfer", "transfer_via")


def _tilt_deg(quat_xyzw) -> float:
    m = pybullet.getMatrixFromQuaternion(list(quat_xyzw))
    r = [m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]
    c = r[2][2]
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def _finite(obj) -> bool:
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, (int, bool)) or obj is None:
        return True
    if isinstance(obj, str):
        return True
    if isinstance(obj, dict):
        return all(_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_finite(v) for v in obj)
    return True


def _xy_dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def evaluate(case: Dict[str, Any], observations: List[Dict[str, Any]], world: Any) -> Dict[str, Any]:
    obs = list(observations) if observations else []
    dest_xyz = list(world.square_xyz(case["dest"]))

    final_xy_mm = 0.0
    final_z_mm = 0.0
    final_tilt = 0.0
    max_lift_mm = 0.0
    transfer_zs = []
    max_track = 0.0
    motor_flag = False
    both_frac_num, both_frac_den = 0, 0
    peak_fixed = 0.0
    peak_moving = 0.0
    contact_den = 0

    for o in obs:
        tpos = list(o["target_pos"])
        q = o.get("target_quat", [0.0, 0.0, 0.0, 1.0])
        tz = tpos[2] if len(tpos) > 2 else 0.0
        phase = o.get("phase", "")

        if phase in _LIFTED_PHASES:
            max_lift_mm = max(max_lift_mm, (tz + 0.09) * 1000.0)
        if phase in _TRANSFER_PHASES:
            transfer_zs.append((tz + 0.09) * 1000.0)

        track = o.get("max_arm_tracking_error_deg")
        if track is not None:
            max_track = max(max_track, float(track))

        mcs = o.get("motor_collision_sum")
        if mcs is not None and float(mcs) > 0.01:
            motor_flag = True

        ff = float(o.get("fixed_jaw_collision_sum", 0.0) or 0.0)
        mf = float(o.get("moving_jaw_collision_sum", 0.0) or 0.0)
        peak_fixed = max(peak_fixed, ff)
        peak_moving = max(peak_moving, mf)

        if phase in _LIFTED_PHASES:
            contact_den += 1
            if ff > 1e-4 and mf > 1e-4:
                both_frac_num += 1

    if obs:
        last = obs[-1]
        tpos = list(last["target_pos"])
        q = last.get("target_quat", [0.0, 0.0, 0.0, 1.0])
        if len(dest_xyz) >= 2 and len(tpos) >= 2:
            final_xy_mm = _xy_dist(tpos, dest_xyz) * 1000.0
        if len(dest_xyz) >= 3 and len(tpos) >= 3:
            final_z_mm = abs((tpos[2] - dest_xyz[2])) * 1000.0
        final_tilt = _tilt_deg(q)

    min_transfer_mm: Optional[float] = None
    if transfer_zs:
        min_transfer_mm = min(transfer_zs)

    # neighbor stats after settle_start
    neighbor_max_disp_mm = 0.0
    neighbor_max_tilt = 0.0
    settle_idx = None
    for i, o in enumerate(obs):
        if o.get("phase") == "settle_start":
            settle_idx = i
    if settle_idx is not None:
        ref = obs[settle_idx].get("object_poses") or {}
        for o in obs[settle_idx + 1:]:
            poses = o.get("object_poses") or {}
            for key, p in poses.items():
                if key == case["target_key"]:
                    continue
                if key not in ref:
                    continue
                old_pos = list(ref[key].get("pos", [0, 0, 0]))
                new_pos = list(p.get("pos", [0, 0, 0]))
                if len(old_pos) >= 2 and len(new_pos) >= 2:
                    neighbor_max_disp_mm = max(
                        neighbor_max_disp_mm, _xy_dist(old_pos, new_pos) * 1000.0
                    )
                pq = p.get("quat", [0, 0, 0, 1])
                neighbor_max_tilt = max(neighbor_max_tilt, abs(_tilt_deg(pq)))

    both_frac = (both_frac_num / contact_den) if contact_den else 0.0

    criteria = {
        "final_xy_error": final_xy_mm <= 3.0,
        "final_z_error": final_z_mm <= 1.0,
        "final_tilt": final_tilt <= 5.0,
        "max_lift": max_lift_mm >= 20.0,
        "min_transfer_lift": (min_transfer_mm is None) or (min_transfer_mm >= 15.0),
        "neighbor_displacement": neighbor_max_disp_mm <= 2.0,
        "neighbor_tilt": neighbor_max_tilt <= 5.0,
        "max_tracking_error": max_track <= 5.0,
        "no_target_motor_contact": not motor_flag,
    }

    criteria["no_unintended_contact"] = not any(c["normalForce"] > .01 for o in obs for name in ("robot_board_contacts", "self_contacts", "neighbor_contacts") for c in o.get(name, [])) and not any(c["normalForce"] > .01 and c["robot_link"] not in ("fixed_jaw_collision", "moving_jaw_so101_v1_link", "gripper_frame_link") for o in obs for c in o["contacts"])
    failed = [k for k, v in criteria.items() if not v]
    ok = all(criteria.values())

    sim_time = getattr(world, "time", None)
    steps = getattr(world, "steps", None)
    masses = getattr(world, "masses", None)

    return {
        "ok": ok,
        "criteria": criteria,
        "failed_criteria": failed,
        "final_xy_error_mm": final_xy_mm,
        "final_bottom_z_error_mm": final_z_mm,
        "final_tilt_deg": final_tilt,
        "max_lift_mm": max_lift_mm,
        "min_transfer_lift_mm": min_transfer_mm,
        "neighbor_max_displacement_mm": neighbor_max_disp_mm,
        "neighbor_max_tilt_deg": neighbor_max_tilt,
        "max_tracking_error_deg": max_track,
        "both_jaws_contact_fraction": both_frac,
        "peak_fixed_force_N": peak_fixed,
        "peak_moving_force_N": peak_moving,
        "simulation_time_s": sim_time,
        "steps": steps,
        "mass_table_kg": masses,
        "no_attachment": True,
        "note": _TILT_THRESHOLD_NOTE,
    }
