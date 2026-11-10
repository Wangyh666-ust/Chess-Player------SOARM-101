import math
import numpy as np
from scipy.spatial.transform import Rotation


_HOLD_SAMPLE_BUDGET_S = 0.15
_STABLE_SUPPORT_S = 0.06
_PROBE_DZ_M = 0.0005
_MAX_PROBES = 6
_FORCE_THRESHOLD_N = 0.005


def _board_force(board_contacts):
    total = 0.0
    for c in (board_contacts or []):
        if isinstance(c, dict):
            total += float(c.get("normalForce", 0.0))
    return total


def _reference_gap_mm(r):
    target_pos = r.last["target_pos"]
    qc = r.query_cfg
    board_top = qc["board_top_m"]
    floor_off = qc["native_collision_floor_offset_m"]
    return float((target_pos[2] - (board_top + floor_off)) * 1000.0)


def _check_support(r, budget_s):
    """Tick hold until support stable for >=0.06s within budget_s.

    Returns (supported, actual_steps_taken). Preserves pressure via tick('support',...,'hold').
    """
    dt = float(r.actual.dt)
    if dt <= 0:
        raise ValueError("actual.dt must be positive")
    max_steps = max(1, int(math.ceil(budget_s / dt)))
    stable_needed = max(1, int(math.ceil(_STABLE_SUPPORT_S / dt)))
    stable_count = 0
    steps = 0
    while steps < max_steps:
        r.tick("support", list(r.qcmd), float(r.gcmd), "hold")
        steps += 1
        force = _board_force(r.last.get("board_contacts") or [])
        if force > _FORCE_THRESHOLD_N:
            stable_count += 1
            if stable_count >= stable_needed:
                return True, steps
        else:
            stable_count = 0
    return False, steps


def _snapshot_to_T(entry):
    pos = np.asarray(entry["pos"], dtype=float).reshape(3)
    quat = np.asarray(entry["quat"], dtype=float).reshape(4)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat).as_matrix()
    T[:3, 3] = pos
    return T


def confirm_support(r, route_fn, pose_fk_fn):
    # Initial support check (needs to tick hold to keep pressure)
    # We do initial check with 0.15s budget to match spec.
    supported, initial_steps = _check_support(r, _HOLD_SAMPLE_BUDGET_S)
    support_samples = initial_steps
    if supported:
        return {
            "ok": True,
            "reason": "initially_supported",
            "probes": [],
            "support_samples": support_samples,
            "reference_gap_mm": _reference_gap_mm(r),
        }

    probe_index = 0
    cumulative_requested_mm = 0.0
    total_budget_mm = _PROBE_DZ_M * 1000.0 * _MAX_PROBES
    probes = []
    last_reference_gap = _reference_gap_mm(r)

    while probe_index < _MAX_PROBES:
        probe_index += 1
        snapshot = r.actual.snapshot()
        entry = snapshot[r.target_key]
        Tpiece = _snapshot_to_T(entry)
        qact = list(r.last["actual_q5"])
        gact = float(r.last["actual_g"])
        r.geo.reset(r.case, snapshot, qact, gact)
        T = pose_fk_fn(r.kin, qact)
        H = np.linalg.inv(T) @ Tpiece

        Goal = np.array(T, dtype=float, copy=True)
        Goal[2, 3] -= _PROBE_DZ_M

        nodes = route_fn(
            r.wq, r.geo, r.kin,
            T.tolist(), Goal.tolist(),
            qact, gact, "lower",
            H=H.tolist(),
        )
        if not nodes:
            return {
                "ok": False,
                "reason": "no_probe_route",
                "probes": probes,
                "support_samples": support_samples,
                "reference_gap_mm": last_reference_gap,
            }

        last_node_q = list(nodes[-1]["q"])
        T_end = pose_fk_fn(r.kin, last_node_q)
        predicted_dz = float(T_end[2, 3] - T[2, 3])
        requested_dz_mm = -(_PROBE_DZ_M * 1000.0)

        if predicted_dz >= 0.0:
            return {
                "ok": False,
                "reason": "probe_not_downward",
                "probes": probes,
                "support_samples": support_samples,
                "reference_gap_mm": last_reference_gap,
            }

        cumulative_requested_mm += _PROBE_DZ_M * 1000.0

        r.move_nodes(nodes, "hold")

        new_q = list(r.last["actual_q5"])
        T_after = pose_fk_fn(r.kin, new_q)
        measured_dz_mm = float((T_after[2, 3] - T[2, 3]) * 1000.0)
        force = _board_force(r.last.get("board_contacts") or [])
        ref_gap = _reference_gap_mm(r)
        last_reference_gap = ref_gap

        probes.append({
            "probe_index": probe_index,
            "requested_dz_mm": requested_dz_mm,
            "predicted_dz_mm": float(predicted_dz * 1000.0),
            "measured_dz_mm": measured_dz_mm,
            "cumulative_requested_descent_mm": cumulative_requested_mm,
            "board_force_N": force,
            "reference_gap_mm": ref_gap,
        })

        supported, post_steps = _check_support(r, _HOLD_SAMPLE_BUDGET_S)
        support_samples += post_steps
        if supported:
            return {
                "ok": True,
                "reason": "supported_after_probe",
                "probes": probes,
                "support_samples": support_samples,
                "reference_gap_mm": _reference_gap_mm(r),
            }

        if cumulative_requested_mm >= total_budget_mm - 1e-12:
            break

    return {
        "ok": False,
        "reason": "support_not_confirmed_after_bounded_probe",
        "probes": probes,
        "support_samples": support_samples,
        "reference_gap_mm": last_reference_gap,
    }
