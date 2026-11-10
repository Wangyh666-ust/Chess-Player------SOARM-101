import time
import numpy as np
from scipy.spatial.transform import Rotation

import tilt_route_v1 as legacy
import tilt_route_base_v1 as base
from tilt_ik_v1 import pose_fk, solve_pose
from query_safety_bound_v1 import pose_dict, board_gap, no_step


def transfer_route(w, geo, kin, A, B, seed, g, H):
    with no_step():
        t0 = time.time()
        audits = getattr(w, "transfer_audits", None)
        if audits is None:
            audits = []
        elif not isinstance(audits, list):
            raise TypeError("w.transfer_audits must be a list")

        Ha = np.asarray(H, dtype=float)

        legacy_reason = None
        try:
            nodes = legacy.route(w, geo, kin, A, B, seed, g, 'transfer', H=H)
            audits.append({
                'method': 'cartesian',
                'legacy_reason': None,
                'ok': True,
                'nodecount': len(nodes),
                'minimum_board_gap_m': None,
                'wall_s': float(time.time() - t0),
            })
            w.transfer_audits = audits
            return nodes
        except RuntimeError as e:
            legacy_reason = str(e)

        joint_reason = None
        try:
            Ba = np.asarray(B, dtype=float)
            quat = Rotation.from_matrix(Ba[:3, :3]).as_quat().tolist()
            res = solve_pose(kin, Ba[:3, 3].tolist(), quat, list(seed))
            if not isinstance(res, dict):
                raise TypeError("solve_pose must return dict")
            if "pass" not in res:
                raise TypeError("solve_pose result missing 'pass'")
            if not isinstance(res["pass"], bool):
                raise TypeError("solve_pose 'pass' must be bool")
            if res["pass"] is not True:
                raise RuntimeError(
                    f"solve_pose rejected endpoint B: {res!r}"
                )
            goalq = res.get("q")
            if goalq is None or len(goalq) != 5:
                raise TypeError("solve_pose result 'q' must have length 5")
            goalq = [float(x) for x in goalq]

            points = base._joint_interp(list(seed), goalq)
            minimum_gap = None
            waypoints = []
            for q in points:
                qlist = [float(x) for x in q]
                actualpiece = pose_fk(kin, qlist) @ Ha
                res_geo = geo.check(qlist, g, 'transfer', pose_dict(actualpiece))
                if not isinstance(res_geo, dict) or "ok" not in res_geo:
                    raise TypeError(
                        f"geo.check invalid result: {res_geo!r}"
                    )
                if not isinstance(res_geo["ok"], bool):
                    raise TypeError("geo.check 'ok' must be bool")
                if res_geo["ok"] is not True:
                    raise RuntimeError(
                        f"geo.check failed at q={qlist} "
                        f"reason={res_geo.get('reason')} "
                        f"clearance={res_geo.get('clearance_mm')}"
                    )
                base._table_check(w, qlist)
                gap = float(board_gap(w, actualpiece))
                if gap < 0.01 - 1e-9:
                    raise RuntimeError(
                        f"board_gap too low at q={qlist}: {gap}"
                    )
                if minimum_gap is None or gap < minimum_gap:
                    minimum_gap = gap
                waypoints.append({
                    'phase': 'transfer',
                    'q': list(qlist),
                    'g': float(g),
                })

            audits.append({
                'method': 'joint',
                'legacy_reason': legacy_reason,
                'ok': True,
                'nodecount': len(waypoints),
                'minimum_board_gap_m': minimum_gap,
                'wall_s': float(time.time() - t0),
            })
            w.transfer_audits = audits
            return waypoints
        except RuntimeError as e:
            joint_reason = str(e)

        audits.append({
            'method': 'joint',
            'legacy_reason': legacy_reason,
            'joint_reason': joint_reason,
            'ok': False,
            'nodecount': 0,
            'minimum_board_gap_m': None,
            'wall_s': float(time.time() - t0),
        })
        w.transfer_audits = audits
        raise RuntimeError(
            f"transfer failed: cartesian={legacy_reason}; joint={joint_reason}"
        )
