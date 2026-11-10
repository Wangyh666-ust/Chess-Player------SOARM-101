import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from core import _homogeneous, _axis_rotation


def pose_fk(robot, q):
    T = np.eye(4)
    for i in range(5):
        j = robot.joints[i]
        T = T @ j['origin'] @ _homogeneous(_axis_rotation(j['axis'], q[i]), np.zeros(3))
    T = T @ robot.joints[5]['origin']
    return T


def solve_pose(robot, pos, quat, seed):
    pos = np.asarray(pos, dtype=float)
    Rgoal = Rotation.from_quat(quat).as_matrix()
    lo = robot.lo + np.deg2rad(5.01)
    hi = robot.hi - np.deg2rad(5.01)
    seed = np.asarray(seed, dtype=float)
    q0 = np.clip(seed, lo, hi)

    def resid(q):
        T = pose_fk(robot, q)
        p = T[:3, 3]
        R = T[:3, :3]
        rv = Rotation.from_matrix(Rgoal.T @ R).as_rotvec()
        return np.concatenate([1000.0 * (p - pos), 60.0 * rv, 0.005 * (q - seed)])

    res = least_squares(resid, q0, bounds=(lo, hi), max_nfev=80,
                        xtol=1e-8, ftol=1e-8, gtol=1e-8)
    q = res.x

    T = pose_fk(robot, q)
    p = T[:3, 3]
    R = T[:3, :3]
    error_mm = float(np.linalg.norm(p - pos) * 1000.0)
    rot_error_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(Rgoal.T @ R).as_rotvec())))
    tool_z = R[:3, 2]
    world_down = np.array([0.0, 0.0, -1.0])
    tilt_deg = float(np.degrees(np.arccos(np.clip(np.dot(tool_z, world_down), -1.0, 1.0))))
    margin_deg = float(robot.margin(q))

    ok = (error_mm <= 5.0 and rot_error_deg <= 10.0 and
          tilt_deg <= 10.0 and margin_deg >= 5.0)
    return {
        'q': [float(v) for v in q],
        'error_mm': error_mm,
        'rot_error_deg': rot_error_deg,
        'tilt_deg': tilt_deg,
        'margin_deg': margin_deg,
        'pass': bool(ok),
    }
