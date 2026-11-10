import contextlib
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from tilt_geometry_v1 import Geometry as BaseGeometry


HARD_TOL = 0.0005


def _as_matrix(pose):
    if isinstance(pose, dict):
        pos = pose.get("pos")
        quat = pose.get("quat")
    else:
        pos = pose["pos"] if isinstance(pose, dict) else pose[0]
        quat = pose["quat"] if isinstance(pose, dict) else pose[1]
    pos = np.asarray(pos, dtype=float).reshape(3)
    quat = np.asarray(quat, dtype=float).reshape(4)
    norm = np.linalg.norm(quat)
    if norm > 0.0:
        quat = quat / norm
    rot = Rotation.from_quat(quat).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def _pose_lists(pose):
    if isinstance(pose, dict):
        pos = pose.get("pos")
        quat = pose.get("quat")
    else:
        pos = pose[0]
        quat = pose[1]
    return [float(x) for x in np.asarray(pos, dtype=float).reshape(3)], [
        float(x) for x in np.asarray(quat, dtype=float).reshape(4)
    ]


def pose_matrix(d):
    return _as_matrix(d)


def pose_dict(T):
    T = np.asarray(T, dtype=float).reshape(4, 4)
    rot = Rotation.from_matrix(T[:3, :3]).as_quat()
    return {"pos": [float(x) for x in T[:3, 3]], "quat": [float(x) for x in rot]}


def tilt_deg(T):
    T = np.asarray(T, dtype=float).reshape(4, 4)
    axis = T[:3, :3].dot(np.array([0.0, 0.0, 1.0]))
    n = np.linalg.norm(axis)
    if n == 0.0:
        return 90.0
    c = float(np.clip(axis[2] / n, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _set_piece(w, pose):
    pos, quat = _pose_lists(pose if isinstance(pose, dict) else pose_dict(pose))
    w.set_query_object(w.case["target_key"], pos, quat)


def _set_robot(w, q, g):
    w.set_query_robot([float(x) for x in np.asarray(q, dtype=float).reshape(5)], float(g))


def _min_closest(cps, fallback=None):
    if not cps:
        if fallback is not None:
            return fallback
        raise ValueError("getClosestPoints returned no contacts")
    best = None
    for cp in cps:
        v = float(cp[8])
        if best is None or v < best:
            best = v
    return best


def board_gap(w, Tpiece):
    _set_piece(w, Tpiece)
    target = w.bodies[w.case["target_key"]]
    cps = p.getClosestPoints(
        bodyA=target,
        bodyB=w.board,
        distance=0.10,
        physicsClientId=w.client,
    )
    return _min_closest(cps)


def target_clearance(w, q, g, piecepose):
    _set_piece(w, piecepose)
    _set_robot(w, q, g)
    target = w.bodies[w.case["target_key"]]
    cps = p.getClosestPoints(
        bodyA=w.robot,
        bodyB=target,
        distance=0.05,
        physicsClientId=w.client,
    )
    if not cps:
        return 0.05
    return _min_closest(cps, fallback=0.05)


@contextlib.contextmanager
def no_step():
    sentinel = object()
    original = getattr(p, "stepSimulation", sentinel)
    had = original is not sentinel

    def _forbidden(*args, **kwargs):
        raise AssertionError("query_physics_forbidden")

    p.stepSimulation = _forbidden
    try:
        yield
    finally:
        if had:
            p.stepSimulation = original
        else:
            try:
                del p.stepSimulation
            except AttributeError:
                pass


class Geometry(BaseGeometry):
    def check(self, q, g, phase, targetpose=None):
        result = super().check(q, g, phase, targetpose=targetpose)
        if not isinstance(result, dict) or not isinstance(result.get("ok"), bool):
            raise TypeError("Geometry verdict must contain bool ok")
        if result["ok"] is not True:
            return result
        if phase in ("lift", "transfer", "lower") and targetpose is not None:
            angle = tilt_deg(pose_matrix(targetpose))
            if angle > 10.0 + 1e-7:
                return {"ok": False, "reason": "planned_piece_tilt %.6f" % angle, "clearance_mm": result.get("clearance_mm")}
        if phase == "empty_home":
            if not isinstance(targetpose, dict):
                raise TypeError("empty_home requires stationary target pose")
            gap = target_clearance(self.w, q, g, targetpose)
            verdict = dict(result)
            verdict["clearance_mm"] = min(float(result.get("clearance_mm", 2.0)), 1000.0 * gap)
            if gap < 0.0005:
                verdict.update(ok=False, reason="empty_home target clearance %.9f" % gap)
            return verdict
        return result
