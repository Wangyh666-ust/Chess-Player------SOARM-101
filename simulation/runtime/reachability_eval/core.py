"""Reachability evaluation core for chess-piece pick-and-place.

Pure library module: URDF kinematics, catalog building, IK solving, piece
geometry normalisation, board/target generation and deterministic self-checks.

All quantities are SI (meters, radians) unless the name carries a
``_mm``/``_deg`` suffix.  Approximate catalog matching is NEVER a proof of
unreachability; only exact FK/IK checks on the true model are meaningful.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path

import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

_PI = 3.14159
_HALF = 0.5
_EPS_INSET = np.deg2rad(0.25)


def _rpy_to_matrix(roll, pitch, yaw):
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_rotation(axis, q):
    """Rodrigues rotation about a unit axis."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n == 0:
        return np.eye(3)
    a = a / n
    c, s = math.cos(q), math.sin(q)
    kx, ky, kz = a
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _homogeneous(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _normalize(v, fallback=None):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        if fallback is None:
            raise ValueError("cannot normalise zero vector")
        return np.asarray(fallback, dtype=float)
    return v / n


def _rot_x180():
    return np.diag([1.0, -1.0, -1.0])


# --------------------------------------------------------------------------
# Robot
# --------------------------------------------------------------------------


class Robot:
    """URDF kinematic chain from base_link to gripper_frame_link.

    The chain is expected to contain five revolute joints followed by one
    fixed tool joint (the fixed joint is stored as ``type='fixed'`` with no
    degree of freedom).
    """

    def __init__(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        joints = []
        for j in root.findall("joint"):
            parent = j.find("parent").get("link")
            child = j.find("child").get("link")
            jtype = j.get("type")
            origin = j.find("origin")
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
            if origin is not None:
                if origin.get("xyz"):
                    xyz = [float(v) for v in origin.get("xyz").split()]
                if origin.get("rpy"):
                    rpy = [float(v) for v in origin.get("rpy").split()]
            R = _rpy_to_matrix(rpy[0], rpy[1], rpy[2])
            T = _homogeneous(R, xyz)
            axis = [0.0, 0.0, 1.0]
            if jtype == "revolute" or jtype == "continuous":
                ax = j.find("axis")
                if ax is not None and ax.get("xyz"):
                    axis = [float(v) for v in ax.get("xyz").split()]
            lo, hi = 0.0, 0.0
            if jtype == "revolute":
                lim = j.find("limit")
                if lim is None:
                    raise ValueError(f"revolute joint missing limit: {j.get('name')}")
                lo = float(lim.get("lower"))
                hi = float(lim.get("upper"))
            joints.append(
                {
                    "name": j.get("name"),
                    "origin": T,
                    "axis": np.asarray(axis, dtype=float),
                    "type": jtype,
                    "parent": parent,
                    "child": child,
                    "lo": lo,
                    "hi": hi,
                }
            )

        child_to_joint = {j["child"]: j for j in joints}
        chain = []
        link = "gripper_frame_link"
        guard = 0
        while link != "base_link":
            if link not in child_to_joint:
                raise ValueError(f"no joint chaining to base_link from {link}")
            j = child_to_joint[link]
            chain.append(j)
            link = j["parent"]
            guard += 1
            if guard > 100:
                raise ValueError("joint chain too deep / cycle")
        chain.reverse()

        movable = [j for j in chain if j["type"] in ("revolute", "continuous")]
        fixed = [j for j in chain if j["type"] == "fixed"]
        if len(movable) != 5:
            raise ValueError(f"expected 5 revolute joints, found {len(movable)}")
        if len(fixed) != 1:
            raise ValueError(f"expected 1 fixed tool joint, found {len(fixed)}")
        self.joints = movable + fixed
        self.names = [j["name"] for j in movable]
        self.lo = np.array([j["lo"] for j in movable], dtype=float)
        self.hi = np.array([j["hi"] for j in movable], dtype=float)
        self._fixed_index = len(movable)
        self._fixed_origin = self.joints[self._fixed_index]["origin"]
        self.model_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    # -- kinematics -------------------------------------------------------

    def _state(self, q, need_jac=False):
        q = np.asarray(q, dtype=float)
        if q.shape != (5,):
            raise ValueError("q must have shape (5,)")
        T = np.eye(4)
        points = [T[:3, 3].copy()]
        axes_w = []
        origins_w = []
        for i in range(5):
            j = self.joints[i]
            T = T @ j["origin"]
            origin = T[:3, 3].copy()
            axis_world = T[:3, :3] @ _normalize(j["axis"])
            origins_w.append(origin)
            axes_w.append(axis_world)
            T = T @ _homogeneous(_axis_rotation(j["axis"], q[i]), np.zeros(3))
            points.append(T[:3, 3].copy())
        T = T @ self._fixed_origin
        points.append(T[:3, 3].copy())
        p = T[:3, 3].copy()
        axis = (T[:3, :3] @ np.array([0.0, 0.0, 1.0])).copy()
        pts = np.asarray(points)
        if not need_jac:
            return p, axis, pts
        axes_w = np.asarray(axes_w)
        origins_w = np.asarray(origins_w)
        Jp = np.zeros((3, 5))
        Ja = np.zeros((3, 5))
        for i in range(5):
            a = axes_w[i]
            Jp[:, i] = np.cross(a, p - origins_w[i])
            Ja[:, i] = np.cross(a, axis)
        return p, axis, pts, Jp, Ja

    def fk(self, q):
        p, axis, pts = self._state(q, need_jac=False)
        return p, axis, pts

    def jac(self, q):
        _p, _axis, _pts, Jp, Ja = self._state(q, need_jac=True)
        return Jp, Ja

    def batch(self, q):
        q = np.asarray(q, dtype=float)
        if q.ndim != 2 or q.shape[1] != 5:
            raise ValueError("q must have shape (N,5)")
        n = q.shape[0]
        R = np.tile(np.eye(3), (n, 1, 1))
        p = np.zeros((n, 3))
        for i in range(5):
            j = self.joints[i]
            Ro = j["origin"][:3, :3]
            po = j["origin"][:3, 3]
            p = p + np.einsum("nij,j->ni", R, po)
            R = R @ Ro
            a = _normalize(j["axis"])
            ang = q[:, i]
            c = np.cos(ang)
            s = np.sin(ang)
            K = np.array(
                [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]
            )
            Rloc = (
                np.eye(3)[None]
                + s[:, None, None] * K[None]
                + (1 - c)[:, None, None] * (K @ K)[None]
            )
            R = R @ Rloc
        Ro = self._fixed_origin[:3, :3]
        po = self._fixed_origin[:3, 3]
        p = p + np.einsum("nij,j->ni", R, po)
        R = R @ Ro
        axis = np.einsum("nij,j->ni", R, np.array([0.0, 0.0, 1.0]))
        return p, axis

    def margin(self, q):
        q = np.asarray(q, dtype=float)
        if q.ndim == 1:
            lo_d = q - self.lo
            hi_d = self.hi - q
            d = min(lo_d.min(), hi_d.min())
            return math.degrees(d)
        lo_d = q - self.lo[None, :]
        hi_d = self.hi[None, :] - q
        d = np.minimum(lo_d.min(axis=1), hi_d.min(axis=1))
        return np.degrees(d)


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------


def _q4_for_down(robot, q123, wristroll):
    """Analytic wristflex that maximises dot(tool_z, down).

    Evaluated on a temporary chain with wristflex in {0, pi/2, pi} and the
    given wristroll; may return orientations outside the true limits.
    """
    q_base = np.array([q123[0], q123[1], q123[2], 0.0, wristroll])

    def dot_at(q4):
        q = q_base.copy()
        q[3] = q4
        _p, axis, _pts = robot.fk(q)
        return float(np.dot(axis, np.array([0.0, 0.0, -1.0])))

    d0 = dot_at(0.0)
    d90 = dot_at(math.pi / 2.0)
    dpi = dot_at(math.pi)
    c = (d0 + dpi) / 2.0
    a = (d0 - dpi) / 2.0
    b = d90 - c
    return math.atan2(b, a)


def build_catalog(robot, step_deg=2, pan_step_deg=3, cache_path=None):
    t0 = time.time()
    step = math.radians(step_deg)
    pan_step = math.radians(pan_step_deg)
    inset = _EPS_INSET

    j0 = robot.joints[0]
    pan_lo = j0["lo"] + inset
    pan_hi = j0["hi"] - inset
    if pan_hi < pan_lo:
        raise ValueError("shoulder pan range empty after inset")

    flex_lo = robot.lo[3] + inset
    flex_hi = robot.hi[3] - inset
    roll_lo = robot.lo[4] + inset
    roll_hi = robot.hi[4] - inset
    if flex_lo >= flex_hi:
        raise ValueError("wrist flex range empty after inset")

    grid_args = (
        float(step_deg),
        float(pan_step_deg),
        float(pan_lo),
        float(pan_hi),
        float(robot.lo[1] + inset),
        float(robot.hi[1] - inset),
        float(robot.lo[2] + inset),
        float(robot.hi[2] - inset),
        float(flex_lo),
        float(flex_hi),
        float(roll_lo),
        float(roll_hi),
    )
    digest = hashlib.sha256()
    digest.update(robot.model_hash.encode())
    digest.update(np.array(grid_args).tobytes())
    digest = digest.hexdigest()

    if cache_path and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=False)
            if str(data["hash"]) == digest:
                xyz = data["xyz"]
                q = data["q"]
                tilt = data["tilt"]
                margin = data["margin"]
                if (
                    xyz.ndim == 2 and xyz.shape[1] == 3
                    and q.ndim == 2 and q.shape[1] == 5
                    and len(q) == len(xyz) == len(tilt) == len(margin)
                    and np.all(np.isfinite(xyz))
                    and np.all(np.isfinite(q))
                    and np.all(np.isfinite(tilt))
                    and np.all(q >= robot.lo[None, :] - 1e-9)
                    and np.all(q <= robot.hi[None, :] + 1e-9)
                    and np.all(tilt <= math.radians(5.0) + 1e-9)
                    and len(q) > 0
                ):
                    print(
                        f"[catalog] loaded cache {cache_path} "
                        f"points={len(q)} time={time.time()-t0:.2f}s",
                        flush=True,
                    )
                    return _catalog_result(robot, xyz, q, tilt, margin)
        except Exception as exc:
            print(f"[catalog] cache unreadable ({exc}); rebuilding", flush=True)

    def _grid(lo, hi, st):
        if st <= 0:
            raise ValueError("st must be positive")
        if hi < lo:
            raise ValueError("hi must be >= lo")
        return np.linspace(lo, hi, int(math.ceil((hi - lo) / st)) + 1)
    pan_vals = _grid(pan_lo, pan_hi, pan_step)
    lift_vals = _grid(robot.lo[1] + inset, robot.hi[1] - inset, step)
    elbow_vals = _grid(robot.lo[2] + inset, robot.hi[2] - inset, step)

    if pan_vals.size == 0 or lift_vals.size == 0 or elbow_vals.size == 0:
        raise ValueError("empty joint grid after inset")

    roll_offsets = []
    for off_deg in (-90.0, 0.0, 90.0):
        val = math.radians(off_deg)
        if roll_lo - 1e-12 <= val <= roll_hi + 1e-12:
            roll_offsets.append(val)
    if not roll_offsets:
        roll_offsets = [0.0]

    down = np.array([0.0, 0.0, -1.0])
    tilt_limit = math.radians(5.0)
    z_limit = -0.20
    chunk = 50000

    Q_list = []
    XYZ_list = []
    TILT_list = []

    # meshgrid order: (pan, lift, elbow) -> flatten
    P, L, E = np.meshgrid(pan_vals, lift_vals, elbow_vals, indexing="ij")
    q123_all = np.stack([P.ravel(), L.ravel(), E.ravel()], axis=1)
    total = q123_all.shape[0]

    processed = 0
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        q123 = q123_all[start:stop]
        n = q123.shape[0]
        qN = np.zeros((n, 5))
        qN[:, :3] = q123

        # base at q4 in {0, pi/2, pi}, roll 0
        base_q = np.zeros((3 * n, 5))
        base_q[:n, :3] = q123
        base_q[:n, 3] = 0.0
        base_q[n:2 * n, :3] = q123
        base_q[n:2 * n, 3] = math.pi / 2.0
        base_q[2 * n:, :3] = q123
        base_q[2 * n:, 3] = math.pi
        p_base, axis_base = robot.batch(base_q)
        d_all = axis_base @ down
        d0 = d_all[:n]
        d90 = d_all[n:2 * n]
        dpi = d_all[2 * n:]
        c = 0.5 * (d0 + dpi)
        a = 0.5 * (d0 - dpi)
        b = d90 - c
        q4 = np.arctan2(b, a)

        for roll in roll_offsets:
            valid_flex = (q4 >= flex_lo - 1e-12) & (q4 <= flex_hi + 1e-12)
            if not np.any(valid_flex):
                continue
            idx = np.nonzero(valid_flex)[0]
            qv = np.zeros((idx.size, 5))
            qv[:, :3] = q123[idx]
            qv[:, 3] = q4[idx]
            qv[:, 4] = roll
            ppos, axis = robot.batch(qv)
            d = axis @ down
            d = np.clip(d, -1.0, 1.0)
            tilt = np.arccos(d)
            ok = (tilt <= tilt_limit) & (ppos[:, 2] > z_limit)
            if not np.any(ok):
                continue
            Q_list.append(qv[ok])
            XYZ_list.append(ppos[ok])
            TILT_list.append(tilt[ok])

        processed = stop
        if processed % 100000 < chunk or processed == total:
            print(
                f"[catalog] chunk {processed}/{total} kept={sum(len(a) for a in Q_list)}",
                flush=True,
            )

    if not Q_list:
        raise ValueError("empty catalog: no reachable samples")

    q = np.concatenate(Q_list, axis=0)
    xyz = np.concatenate(XYZ_list, axis=0)
    tilt = np.concatenate(TILT_list, axis=0)
    margin = robot.margin(q)

    if cache_path:
        try:
            np.savez_compressed(
                cache_path,
                hash=np.array(digest),
                xyz=xyz,
                q=q,
                tilt=tilt,
                margin=margin,
            )
        except Exception as exc:
            print(f"[catalog] cache write failed: {exc}", flush=True)

    print(
        f"[catalog] points={len(q)} time={time.time()-t0:.2f}s",
        flush=True,
    )
    return _catalog_result(robot, xyz, q, tilt, margin)


def _catalog_result(robot, xyz, q, tilt, margin):
    tree = cKDTree(xyz)
    robust_mask = margin >= 5.0
    robust_indices = np.nonzero(robust_mask)[0]
    if len(robust_indices) > 0:
        robust_tree = cKDTree(xyz[robust_indices])
    else:
        robust_tree = cKDTree(np.zeros((0, 3)))
    return {
        "xyz": xyz,
        "q": q,
        "tilt": tilt,
        "margin": margin,
        "tree": tree,
        "robust_tree": robust_tree,
        "robust_indices": robust_indices,
    }


# --------------------------------------------------------------------------
# solve
# --------------------------------------------------------------------------


def solve(robot, target, seed_array, desired_axis=None, max_nfev=80):
    target = np.asarray(target, dtype=float)
    if target.shape != (3,):
        raise ValueError("target must be shape (3,)")
    if desired_axis is None:
        desired_axis = np.array([0.0, 0.0, -1.0])
    desired_axis = _normalize(desired_axis)
    seeds = np.asarray(seed_array, dtype=float)
    if seeds.ndim != 2 or seeds.shape[1] != 5:
        raise ValueError("seed_array must be shape (N,5)")
    if seeds.shape[0] == 0:
        raise ValueError("seed_array must be non-empty")

    inset = _EPS_INSET
    lb = robot.lo + inset
    ub = robot.hi - inset
    if np.any(lb >= ub):
        raise ValueError("joint bounds empty after inset")

    cache = {"q": None, "state": None}

    def _eval_at(qv):
        qv = np.asarray(qv, dtype=float)
        if cache["q"] is not None and np.array_equal(cache["q"], qv):
            return cache["state"]
        p, axis, _pts, Jp, Ja = robot._state(qv, need_jac=True)
        state = (p, axis, Jp, Ja)
        cache["q"] = qv.copy()
        cache["state"] = state
        return state

    def residual(qv):
        p, axis, _Jp, _Ja = _eval_at(qv)
        r_pos = 1000.0 * (p - target)
        r_axis = 100.0 * (axis - desired_axis)
        return np.concatenate([r_pos, r_axis])

    def jacobian(qv):
        _p, _axis, Jp, Ja = _eval_at(qv)
        return np.vstack([1000.0 * Jp, 100.0 * Ja])

    results = []
    for s in seeds:
        s_clipped = np.clip(s, lb, ub)
        try:
            sol = least_squares(
                residual,
                s_clipped,
                jac=jacobian,
                bounds=(lb, ub),
                max_nfev=max_nfev,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
            )
        except Exception as exc:
            raise RuntimeError(f"least_squares failed: {exc}") from exc
        q = sol.x
        p, axis, _ = robot.fk(q)
        err = p - target
        err_pos_mm = float(np.linalg.norm(err) * 1000.0)
        d = float(np.dot(axis, desired_axis))
        d = max(-1.0, min(1.0, d))
        tilt_deg = math.degrees(math.acos(d))
        m = float(robot.margin(q))
        if not (math.isfinite(err_pos_mm) and math.isfinite(tilt_deg) and math.isfinite(m)):
            continue
        pass_geom = err_pos_mm <= 2.0 and tilt_deg <= 5.0
        pass_margin = pass_geom and m >= 5.0
        entry = {
            "q": [float(x) for x in q],
            "error_mm": err_pos_mm,
            "tilt_deg": tilt_deg,
            "margin_deg": m,
            "pass_geometry": bool(pass_geom),
            "pass_margin": bool(pass_margin),
            "_score": (
                0 if pass_margin else (1 if pass_geom else 2),
                max(err_pos_mm / 2.0, tilt_deg / 5.0),
                err_pos_mm + tilt_deg,
            ),
        }
        if (
            err_pos_mm <= 0.2
            and tilt_deg <= 3.0
            and m >= 7.0
        ):
            entry.pop("_score", None)
            return entry
        results.append(entry)

    if not results:
        raise RuntimeError("solve: no finite candidate produced")
    results.sort(key=lambda r: r["_score"])
    best = results[0]
    best.pop("_score", None)
    return best


# --------------------------------------------------------------------------
# Pieces
# --------------------------------------------------------------------------


_STL_NAMES = {
    "king": "obj_1",
    "queen": "obj_2",
    "knight": "obj_3",
    "rook": "obj_4",
    "pawn": "obj_5",
    "bishop": "obj_13",
}
_BISHOP_HEAD = "obj_12"


def _read_stl_bounds(path):
    import vtk

    reader = vtk.vtkSTLReader()
    reader.SetFileName(path)
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise ValueError(f"STL has no points: {path}")
    b = polydata.GetBounds()  # xmin,xmax,ymin,ymax,zmin,zmax
    return b


def load_pieces(folder):
    folder = Path(folder)
    pieces = {}
    for ptype, stem in _STL_NAMES.items():
        source_hashes = {}
        matches = sorted(folder.glob(stem + "_*.stl"))
        if len(matches) == 0:
            raise FileNotFoundError(f"no STL matching {stem}_*.stl in {folder}")
        if len(matches) > 1:
            raise AssertionError(
                f"expected exactly one STL for {stem}, found {len(matches)}"
            )
        fpath = matches[0]
        b = _read_stl_bounds(str(fpath))
        h = b[5] - b[4]
        w = max(b[1] - b[0], b[3] - b[2])
        source_hashes[fpath.name] = hashlib.sha256(fpath.read_bytes()).hexdigest()
        if ptype == "bishop":
            head_matches = sorted(folder.glob(_BISHOP_HEAD + "_*.stl"))
            if len(head_matches) == 0:
                raise FileNotFoundError(
                    f"no STL matching {_BISHOP_HEAD}_*.stl in {folder}"
                )
            if len(head_matches) > 1:
                raise AssertionError(
                    f"expected exactly one STL for {_BISHOP_HEAD}, "
                    f"found {len(head_matches)}"
                )
            head_path = head_matches[0]
            hb = _read_stl_bounds(str(head_path))
            head_x_width = hb[1] - hb[0]
            head_w = max(hb[1] - hb[0], hb[3] - hb[2])
            h = h + head_x_width
            w = max(w, head_w)
            source_hashes[head_path.name] = hashlib.sha256(
                head_path.read_bytes()
            ).hexdigest()
        pieces[ptype] = {
            "raw_height_mm": h,
            "raw_width_mm": w,
            "name": fpath.name,
            "source_hashes": source_hashes,
        }
    return pieces


def scaled_pieces(raw, square_mm):
    king_h = raw["king"]["raw_height_mm"]
    if king_h <= 0:
        raise ValueError("king raw height must be positive")
    king_desired = 1.56 * square_mm
    scale = king_desired / king_h

    per_type = {}
    for ptype in ("king", "queen", "knight", "rook", "pawn", "bishop"):
        info = raw[ptype]
        h = info["raw_height_mm"] * scale
        w = info["raw_width_mm"] * scale
        grasp = float(np.clip(0.60 * h, 10.0, h - 6.0))
        per_type[ptype] = {
            "height_mm": h,
            "width_mm": w,
            "grasp_mm": grasp,
            "raw_scale": scale,
        }
    max_h = max(v["height_mm"] for v in per_type.values())
    for v in per_type.values():
        v["carry_above_board_mm"] = max_h + v["grasp_mm"] + 10.0
    return per_type


# --------------------------------------------------------------------------
# Board / targets
# --------------------------------------------------------------------------


def _rot2(theta_deg):
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


def board_xy(square_mm, cx_mm, cy_mm, theta_deg):
    s = square_mm
    cx, cy = cx_mm, cy_mm
    R = _rot2(theta_deg)
    out = np.zeros((64, 2))
    ranks = "abcdefgh"
    idx = 0
    for r in range(8):
        for f in range(8):
            local = np.array([(f - 3.5) * s, (r - 3.5) * s])
            world = R @ local
            out[idx, 0] = (cx + world[0]) / 1000.0
            out[idx, 1] = (cy + world[1]) / 1000.0
            idx += 1
            _ = ranks
    return out


def board_polygon(square_mm, cx_mm, cy_mm, theta_deg):
    s = square_mm
    R = _rot2(theta_deg)
    corners = np.array(
        [
            [-4 * s, -4 * s],
            [4 * s, -4 * s],
            [4 * s, 4 * s],
            [-4 * s, 4 * s],
        ]
    )
    world = (R @ corners.T).T + np.array([cx_mm, cy_mm])
    return world


def overlaps_base(poly):
    poly = np.asarray(poly, dtype=float)
    if poly.shape != (4, 2):
        raise ValueError("poly must be 4x2")
    rect = np.array([[-75.0, -65.0], [85.0, -65.0], [85.0, 65.0], [-75.0, 65.0]])

    def axes(p):
        out = []
        for i in range(4):
            e = p[(i + 1) % 4] - p[i]
            n = np.array([-e[1], e[0]])
            ln = np.linalg.norm(n)
            if ln > 0:
                out.append(n / ln)
        return out

    for ax in axes(poly) + axes(rect):
        pa = poly @ ax
        ra = rect @ ax
        if pa.max() < ra.min() - 1e-12 or ra.max() < pa.min() - 1e-12:
            return False
    return True


def targets(layout, pieces):
    square_mm = layout["square_mm"]
    cx = layout["cx_mm"]
    cy = layout["cy_mm"]
    theta = layout["theta_deg"]
    board_z = layout["board_z_mm"]

    coords = board_xy(square_mm, cx, cy, theta)
    square_names = []
    for r in range(8):
        for f in range(8):
            square_names.append(f"{chr(ord('a')+f)}{r+1}")

    ptypes = sorted(pieces.keys())

    xyz = []
    labels = []
    for i, sq in enumerate(square_names):
        bx, by = coords[i]
        for ptype in ptypes:
            info = pieces[ptype]
            grasp = info["grasp_mm"]
            carry = info["carry_above_board_mm"]
            grasp_z = (board_z + grasp) / 1000.0
            carry_z = (board_z + carry) / 1000.0
            xyz.append([bx, by, grasp_z])
            labels.append(
                {
                    "square": sq,
                    "type": ptype,
                    "stage": "grasp",
                    "grasp_mm": grasp,
                }
            )
            xyz.append([bx, by, carry_z])
            labels.append(
                {
                    "square": sq,
                    "type": ptype,
                    "stage": "lift",
                    "grasp_mm": grasp,
                }
            )
    return np.asarray(xyz), labels


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


def verify(robot, raw):
    metrics = {}

    rng = np.random.default_rng(12345)

    # batch vs scalar FK
    max_err_fk = 0.0
    qs = np.zeros((50, 5))
    for i in range(50):
        qs[i] = robot.lo + rng.random(5) * (robot.hi - robot.lo)
    bp, bax = robot.batch(qs)
    for i in range(50):
        p, a, _pts = robot.fk(qs[i])
        max_err_fk = max(max_err_fk, float(np.abs(p - bp[i]).max()))
        max_err_fk = max(max_err_fk, float(np.abs(a - bax[i]).max()))
    metrics["fk_batch_max_err"] = max_err_fk
    assert max_err_fk < 1e-10, "batch/scalar FK mismatch"

    # jacobian vs central finite differences
    max_jac = 0.0
    eps = 1e-6
    for i in range(10):
        q = robot.lo + rng.random(5) * (robot.hi - robot.lo)
        Jp, Ja = robot.jac(q)
        Jp_fd = np.zeros((3, 5))
        Ja_fd = np.zeros((3, 5))
        for k in range(5):
            if q[k] - eps < robot.lo[k] or q[k] + eps > robot.hi[k]:
                continue
            qp = q.copy()
            qm = q.copy()
            qp[k] += eps
            qm[k] -= eps
            pp, ap, _ = robot.fk(qp)
            pm, am, _ = robot.fk(qm)
            Jp_fd[:, k] = (pp - pm) / (2 * eps)
            Ja_fd[:, k] = (ap - am) / (2 * eps)
        max_jac = max(max_jac, float(np.abs(Jp - Jp_fd).max()))
        max_jac = max(max_jac, float(np.abs(Ja - Ja_fd).max()))
    metrics["jac_max_err"] = max_jac
    assert max_jac < 1e-6, "jacobian finite-difference mismatch"

    # roll invariance under TCP rpy rounding: rotating tool about its own
    # Z (wrist roll) should leave TCP z-axis dot-down constant up to
    # rounding tolerance 1e-5.
    worst = 0.0
    for i in range(10):
        q4 = rng.uniform(robot.lo[3], robot.hi[3])
        q = np.array(
            [
                rng.uniform(robot.lo[0], robot.hi[0]),
                rng.uniform(robot.lo[1], robot.hi[1]),
                rng.uniform(robot.lo[2], robot.hi[2]),
                q4,
                0.0,
            ]
        )
        _, a0, _ = robot.fk(q)
        q[4] = robot.hi[4]
        _, a1, _ = robot.fk(q)
        worst = max(worst, float(np.linalg.norm(a0 - a1)))
    metrics["roll_axis_rounding"] = worst
    assert worst <= 1e-5 + 1e-9, "roll axis rounding exceeds tolerance"

    # analytic q4 down-projection vs sampled
    best_gain = 0.0
    for i in range(10):
        q123 = np.array(
            [
                rng.uniform(robot.lo[0], robot.hi[0]),
                rng.uniform(robot.lo[1], robot.hi[1]),
                rng.uniform(robot.lo[2], robot.hi[2]),
            ]
        )
        q4_star = _q4_for_down(robot, q123, 0.0)
        base = np.array([q123[0], q123[1], q123[2], q4_star, 0.0])
        _, ax_star, _ = robot.fk(base)
        d_star = float(np.dot(ax_star, np.array([0.0, 0.0, -1.0])))
        samples = np.linspace(0.0, 2 * math.pi, 91, endpoint=False)
        d_best = -2.0
        for s in samples:
            qv = base.copy()
            qv[3] = s
            _, axs, _ = robot.fk(qv)
            d_best = max(d_best, float(np.dot(axs, np.array([0.0, 0.0, -1.0]))))
        best_gain = max(best_gain, d_best - d_star)
    metrics["q4_vs_sampled_gain"] = best_gain
    assert best_gain <= 1e-6, "analytic q4 worse than sampled projection"

    # board basis and 7-square diagonal
    s = 50.0
    xy = board_xy(s, 0.0, 0.0, 0.0)
    a1 = xy[0]
    assert np.allclose(a1, [-3.5 * s / 1000.0, -3.5 * s / 1000.0])
    h8 = xy[63]
    assert np.allclose(h8, [3.5 * s / 1000.0, 3.5 * s / 1000.0])
    diag = np.linalg.norm(h8 - a1) * 1000.0
    assert abs(diag - 7 * s * math.sqrt(2)) < 1e-9, "board diagonal mismatch"
    metrics["board_diagonal_mm"] = diag

    # SAT: rotated intersect and far away not
    rect = np.array([[-75.0, -65.0], [85.0, -65.0], [85.0, 65.0], [-75.0, 65.0]])
    assert overlaps_base(rect), "identical rect should overlap"
    poly_rot = board_polygon(50.0, 0.0, 0.0, 45.0)
    assert overlaps_base(poly_rot), "rotated polygon intersecting base"
    far = board_polygon(50.0, 1000.0, 1000.0, 0.0)
    assert not overlaps_base(far), "far polygon should not overlap"
    metrics["sat_checks"] = True

    # carried bottom = maxH + grasp + 10 for every piece
    scaled = scaled_pieces(raw, 50.0)
    max_h = max(v["height_mm"] for v in scaled.values())
    min_clearance = None
    for ptype, v in scaled.items():
        expected = max_h + v["grasp_mm"] + 10.0
        assert abs(v["carry_above_board_mm"] - expected) < 1e-9, (
            f"carried bottom mismatch for {ptype}"
        )
        clearance = v["carry_above_board_mm"] - v["grasp_mm"] - max_h
        if min_clearance is None or clearance < min_clearance:
            min_clearance = clearance
    metrics["carried_bottom_mm"] = max_h + 10.0
    metrics["carried_bottom_min_clearance_mm"] = float(min_clearance)
    assert abs(min_clearance - 10.0) < 1e-9, "carried bottom clearance != 10mm"

    # IK roundtrip on a target generated by FK
    q_true = np.array(
        [
            robot.lo[0] + 0.4 * (robot.hi[0] - robot.lo[0]),
            robot.lo[1] + 0.4 * (robot.hi[1] - robot.lo[1]),
            robot.lo[2] + 0.4 * (robot.hi[2] - robot.lo[2]),
            robot.lo[3] + 0.4 * (robot.hi[3] - robot.lo[3]),
            robot.lo[4] + 0.4 * (robot.hi[4] - robot.lo[4]),
        ]
    )
    p_true, axis_true, _ = robot.fk(q_true)
    seed = q_true + 0.05
    res = solve(robot, p_true, seed[None, :], desired_axis=axis_true)
    metrics["ik_roundtrip_error_mm"] = res["error_mm"]
    metrics["ik_roundtrip_tilt_deg"] = res["tilt_deg"]
    assert res["error_mm"] <= 0.1, "IK roundtrip position error"
    assert res["tilt_deg"] <= 0.1, "IK roundtrip tilt error"

    metrics["model_hash"] = robot.model_hash
    return metrics
