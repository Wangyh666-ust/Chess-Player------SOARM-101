"""pilot_world_v2.py

Persistent complete-collision PyBullet worlds for SINGLE-STEP pilot.

Holds a pool of 32 pre-generated piece bodies (the standard chess starting
setup) that can be reactivated / parked across independent trials without
reloading the URDFs.  Inherits AuditedWorld so every observation and dynamics
readback follows the audited path.

Notes:
  * This module is geometry / bookkeeping only.  It never steps the physics.
  * set_case / activate_subcase / set_query_robot / set_query_object are
    "setup / query only" methods; they must not be invoked inside a measured
    rollout.
  * No XML is regenerated; the pre-existing full_piece_URDF/{type}_650.urdf
    files are loaded with complete collision hulls and file inertia.
"""

from __future__ import annotations

import copy
from pathlib import Path

import chess
import pybullet as p

from friction_audit_v1 import AuditedWorld
from physics_world import (
    ASSETS,
    JOINT_NAMES,
    GRIPPER_NAME,
    TCP_LINK,
    _parse_srdf_disables,
)

# PyBullet flag: stop loading visual shapes (graphics only; collisions intact).
_IGNORE_VISUAL = getattr(p, "URDF_IGNORE_VISUAL_SHAPES", 1048576)

# type -> expected hull count from geometry.json (king32, queen28, bishop24, ...)
_HULL_COUNTS = {
    "king": 32,
    "queen": 28,
    "bishop": 24,
    "knight": 22,
    "rook": 18,
    "pawn": 16,
}

_PIECE_KINDS = ("king", "queen", "rook", "bishop", "knight", "pawn")


def _load_geometry():
    import json

    path = ASSETS / "geometry.json"
    if path.exists():
        return json.loads(path.read_text())
    # Fall back to the advanced_30_v1 asset geometry used by full_piece_v2.
    alt = ASSETS.parent / "advanced_30_v1" / "assets" / "geometry.json"
    return json.loads(alt.read_text())


class PilotWorld(AuditedWorld):
    """Persistent 32-piece pool with full collision, single-step pilot scope."""

    def __init__(
        self,
        home_q,
        home_g,
        *,
        dt=1.0 / 240.0,
        grip_force=0.05,
        visuals=False,
    ):
        # self.visuals must be set before super().__init__ because _load_robot
        # and _build_piece_multibody both consult it.
        self.visuals = bool(visuals)
        self.home_q = list(home_q)
        self.home_g = float(home_g)

        board = chess.Board()
        pieces = []
        for square, piece in board.piece_map().items():
            pieces.append({
                "key": chess.square_name(square),
                "square": chess.square_name(square),
                "type": chess.piece_name(piece.piece_type),
                "color": bool(piece.color),
            })

        case = {
            "id": "pool",
            "target_key": "e2",
            "source": "e2",
            "dest": "e3",
            "pieces": pieces,
            "fen": board.fen(),
            "uci": "e2e3",
        }

        super().__init__(
            case,
            list(home_q),
            float(home_g),
            dt=dt,
            grip_force=grip_force,
            board_mu=0.6,
            jaw_mu=0.6,
            piece_friction=0.6,
            rolling=0.001,
            spinning=0.001,
        )

        # ----- pool metadata (mass / inertia / com read back from PyBullet) --
        self.pool_records = []
        for piece in case["pieces"]:
            key = piece["key"]
            body = self.bodies[key]
            dyn = p.getDynamicsInfo(body, -1, physicsClientId=self.client)
            mass = float(dyn[0])
            local_inertia = list(dyn[2])
            # self.com[key] is the LOCAL inertial offset from the base link
            # frame. The URDF base frame is the link (bottom) frame, so the
            # local inertial offset equals the geometric com offset.
            com = list(self.com[key])
            self.pool_records.append(
                {
                    "key": key,
                    "body": body,
                    "type": piece["type"],
                    "color": piece["color"],
                    "com": com,
                    "mass": mass,
                    "inertia": local_inertia,
                }
            )

        # ----- capture tray (static box, top surface at z = -0.09) ---------
        half = [0.045, 0.016, 0.0015]
        center = [0.195, 0.16, -0.0915]
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half,
            physicsClientId=self.client,
        )
        if self.visuals:
            rgb = [0.35, 0.35, 0.35, 1.0]
        else:
            rgb = [-1, -1, -1, -1]
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half,
            rgbaColor=rgb,
            physicsClientId=self.client,
        )
        self.tray = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=center,
            physicsClientId=self.client,
        )
        p.changeDynamics(
            self.tray, -1,
            restitution=0.0,
            collisionMargin=5e-5,
            physicsClientId=self.client,
        )
        # lateralFriction in a separate call AFTER collisionMargin/damping,
        # because the combined call silently drops it.
        p.changeDynamics(
            self.tray, -1,
            lateralFriction=1.0,
            physicsClientId=self.client,
        )

        # Cache material readback (single authoritative snapshot).
        self.material_readback = self.runtime_materials()

    # ------------------------------------------------------------------
    # Robot loading
    # ------------------------------------------------------------------
    def _load_robot(self):
        flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
        if not self.visuals:
            flags |= _IGNORE_VISUAL
        self.robot = p.loadURDF(
            str(ASSETS / "robot_windows.urdf"),
            useFixedBase=True,
            flags=flags,
            physicsClientId=self.client,
        )

        self.joint_index = {}
        self.link_index = {}
        n = p.getNumJoints(self.robot, physicsClientId=self.client)
        for i in range(n):
            info = p.getJointInfo(self.robot, i, physicsClientId=self.client)
            jname = info[1].decode("utf-8")
            lname = info[12].decode("utf-8")
            self.joint_index[jname] = i
            self.link_index[lname] = i
        self.link_index["base_link"] = -1

        self.arm_joints = [self.joint_index[n] for n in JOINT_NAMES]
        self.gripper_joint = self.joint_index[GRIPPER_NAME]
        self.tcp_link = self.link_index[TCP_LINK]

        for i in range(n):
            p.changeDynamics(
                self.robot, i,
                lateralFriction=self.friction,
                restitution=0.0,
                collisionMargin=5e-5,
                linearDamping=0.02,
                angularDamping=0.02,
                physicsClientId=self.client,
            )
        p.changeDynamics(
            self.robot, -1,
            lateralFriction=self.friction,
            restitution=0.0,
            collisionMargin=5e-5,
            linearDamping=0.02,
            angularDamping=0.02,
            physicsClientId=self.client,
        )

        srdf_pairs = _parse_srdf_disables(ASSETS / "robot.srdf")
        for l1, l2 in srdf_pairs:
            i1 = self.link_index.get(l1, -1)
            i2 = self.link_index.get(l2, -1)
            p.setCollisionFilterPair(
                self.robot, self.robot, i1, i2, 0,
                physicsClientId=self.client,
            )

        # name->link index map (used for readable self checks)
        self.names = {}
        for i in range(n):
            info = p.getJointInfo(self.robot, i, physicsClientId=self.client)
            self.names[i] = info[12].decode("utf-8")
        self.names[-1] = "base_link"

        # Sorted pairs of link indices to introspect geometric self collision
        self.excluded_pairs = set()
        for l1, l2 in srdf_pairs:
            i1 = self.link_index.get(l1, -1)
            i2 = self.link_index.get(l2, -1)
            self.excluded_pairs.add(tuple(sorted((i1, i2))))

    # ------------------------------------------------------------------
    # Piece loading
    # ------------------------------------------------------------------
    def _build_piece_multibody(self, piece):
        kind = piece["type"]
        color = piece["color"]
        square = piece["square"]
        key = piece["key"]

        if self.geometry is None or kind not in self.geometry:
            raise RuntimeError(
                f"geometry for piece type {kind!r} not available; cannot build"
            )
        hulls = self.geometry[kind]["hulls"]
        vol = self.geometry[kind]["approximate_volume_m3"]
        mass = vol * 650.0
        if abs(mass - 650.0 * vol) > 1e-9:
            raise RuntimeError("density mismatch; refusing to load wrong URDF")

        urdf = ASSETS / "full_piece_URDF" / f"{kind}_650.urdf"
        if not urdf.exists():
            raise FileNotFoundError(f"missing piece URDF: {urdf}")

        # PyBullet loadURDF basePosition is the LINK frame bottom (see full
        # piece URDFs). Later getBasePositionAndOrientation returns the COM
        # frame position, so base = bottom + R @ localcom.
        bottom = list(self.square_xyz(square))
        bottom[2] += 0.001

        if kind == "knight" and not color:
            base_orn = p.getQuaternionFromEuler([0.0, 0.0, 3.141592653589793])
        else:
            base_orn = [0.0, 0.0, 0.0, 1.0]

        flags = p.URDF_USE_INERTIA_FROM_FILE
        if not self.visuals:
            flags |= _IGNORE_VISUAL
        body = p.loadURDF(
            str(urdf),
            basePosition=bottom,
            baseOrientation=base_orn,
            useFixedBase=False,
            flags=flags,
            physicsClientId=self.client,
        )

        p.changeDynamics(
            body, -1,
            lateralFriction=0.6,
            linearDamping=0.02,
            angularDamping=0.02,
            spinningFriction=0.001,
            rollingFriction=0.001,
            restitution=0.0,
            collisionMargin=5e-5,
            physicsClientId=self.client,
        )
        if self.visuals:
            rgb = [0.96, 0.93, 0.81, 1.0] if color else [0.0, 0.0, 0.0, 1.0]
            p.changeVisualShape(body, -1, rgbaColor=rgb,
                                physicsClientId=self.client)

        n_coll = len(p.getCollisionShapeData(body, -1,
                                             physicsClientId=self.client))
        expected = _HULL_COUNTS.get(kind, len(hulls))
        if n_coll != len(hulls):
            raise AssertionError(
                f"collision layers omitted for {kind}: {n_coll} != {len(hulls)}"
            )
        if n_coll != expected:
            raise AssertionError(
                f"{kind} hull count mismatch: {n_coll} != {expected}"
            )

        self.com[key] = list(self.geometry[kind]["com_m"])
        return body, mass

    # ------------------------------------------------------------------
    # Convenience geometry helpers
    # ------------------------------------------------------------------
    def square_xyz(self, square):
        if square == "tray":
            return [0.195, 0.16, -0.09]
        return super().square_xyz(square)

    def support_body(self, dest):
        return self.tray if dest == "tray" else self.board

    # ------------------------------------------------------------------
    # Trial / subcase manipulation
    # ------------------------------------------------------------------
    def set_case(self, case, home_q, home_g, poses=None):
        """Setup-only: full reset of a trial's piece layout + robot pose.

        Allowed between independent trials, never during measured dynamics.
        """
        self.case = copy.deepcopy(case)

        case_pieces = self.case.get("pieces") or []
        # Normalize list form to keyed lookup.
        if isinstance(case_pieces, dict):
            pieces_by_key = dict(case_pieces)
        else:
            pieces_by_key = {piece["key"]: piece for piece in case_pieces}

        # match each case piece to one unused pool body deterministically
        used = set()
        assignments = {}
        for ckey, cpiece in pieces_by_key.items():
            ctype = cpiece["type"]
            ccolor = cpiece["color"]
            chosen = None
            for rec in self.pool_records:
                if rec["key"] in used:
                    continue
                if rec["type"] == ctype and rec["color"] == ccolor:
                    chosen = rec
                    break
            if chosen is None:
                raise RuntimeError(
                    f"no unused pool body for piece {ckey} "
                    f"(type={ctype}, color={ccolor})"
                )
            used.add(chosen["key"])
            assignments[ckey] = chosen

        self.bodies = {}
        self.com = {}
        self.masses = {}

        # Park unused pool bodies.
        parked = 0
        for rec in self.pool_records:
            if rec["key"] in used:
                continue
            body = rec["body"]
            p.setCollisionFilterGroupMask(
                body, -1,
                collisionFilterGroup=0,
                collisionFilterMask=0,
                physicsClientId=self.client,
            )
            p.changeDynamics(body, -1, mass=0.0,
                             physicsClientId=self.client)
            p.resetBasePositionAndOrientation(
                body,
                [2.0 + 0.05 * parked, 2.0, -2.0],
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.client,
            )
            p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0],
                                physicsClientId=self.client)
            parked += 1

        # Activate chosen bodies.
        for ckey, rec in assignments.items():
            body = rec["body"]
            cpiece = pieces_by_key[ckey]
            mass = rec["mass"]
            inertia = rec["inertia"]

            p.changeDynamics(body, -1, mass=mass,
                             physicsClientId=self.client)
            p.setCollisionFilterGroupMask(
                body, -1,
                collisionFilterGroup=1,
                collisionFilterMask=1,
                physicsClientId=self.client,
            )
            diag = list(inertia)
            if len(diag) < 3:
                diag = diag + [0.0, 0.0, 0.0]
            p.changeDynamics(body, -1,
                             localInertiaDiagonal=diag[:3],
                             physicsClientId=self.client)

            if poses is not None and ckey in poses:
                pos = list(poses[ckey]["pos"])
                quat = list(poses[ckey]["quat"])
                com = rec["com"]
                rot = p.getMatrixFromQuaternion(quat, physicsClientId=self.client)
                rotated = [
                    rot[0] * com[0] + rot[1] * com[1] + rot[2] * com[2],
                    rot[3] * com[0] + rot[4] * com[1] + rot[5] * com[2],
                    rot[6] * com[0] + rot[7] * com[1] + rot[8] * com[2],
                ]
                # poses[ckey]['pos'] is the bottom (link frame) position, so
                # shift by the rotated local com to obtain the base/COM frame.
                base = [pos[i] + rotated[i] for i in range(3)]
            else:
                square = cpiece.get("square") or self.case.get("source")
                bottom = list(self.square_xyz(square))
                bottom[2] += 0.001
                if cpiece["type"] == "knight" and not cpiece["color"]:
                    quat = p.getQuaternionFromEuler([0, 0, 3.141592653589793])
                else:
                    quat = [0.0, 0.0, 0.0, 1.0]
                com = rec["com"]
                rot = p.getMatrixFromQuaternion(quat, physicsClientId=self.client)
                rotated = [
                    rot[0] * com[0] + rot[1] * com[1] + rot[2] * com[2],
                    rot[3] * com[0] + rot[4] * com[1] + rot[5] * com[2],
                    rot[6] * com[0] + rot[7] * com[1] + rot[8] * com[2],
                ]
                base = [bottom[i] + rotated[i] for i in range(3)]

            p.resetBasePositionAndOrientation(body, base, quat,
                                              physicsClientId=self.client)
            p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0],
                                physicsClientId=self.client)

            self.bodies[ckey] = body
            self.com[ckey] = list(rec["com"])
            self.masses[cpiece["type"]] = float(mass)

        # Reset robot to home.
        for i, q in zip(self.arm_joints, home_q):
            p.resetJointState(self.robot, i, q,
                              targetVelocity=0.0,
                              physicsClientId=self.client)
        p.resetJointState(self.robot, self.gripper_joint, home_g,
                          targetVelocity=0.0, physicsClientId=self.client)

        self.time = 0.0
        self.steps = 0

        # Refresh motor targets after reset (no stepping).
        self.command(list(home_q), float(home_g))

        # Refresh collision state and re-snapshot material readback.
        p.performCollisionDetection(physicsClientId=self.client)
        self.material_readback = self.runtime_materials()

        self._validate_readback()
        self._validate_tray_lateral()

    def activate_subcase(self, case):
        """Setup-only: swap the case descriptor; active objects untouched."""
        self.case = copy.deepcopy(case)

    def _validate_tray_lateral(self):
        dyn = p.getDynamicsInfo(self.tray, -1, physicsClientId=self.client)
        if abs(dyn[1] - 1.0) > 1e-6:
            raise AssertionError(
                f"tray lateral friction changed: {dyn[1]}"
            )

    # ------------------------------------------------------------------
    # Query-only helpers (never used inside a measured rollout)
    # ------------------------------------------------------------------
    def set_query_robot(self, q5, g):
        """Query / display only: reset arm and gripper joints."""
        for i, q in zip(self.arm_joints, q5):
            p.resetJointState(self.robot, i, q,
                              targetVelocity=0.0,
                              physicsClientId=self.client)
        p.resetJointState(self.robot, self.gripper_joint, g,
                          targetVelocity=0.0,
                          physicsClientId=self.client)
        # getClosestPoints and AABBs follow reset joint transforms directly.
        # A full dynamics-world contact update is unnecessary for this query.

    def set_query_object(self, key, pos, quat):
        """Query / display only: place an active body at a target pose."""
        body = self.bodies[key]
        com = self.com[key]
        rot = p.getMatrixFromQuaternion(quat, physicsClientId=self.client)
        rotated = [
            rot[0] * com[0] + rot[1] * com[1] + rot[2] * com[2],
            rot[3] * com[0] + rot[4] * com[1] + rot[5] * com[2],
            rot[6] * com[0] + rot[7] * com[1] + rot[8] * com[2],
        ]
        # pos is bottom/link frame; add rotated local com to get COM base.
        base = [pos[i] + rotated[i] for i in range(3)]
        p.resetBasePositionAndOrientation(body, base, quat,
                                          physicsClientId=self.client)
        p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self.client)

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------
    def bottom_snapshot(self):
        """Return the bottom pose of every active body keyed by key."""
        out = {}
        for key in self.bodies:
            pos, quat = self.bottom_pose(key)
            out[key] = {"pos": list(pos), "quat": list(quat)}
        return out

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def observe(self, phase, commanded_q, commanded_g):
        obs = super().observe(phase, commanded_q, commanded_g)

        robot_tray = []
        tray_contacts = []
        for c in p.getContactPoints(
            bodyA=self.robot, bodyB=self.tray,
            physicsClientId=self.client,
        ):
            entry = {
                "normalForce": float(c[9]),
                "distance": float(c[8]),
                "linkA": int(c[3]),
                "linkB": int(c[4]),
                "surface": "tray",
            }
            robot_tray.append(entry)
            tray_contacts.append(dict(entry))

        board_contacts = obs.setdefault("board_contacts", [])
        target_key = None
        if getattr(self, "case", None):
            target_key = self.case.get("target_key")
        if target_key is not None and target_key in self.bodies:
            tb = self.bodies[target_key]
            for c in p.getContactPoints(
                bodyA=tb, bodyB=self.tray,
                physicsClientId=self.client,
            ):
                entry = {
                    "normalForce": float(c[9]),
                    "distance": float(c[8]),
                    "linkA": int(c[3]),
                    "linkB": int(c[4]),
                    "surface": "tray",
                }
                board_contacts.append(dict(entry))
                tray_contacts.append(dict(entry))

        obs["robot_board_contacts"].extend(robot_tray)
        obs["tray_contacts"] = tray_contacts
        return obs

    # ------------------------------------------------------------------
    # Introspection / smoke
    # ------------------------------------------------------------------
    def geometry_signature(self):
        counts = {k: 0 for k in _PIECE_KINDS}
        for rec in self.pool_records:
            counts[rec["type"]] = counts.get(rec["type"], 0) + 1
        robot_shapes = 0
        n = p.getNumJoints(self.robot, physicsClientId=self.client)
        for i in [-1] + list(range(n)):
            shapes = p.getCollisionShapeData(self.robot, i,
                                             physicsClientId=self.client)
            robot_shapes += len(shapes)
        return {
            "piece_counts": counts,
            "robot_collision_shapes": robot_shapes,
            "material_readback": self.runtime_materials(),
            "visuals": self.visuals,
        }
