import math
import numpy as np
from pathlib import Path

import pybullet as p

import physics_world
from physics_world import PhysicsWorld, _parse_srdf_disables
from pilot_world import PilotWorld


class QueryWorld(PhysicsWorld):
    def __init__(self, config, case, q5, g):
        self.cfg = config
        self.tray = None

        old_assets = physics_world.ASSETS
        old_load_geometry = physics_world._load_geometry
        physics_world.ASSETS = Path(config["asset_dir"])
        physics_world._load_geometry = lambda: config["geometry"]
        try:
            super().__init__(
                case, q5, g,
                friction=math.sqrt(0.4),
                dt=1.0 / 960.0,
                grip_force=0.05,
            )
        finally:
            physics_world.ASSETS = old_assets
            physics_world._load_geometry = old_load_geometry

        self.names = {i: n for n, i in self.link_index.items()}

        excluded = set()
        srdf_pairs = _parse_srdf_disables(
            Path(self.cfg["asset_dir"]) / "robot.srdf"
        )
        for l1, l2 in srdf_pairs:
            i1 = self.link_index.get(l1, -1)
            i2 = self.link_index.get(l2, -1)
            a, b = sorted((i1, i2))
            excluded.add((a, b))
        self.excluded_pairs = excluded

    def square_xyz(self, square):
        s = self.cfg["square_m"]
        yaw = self.cfg["yaw_rad"]
        cx, cy = self.cfg["center_m"]
        fi = (ord(square[0].lower()) - ord("a")) - 3.5
        ri = (int(square[1:]) - 1) - 3.5
        lx = fi * s
        ly = ri * s
        c = math.cos(yaw)
        sn = math.sin(yaw)
        x = c * lx - sn * ly + cx
        y = sn * lx + c * ly + cy
        z = self.cfg["board_top_m"]
        return [x, y, z]

    def _load_board(self):
        top = self.cfg["board_top_m"]
        bot = self.cfg["board_bottom_m"]
        hx, hy = self.cfg["board_half_xy_m"]
        cx, cy = self.cfg["center_m"]
        yaw = self.cfg["yaw_rad"]
        half = [hx, hy, (top - bot) / 2.0]
        orn = p.getQuaternionFromEuler([0, 0, yaw])

        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half,
            physicsClientId=self.client,
        )
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half,
            physicsClientId=self.client,
        )
        self.board = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[cx, cy, (top + bot) / 2.0],
            baseOrientation=orn,
            physicsClientId=self.client,
        )
        p.changeDynamics(
            self.board, -1,
            lateralFriction=self.friction,
            restitution=0.0,
            collisionMargin=5e-5,
            physicsClientId=self.client,
        )

        table_top = self.cfg["table_top_m"]
        table_half = [0.4, 0.4, 0.015]
        tcol = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=table_half,
            physicsClientId=self.client,
        )
        tvis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=table_half,
            physicsClientId=self.client,
        )
        self.table = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tcol,
            baseVisualShapeIndex=tvis,
            basePosition=[0.3, 0.0, table_top - 0.015],
            physicsClientId=self.client,
        )
        p.changeDynamics(
            self.table, -1,
            lateralFriction=self.friction,
            restitution=0.0,
            collisionMargin=5e-5,
            physicsClientId=self.client,
        )

    def _build_piece_multibody(self, piece):
        key = piece["key"]
        ptype = piece["type"]
        g = self.cfg["geometry"][ptype]
        flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
        base = list(self.square_xyz(piece["square"]))
        base[2] += 0.0002
        orn = [0, 0, 0, 1]
        body = p.loadURDF(
            str(g["urdf"]),
            basePosition=base,
            baseOrientation=orn,
            useFixedBase=False,
            flags=flags,
            physicsClientId=self.client,
        )
        return body, float(g["mass_kg"])

    set_query_robot = PilotWorld.set_query_robot
    set_query_object = PilotWorld.set_query_object
    snapshot = PilotWorld.bottom_snapshot

    def set_case(self, case, startq, startg, poses=None):
        existing = {pc["key"]: pc["type"] for pc in self.case["pieces"]}
        incoming = {pc["key"]: pc["type"] for pc in case["pieces"]}
        assert existing == incoming, "piece key/type set mismatch"

        if poses is None:
            for pc in case["pieces"]:
                key = pc["key"]
                pos = list(self.square_xyz(pc["square"]))
                pos[2] += 0.0002
                self.set_query_object(key, pos, [0, 0, 0, 1])
        else:
            for key, pose in poses.items():
                self.set_query_object(key, pose["pos"], pose["quat"])

        self.set_query_robot(startq, startg)
        self.case = case

    def support_body(self, key):
        return self.board

    def step(self):
        raise AssertionError("QueryWorld is static query-only; step unsupported")
