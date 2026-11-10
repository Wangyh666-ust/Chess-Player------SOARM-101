from pathlib import Path
import json
import numpy as np
import pybullet as p
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "results" / "advanced_30_v1" / "assets"

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_NAME = "gripper"
TCP_JOINT = "gripper_frame_joint"
TCP_LINK = "gripper_frame_link"


def _load_geometry():
    with open(ASSETS / "geometry.json", "r") as f:
        return json.load(f)


def _load_layers():
    with open(HERE / "path_assets" / "layers" / "layers_meta.json", "r") as f:
        return json.load(f)


def _parse_srdf_disables(srdf_path):
    tree = ET.parse(str(srdf_path))
    root = tree.getroot()
    pairs = []
    for elem in root.findall("disable_collisions"):
        l1 = elem.get("link1")
        l2 = elem.get("link2")
        if l1 is not None and l2 is not None:
            pairs.append((l1, l2))
    return pairs


class PhysicsWorld:
    def __init__(self, case, q5, gscalar, friction=0.6, density=650,
                 dt=1.0 / 240.0, arm_force=1.5, grip_force=0.2):
        assert isinstance(q5, (list, tuple, np.ndarray))
        assert len(q5) == 5
        assert isinstance(gscalar, float) or isinstance(gscalar, int)

        self.case = case
        self.dt = float(dt)
        self.friction = float(friction)
        self.density = float(density)
        self.arm_force = float(arm_force)
        self.grip_force = float(grip_force)
        self.steps = 0
        self.time = 0.0
        self._closed = False
        self._gscalar_init = float(gscalar)
        self._q5_init = [float(x) for x in q5]

        self.geometry = _load_geometry()
        self.layers = _load_layers()

        self._connect()
        self._load_robot()
        self._load_board()
        self._load_pieces()
        self._reset_init(q5, gscalar)

    # ---------------- setup ----------------
    def _connect(self):
        self.client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            numSolverIterations=150,
            contactBreakingThreshold=0.0002,
            deterministicOverlappingPairs=1,
            physicsClientId=self.client,
        )

    def _load_robot(self):
        flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
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

        # friction / restitution / margin / damping across links
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

        # SRDF disable collisions
        srdf_pairs = _parse_srdf_disables(ASSETS / "robot.srdf")
        for l1, l2 in srdf_pairs:
            i1 = self.link_index.get(l1, -1)
            i2 = self.link_index.get(l2, -1)
            if l1 in self.link_index:
                i1 = self.link_index[l1]
            else:
                i1 = -1
            if l2 in self.link_index:
                i2 = self.link_index[l2]
            else:
                i2 = -1
            p.setCollisionFilterPair(self.robot, self.robot, i1, i2, 0,
                                     physicsClientId=self.client)

    def _load_board(self):
        half = [0.125, 0.125, 0.0015]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half,
                                     physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                  rgbaColor=[0.1, 0.1, 0.1, 1],
                                  physicsClientId=self.client)
        self.board = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0.195, 0.0, -0.0915],
            physicsClientId=self.client,
        )
        # visual tiles (optional, cosmetic)
        for i in range(8):
            for j in range(8):
                x = 0.195 + (i - 3.5) * 0.03
                y = (j - 3.5) * 0.03
                color = [0.9, 0.9, 0.9, 1] if (i + j) % 2 == 0 else [0.05, 0.05, 0.05, 1]
                p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[0.0145, 0.0145, 0.0005],
                    rgbaColor=color,
                    visualFramePosition=[0, 0, 0.0015],
                    physicsClientId=self.client,
                )

    def _build_piece_multibody(self, piece):
        from full_piece_v2 import load_piece
        ptype = piece['type']
        bottom = self.square_xyz(piece['square'])
        bottom[2] += .001
        yaw = 3.141592653589793 if ptype == 'knight' and not piece['color'] else 0
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        body = load_piece(self.client, ptype, bottom, orn, color=piece['color'],
                          density=self.density, fixed=False, friction=self.friction)
        mass = self.geometry[ptype]['approximate_volume_m3'] * self.density
        return (body, mass)
    def _load_pieces(self):
        self.bodies = {}
        self.masses = {}
        self.com = {}
        for piece in self.case["pieces"]:
            key = piece["key"]
            body, mass = self._build_piece_multibody(piece)
            self.bodies[key] = body
            self.masses[piece["type"]] = mass
            self.com[key] = list(self.geometry[piece["type"]]["com_m"])

    def _reset_init(self, q5, gscalar):
        for j, val in zip(self.arm_joints, q5):
            p.resetJointState(self.robot, j, float(val), physicsClientId=self.client)
        p.resetJointState(self.robot, self.gripper_joint, float(gscalar),
                          physicsClientId=self.client)

    # ---------------- helpers ----------------
    def square_xyz(self, square):
        f = square[0].lower()
        rank = square[1:]
        x = 0.195 + (ord(f) - ord('a') - 3.5) * 0.03
        y = (int(rank) - 1 - 3.5) * 0.03
        z = -0.09
        return [x, y, z]

    def command(self, q5, gscalar):
        assert len(q5) == 5
        for j, val in zip(self.arm_joints, q5):
            p.setJointMotorControl2(
                self.robot, j, p.POSITION_CONTROL,
                targetPosition=float(val),
                maxVelocity=0.6,
                positionGain=0.2,
                velocityGain=1.0,
                force=self.arm_force,
                physicsClientId=self.client,
            )
        p.setJointMotorControl2(
            self.robot, self.gripper_joint, p.POSITION_CONTROL,
            targetPosition=float(gscalar),
            maxVelocity=1.0,
            positionGain=0.2,
            velocityGain=1.0,
            force=self.grip_force,
            physicsClientId=self.client,
        )

    def step(self):
        p.stepSimulation(physicsClientId=self.client)
        self.steps += 1
        self.time = self.steps * self.dt

    def bottom_pose(self, key):
        body = self.bodies[key]
        pos, orn = p.getBasePositionAndOrientation(body, physicsClientId=self.client)
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        com = np.array(self.com[key])
        bottom = np.array(pos) - rot @ com
        return bottom.tolist(), list(orn)

    # ---------------- observation ----------------
    def observe(self, phase, commanded_q, commanded_g):
        assert len(commanded_q) == 5

        actual_q5 = []
        actual_tau_arm = []
        for j in self.arm_joints:
            st = p.getJointState(self.robot, j, physicsClientId=self.client)
            actual_q5.append(float(st[0]))
            actual_tau_arm.append(float(st[3]))

        gst = p.getJointState(self.robot, self.gripper_joint, physicsClientId=self.client)
        actual_g = float(gst[0])
        tau_grip = float(gst[3])

        tcp_state = p.getLinkState(self.robot, self.tcp_link,
                                   computeForwardKinematics=True,
                                   physicsClientId=self.client)
        tcp_pos = list(tcp_state[4])
        tcp_quat = list(tcp_state[5])

        object_poses = {}
        for key, body in self.bodies.items():
            pos, orn = self.bottom_pose(key)
            object_poses[key] = {"pos": list(pos), "quat": list(orn)}

        target_key = self.case["target_key"]
        target_body = self.bodies[target_key]
        tpos, torn = self.bottom_pose(target_key)

        max_err = 0.0
        for c, a in zip(commanded_q, actual_q5):
            d = np.degrees(abs(a - c))
            if d > max_err:
                max_err = d

        contacts = []
        cps = p.getContactPoints(self.robot, target_body, physicsClientId=self.client)
        link_names = {}
        for name, idx in self.link_index.items():
            link_names[idx] = name
        for cp in cps:
            linkA = cp[3]
            contacts.append({
                "robot_link": link_names.get(linkA, str(linkA)),
                "normalForce": float(cp[9]),
                "distance": float(cp[8]),
                "frictionForce1": float(cp[10]),
                "frictionForce2": float(cp[12]),
            })

        board_contacts = []
        bc = p.getContactPoints(self.board, target_body, physicsClientId=self.client)
        for cp in bc:
            linkA = cp[3]
            board_contacts.append({
                "link": "board",
                "robot_link": link_names.get(linkA, str(linkA)),
                "normalForce": float(cp[9]),
                "distance": float(cp[8]),
            })

        neighbor_contacts = []
        for key, body in self.bodies.items():
            if key == target_key:
                continue
            nc = p.getContactPoints(self.robot, body, physicsClientId=self.client)
            for cp in nc:
                linkA = cp[3]
                neighbor_contacts.append({
                    "neighbor": key,
                    "robot_link": link_names.get(linkA, str(linkA)),
                    "normalForce": float(cp[9]),
                    "distance": float(cp[8]),
                })

        sum_fixed_jaw = 0.0
        sum_moving_jaw = 0.0
        sum_motor_collision = 0.0
        for c in contacts:
            ln = c["robot_link"]
            if "fixed_jaw" in ln:
                sum_fixed_jaw += c["normalForce"]
            if "moving_jaw_so101_v1_link" in ln:
                sum_moving_jaw += c["normalForce"]
            if "motor" in ln:
                sum_motor_collision += c["normalForce"]

        # validate poses
        all_pose_vecs = list(tcp_pos) + list(tcp_quat)
        for k, v in object_poses.items():
            all_pose_vecs.extend(v["pos"])
            all_pose_vecs.extend(v["quat"])
        for v in all_pose_vecs:
            if not np.isfinite(v):
                raise RuntimeError("Non-finite pose component encountered")
        for k, v in object_poses.items():
            nrm = np.linalg.norm(v["pos"])
            if nrm > 2.0:
                raise RuntimeError(f"Object {k} position norm {nrm} exceeds 2m")

        return {
            "time": self.time,
            "phase": phase,
            "commanded_q": list(commanded_q),
            "commanded_g": float(commanded_g),
            "actual_q5": actual_q5,
            "actual_g": actual_g,
            "tcp_pos": tcp_pos,
            "tcp_quat": tcp_quat,
            "object_poses": object_poses,
            "target_pos": list(tpos),
            "target_quat": list(torn),
            "max_arm_tracking_error_deg": float(max_err),
            "tau_arm": actual_tau_arm,
            "tau_gripper": tau_grip,
            "contacts": contacts,
            "robot_board_contacts": [dict(normalForce=float(cp[9]), distance=float(cp[8]), linkA=cp[3], linkB=cp[4]) for cp in p.getContactPoints(self.robot, self.board, physicsClientId=self.client)],
            "self_contacts": [dict(normalForce=float(cp[9]), distance=float(cp[8]), linkA=cp[3], linkB=cp[4]) for cp in p.getContactPoints(self.robot, self.robot, physicsClientId=self.client)],
            "board_contacts": board_contacts,
            "neighbor_contacts": neighbor_contacts,
            "fixed_jaw_collision_sum": sum_fixed_jaw,
            "moving_jaw_collision_sum": sum_moving_jaw,
            "motor_collision_sum": sum_motor_collision,
        }

    def close(self):
        if not self._closed:
            p.disconnect(self.client)
            self._closed = True
