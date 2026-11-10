import argparse
import copy
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import struct
import sys
import zlib
from pathlib import Path

ENTRY_REL = Path(
    "calibration/manipulation_20261109_v1/staged_continuous_3x10_20261110_v1/work/run_three_staged_v2.py"
)

SCOPE = "Static load only; no manipulation or physics validation"


class StepTrap:
    def __init__(self, p):
        self.p = p
        self.count = 0
        self._orig = None

    def install(self):
        self._orig = self.p.stepSimulation

        def wrapped(*args, **kwargs):
            self.count += 1
            raise RuntimeError("stepSimulation is forbidden during environment inspection")

        self.p.stepSimulation = wrapped

    def restore(self):
        if self._orig is not None:
            self.p.stepSimulation = self._orig


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve(base_dir, value):
    if value is None:
        return None
    p = Path(str(value))
    if p.is_absolute():
        return p
    return (Path(base_dir) / p).resolve()


def _png_chunk(tag, data):
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _write_png(path, width, height, rgba):
    if len(rgba) != width * height * 4:
        raise ValueError(f"pixel buffer size mismatch: {len(rgba)} != {width*height*4}")
    raw = bytearray()
    stride = width * 4
    for y in range(height):
        raw.append(0)
        raw.extend(rgba[y * stride:(y + 1) * stride])
    compressed = zlib.compress(bytes(raw), 9)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_png_chunk(b"IHDR", ihdr))
        f.write(_png_chunk(b"IDAT", compressed))
        f.write(_png_chunk(b"IEND", b""))


def _package_versions():
    out = {}
    for name in ("numpy", "scipy", "pybullet", "python-chess"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Static environment inspection")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--session-index", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f"workspace is not a directory: {workspace}")

    out_dir = Path(args.out)
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")

    entry_path = workspace / ENTRY_REL
    if not entry_path.is_file():
        raise FileNotFoundError(f"entry module not found: {entry_path}")

    import pybullet as p
    import numpy as np

    trap = StepTrap(p)
    trap.install()

    entry = None
    world = None
    info_list = []
    try:
        entry = _load_module("run_three_staged_v2", entry_path)

        recorder = getattr(entry, "recorder", None)
        if recorder is None or not hasattr(recorder, "E"):
            raise RuntimeError("entry.recorder.E is not available")
        query_world_cls = recorder.E.QueryWorld

        build_factory = getattr(entry, "build_factory", None)
        if build_factory is None:
            raise RuntimeError("entry.build_factory is not available")

        config_dir = entry_path.parent.parent / "config"
        batch_path = config_dir / "batch.json"
        batch = _read_json(batch_path)
        sessions = batch.get("sessions") or []
        idx = args.session_index
        if idx < 0 or idx >= len(sessions):
            raise IndexError(f"session index {idx} out of range (0..{len(sessions)-1})")
        session_cfg_path = _resolve(config_dir, sessions[idx].get("config"))
        session = _read_json(session_cfg_path)

        base_query_path = _resolve(config_dir, session.get("base_query"))
        base_query = _read_json(base_query_path)
        home_path = _resolve(config_dir, session.get("initial_home_plan"))
        home = _read_json(home_path)

        home_q = home.get("home_q")
        if not isinstance(home_q, list) or len(home_q) != 5:
            raise ValueError("initial_home_plan.home_q must be a list of 5 joint values")
        g_open = float(home.get("g_open"))

        pieces = session.get("pieces")
        if not isinstance(pieces, list) or len(pieces) != 12:
            raise ValueError(f"session.pieces must contain exactly 12 pieces, found {0 if pieces is None else len(pieces)}")
        steps = session.get("steps") or []
        if not steps:
            raise ValueError("session.steps must be non-empty")
        first_step = steps[0]

        cfg = copy.deepcopy(base_query)
        if not isinstance(cfg, dict):
            raise ValueError("base_query must be a JSON object")
        cfg["diagnostic_threshold"] = 0.002

        case = {
            "id": "environment_smoke",
            "pieces": copy.deepcopy(pieces),
            "target_key": first_step.get("key"),
            "source": first_step.get("source"),
            "dest": first_step.get("dest"),
            "type": first_step.get("type"),
        }

        factory = build_factory(p, query_world_cls, info_list)
        world = factory(cfg, case, list(home_q), g_open)
        if world is None:
            raise RuntimeError("build_factory returned no world")

        query_worlds = 1
        physics_steps = int(getattr(world, "steps", 0))
        attempted_steps = int(trap.count)

        arm_joint_names = [p.getJointInfo(world.robot, j, physicsClientId=world.client)[1].decode("utf-8") for j in world.arm_joints]
        gripper_joint_name = p.getJointInfo(world.robot, world.gripper_joint, physicsClientId=world.client)[1].decode("utf-8")

        if len(arm_joint_names) != 5:
            raise RuntimeError(f"expected 5 arm joints, found {len(arm_joint_names)}: {arm_joint_names}")
        if gripper_joint_name is None:
            raise RuntimeError("gripper joint not identified")

        piece_count = len(world.bodies)
        geometry_types = sorted(cfg["geometry"])
        if len(geometry_types) != 6:
            raise RuntimeError("expected six geometry types")

        if piece_count != 12:
            raise RuntimeError(f"expected 12 pieces, got {piece_count}")

        if len(info_list) != 1:
            raise RuntimeError(f"expected exactly one threshold info entry, got {len(info_list)}")
        info = info_list[0]
        if not info.get("constructor_completed"):
            raise RuntimeError("constructor did not complete")
        forwarded = info.get("forwarded_updates") or []
        if len(forwarded) < 1:
            raise RuntimeError("no threshold updates were forwarded")

        for upd in forwarded:
            if not math.isclose(float(upd.get("requested")), 0.002, rel_tol=0.0, abs_tol=0.0):
                raise RuntimeError(f"threshold forwarded not 0.002: {upd}")

        if physics_steps != 0:
            raise RuntimeError(f"physics steps must be 0, got {physics_steps}")
        if attempted_steps != 0:
            raise RuntimeError(f"attempted stepSimulation calls must be 0, got {attempted_steps}")

        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.18, 0.0, 0.06],
            distance=0.65,
            yaw=45,
            pitch=-40,
            roll=0,
            upAxisIndex=2,
        )
        proj = p.computeProjectionMatrixFOV(45, 1.5, 0.01, 2)
        render = p.getCameraImage(
            width=1200,
            height=800,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=world.client,
        )

        if isinstance(render, dict):
            width = int(render["width"])
            height = int(render["height"])
            rgba = render["rgba"]
        else:
            width, height, rgba = int(render[0]), int(render[1]), render[2]

        rgba_arr = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
        png_bytes = rgba_arr.tobytes()

        result = {
            "category": "static_environment_check",
            "query_worlds": query_worlds,
            "physics_steps": physics_steps,
            "attempted_step_calls": attempted_steps,
            "piece_count": piece_count,
            "geometry_types": geometry_types,
            "arm_joint_names": arm_joint_names,
            "gripper_joint_name": gripper_joint_name,
            "dt_s": float(getattr(world, "dt")),
            "threshold_updates": info,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": _package_versions(),
            "scope": SCOPE,
        }
    finally:
        try:
            if world is not None:
                world.close()
        finally:
            trap.restore()

    out_dir.mkdir(parents=True, exist_ok=False)
    image_path = out_dir / "preview.png"
    _write_png(image_path, width, height, png_bytes)

    result["image"] = str(image_path)
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
