import copy
import json
import gzip
import hashlib
import math
from pathlib import Path

import numpy as np
import pybullet as p
import chess

ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "reachability_eval" / "results"


def load_inputs():
    with open(RESULTS_DIR / "continuous_games_v1" / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(RESULTS_DIR / "tabletop_height_v8" / "home_setup.json", "r", encoding="utf-8") as f:
        home = json.load(f)
    with open(RESULTS_DIR / "single_step_pilot_v1" / "profiles.json", "r", encoding="utf-8") as f:
        profiles = json.load(f)
    if isinstance(profiles, dict) and "profiles" in profiles:
        profiles = profiles["profiles"]
    return manifest, home, profiles


def validate(manifest):
    games = manifest["games"]
    expected_names = ["game01", "game02", "game03", "game04", "game05"]
    total_plies = 0
    total_subactions = 0
    captures = []
    for name in expected_names:
        game = games[name]
        fen = game["initial_fen"]
        board = chess.Board(fen)
        init_pieces = game["initial_pieces"]
        expected = {}
        for sq in chess.SQUARES:
            pc = board.piece_at(sq)
            if pc is not None:
                expected[chess.square_name(sq)] = (chess.piece_name(pc.piece_type), pc.color)
        got = {}
        for piece in init_pieces:
            got[piece["square"]] = (piece["type"], piece["color"])
        assert set(got.keys()) == set(expected.keys())
        for sq, val in got.items():
            assert val == expected[sq], (name, sq, val, expected[sq])
        count_capture = 0
        for i, ply in enumerate(game["plies"][:10]):
            total_plies += 1
            if i < 10:
                b_before = chess.Board(ply["fen_before"])
                assert board.fen() == ply["fen_before"]
                b_after = chess.Board(ply["fen_after"])
                move = chess.Move.from_uci(ply["uci"])
                assert b_before.is_legal(move)
                piece = b_before.piece_at(move.from_square)
                assert piece is not None
                assert ply["type"] == chess.piece_name(piece.piece_type)
                assert ply["color"] == bool(piece.color)
                is_capture = b_before.is_capture(move)
                is_castle = b_before.is_castling(move)
                is_enpassant = b_before.is_en_passant(move)
                is_promo = move.promotion is not None
                assert not is_castle
                assert not is_enpassant
                assert not is_promo
                assert bool(ply["capture"]) == is_capture
                b_before.push(move)
                assert b_before.fen() == ply["fen_after"]
                if is_capture:
                    total_subactions += 2
                    count_capture += 1
                else:
                    total_subactions += 1
            if i >= 10:
                board.push(chess.Move.from_uci(ply["uci"]))
            else:
                board = chess.Board(ply["fen_after"])
        captures.append(count_capture)
    assert total_plies == 50, total_plies
    assert total_subactions == 53, total_subactions
    assert captures == [0, 0, 1, 0, 2], captures
    return {
        "total_plies": total_plies,
        "total_subactions": total_subactions,
        "captures": captures,
    }


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError("not serializable")


def save(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, default=_json_default)
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(w):
    bottom = w.bottom_snapshot()
    vel = {}
    for key, bid in w.bodies.items():
        lin, ang = p.getBaseVelocity(bid, physicsClientId=w.client)
        vel[key] = {"lin": list(lin), "ang": list(ang)}
    joints = {}
    for j in list(w.arm_joints) + [w.gripper_joint]:
        q, qd, _, _ = p.getJointState(w.robot, int(j), physicsClientId=w.client)
        joints[str(int(j))] = {"q": q, "qd": qd}
    return {
        "bodies": bottom,
        "vel": vel,
        "joints": joints,
        "body_ids": dict(w.bodies),
        "steps": int(w.steps),
        "time": float(w.time),
    }


def _quat_equal(a, b):
    if np.allclose(a, b, rtol=0, atol=1e-12):
        return True
    return np.allclose(a, -np.asarray(b), rtol=0, atol=1e-12)


def same(a, b):
    if set(a["body_ids"].keys()) != set(b["body_ids"].keys()):
        return False
    for k in a["body_ids"]:
        if a["body_ids"][k] != b["body_ids"][k]:
            return False
    if a["steps"] != b["steps"]:
        return False
    if a["time"] != b["time"]:
        return False
    if set(a["bodies"].keys()) != set(b["bodies"].keys()):
        return False
    for k in a["bodies"]:
        pa = a["bodies"][k]
        pb = b["bodies"][k]
        pos_a = np.asarray(pa["pos"] if isinstance(pa, dict) else pa[0])
        pos_b = np.asarray(pb["pos"] if isinstance(pb, dict) else pb[0])
        if not np.allclose(pos_a, pos_b, rtol=0, atol=1e-12):
            return False
        quat_a = pa["quat"] if isinstance(pa, dict) and "quat" in pa else pa[1]
        quat_b = pb["quat"] if isinstance(pb, dict) and "quat" in pb else pb[1]
        if not _quat_equal(np.asarray(quat_a), np.asarray(quat_b)):
            return False
    if set(a["vel"].keys()) != set(b["vel"].keys()):
        return False
    for k in a["vel"]:
        if not np.allclose(a["vel"][k]["lin"], b["vel"][k]["lin"], rtol=0, atol=1e-12):
            return False
        if not np.allclose(a["vel"][k]["ang"], b["vel"][k]["ang"], rtol=0, atol=1e-12):
            return False
    if set(a["joints"].keys()) != set(b["joints"].keys()):
        return False
    for k in a["joints"]:
        if not np.allclose([a["joints"][k]["q"]], [b["joints"][k]["q"]], rtol=0, atol=1e-12):
            return False
        if not np.allclose([a["joints"][k]["qd"]], [b["joints"][k]["qd"]], rtol=0, atol=1e-12):
            return False
    return True


def case_for(case_id, piece, dest, ply, pieces, tray_slot=1):
    return {
        "id": case_id,
        "type": piece["type"],
        "piece_type": piece["type"],
        "color": piece["color"],
        "target_key": piece["key"],
        "source": piece["square"],
        "dest": dest,
        "fen": ply["fen_before"],
        "fen_after": ply["fen_after"],
        "uci": ply["uci"],
        "san": ply["san"],
        "pieces": copy.deepcopy(pieces),
        "is_capture": False,
        "capture_key": None,
        "capture_square": None,
        "tray_slot": tray_slot,
    }


def install_layout(w, mount, layout):
    r1 = mount.install_world(w)
    r2 = layout.apply_layout(w, 90.0, 0.0006)
    original = w.square_xyz

    def wrapper(square):
        if square == "tray":
            slot = int(w.case.get("tray_slot", 1))
            return [0.165 + 0.03 * slot, 0.16, 0.0006]
        return original(square)

    w.square_xyz = wrapper
    return r1, r2


def poses(w, pieces):
    out = {}
    for pc in pieces:
        xyz = w.square_xyz(pc["square"])
        pos = [xyz[0], xyz[1], xyz[2] + 0.001]
        yaw = 90.0
        if pc["type"] == "knight" and pc["color"] is False:
            yaw += 180.0
        quat = p.getQuaternionFromEuler([0.0, 0.0, math.radians(yaw)])
        out[pc["key"]] = {"pos": pos, "quat": list(quat)}
    return out


def scene(w, expected_pieces):
    bottom = w.bottom_snapshot()
    per_key = {}
    all_ok = True
    max_xy = 0.0
    max_z = 0.0
    max_tilt = 0.0
    for pc in expected_pieces:
        key = pc["key"]
        if key not in bottom:
            raise KeyError(key)
        entry = bottom[key]
        pos = np.asarray(entry["pos"] if isinstance(entry, dict) else entry[0])
        quat = np.asarray(entry["quat"] if isinstance(entry, dict) and "quat" in entry else entry[1])
        if pc["square"] == "tray":
            slot = pc.get("tray_slot", 1)
            expected = np.array([0.165 + 0.03 * slot, 0.16, 0.0006])
        else:
            expected = np.asarray(w.square_xyz(pc["square"]))
        xy = float(np.linalg.norm(pos[:2] - expected[:2]))
        z = float(abs(pos[2] - expected[2]))
        mat = np.asarray(p.getMatrixFromQuaternion(list(quat))).reshape(3, 3)
        tilt = float(math.degrees(math.acos(max(-1.0, min(1.0, mat[2, 2])))))
        ok = xy <= 0.003 and z <= 0.001 and tilt <= 5.0
        if not ok:
            all_ok = False
        max_xy = max(max_xy, xy)
        max_z = max(max_z, z)
        max_tilt = max(max_tilt, tilt)
        per_key[key] = {"xy": xy, "z": z, "tilt": tilt, "ok": ok}
    return {
        "all_ok": all_ok,
        "per_key": per_key,
        "max_xy": max_xy,
        "max_z": max_z,
        "max_tilt": max_tilt,
    }


def protect(w):
    calls = []
    originals = {}
    instance_orig = {}

    def make_wrapper(name, arg_index):
        def wrapper(*args, **kwargs):
            if "physicsClientId" in kwargs:
                client = kwargs["physicsClientId"]
            elif arg_index is not None and len(args) > arg_index:
                client = args[arg_index]
            else:
                client = 0
            if client == w.client:
                calls.append(name)
                raise RuntimeError("protected call: " + name)
            return originals[name](*args, **kwargs)
        return wrapper

    names = [
        ("resetJointState", 4),
        ("resetBasePositionAndOrientation", 3),
        ("resetBaseVelocity", 3),
        ("resetSimulation", 0),
        ("createConstraint", None),
    ]
    for name, arg_index in names:
        originals[name] = getattr(p, name)
        setattr(p, name, make_wrapper(name, arg_index))

    def make_inst_wrapper(name):
        def wrapper(*args, **kwargs):
            calls.append(name)
            raise RuntimeError("protected instance call: " + name)
        return wrapper

    for name in ("set_case", "set_query_robot", "set_query_object"):
        if hasattr(w, name):
            instance_orig[name] = getattr(w, name)
            setattr(w, name, make_inst_wrapper(name))

    def restore():
        for name, _ in names:
            setattr(p, name, originals[name])
        for name, fn in instance_orig.items():
            setattr(w, name, fn)

    return restore, calls


def _has_error(value):
    if isinstance(value, dict):
        for k in ("code_error", "restore_error", "guard_calls"):
            if k in value and value[k]:
                return True
        for v in value.values():
            if _has_error(v):
                return True
    elif isinstance(value, (list, tuple)):
        for v in value:
            if _has_error(v):
                return True
    return False


def bad_records(value):
    return _has_error(value)


def trace(path, raw, pl, before, after):
    controller = copy.deepcopy(raw) if raw else {}
    omitted = ["all_piece_board_contacts", "piece_pair_contacts", "robot_piece_contact_vectors"]
    frames = controller.get("frames") if isinstance(controller, dict) else None
    if isinstance(frames, list):
        for frame in frames:
            if isinstance(frame, dict):
                for field in omitted:
                    frame.pop(field, None)
    payload = {
        "controller": controller,
        "prefix_records": pl.prefix_records,
        "tail_records": pl.tail_records,
        "before": before,
        "after": after,
        "omitted_frame_fields": list(omitted),
    }
    path = Path(path)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=3) as f:
        json.dump(payload, f, default=_json_default)
    frame_count = 0
    if isinstance(frames, list):
        frame_count = len(frames)
    raw_path = str(path.resolve().relative_to(ROOT))
    raw_hash = digest(path)
    return {"raw_path": raw_path, "raw_hash": raw_hash, "frame_count": frame_count}
