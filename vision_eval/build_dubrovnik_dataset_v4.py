"""Build a reproducible, automatically annotated Dubrovnik finetune dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import chess
import numpy as np
from scipy.spatial import ConvexHull
import yaml
from PIL import Image, ImageDraw, ImageFilter
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from prepare_scenes import load_sets, actor_for, plane, SQUARE, KING_HEIGHT

ROOT = Path(__file__).resolve().parent

WIDTH, HEIGHT = 960, 720
DEFAULT_OUTPUT = ROOT / "dubrovnik_finetune" / "dataset_v1"
DEFAULT_GROUPS = 375
DEFAULT_VIEWS = 8
DEFAULT_SEED = 20261007
TOTAL_IMAGES = DEFAULT_GROUPS * DEFAULT_VIEWS

PIECE_TYPES = ("pawn", "knight", "bishop", "rook", "queen", "king")

CLASS_NAMES = [
    "white-pawn", "white-rook", "white-knight", "white-bishop", "white-queen", "white-king",
    "black-pawn", "black-rook", "black-knight", "black-bishop", "black-queen", "black-king",
    "empty",
]

CLASS_IDS = {
    ("pawn", True): 0, ("rook", True): 1, ("knight", True): 2, ("bishop", True): 3, ("queen", True): 4, ("king", True): 5,
    ("pawn", False): 6, ("rook", False): 7, ("knight", False): 8, ("bishop", False): 9, ("queen", False): 10, ("king", False): 11,
}

PIECE_SYMBOLS = {chess.PAWN: "P", chess.ROOK: "R", chess.KNIGHT: "N", chess.BISHOP: "B", chess.QUEEN: "Q", chess.KING: "K"}


def piece_type_name(piece_type: int) -> str:
    return chess.piece_name(piece_type)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gather_mesh_sources() -> dict:
    """Return a dict describing mesh source files and their SHA256 hashes."""
    prepared = ROOT / "prepared"
    dub_folder = prepared / "ezeeone_Dubrovnik_1960_set_02_stls"
    dub_prefix = dict(zip(PIECE_TYPES, ("obj_5_", "obj_3_", "obj_13_", "obj_4_", "obj_2_", "obj_1_")))
    info = {}
    for name, prefix in dub_prefix.items():
        path = next(dub_folder.glob(prefix + "*.stl"))
        info[f"dubrovnik_{name}"] = {"path": str(path.relative_to(ROOT).as_posix()), "sha256": sha256_file(path)}
    bishop_head = next(dub_folder.glob("obj_12_*.stl"))
    info["dubrovnik_bishop_head"] = {"path": str(bishop_head.relative_to(ROOT).as_posix()), "sha256": sha256_file(bishop_head)}
    return info


def load_excluded_placements() -> list[str]:
    manifest_path = ROOT / "scenes" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing old scenes manifest: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    excluded = []
    seen = set()
    for scene in data.get("scenes", []):
        if scene.get("set") != "dubrovnik":
            continue
        fen = scene.get("fen")
        if not fen:
            continue
        placement = fen.split(" ")[0]
        if placement not in seen:
            seen.add(placement)
            excluded.append(placement)
    return excluded


def cube_homogeneous_project(points_xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    n = points_xyz.shape[0]
    pts_h = np.concatenate([points_xyz, np.ones((n, 1), dtype=np.float64)], axis=1)
    out = pts_h @ matrix.T
    return out


def project_vertices_world_to_display(renderer, points_xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Project all provided mesh vertices via pipeline transform + GetCompositeProjectionTransformMatrix."""
    pts_h = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    world = pts_h @ matrix.T
    # Renormalise world coordinates: matrix may not be a full rigid transform (e.g. scaling),
    # actor matrix is a vtkMatrix4x4 -> use it as full 4x4.
    w_world = world[:, 3:4]
    # Only valid if actor matrix is affine with homogeneous row [0,0,0,1]; guard anyway.
    safe_w = np.where(np.abs(w_world) > 1e-9, w_world, 1.0)
    world_xyz = world[:, :3] / safe_w
    cam = renderer.GetActiveCamera()
    aspect = WIDTH / HEIGHT
    vtk_matrix = cam.GetCompositeProjectionTransformMatrix(aspect, -1.0, 1.0)
    # Build numpy from vtkMatrix4x4 manually.
    m = np.zeros((4, 4), dtype=np.float64)
    for r in range(4):
        for c in range(4):
            m[r, c] = vtk_matrix.GetElement(r, c)
    pts_h2 = np.concatenate([world_xyz, np.ones((world_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    clip = pts_h2 @ m.T
    w = clip[:, 3]
    safe_w2 = np.where(np.abs(w) > 1e-9, w, np.nan)
    ndc = clip[:, :3] / safe_w2[:, None]
    x = (ndc[:, 0] + 1.0) * (WIDTH / 2.0)
    y = HEIGHT - 1.0 - (ndc[:, 1] + 1.0) * (HEIGHT / 2.0)
    valid = np.isfinite(x) & np.isfinite(y) & (w > 1e-6)
    return np.stack([x, y, w], axis=1), valid


def actor_matrix(actor: vtk.vtkActor) -> np.ndarray:
    m = actor.GetMatrix()
    arr = np.zeros((4, 4), dtype=np.float64)
    for r in range(4):
        for c in range(4):
            arr[r, c] = m.GetElement(r, c)
    return arr


def cache_local_points(mesh: vtk.vtkPolyData) -> np.ndarray:
    pts = vtk_to_numpy(mesh.GetPoints().GetData())
    array = np.asarray(pts, dtype=np.float64)
    if array.size == 0:
        return array
    hull = ConvexHull(array)
    return array[hull.vertices]


class Renderer360:
    """Persistent offscreen renderer with reusable actors."""

    def __init__(self, pieces: dict):
        self.pieces = pieces
        self.window = vtk.vtkRenderWindow()
        self.window.SetOffScreenRendering(1)
        self.window.SetSize(WIDTH, HEIGHT)
        self.window.SetMultiSamples(0)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.36, 0.39, 0.42)
        self.renderer.SetUseFXAA(False)
        self.renderer.AutomaticLightCreationOff()
        self.light = vtk.vtkLight()
        self.light.SetLightTypeToSceneLight()
        self.light.SetConeAngle(60)
        self.renderer.AddLight(self.light)
        self.renderer.SetLightFollowCamera(False)
        self.window.AddRenderer(self.renderer)

        # Table + board: reuse created actors once.
        self.table_actor = actor_for(plane(-500, -500, 900, 900, -4), (0.46, 0.39, 0.31))
        self.board_actor = actor_for(plane(-15, -15, 415, 415, -1), (0.25, 0.18, 0.11))
        self.renderer.AddActor(self.table_actor)
        self.renderer.AddActor(self.board_actor)
        self.square_actors = []
        for rank in range(8):
            for file in range(8):
                color = (0.74, 0.64, 0.46) if (rank + file) % 2 else (0.22, 0.16, 0.11)
                square = actor_for(plane(file * SQUARE, rank * SQUARE, (file + 1) * SQUARE, (rank + 1) * SQUARE), color)
                self.renderer.AddActor(square)
                self.square_actors.append((rank, file, square))

        self.piece_mesh_points = {name: cache_local_points(mesh) for name, mesh in pieces.items()}

        self.capture_rgb = vtk.vtkWindowToImageFilter()
        self.capture_rgb.SetInput(self.window)
        self.capture_rgb.SetInputBufferTypeToRGB()
        self.capture_rgb.ReadFrontBufferOff()

        self.capture_id = vtk.vtkWindowToImageFilter()
        self.capture_id.SetInput(self.window)
        self.capture_id.SetInputBufferTypeToRGB()
        self.capture_id.ReadFrontBufferOff()

        self.piece_actors: dict[int, vtk.vtkActor] = {}

    def build_scene(self, board: chess.Board, jitter: dict[int, tuple[float, float]], yaw: dict[int, float]):
        # Clear old piece actors.
        for actor in self.piece_actors.values():
            self.renderer.RemoveActor(actor)
        self.piece_actors = {}

        for square, piece in board.piece_map().items():
            name = piece_type_name(piece.piece_type)
            color = (0.88, 0.85, 0.76) if piece.color else (0.085, 0.09, 0.10)
            actor = actor_for(self.pieces[name], color)
            jx, jy = jitter.get(square, (0.0, 0.0))
            actor.SetPosition((chess.square_file(square) + 0.5) * SQUARE + jx,
                              (chess.square_rank(square) + 0.5) * SQUARE + jy, 0.2)
            actor.SetOrientation(0, 0, yaw.get(square, 0.0))
            actor.SetVisibility(1)
            self.renderer.AddActor(actor)
            self.piece_actors[square] = actor

    def set_piece_colors(self, board: chess.Board, colors: dict[int, tuple[float, float, float]]):
        for square, actor in self.piece_actors.items():
            color = colors[square]
            actor.GetProperty().SetColor(*color)

    def set_piece_ambient_diffuse_specular(self, board: chess.Board, values: dict[int, tuple[float, float, float]]):
        for square, actor in self.piece_actors.items():
            ambient, diffuse, specular = values[square]
            actor.GetProperty().SetAmbient(ambient)
            actor.GetProperty().SetDiffuse(diffuse)
            actor.GetProperty().SetSpecular(specular)
            actor.GetProperty().SetSpecularPower(20)

    def set_light(self, position: tuple[float, float, float], intensity: float):
        self.light.SetPosition(*position)
        self.light.SetFocalPoint(200, 200, 25)
        self.light.SetIntensity(intensity)

    def configure_camera(self, elevation_deg: float, azimuth_deg: float, distance: float, fov: float):
        camera = self.renderer.GetActiveCamera()
        elev = math.radians(elevation_deg)
        azim = math.radians(azimuth_deg)
        # base direction from board centre (200,200) with slight offset for perspective
        base_x = 200.0
        base_y = 200.0
        horiz = distance * math.cos(elev)
        # Compute horizontal unit vector around the board centre: rotate the white-front direction.
        white_front = np.array([0.0, -1.0])
        cos_a, sin_a = math.cos(azim), math.sin(azim)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        offset = rotation @ white_front * horiz
        cam_x = base_x + offset[0]
        cam_y = base_y + offset[1]
        cam_z = distance * math.sin(elev)
        camera.SetPosition(cam_x, cam_y, cam_z)
        camera.SetFocalPoint(200, 200, 25)
        camera.SetViewUp(0, 0, 1)
        camera.SetViewAngle(fov)
        self.renderer.ResetCameraClippingRange()

    def render_rgb(self) -> np.ndarray:
        self.window.Render()
        self.capture_rgb.Modified()
        self.capture_rgb.Update()
        pixels = vtk_to_numpy(self.capture_rgb.GetOutput().GetPointData().GetScalars())
        img = pixels.reshape(HEIGHT, WIDTH, 3)[::-1].copy()
        return img

    def render_id(self, board: chess.Board) -> tuple[np.ndarray, dict[int, tuple[int, int, int]]]:
        # Save state
        saved_bg = self.renderer.GetBackground()
        saved_bg = tuple(saved_bg)
        saved_visibility = {sq: act.GetVisibility() for sq, act in self.piece_actors.items()}
        saved_lighting = {}
        saved_color = {}
        saved_opacity = {}
        saved_material = {}
        saved_table = self.table_actor.GetVisibility()
        saved_board = self.board_actor.GetVisibility()
        saved_squares = [a.GetVisibility() for _, _, a in self.square_actors]

        self.renderer.SetBackground(0.0, 0.0, 0.0)
        self.table_actor.SetVisibility(0)
        self.board_actor.SetVisibility(0)
        for _, _, act in self.square_actors:
            act.SetVisibility(0)

        squares = sorted(self.piece_actors.keys())
        palette: dict[int, tuple[int, int, int]] = {}
        for i, sq in enumerate(squares):
            if i >= 32:
                raise RuntimeError("more than 32 pieces on board")
            r = (i + 1) * 7
            g = (i + 1) * 5
            b = (i + 1) * 3
            palette[sq] = (r, g, b)
            act = self.piece_actors[sq]
            prop = act.GetProperty()
            saved_lighting[sq] = prop.GetLighting()
            saved_color[sq] = prop.GetColor()
            saved_opacity[sq] = prop.GetOpacity()
            saved_material[sq] = (prop.GetAmbient(), prop.GetDiffuse(), prop.GetSpecular())
            prop.LightingOff()
            # Palette RGB must be exact. VTK still multiplies actor RGB by
            # (Ambient + Diffuse) even with lighting off, so pin those
            # factors to sum to 1.0 (Ambient=1, Diffuse=0). Specular is
            # irrelevant with lighting off but pin it too for determinism.
            prop.SetAmbient(1.0)
            prop.SetDiffuse(0.0)
            prop.SetSpecular(0.0)
            prop.SetColor(r / 255.0, g / 255.0, b / 255.0)
            prop.SetOpacity(1.0)
            act.SetVisibility(1)

        self.window.Render()
        self.capture_id.Modified()
        self.capture_id.Update()
        pixels = vtk_to_numpy(self.capture_id.GetOutput().GetPointData().GetScalars())
        img = pixels.reshape(HEIGHT, WIDTH, 3)[::-1].copy()

        # Restore
        self.renderer.SetBackground(*saved_bg)
        self.table_actor.SetVisibility(saved_table)
        self.board_actor.SetVisibility(saved_board)
        for (_, _, act), vis in zip(self.square_actors, saved_squares):
            act.SetVisibility(vis)
        for sq, act in self.piece_actors.items():
            act.SetVisibility(saved_visibility[sq])
            prop = act.GetProperty()
            if saved_lighting[sq]:
                prop.LightingOn()
            else:
                prop.LightingOff()
            c = saved_color[sq]
            prop.SetColor(c[0], c[1], c[2])
            prop.SetOpacity(saved_opacity[sq])
            ambient, diffuse, specular = saved_material[sq]
            prop.SetAmbient(ambient)
            prop.SetDiffuse(diffuse)
            prop.SetSpecular(specular)

        return img, palette


def compute_instance_mask(id_img: np.ndarray, squares_order: list[int], palette: dict[int, tuple[int, int, int]]) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for idx, sq in enumerate(squares_order, start=1):
        r, g, b = palette[sq]
        match = (np.abs(id_img[:, :, 0].astype(np.int16) - r) <= 1) & \
                (np.abs(id_img[:, :, 1].astype(np.int16) - g) <= 1) & \
                (np.abs(id_img[:, :, 2].astype(np.int16) - b) <= 1)
        mask[match] = idx
    return mask


def compute_visible_box(mask: np.ndarray, instance_index: int):
    ys, xs = np.where(mask == instance_index)
    if xs.size == 0:
        return None, 0
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], int(xs.size)


def project_actor_amodal(renderer: Renderer360, actor: vtk.vtkActor, local_points: np.ndarray):
    matrix = actor_matrix(actor)
    proj, valid = project_vertices_world_to_display(renderer.renderer, local_points, matrix)
    if not np.any(valid):
        return None
    pts = proj[valid]
    xs = pts[:, 0]
    ys = pts[:, 1]
    x0 = float(np.min(xs))
    x1 = float(np.max(xs))
    y0 = float(np.min(ys))
    y1 = float(np.max(ys))
    # clip
    cx0 = max(0.0, min(float(WIDTH - 1), x0))
    cx1 = max(0.0, min(float(WIDTH - 1), x1))
    cy0 = max(0.0, min(float(HEIGHT - 1), y0))
    cy1 = max(0.0, min(float(HEIGHT - 1), y1))
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return [cx0, cy0, cx1, cy1]


def compute_board_corners(renderer: Renderer360) -> list[list[float]]:
    corners = []
    for x, y in ((0, 400), (400, 400), (400, 0), (0, 0)):
        renderer.SetWorldPoint(x, y, 0, 1)
        renderer.WorldToDisplay()
        px, py, _ = renderer.GetDisplayPoint()
        corners.append([float(px), float(HEIGHT - 1 - py)])
    return corners


def choose_elevation(rng: random.Random) -> float:
    bucket = rng.random()
    if bucket < 1.0 / 3.0:
        return rng.uniform(25.0, 40.0)
    elif bucket < 2.0 / 3.0:
        return rng.uniform(40.0, 60.0)
    else:
        return rng.uniform(60.0, 80.0)


def photometric_augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.astype(np.float32)
    brightness = rng.uniform(0.85, 1.15)
    out = out * brightness
    pil = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    blur_sigma = rng.uniform(0.0, 0.6)
    if blur_sigma > 0.01:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    arr = np.asarray(pil).astype(np.float32)
    noise_std = rng.uniform(0.0, 2.0)
    if noise_std > 0.01:
        noise = np.random.default_rng(rng.getrandbits(32)).normal(0.0, noise_std, arr.shape)
        arr = arr + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def make_jitter(rng: random.Random, board: chess.Board) -> dict[int, tuple[float, float]]:
    jitter = {}
    for square in board.piece_map():
        jitter[square] = (rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0))
    return jitter


def make_yaw(rng: random.Random, board: chess.Board) -> dict[int, float]:
    yaw = {}
    for square, piece in board.piece_map().items():
        base = 0.0 if piece.color else 180.0
        yaw[square] = base + rng.uniform(-12.0, 12.0)
    return yaw


def make_colors(rng: random.Random, board: chess.Board) -> dict[int, tuple[float, float, float]]:
    colors = {}
    for square, piece in board.piece_map().items():
        if piece.color:
            v = rng.uniform(0.72, 0.95)
            colors[square] = (v, v * rng.uniform(0.96, 1.0), v * rng.uniform(0.90, 0.98))
        else:
            v = rng.uniform(0.035, 0.18)
            colors[square] = (v, v, v * rng.uniform(0.95, 1.05))
    return colors


def make_material(rng: random.Random, board: chess.Board) -> dict[int, tuple[float, float, float]]:
    values = {}
    for square, piece in board.piece_map().items():
        if piece.color:
            ambient = rng.uniform(0.16, 0.30)
            diffuse = rng.uniform(0.68, 0.84)
            specular = rng.uniform(0.04, 0.18)
        else:
            ambient = rng.uniform(0.10, 0.24)
            diffuse = rng.uniform(0.55, 0.78)
            specular = rng.uniform(0.06, 0.22)
        values[square] = (ambient, diffuse, specular)
    return values


def generate_group_fen(seed: int):
    board = chess.Board()
    rng = random.Random(seed)
    target_ply = rng.randint(10, 120)
    plies = 0
    while plies < target_ply and not board.is_game_over():
        legal = list(board.legal_moves)
        if not legal:
            break
        move = rng.choice(legal)
        board.push(move)
        plies += 1
    return board.fen()


def build_plan(groups: int, views: int, seed: int, output: Path, mesh_sources: dict, excluded: list[str]) -> dict:
    rng = random.Random(seed)
    excluded_set = set(excluded)
    placements_seen = set()
    placement_seen_fens = {}
    candidate_seed_rng = random.Random(seed ^ 0x5F3759DF)
    raw_candidates = []
    attempts = 0
    max_attempts = groups * 500
    while len(raw_candidates) < groups and attempts < max_attempts:
        attempts += 1
        candidate_seed = candidate_seed_rng.randrange(1, 1 << 62)
        try:
            fen = generate_group_fen(candidate_seed)
        except Exception:
            continue
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        placement = fen.split(" ")[0]
        if placement in excluded_set:
            continue
        if placement in placements_seen:
            continue
        placements_seen.add(placement)
        raw_candidates.append({"fen": fen, "seed": candidate_seed})

    if len(raw_candidates) < groups:
        raise RuntimeError(f"only generated {len(raw_candidates)} unique placements, wanted {groups}")

    # Shuffle group IDs deterministically.
    id_rng = random.Random(seed ^ 0xA5A5A5A5)
    order = list(range(groups))
    id_rng.shuffle(order)

    n_train = round(0.70 * groups)
    n_val = round(0.15 * groups)
    n_test = groups - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = groups - n_train

    groups_list = []
    for slot, src_idx in enumerate(order):
        candidate = raw_candidates[src_idx]
        group_id = f"g{slot:04d}"
        if slot < n_train:
            split = "train"
        elif slot < n_train + n_val:
            split = "val"
        else:
            split = "test"
        # Per-image seeds
        per_view_seeds = []
        for v in range(views):
            per_view_seeds.append((seed + slot * 1000003 + v * 7919) & 0x7FFFFFFFFFFFFFFF)
        groups_list.append({
            "group_id": group_id,
            "fen": candidate["fen"],
            "placement": candidate["fen"].split(" ")[0],
            "generation_seed": candidate["seed"],
            "split": split,
            "image_seeds": per_view_seeds,
        })

    plan = {
        "config": {
            "groups": groups,
            "views": views,
            "seed": seed,
            "width": WIDTH,
            "height": HEIGHT,
            "square_size_mm": SQUARE,
            "king_height_mm": KING_HEIGHT,
            "total_images": groups * views,
            "class_names": CLASS_NAMES,
            "renderer_policy_version": "v2",
        },
        "mesh_sources": mesh_sources,
        "output": str(output),
        "excluded_placements": excluded,
        "groups": groups_list,
    }
    return plan


def plan_hash(plan: dict) -> str:
    payload = json.dumps({k: v for k, v in plan.items() if k not in ("output",)}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_or_create_plan(args, output: Path) -> dict:
    plan_path = output / "plan.json"
    mesh_sources = gather_mesh_sources()
    excluded = load_excluded_placements()
    fresh_plan = build_plan(args.groups, args.views, args.seed, output, mesh_sources, excluded)
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        existing_norm = {k: v for k, v in existing.items() if k != "output"}
        fresh_norm = {k: v for k, v in fresh_plan.items() if k != "output"}
        if existing_norm != fresh_norm:
            raise RuntimeError("existing plan.json does not match requested configuration; refusing to reuse output directory")
        return existing
    output.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(fresh_plan, indent=2), encoding="utf-8")
    return fresh_plan


def image_id(group_id: str, view_idx: int) -> str:
    return f"{group_id}_v{view_idx:02d}"


def sample_paths(output: Path, split: str, sid: str) -> dict:
    return {
        "image": output / "images" / split / f"{sid}.png",
        "label": output / "labels" / split / f"{sid}.txt",
        "mask": output / "masks" / split / f"{sid}.png",
        "record": output / "records" / split / f"{sid}.json",
    }


def file_sha256(path: Path) -> str:
    return sha256_file(path)


def render_sample(render360: Renderer360, plan: dict, group: dict, view_idx: int, output: Path, existing_record: dict | None) -> dict:
    sid = image_id(group["group_id"], view_idx)
    split = group["split"]
    paths = sample_paths(output, split, sid)
    for key in ("image", "label", "mask", "record"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)

    seed = group["image_seeds"][view_idx]
    rng = random.Random(seed)

    board = chess.Board(group["fen"])

    # Chooses.
    elevation = choose_elevation(rng)
    azimuth = rng.uniform(-15.0, 15.0)
    distance = rng.uniform(800.0, 1050.0)
    fov = rng.uniform(35.0, 43.0)

    jitter = make_jitter(rng, board)
    yaw = make_yaw(rng, board)

    board_color = (rng.uniform(0.55, 0.85), rng.uniform(0.45, 0.75), rng.uniform(0.30, 0.65))
    table_color = (rng.uniform(0.20, 0.55), rng.uniform(0.15, 0.45), rng.uniform(0.10, 0.40))
    second_board_color = (rng.uniform(0.10, 0.35), rng.uniform(0.06, 0.30), rng.uniform(0.04, 0.25))
    bg_color = (rng.uniform(0.10, 0.45), rng.uniform(0.10, 0.45), rng.uniform(0.10, 0.45))

    render360.table_actor.GetProperty().SetColor(*table_color)
    render360.board_actor.GetProperty().SetColor(*second_board_color)
    # apply board square colors deterministically: mix of two tones around board_color
    for idx, (rank, file, act) in enumerate(render360.square_actors):
        if (rank + file) % 2:
            act.GetProperty().SetColor(board_color[0], board_color[1], board_color[2])
        else:
            act.GetProperty().SetColor(second_board_color[0] * 0.9, second_board_color[1] * 0.9, second_board_color[2] * 0.9)
    render360.renderer.SetBackground(*bg_color)

    # Light
    light_pos = (rng.uniform(-200.0, 600.0), rng.uniform(-200.0, 600.0), rng.uniform(300.0, 800.0))
    light_intensity = rng.uniform(0.6, 1.2)
    render360.set_light(light_pos, light_intensity)

    # Build pieces.
    render360.build_scene(board, jitter, yaw)
    piece_colors = make_colors(rng, board)
    piece_materials = make_material(rng, board)
    render360.set_piece_colors(board, piece_colors)
    render360.set_piece_ambient_diffuse_specular(board, piece_materials)

    render360.configure_camera(elevation, azimuth, distance, fov)

    rgb = render360.render_rgb()
    rgb = photometric_augment(rgb, rng)

    # Board corners
    corners = compute_board_corners(render360.renderer)

    # Amodal boxes per piece.
    squares_order = sorted(render360.piece_actors.keys())
    amodal_boxes = {}
    base_points = {}
    for square in squares_order:
        actor = render360.piece_actors[square]
        name = piece_type_name(board.piece_at(square).piece_type)
        local_pts = render360.piece_mesh_points[name]
        box = project_actor_amodal(render360, actor, local_pts)
        amodal_boxes[square] = box
        # base
        m = actor_matrix(actor)
        base_h = np.array([0.0, 0.0, 0.0, 1.0])
        world = m @ base_h
        w_scale = world[3] if abs(world[3]) > 1e-9 else 1.0
        bx, by, bz = world[0] / w_scale, world[1] / w_scale, world[2] / w_scale
        render360.renderer.SetWorldPoint(float(bx), float(by), float(bz), 1.0)
        render360.renderer.WorldToDisplay()
        px, py, _ = render360.renderer.GetDisplayPoint()
        base_points[square] = [float(px), float(HEIGHT - 1 - py)]

    # Instance mask
    id_img, palette = render360.render_id(board)
    mask = compute_instance_mask(id_img, squares_order, palette)

    # Verify at least some IDs
    total_ids = int(np.count_nonzero(mask))
    if total_ids == 0:
        raise RuntimeError(f"palette id rendering failed for sample {sid}: no instance pixels")

    # Save mask (lossless, fast deflate)
    Image.fromarray(mask, mode="L").save(paths["mask"], compress_level=1)

    # Build instances
    instances = []
    yolo_lines = []
    for idx, square in enumerate(squares_order, start=1):
        piece = board.piece_at(square)
        name = piece_type_name(piece.piece_type)
        color = piece.color
        class_id = CLASS_IDS[(name, color)]
        symbol = piece.symbol()
        amodal = amodal_boxes[square]
        visible_box, visible_pixels = compute_visible_box(mask, idx)
        labelled = False
        if visible_pixels >= 20 and amodal is not None:
            x0, y0, x1, y1 = amodal
            if x1 > x0 and y1 > y0:
                labelled = True
                w = x1 - x0
                h = y1 - y0
                xc = (x0 + x1) / 2.0 / WIDTH
                yc = (y0 + y1) / 2.0 / HEIGHT
                nw = w / WIDTH
                nh = h / HEIGHT
                # Clip normalised ranges
                xc = min(max(xc, 0.0), 1.0)
                yc = min(max(yc, 0.0), 1.0)
                nw = min(max(nw, 0.0), 1.0)
                nh = min(max(nh, 0.0), 1.0)
                yolo_lines.append(f"{class_id} {xc:.8f} {yc:.8f} {nw:.8f} {nh:.8f}")
        instances.append({
            "symbol": symbol,
            "square": chess.square_name(square),
            "class_id": class_id,
            "amodal_box": [] if amodal is None else [float(v) for v in amodal],
            "visible_box": [] if visible_box is None else [float(v) for v in visible_box],
            "visible_pixels": int(visible_pixels),
            "base_image_point": base_points[square],
            "labelled": labelled,
        })

    # Save image (lossless, fast deflate)
    Image.fromarray(rgb).save(paths["image"], compress_level=1)

    # Save labels
    paths["label"].write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

    # Truth board: 64 symbols in row-major order a8..h1 (rank 7 down to 0).
    truth_symbols = []
    for rank in range(7, -1, -1):
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece is None:
                truth_symbols.append("_")
            else:
                symbol = PIECE_SYMBOLS[piece.piece_type]
                truth_symbols.append(symbol if piece.color else symbol.lower())

    image_rel = str(paths["image"].relative_to(output).as_posix())
    record = {
        "id": sid,
        "group_id": group["group_id"],
        "split": split,
        "fen": group["fen"],
        "truth": list(truth_symbols),
        "corners": corners,
        "image": image_rel,
        "width": WIDTH,
        "height": HEIGHT,
        "elevation": float(elevation),
        "azimuth": float(azimuth),
        "draw_parameters": {
            "distance": float(distance),
            "fov": float(fov),
            "jitter": {chess.square_name(sq): list(v) for sq, v in jitter.items()},
            "yaw": {chess.square_name(sq): float(v) for sq, v in yaw.items()},
            "board_color": list(board_color),
            "second_board_color": list(second_board_color),
            "table_color": list(table_color),
            "background_color": list(bg_color),
            "light_position": list(light_pos),
            "light_intensity": float(light_intensity),
            "piece_colors": {chess.square_name(sq): list(v) for sq, v in piece_colors.items()},
            "piece_materials": {chess.square_name(sq): list(v) for sq, v in piece_materials.items()},
        },
        "instances": instances,
        "seed": int(seed),
        "renderer_info": {
            "type": "vtk_offscreen",
            "multisamples": 0,
            "size": [WIDTH, HEIGHT],
            "annotation_policy": "amodal_projected_full_mesh",
        },
    }
    record["image_sha256"] = file_sha256(paths["image"])
    record["label_sha256"] = file_sha256(paths["label"])
    record["mask_sha256"] = file_sha256(paths["mask"])
    record["annotation_policy"] = "amodal_projected_full_mesh"

    # Sidecar written last as completion marker.
    paths["record"].write_text(json.dumps(record, indent=2), encoding="utf-8")
    if not validate_record(record, output):
        raise RuntimeError(f"self-validation failed for sample {sid}")
    return record


def validate_record(record: dict, output: Path) -> bool:
    try:
        sid = record["id"]
        split = record["split"]
        paths = sample_paths(output, split, sid)
        for key in ("image", "label", "mask", "record"):
            if not paths[key].exists():
                return False
        if file_sha256(paths["image"]) != record.get("image_sha256"):
            return False
        if file_sha256(paths["label"]) != record.get("label_sha256"):
            return False
        if file_sha256(paths["mask"]) != record.get("mask_sha256"):
            return False
        # Basic content sanity
        if record.get("width") != WIDTH or record.get("height") != HEIGHT:
            return False
        for inst in record.get("instances", []):
            cid = inst.get("class_id")
            if not isinstance(cid, int) or cid < 0 or cid > 11:
                return False
            for key in ("amodal_box", "visible_box"):
                box = inst.get(key) or []
                if box:
                    if len(box) != 4:
                        return False
                    for v in box:
                        if not math.isfinite(v):
                            return False
                    if box[2] < box[0] or box[3] < box[1]:
                        return False

        # Deep checks: images/mask/truth/instances/labels.
        with Image.open(paths["image"]) as img:
            if img.size != (WIDTH, HEIGHT):
                return False
        with Image.open(paths["mask"]) as mimg:
            if mimg.size != (WIDTH, HEIGHT):
                return False
            mask_np = np.asarray(mimg.convert("L"), dtype=np.uint8)
        if mask_np.shape != (HEIGHT, WIDTH) or mask_np.dtype != np.uint8:
            return False

        truth = record.get("truth")
        if not isinstance(truth, list) or len(truth) != 64:
            return False
        board = chess.Board(record["fen"])
        expected_truth = []
        for rank in range(7, -1, -1):
            for file in range(8):
                sq = chess.square(file, rank)
                piece = board.piece_at(sq)
                if piece is None:
                    expected_truth.append("_")
                else:
                    expected_truth.append(piece.symbol())
        if truth != expected_truth:
            return False

        instances = record.get("instances", [])
        for idx, inst in enumerate(instances, start=1):
            pixels = int(np.count_nonzero(mask_np == idx))
            if pixels != int(inst.get("visible_pixels", -1)):
                return False
            vbox = inst.get("visible_box") or []
            abox = inst.get("amodal_box") or []
            if vbox:
                if not abox:
                    return False
                if (vbox[0] < abox[0] - 2.0 or vbox[1] < abox[1] - 2.0 or
                        vbox[2] > abox[2] + 2.0 or vbox[3] > abox[3] + 2.0):
                    return False
            sq_name = inst.get("square")
            if isinstance(sq_name, str) and len(sq_name) == 2:
                sq_idx = chess.parse_square(sq_name)
                piece = board.piece_at(sq_idx)
                if piece is None:
                    return False
                if CLASS_IDS[(piece_type_name(piece.piece_type), piece.color)] != inst.get("class_id"):
                    return False

        # Parse YOLO label lines and cross-check against instances.
        label_lines = [ln for ln in paths["label"].read_text(encoding="utf-8").splitlines() if ln.strip()]
        labelled_instances = [inst for inst in instances if inst.get("labelled")]
        if len(label_lines) != len(labelled_instances):
            return False
        for line, inst in zip(label_lines, labelled_instances):
            parts = line.split()
            if len(parts) != 5:
                return False
            cid_label = int(parts[0])
            vals = [float(p) for p in parts[1:]]
            for v in vals:
                if not math.isfinite(v) or v < 0.0 or v > 1.0:
                    return False
            if vals[2] <= 0.0 or vals[3] <= 0.0:
                return False
            if cid_label != int(inst["class_id"]):
                return False
            abox = inst.get("amodal_box") or []
            if len(abox) != 4:
                return False
            x0, y0, x1, y1 = abox
            xc = (x0 + x1) / 2.0 / WIDTH
            yc = (y0 + y1) / 2.0 / HEIGHT
            nw = (x1 - x0) / WIDTH
            nh = (y1 - y0) / HEIGHT
            if (abs(vals[0] - xc) > 1e-7 or abs(vals[1] - yc) > 1e-7 or
                    abs(vals[2] - nw) > 1e-7 or abs(vals[3] - nh) > 1e-7):
                return False
        return True
    except (ValueError, KeyError, OSError):
        return False


def load_completed_records(output: Path, plan: dict) -> dict:
    completed = {}
    for group in plan["groups"]:
        for view_idx in range(plan["config"]["views"]):
            sid = image_id(group["group_id"], view_idx)
            paths = sample_paths(output, group["split"], sid)
            if not paths["record"].exists():
                continue
            try:
                rec = json.loads(paths["record"].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if rec.get("id") != sid:
                continue
            if validate_record(rec, output):
                completed[sid] = rec
    return completed


def write_manifests(output: Path, plan: dict, completed: dict, elapsed_seconds: float):
    records_path = output / "records.jsonl"
    with open(records_path, "w", encoding="utf-8") as handle:
        for group in plan["groups"]:
            for view_idx in range(plan["config"]["views"]):
                sid = image_id(group["group_id"], view_idx)
                rec = completed.get(sid)
                if rec is None:
                    continue
                handle.write(json.dumps(rec, sort_keys=True) + "\n")

    dataset_yaml = {
        "path": str(output.resolve().as_posix()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")

    split_counts = {"train": 0, "val": 0, "test": 0}
    planned_counts = {"train": 0, "val": 0, "test": 0}
    class_counts = {i: 0 for i in range(12)}
    unlabelled_objects = 0
    labelled_objects = 0
    for group in plan["groups"]:
        planned_counts[group["split"]] += plan["config"]["views"]
    for rec in completed.values():
        split_counts[rec["split"]] += 1
        for inst in rec["instances"]:
            if inst["labelled"]:
                class_counts[inst["class_id"]] += 1
                labelled_objects += 1
            else:
                unlabelled_objects += 1

    # Old placement intersection check MUST be zero.
    excluded = set(plan["excluded_placements"])
    placements_intersection_with_old = 0
    for group in plan["groups"]:
        if group["placement"] in excluded:
            placements_intersection_with_old += 1
    if placements_intersection_with_old != 0:
        raise RuntimeError(
            f"planned placements intersect excluded old scenes: {placements_intersection_with_old}"
        )

    # Split-set intersections of placements.
    split_placements = {"train": set(), "val": set(), "test": set()}
    for group in plan["groups"]:
        split_placements[group["split"]].add(group["placement"])
    tv = len(split_placements["train"] & split_placements["val"])
    tt = len(split_placements["train"] & split_placements["test"])
    vt = len(split_placements["val"] & split_placements["test"])
    if tv or tt or vt:
        raise RuntimeError(
            f"split placement leakage detected: train_val={tv} train_test={tt} val_test={vt}"
        )
    split_placement_intersections = {
        "train_val": tv,
        "train_test": tt,
        "val_test": vt,
    }

    planned_total = plan["config"]["total_images"]
    complete_total = len(completed)
    complete = complete_total == planned_total

    statistics = {
        "counts_per_split": split_counts,
        "planned_counts_per_split": planned_counts,
        "planned_total": planned_total,
        "complete_total": complete_total,
        "complete": complete,
        "label_classes": class_counts,
        "labelled_objects": labelled_objects,
        "unlabelled_objects": unlabelled_objects,
        "placements_intersection_with_old": placements_intersection_with_old,
        "split_placement_intersections": split_placement_intersections,
        "annotation_policy": "amodal_projected_full_mesh",
        "visible_policy": "id_mask_visible_bbox",
        "min_visible_pixels_for_label": 20,
        "elapsed_seconds": elapsed_seconds,
    }
    (output / "statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    return statistics


def make_audit_contact_sheet(output: Path, plan: dict, completed: dict):
    wanted_splits = {"train": [], "val": [], "test": []}
    for group in plan["groups"]:
        split = group["split"]
        for view_idx in range(plan["config"]["views"]):
            sid = image_id(group["group_id"], view_idx)
            if sid in completed:
                wanted_splits[split].append(sid)

    # First 18 samples overall ordered by split priority: train, val, test.
    chosen = []
    for split in ("train", "val", "test"):
        for sid in wanted_splits[split]:
            if len(chosen) < 18:
                chosen.append(sid)

    if not chosen:
        return

    cell_w, cell_h = 320, 240
    cols = 6
    rows = math.ceil(len(chosen) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * (cell_h + 18)), "#101820")
    draw = ImageDraw.Draw(sheet)
    for i, sid in enumerate(chosen):
        rec = completed[sid]
        path = output / rec["image"]
        with Image.open(path).convert("RGB") as img:
            img = img.resize((cell_w, cell_h))
            col = i % cols
            row = i // cols
            sheet.paste(img, (col * cell_w, row * (cell_h + 18)))
        overlay = ImageDraw.Draw(sheet)
        for inst in rec["instances"]:
            if inst["amodal_box"]:
                x0, y0, x1, y1 = inst["amodal_box"]
                sx = cell_w / WIDTH
                sy = cell_h / HEIGHT
                overlay.rectangle([col * cell_w + x0 * sx, row * (cell_h + 18) + y0 * sy,
                                   col * cell_w + x1 * sx, row * (cell_h + 18) + y1 * sy],
                                  outline=(0, 255, 0))
            if inst["visible_box"]:
                x0, y0, x1, y1 = inst["visible_box"]
                sx = cell_w / WIDTH
                sy = cell_h / HEIGHT
                overlay.rectangle([col * cell_w + x0 * sx, row * (cell_h + 18) + y0 * sy,
                                   col * cell_w + x1 * sx, row * (cell_h + 18) + y1 * sy],
                                  outline=(255, 165, 0))
        overlay.text((col * cell_w + 4, row * (cell_h + 18) + cell_h + 2),
                     f"{sid} elev={rec['elevation']:.1f}", fill="white")
    sheet.save(output / "audit.jpg")


def make_qa_images(output: Path, plan: dict, completed: dict):
    qa_dir = output / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    # find lowest & highest elevation samples
    records = list(completed.values())
    if not records:
        return
    lowest = min(records, key=lambda r: r["elevation"])
    highest = max(records, key=lambda r: r["elevation"])
    selected = [lowest, highest]
    # add a couple more from mid elevations if present
    mids = sorted(records, key=lambda r: abs(r["elevation"] - 50.0))[:2]
    for rec in mids:
        if rec["id"] not in {s["id"] for s in selected}:
            selected.append(rec)
    for rec in selected:
        path = output / rec["image"]
        with Image.open(path).convert("RGB") as img:
            draw = ImageDraw.Draw(img)
            for inst in rec["instances"]:
                if inst["amodal_box"]:
                    x0, y0, x1, y1 = inst["amodal_box"]
                    draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
                    symbol = inst.get("symbol", "")
                    if symbol:
                        draw.text((x0 + 2, max(0, y0 - 12)), symbol, fill=(0, 255, 0))
                if inst["visible_box"]:
                    x0, y0, x1, y1 = inst["visible_box"]
                    draw.rectangle([x0, y0, x1, y1], outline=(255, 165, 0), width=2)
            draw.text((6, 6), f"{rec['id']} elev={rec['elevation']:.1f} azim={rec['azimuth']:.1f}", fill="white")
            img.save(qa_dir / f"{rec['id']}.jpg", quality=90)


def audit_only(output: Path, plan: dict):
    completed = load_completed_records(output, plan)
    elapsed = 0.0
    write_manifests(output, plan, completed, elapsed)
    make_audit_contact_sheet(output, plan, completed)
    make_qa_images(output, plan, completed)
    all_planned = plan["config"]["total_images"]
    if len(completed) < all_planned:
        print(json.dumps({"status": "incomplete", "count": len(completed), "planned": all_planned}), flush=True)
    else:
        print(json.dumps({"status": "complete", "count": len(completed), "planned": all_planned}), flush=True)


def run(args):
    output = Path(args.output).resolve() if args.output else DEFAULT_OUTPUT.resolve()
    if not output.is_relative_to(ROOT):
        raise SystemExit(f"refusing output directory outside repository root: {output}")
    output.mkdir(parents=True, exist_ok=True)
    plan = load_or_create_plan(args, output)
    # Persist/refresh plan file (only if new).
    plan_path = output / "plan.json"
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    if args.audit_only:
        audit_only(output, plan)
        return

    sets, scales = load_sets()
    pieces = sets["dubrovnik"]

    started = time.time()
    completed = load_completed_records(output, plan)
    print(json.dumps({"resumed": len(completed), "planned": plan["config"]["total_images"]}), flush=True)

    render360 = Renderer360(pieces)

    total_planned = plan["config"]["total_images"]
    processed = 0

    # Enumerate all planned slots in order, then apply limit to the first N
    # planned slots (including ones already completed).
    planned_slots = []
    for group in plan["groups"]:
        for view_idx in range(plan["config"]["views"]):
            sid = image_id(group["group_id"], view_idx)
            planned_slots.append((group, view_idx, sid))

    if args.limit and args.limit > 0:
        planned_slots = planned_slots[: args.limit]

    to_render = []
    for group, view_idx, sid in planned_slots:
        if sid in completed:
            continue
        to_render.append((group, view_idx, sid))

    for group, view_idx, sid in to_render:
        record = render_sample(render360, plan, group, view_idx, output, None)
        completed[sid] = record
        processed += 1
        if processed % 25 == 0:
            print(json.dumps({"rendered": processed, "total_done": len(completed), "planned": total_planned}), flush=True)

    render360.window.Finalize()

    elapsed = time.time() - started
    statistics = write_manifests(output, plan, completed, elapsed)
    make_audit_contact_sheet(output, plan, completed)
    make_qa_images(output, plan, completed)

    summary = {
        "output": str(output),
        "planned": total_planned,
        "completed": len(completed),
        "statistics": statistics,
        "complete": len(completed) >= total_planned,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Build Dubrovnik finetune dataset")
    parser.add_argument("--groups", type=int, default=DEFAULT_GROUPS)
    parser.add_argument("--views", type=int, default=DEFAULT_VIEWS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--audit-only", action="store_true", dest="audit_only")
    args = parser.parse_args()
    if args.groups < 3:
        parser.error("--groups must be >= 3")
    if args.views < 1:
        parser.error("--views must be >= 1")
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    run(args)


if __name__ == "__main__":
    main()
