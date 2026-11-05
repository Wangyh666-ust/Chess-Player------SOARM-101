"""Plan lighting, render paired views, and validate a saved batch of legal setups.

Run outside Blender. Source configs and assets are never modified. A run is ready
only when run.json says 'passed' and data/dataset.yaml has been published.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess

import chess

from randomized_lighting import sample_lighting
from randomized_cameras import framing_points, sample_camera

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def validate_positions(source):
    """Reject duplicate layouts, unsafe filenames and mismatched game histories."""
    if not source["positions"]:
        raise ValueError("No positions to render")
    games = {game["id"]: game for game in source.get("setup_generation", {}).get("games", [])}
    ids, placements, game_splits = set(), set(), {}
    for entry in source["positions"]:
        name, split = entry["id"], entry["split"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in ids:
            raise ValueError(f"Invalid or duplicate position ID: {name!r}")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        board = chess.Board(entry["fen"])
        if not board.is_valid() or board.board_fen() in placements:
            raise ValueError(f"Invalid or repeated board placement: {name}")
        ids.add(name)
        placements.add(board.board_fen())
        game_id = entry.get("game_id")
        if game_id is not None:
            if game_splits.setdefault(game_id, split) != split:
                raise ValueError(f"Game {game_id} crosses dataset splits")
            game = games[game_id]
            ply = entry["ply"]
            if type(ply) is not int or not 0 <= ply <= len(game["moves_uci"]):
                raise ValueError(f"Invalid game ply: {name}")
            replay = chess.Board()
            for uci in game["moves_uci"][:ply]:
                replay.push_uci(uci)  # Includes legal-move validation.
            if replay.fen() != entry["fen"] or game["split"] != split:
                raise ValueError(f"Game history does not match setup: {name}")


def draw_seed(seed, position_id, stream):
    """Independent streams: changing camera bounds must not change piece poses/lights."""
    digest = hashlib.sha256(f"{seed}:{position_id}:{stream}".encode()).digest()
    return int.from_bytes(digest[:6], "big")


def validate_piece_bounds(settings, square_mm, diameters):
    if set(settings) != {"offset_mm", "cell_margin_mm", "yaw_degrees"}:
        raise ValueError("piece_randomization needs offset_mm, cell_margin_mm and yaw_degrees")
    for key, value in settings.items():
        if key == "offset_mm" and value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid piece randomization bound: {key}")
    if settings["yaw_degrees"] > 180:
        raise ValueError("yaw_degrees must be at most 180 (a full turn)")
    if any(not math.isfinite(v) or v <= 0 for v in [square_mm, *diameters.values()]):
        raise ValueError("Piece/board dimensions must be positive and finite")
    if max(diameters.values()) + 2 * settings["cell_margin_mm"] > square_mm:
        raise ValueError("Largest base plus margin cannot fit inside a cell")


def sample_piece_transforms(board, rng, settings, square_mm, diameters):
    """Use each circular base's available cell space; numeric offset caps each axis."""
    poses = {}
    for square, piece in sorted(board.piece_map().items()):
        diameter = diameters[chess.piece_name(piece.piece_type)]
        limit = (square_mm - diameter) / 2 - settings["cell_margin_mm"]
        if settings["offset_mm"] is not None:
            limit = min(limit, settings["offset_mm"])
        poses[chess.square_name(square)] = {
            "offset_mm": [rng.uniform(-limit, limit), rng.uniform(-limit, limit)],
            "yaw_degrees": ((90 if piece.color else -90)
                            + rng.uniform(-settings["yaw_degrees"], settings["yaw_degrees"])) % 360,
        }
    return poses


def prepare_dataset(source, settings, presets, base_config):
    """Return the exact render plan without starting Blender or writing files."""
    validate_positions(source)
    variant_count = settings.get("variants_per_setup", 1)
    if type(variant_count) is not int or variant_count < 1:
        raise ValueError("variants_per_setup must be a positive integer")
    for key in ("samples", "threads"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(settings["seed"]) is not int:
        raise ValueError("seed must be an integer")
    scale = settings["resolution_scale"]
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("resolution_scale must be finite and positive")
    variants = {variant["id"]: variant for variant in presets["variants"]}
    names = settings["lighting_presets"]
    if not isinstance(names, list) or not names or any(name not in variants for name in names):
        raise ValueError("lighting_presets must name at least one known lighting preset")
    # Validate every selected preset, even one the random draw might not choose.
    for name in names:
        overrides = variants[name]["overrides"]
        if set(overrides) != {"lights", "world_strength", "world_color", "exposure"}:
            raise ValueError(f"Unsupported preset overrides: {name}")
        sample_lighting(overrides["lights"], overrides["world_strength"],
                        seed=settings["seed"], bounds=settings["lighting_bounds"])
    result = copy.deepcopy(base_config)
    if set(result["cameras"]) != {"fixed", "arm"}:
        raise ValueError("Scene config must contain fixed and arm cameras")
    if set(settings["camera_randomization"]) != set(result["cameras"]):
        raise ValueError("Camera randomization must configure fixed and arm views")
    # All physical variants retain the source FEN, game and split. Their IDs give
    # them independent random streams while keeping the source position traceable.
    result["positions"] = []
    for entry in source["positions"]:
        for variant in range(variant_count):
            expanded = copy.deepcopy(entry)
            expanded["source_setup_id"] = entry["id"]
            expanded["variant_index"] = variant
            if variant_count > 1:
                expanded["id"] = f"{entry['id']}_v{variant:03d}"
            result["positions"].append(expanded)
    if len({entry["id"] for entry in result["positions"]}) != len(result["positions"]):
        raise ValueError("Expanded setup IDs collide")
    result["setup_generation"] = copy.deepcopy(source.get("setup_generation", {}))
    result.pop("reference_note", None)
    dimensions = {
        kind: json.loads((ROOT / asset["file"]).with_name("dimensions.json").read_text())
        for kind, asset in result["assets"].items()
    }
    pitch = dimensions["board"]["square_mm"]
    diameters = {kind: value["base_diameter_mm"] for kind, value in dimensions.items() if kind != "board"}
    piece_settings = settings["piece_randomization"]
    validate_piece_bounds(piece_settings, pitch, diameters)
    tallest = max(value["height_mm"] for kind, value in dimensions.items() if kind != "board")
    frame = framing_points(dimensions["board"], tallest)
    # Every physical draw is explicit; do not jitter it again inside Blender.
    for key in ["piece_jitter_mm", "piece_yaw_jitter_degrees",
                "camera_position_jitter_mm", "camera_target_jitter_mm"]:
        result[key] = 0
    result.update(seed=settings["seed"], samples=settings["samples"], require_kings=True)
    result["lighting_note"] = "Per-setup preset and bounded lighting draw saved in positions[].lighting_setup."
    result.pop("lighting_randomization", None)  # Lighting is now explicitly saved per setup.
    for camera in result["cameras"].values():
        camera["resolution"] = [max(1, round(value * scale))
                                for value in camera.get("resolution", result["resolution"])]
    for entry in result["positions"]:
        piece_seed = draw_seed(settings["seed"], entry["id"], "pieces")
        # Legacy setup files may contain physical poses; the dataset now owns them.
        entry["piece_transforms"] = sample_piece_transforms(
            chess.Board(entry["fen"]), random.Random(piece_seed), piece_settings, pitch, diameters)
        entry["piece_seed"] = piece_seed
        entry["camera_setup"] = {
            view: sample_camera(spec, settings["camera_randomization"][view], frame,
                                seed=draw_seed(settings["seed"], entry["id"], view))
            for view, spec in result["cameras"].items()
        }
        rng = random.Random(draw_seed(settings["seed"], entry["id"], "lighting"))
        name = rng.choice(names)
        overrides = variants[name]["overrides"]
        sampled = sample_lighting(overrides["lights"], overrides["world_strength"],
                                  seed=rng.getrandbits(53), bounds=settings["lighting_bounds"])
        entry["lighting_setup"] = {
            **sampled, "preset_id": name, "world_color": overrides["world_color"],
            "exposure": rng.uniform(*overrides["exposure"]),
        }
    return result


def run_dataset(source_path, settings_path, *, output=None, limit=None, dry_run=False):
    source_path, settings_path = Path(source_path).resolve(), Path(settings_path).resolve()
    source = json.loads(source_path.read_text())
    settings = json.loads(settings_path.read_text())
    presets_path = (settings_path.parent / settings["presets_file"]).resolve()
    presets = json.loads(presets_path.read_text())
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        source["positions"] = source["positions"][:limit]
    base_path = (settings_path.parent / settings["base_config"]).resolve()
    base = json.loads(base_path.read_text())
    render_config = prepare_dataset(source, settings, presets, base)
    destination = (ROOT / (output or source["output"])).resolve()
    if not destination.is_relative_to(ROOT) or destination.exists():
        raise ValueError("Output must be a new directory inside this project")
    render_config["output"] = str((destination / "data").relative_to(ROOT))
    summary = {"source_positions": len(source["positions"]),
               "variants_per_setup": settings.get("variants_per_setup", 1),
               "positions": len(render_config["positions"]),
               "images": 2 * len(render_config["positions"]), "output": str(destination)}
    print(json.dumps(summary, indent=2), flush=True)
    if dry_run:
        return summary

    # Fail on missing checker dependencies before spending time rendering.
    from check_dataset import check as check_dataset
    from check_lighting import check as check_lighting

    destination.mkdir(parents=True, exist_ok=False)
    plan_path = destination / "render_config.json"
    plan_path.write_text(json.dumps(render_config, indent=2, allow_nan=False) + "\n")
    # Embed inputs so changing a preset file later cannot change this run's record.
    snapshot = {"settings": settings, "presets": presets, "setups_file": str(source_path),
                "setups_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "base_config": base, "base_config_file": str(base_path)}
    (destination / "settings.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    state = {**summary, "status": "rendering"}

    def save_state():
        (destination / "run.json").write_text(json.dumps(state, indent=2) + "\n")

    save_state()
    data = destination / "data"
    dataset_yaml = data / "dataset.yaml"
    pending_yaml = data / "dataset.unvalidated.yaml"
    try:
        with (destination / "render.log").open("w") as log:
            subprocess.run([
                str(ROOT / ".local-blender/start-blender.sh"), "--background",
                "--threads", str(settings["threads"]), "--python-exit-code", "1",
                "--python", str(HERE / "render_dataset.py"), "--", "--config", str(plan_path),
            ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        # Training entry point is published only after both independent checks pass.
        dataset_yaml.rename(pending_yaml)
        state["status"] = "checking"
        save_state()
        print("Rendering complete; checking labels, paired views and lighting.", flush=True)
        errors = {}
        for name, check in (("dataset", lambda: check_dataset(data)),
                            ("lighting", lambda: check_lighting(data, presets["quality_limits"]))):
            try:
                check()
            except Exception as error:
                errors[name] = f"{type(error).__name__}: {error}"
        if errors:
            state["check_errors"] = errors
            raise RuntimeError("Dataset checks failed; inspect run.json and data/*qa.json")
        pending_yaml.rename(dataset_yaml)
        state["status"] = "passed"
        save_state()
    except BaseException as error:
        # Keep partial renders/logs for diagnosis, but never leave a failed training entry point.
        if dataset_yaml.exists():
            dataset_yaml.rename(pending_yaml)
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_state()
        raise
    print("Dataset ready:", dataset_yaml, flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setups", type=Path, required=True, help="JSON saved by create_setups.py")
    parser.add_argument("--config", type=Path, default=HERE / "configs/dataset.json")
    parser.add_argument("--output", help="New run directory, relative to project root")
    parser.add_argument("--limit", type=int, help="Use only the first N source setups, before expanding variants")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing or rendering")
    args = parser.parse_args()
    run_dataset(args.setups, args.config, output=args.output, limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
