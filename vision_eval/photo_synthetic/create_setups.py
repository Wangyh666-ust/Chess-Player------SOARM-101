"""Play legal games and save logical positions; no scene assets or Blender required.

Stockfish self-play is the default. The explicit random mode is a legal-position
stress test, not a model of human play. Physical randomization belongs to create_dataset.py.
"""

import argparse
from collections import Counter
from contextlib import nullcontext
import copy
import json
import math
from pathlib import Path
import random

import chess
import chess.engine

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PHASES = ("opening", "middlegame", "endgame")


def phase_of(board, ply):
    """Visual sampling buckets, not a precise strategic phase classification."""
    if len(board.piece_map()) <= 12:
        return "endgame"
    return "opening" if ply <= 20 else "middlegame"


def choose_move(board, rng, engine, settings, game_id):
    if engine is None:
        return rng.choice(sorted(board.legal_moves, key=lambda move: move.uci()))
    # Several good moves provide diversity without intentionally making blunders.
    lines = engine.analyse(
        board, chess.engine.Limit(nodes=settings["engine_nodes"]),
        multipv=settings["candidate_moves"], game=game_id,
    )
    scored = [(line["score"].pov(board.turn).score(mate_score=100000),
               line["pv"][0]) for line in lines if line.get("pv")]
    best = max(score for score, _ in scored)
    return rng.choice([move for score, move in scored
                       if best - score <= settings["max_loss_cp"]])


def validate_settings(settings):
    """Validate game-generation controls before starting any games."""
    integer_fields = ("count", "games", "max_plies", "sample_every_plies",
                      "positions_per_phase_per_game", "engine_nodes", "candidate_moves")
    for key in integer_fields:
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(settings["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if (type(settings["min_ply"]) is not int
            or not 0 <= settings["min_ply"] <= settings["max_plies"]):
        raise ValueError("min_ply must be between zero and max_plies")
    value = settings["max_loss_cp"]
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("max_loss_cp must be finite and nonnegative")
    splits = settings["split_weights"]
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("split_weights must contain train, val and test")
    if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in splits.values()) or sum(splits.values()) <= 0:
        raise ValueError("split_weights must be finite, nonnegative and have a positive sum")
    if settings["mode"] not in ("engine", "random"):
        raise ValueError("mode must be engine or random")


def create_setups(settings, *, engine=None):
    """Return legal FENs, histories and game-based splits without physical poses."""
    validate_settings(settings)
    if (engine is None) != (settings["mode"] == "random"):
        raise ValueError("engine mode requires an engine; random mode requires no engine")
    if engine is not None:
        engine.configure({"Threads": 1, "Hash": 32})

    rng = random.Random(settings["seed"])
    pools = {phase: [] for phase in PHASES}
    seen_placements = set()
    games = []
    for game_index in range(settings["games"]):
        game_seed = rng.getrandbits(64)
        move_rng = random.Random(game_seed)
        game_id = f"game_{game_index:05d}"
        board = chess.Board()
        candidates = {phase: [] for phase in PHASES}
        moves = []
        for ply in range(settings["max_plies"] + 1):
            if ply >= settings["min_ply"] and (ply - settings["min_ply"]) % settings["sample_every_plies"] == 0:
                phase = phase_of(board, ply)
                candidates[phase].append({"fen": board.fen(), "ply": ply, "phase": phase})
            if board.is_game_over(claim_draw=True) or ply == settings["max_plies"]:
                break
            move = choose_move(board, move_rng, engine, settings, game_id)
            if move not in board.legal_moves:
                raise ValueError(f"Illegal move from player: {move}")
            moves.append(move.uci())
            board.push(move)
        games.append({"id": game_id, "seed": game_seed,
                      "moves_uci": moves, "result": board.result(claim_draw=True),
                      "termination": "game_over" if board.is_game_over(claim_draw=True) else "ply_limit"})
        for phase, entries in candidates.items():
            rng.shuffle(entries)
            accepted = 0
            for entry in entries:
                placement = entry["fen"].split()[0]
                if placement in seen_placements:
                    continue
                seen_placements.add(placement)
                pools[phase].append({**entry, "game_id": game_id})
                accepted += 1
                if accepted == settings["positions_per_phase_per_game"]:
                    break

    available = sum(map(len, pools.values()))
    if available < settings["count"]:
        raise ValueError(f"Only {available} distinct setups for requested {settings['count']}; increase games or positions_per_phase_per_game")
    for pool in pools.values():
        rng.shuffle(pool)
    # Round-robin available phases instead of allowing long middlegames to dominate.
    selected = []
    while len(selected) < settings["count"]:
        for phase in PHASES:
            if pools[phase] and len(selected) < settings["count"]:
                selected.append(pools[phase].pop())
    rng.shuffle(selected)
    # Allocate splits after sampling so unused games cannot consume a val/test slot.
    # Largest-remainder rounding approximates the requested ratios in whole games.
    game_ids = sorted({entry["game_id"] for entry in selected})
    rng.shuffle(game_ids)
    weights = settings["split_weights"]
    quotas = {split: len(game_ids) * weight / sum(weights.values())
              for split, weight in weights.items()}
    counts = {split: math.floor(quota) for split, quota in quotas.items()}
    remainders = sorted(quotas, key=lambda split: quotas[split] - counts[split], reverse=True)
    for split in remainders[:len(game_ids) - sum(counts.values())]:
        counts[split] += 1
    assignments = [split for split, count in counts.items() for _ in range(count)]
    game_splits = dict(zip(game_ids, assignments))
    for game in games:
        # Unsampled histories stay available for inspection but have no split.
        game["split"] = game_splits.get(game["id"])
    for index, entry in enumerate(selected):
        entry["id"] = f"setup_{index:06d}"
        entry["split"] = game_splits[entry["game_id"]]

    result = {"seed": settings["seed"], "positions": selected}
    result["setup_generation"] = {
        "settings": copy.deepcopy(settings), "chess_version": chess.__version__,
        "engine": dict(engine.id) if engine is not None else None,
        "games": games,
        "phase_counts": dict(Counter(p["phase"] for p in selected)),
        "split_counts": dict(Counter(p["split"] for p in selected)),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "configs/setups.json")
    parser.add_argument("--output", type=Path, required=True, help="New legal-game setup JSON")
    parser.add_argument("--dataset-output", required=True, help="Future render directory inside project")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    destination = (ROOT / args.dataset_output).resolve()
    if not destination.is_relative_to(ROOT) or destination.exists():
        parser.error("dataset-output must be a new directory inside this project")
    settings = json.loads(args.config.read_text())
    engine_path = (ROOT / settings["engine_path"]).resolve()
    if settings["mode"] == "engine" and not engine_path.is_file():
        parser.error(f"Engine missing: {engine_path}. Set engine_path or explicitly choose mode=random.")
    context = chess.engine.SimpleEngine.popen_uci(str(engine_path)) if settings["mode"] == "engine" else nullcontext(None)
    with context as engine:
        result = create_setups(settings, engine=engine)
    result["output"] = str(destination.relative_to(ROOT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    summary = result["setup_generation"]
    print(f"Saved {len(result['positions'])} setups to {args.output}")
    print("Phases:", summary["phase_counts"], "Splits:", summary["split_counts"])


if __name__ == "__main__":
    main()
