from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import pathlib
import sys

from physics_world import PhysicsWorld
from dynamics_drive import run
from dynamics_evaluate import evaluate
from opening_profile_v1 import tighten_opening

HERE = pathlib.Path(__file__).resolve().parent
OLD = HERE / "results" / "paths_30_v1"
OUT = HERE / "results" / "advanced_30_v1" / "dynamics"

SOURCE_FILES = ("dynamics_cli.py", "physics_world.py", "dynamics_drive.py", "dynamics_evaluate.py", "full_piece_v2.py", "opening_profile_v1.py")
ASSET_EXTS = (".urdf", ".srdf")


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_label(label: str) -> str:
    if not label or not all(c.isalnum() or c in "-_" for c in label):
        raise SystemExit(f"invalid label: {label!r}")
    return label


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--source", choices=("paths", "diagnostics"), default="paths")
    p.add_argument("--friction", type=float, default=0.6)
    p.add_argument("--density", type=float, default=650.0)
    p.add_argument("--preload-deg", type=float, default=2.0)
    p.add_argument("--arm-force", type=float, default=1.5)
    p.add_argument("--grip-force", type=float, default=0.2)
    p.add_argument("--dt", type=float, default=1.0 / 240.0)
    p.add_argument("--speed", type=float, default=0.025)
    p.add_argument("--no-grip", action="store_true")
    p.add_argument('--opening-delta-deg', type=float, default=None)
    return p.parse_args(argv)


def _load_source(case: str, source: str):
    path = OLD / source / f"{case}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        raw = json.load(fh)
    assert isinstance(raw, dict), "source not a dict"
    assert raw.get("ok") is True, "source ok flag false"
    result = raw.get("result") or {}
    frames = result.get("frames")
    assert frames, "source frames empty"
    case_obj = raw.get("case")
    assert isinstance(case_obj, dict), "source case missing"
    assert case_obj.get("id") == case, "source case id mismatch"
    q5 = frames[0]["q"]
    assert isinstance(q5, list) and q5, "source q5 missing"
    g = float(frames[0]["g"])
    assert isinstance(g, float), "source g missing"
    return path, case_obj, q5, g, frames


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    label = _safe_label(args.label)

    source_path, case_obj, q5, g, frames = _load_source(args.case, args.source)
    if args.opening_delta_deg is not None:
        frames = tighten_opening(frames, args.opening_delta_deg)
        g = float(frames[0]['g'])


    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{args.case}_{label}.json.gz"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite {out_path}")

    world = PhysicsWorld(
        case_obj,
        q5,
        g,
        args.friction,
        args.density,
        args.dt,
        args.arm_force,
        args.grip_force,
    )
    try:
        observations = run(
            world,
            frames,
            args.preload_deg,
            args.speed,
            args.no_grip,
        )
        summary = evaluate(case_obj, observations, world)
    finally:
        world.close()

    digest_of = {}
    digest_of["dynamics_cli.py"] = _sha256(pathlib.Path(__file__).resolve())
    for name in SOURCE_FILES[1:]:
        digest_of[name] = _sha256(HERE / name)
    assets = {
        "urdf": sorted(str(p.relative_to(HERE)) for p in HERE.rglob("*.urdf")),
        "srdf": sorted(str(p.relative_to(HERE)) for p in HERE.rglob("*.srdf")),
    }

    configuration = dict(vars(args))
    configuration["source"] = args.source
    configuration["source_gzip_sha256"] = _sha256(source_path)
    configuration["source_hashes"] = digest_of
    configuration["assets"] = assets
    configuration["asset_hashes"] = {name: _sha256(HERE / "results" / "advanced_30_v1" / "assets" / name) for name in ("robot_windows.urdf","robot.srdf","geometry.json")}
    configuration["pybullet_version"] = importlib.metadata.version("pybullet")
    configuration["no_attachment"] = True
    configuration["note"] = (
        "geometry assumed; masses/inertia uncalibrated"
    )
    steps = getattr(world, "steps", None)
    world_time = getattr(world, "time", None)
    if steps is None or world_time is None:
        steps = getattr(world, "simulator_steps", steps)
        world_time = getattr(world, "simulator_time", world_time)
    configuration["simulator_steps"] = steps
    configuration["simulator_world_time"] = world_time

    summary["case_id"] = args.case
    summary["configuration"] = configuration
    payload = {
        "case": case_obj,
        "summary": summary,
        "frames": observations,
        "configuration": configuration,
    }
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)

    compact_path = OUT / f"{args.case}_{label}_summary.json"
    with compact_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
