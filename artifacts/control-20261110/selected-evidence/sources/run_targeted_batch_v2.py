import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
for p in (
    Path(__file__).resolve().parent,
    ROOT / "calibration/manipulation_20261109_v1/adaptive_continuous_20261110_v1/work",
    ROOT / "calibration/manipulation_20261109_v1/first_contact_root_20261110_v1/work",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import targeted_batch_v1 as recorder
sys.path.insert(0, str(ROOT / "calibration/manipulation_20261109_v1/planning_repair_20261110_v1/work"))
import tilt_route_placeyaw_v1 as placement_routes
from placement_yaw_bindings_v1 import bind_routes
from threshold_factory_v2 import build_factory


def _validate(cfg):
    raw = cfg.get("diagnostic_threshold")
    if isinstance(raw, bool) or raw != 0.002:
        raise ValueError("diagnostic_threshold must be exactly 0.002")
    if cfg.get("mode") != "games" or len(cfg.get("sessions", [])) != 3:
        raise ValueError("exactly three reviewed targeted sessions required")
    return cfg["sessions"]


def run(cfg, outdir):
    outdir = Path(outdir)
    _validate(cfg)

    if (outdir / "result.json").exists():
        raise FileExistsError("result.json already exists: refusing to overwrite")
    outdir.mkdir(parents=True, exist_ok=True)

    info = []

    def operation(c, _case, _q, _g):
        return recorder.run_batch(c, outdir)

    try:
        return build_factory(recorder.E.p, operation, info)(cfg, None, None, None)
    finally:
        payload = {
            "scope": "whole_three_game_batch",
            "threshold": 0.002,
            "info": info,
        }
        with open(outdir / "threshold_forwarding.json", "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    with open(args.config) as fh:
        cfg = json.load(fh)

    sys.exit(run(cfg, Path(args.out)))


if __name__ == "__main__":
    main()
