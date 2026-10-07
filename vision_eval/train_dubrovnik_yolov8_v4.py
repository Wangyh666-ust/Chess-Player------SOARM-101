from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

NAMES = {
    0: "white-pawn",
    1: "white-rook",
    2: "white-knight",
    3: "white-bishop",
    4: "white-queen",
    5: "white-king",
    6: "black-pawn",
    7: "black-rook",
    8: "black-knight",
    9: "black-bishop",
    10: "black-queen",
    11: "black-king",
    12: "empty",
}

SOURCE_CKPT = ROOT / "alternatives" / "vendor" / "chessred_yolov8" / "models" / "piece_best.pt"
DEFAULT_DATA = ROOT / "dubrovnik_finetune" / "dataset_v1" / "dataset.yaml"
RUNS_DIR = ROOT / "dubrovnik_finetune" / "runs"
EXP_DIR = ROOT / "dubrovnik_finetune" / "experiments"
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_json(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [safe_json(v) for v in obj]
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return safe_json(obj.item())
        if isinstance(obj, np.ndarray):
            return safe_json(obj.tolist())
    except Exception:
        pass
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return safe_json(obj.detach().cpu().tolist())
    except Exception:
        pass
    return str(obj)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(safe_json(obj), f, indent=2, sort_keys=True)
        f.write("\n")


def load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"dataset yaml not a mapping: {path}")
    return data


def check_dataset_yaml(path: Path) -> dict:
    data = load_yaml(path)
    path_val = data.get("path")
    if not path_val:
        raise ValueError("dataset.yaml missing 'path'")
    abs_root = Path(path_val)
    if not abs_root.is_absolute():
        raise ValueError("dataset.yaml 'path' must be absolute")
    resolved = abs_root.resolve()
    if ROOT not in resolved.parents and resolved != ROOT:
        raise ValueError(f"dataset.yaml path outside ROOT: {resolved}")
    for key in ("train", "val", "test"):
        if not data.get(key):
            raise ValueError(f"dataset.yaml missing '{key}'")
    names = data.get("names")
    if not isinstance(names, dict):
        raise ValueError("dataset.yaml names must be mapping")
    norm = {int(k): str(v) for k, v in names.items()}
    if norm != NAMES:
        raise ValueError(f"dataset.yaml names mismatch: {norm}")
    return data


def label_path_for(image_path: Path, split: str) -> Path:
    root = None
    parts = image_path.parts
    if "images" in parts:
        idx = len(parts) - 1 - list(reversed(parts)).index("images")
        root = Path(*parts[:idx])
    else:
        root = image_path.parent.parent
    return root / "labels" / split / (image_path.stem + ".txt")


def validate_label_file(lp: Path) -> None:
    if not lp.is_file():
        raise FileNotFoundError(f"missing label file: {lp}")
    with lp.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{lp}:{i} expected 5 fields got {len(fields)}")
            cid = int(fields[0])
            if cid < 0 or cid > 11:
                raise ValueError(f"{lp}:{i} class id out of range: {cid}")
            vals = [float(x) for x in fields[1:]]
            for v in vals:
                if not math.isfinite(v):
                    raise ValueError(f"{lp}:{i} non-finite coord")
            if not all(0.0 <= v <= 1.0 for v in vals[:2]):
                raise ValueError(f"{lp}:{i} center coords out of [0,1]")
            if vals[2] <= 0 or vals[3] <= 0:
                raise ValueError(f"{lp}:{i} w/h must be > 0")
            if vals[0] - vals[2] / 2 < -1e-9 or vals[0] + vals[2] / 2 > 1 + 1e-9:
                raise ValueError(f"{lp}:{i} x extends out of [0,1]")
            if vals[1] - vals[3] / 2 < -1e-9 or vals[1] + vals[3] / 2 > 1 + 1e-9:
                raise ValueError(f"{lp}:{i} y extends out of [0,1]")


def list_split_images(data_root: Path, split: str) -> list[Path]:
    d = data_root / "images" / split
    if not d.is_dir():
        raise FileNotFoundError(f"missing image dir: {d}")
    imgs = sorted([p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    return imgs


def full_validation(data_root: Path, stats_path: Path, plan_path: Path) -> None:
    if not stats_path.is_file():
        raise FileNotFoundError(f"missing statistics.json: {stats_path}")
    if not plan_path.is_file():
        raise FileNotFoundError(f"missing plan.json: {plan_path}")
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    with plan_path.open("r", encoding="utf-8") as f:
        plan = json.load(f)

    if stats.get("complete") is not True:
        raise ValueError("statistics complete is not true")
    cps = stats.get("counts_per_split")
    if not isinstance(cps, dict):
        raise ValueError("counts_per_split missing")
    for s in ("train", "val", "test"):
        if int(cps.get(s, -1)) <= 0:
            raise ValueError(f"counts_per_split[{s}] must be >=1")
    views = int(plan.get("config", {}).get("views", 0))
    if views <= 0:
        raise ValueError("plan config.views must be positive")

    excluded = set(plan.get("excluded_placements", []))
    groups = plan.get("groups", [])
    if not isinstance(groups, list):
        raise ValueError("plan groups must be list")

    split_gids: dict[str, set] = {"train": set(), "val": set(), "test": set()}
    split_placements: dict[str, set] = {"train": set(), "val": set(), "test": set()}
    for g in groups:
        gid = g.get("group_id")
        sp = g.get("split")
        pl = g.get("placement")
        if sp not in split_gids:
            raise ValueError(f"unknown split in group: {sp}")
        if gid in split_gids[sp]:
            raise ValueError(f"duplicate group_id {gid} in split {sp}")
        split_gids[sp].add(gid)
        split_placements[sp].add(pl)
        seeds = g.get("image_seeds")
        if not isinstance(seeds, list) or len(seeds) != views:
            raise ValueError(f"group {gid} image_seeds length {None if not isinstance(seeds, list) else len(seeds)} != {views}")
        if not g.get("fen"):
            raise ValueError(f"group {gid} missing fen")

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    intersections: dict[str, int] = {}
    for a, b in pairs:
        gi = split_gids[a] & split_gids[b]
        if gi:
            raise ValueError(f"group_id intersection between {a} and {b}: {sorted(gi)[:5]}")
        pl = split_placements[a] & split_placements[b]
        if pl:
            raise ValueError(f"placement intersection between {a} and {b}: {sorted(pl)[:5]}")
        intersections[f"{a}_{b}"] = 0

    for a in ("train", "val", "test"):
        if split_placements[a] & excluded:
            raise ValueError(f"split {a} includes excluded placements: {sorted(split_placements[a] & excluded)[:5]}")
    if int(stats.get("placements_intersection_with_old", -1)) != 0:
        raise ValueError("placements_intersection_with_old must be 0")

    total_expected = int(stats.get("complete_total", -1))
    planned_total = int(stats.get("planned_total", -1))
    summed = 0
    for s in ("train", "val", "test"):
        expected_count = int(cps[s])
        n_groups = len(split_gids[s])
        if n_groups * views != expected_count:
            raise ValueError(f"{s}: groups*views {n_groups * views} != counts {expected_count}")
        imgs = list_split_images(data_root, s)
        if len(imgs) != expected_count:
            raise ValueError(f"{s}: images on disk {len(imgs)} != counts {expected_count}")
        for img in imgs:
            lp = label_path_for(img, s)
            validate_label_file(lp)
        summed += expected_count

    if summed != total_expected:
        raise ValueError(f"split sum {summed} != complete_total {total_expected}")
    if total_expected != planned_total:
        raise ValueError(f"complete_total {total_expected} != planned_total {planned_total}")
    if summed != 3000:
        raise ValueError(f"total {summed} != 3000")

    si = stats.get("split_placement_intersections")
    if isinstance(si, dict):
        for k, v in si.items():
            if int(v) != 0:
                raise ValueError(f"statistics split_placement_intersections[{k}] != 0")


def smoke_setup(data_root: Path, out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_imgs = list_split_images(data_root, "train")[:16]
    val_imgs = list_split_images(data_root, "val")[:4]
    if len(train_imgs) < 16:
        raise ValueError(f"smoke needs >=16 train images, got {len(train_imgs)}")
    if len(val_imgs) < 4:
        raise ValueError(f"smoke needs >=4 val images, got {len(val_imgs)}")

    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"
    with train_txt.open("w", encoding="utf-8") as f:
        for img in train_imgs:
            f.write(str(img.resolve()) + "\n")
    with val_txt.open("w", encoding="utf-8") as f:
        for img in val_imgs:
            f.write(str(img.resolve()) + "\n")

    for img in train_imgs:
        validate_label_file(label_path_for(img, "train"))
    for img in val_imgs:
        validate_label_file(label_path_for(img, "val"))

    yaml_path = out_dir / "smoke_data.yaml"
    body = {
        "path": str(data_root.resolve()),
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "test": str(val_txt.resolve()),
        "names": {int(k): v for k, v in NAMES.items()},
    }
    import yaml

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(body, f, sort_keys=False)
    return yaml_path, train_txt, val_txt


def gpu_vram_info() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = int(props.total_memory)
            free = None
            try:
                t = torch.cuda.mem_get_info(0)
                free = int(t[0])
            except Exception:
                pass
            return {
                "name": props.name,
                "total_bytes": total,
                "free_bytes": free,
            }
    except Exception as e:
        return {"error": str(e)}
    return {"available": False}


def build_train_kwargs(args, project: Path, name: str) -> dict:
    return {
        "data": str(args.data_resolved),
        "epochs": int(args.epochs),
        "batch": int(args.batch),
        "imgsz": int(args.imgsz),
        "device": 0,
        "workers": 0,
        "seed": 20261007,
        "deterministic": True,
        "amp": True,
        "optimizer": "AdamW",
        "lr0": 0.0005,
        "lrf": 0.1,
        "cos_lr": True,
        "weight_decay": 0.0005,
        "warmup_epochs": 1,
        "patience": 6,
        "nbs": 16,
        "cache": False,
        "save": True,
        "plots": True,
        "verbose": True,
        "exist_ok": False,
        "project": str(project),
        "name": name,
        "hsv_h": 0.005,
        "hsv_s": 0.15,
        "hsv_v": 0.15,
        "translate": 0.03,
        "scale": 0.1,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train_dubrovnik_yolov8_v3")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--name", type=str, default="trial_v1")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    if not NAME_RE.match(args.name):
        print(json.dumps({"status": "error", "error": "name must match [A-Za-z0-9_-]+"}))
        return 1
    if args.epochs <= 0 or args.batch <= 0 or args.imgsz <= 0:
        print(json.dumps({"status": "error", "error": "epochs/batch/imgsz must be positive"}))
        return 1

    base_name = args.name + ("_smoke" if args.smoke else "")
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (ROOT / data_path).resolve()
    if ROOT not in data_path.parents and data_path != ROOT:
        print(json.dumps({"status": "error", "error": f"data outside ROOT: {data_path}"}))
        return 1
    if not data_path.is_file():
        print(json.dumps({"status": "error", "error": f"missing data file: {data_path}"}))
        return 1

    if not SOURCE_CKPT.is_file():
        print(json.dumps({"status": "error", "error": f"missing checkpoint: {SOURCE_CKPT}"}))
        return 1

    project_dir = RUNS_DIR
    run_dir = project_dir / base_name
    exp_file = EXP_DIR / (base_name + ".json")
    if run_dir.exists():
        print(json.dumps({"status": "error", "error": f"run dir exists: {run_dir}"}))
        return 1
    if exp_file.exists():
        print(json.dumps({"status": "error", "error": f"experiment file exists: {exp_file}"}))
        return 1

    import torch

    torch.set_num_threads(4)
    try:
        import cv2

        cv2.setNumThreads(0)
    except Exception:
        pass

    os.environ.setdefault("YOLO_VERBOSE", "false")

    epochs = 1 if args.smoke else args.epochs
    batch = 2 if args.smoke else args.batch

    metadata: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "cli": {
            "data": str(data_path),
            "name": args.name,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "smoke": bool(args.smoke),
        },
        "start_time": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(SOURCE_CKPT),
        "source_checkpoint_sha256": sha256_file(SOURCE_CKPT),
        "data_config": str(data_path),
        "data_config_sha256": sha256_file(data_path),
        "env": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu": gpu_vram_info(),
            "cv2_threads": 0,
            "torch_threads": 4,
        },
    }

    try:
        import ultralytics

        metadata["env"]["ultralytics"] = getattr(ultralytics, "__version__", "unknown")
    except Exception as e:
        metadata["env"]["ultralytics"] = f"unavailable: {e}"

    if not torch.cuda.is_available():
        metadata["status"] = "error"
        metadata["error"] = "CUDA not available"
        metadata["finish_time"] = datetime.now(timezone.utc).isoformat()
        write_json(exp_file, metadata)
        print(json.dumps({"status": "error", "error": "CUDA not available"}))
        return 1

    data_root = Path(check_dataset_yaml(data_path)["path"]).resolve()

    try:
        if args.smoke:
            fake_root = data_root
            effective_yaml, _, _ = smoke_setup(fake_root, RUNS_DIR / (base_name + "_data"))
            data_path_effective = effective_yaml
        else:
            stats_path = data_root / "statistics.json"
            plan_path = data_root / "plan.json"
            full_validation(data_root, stats_path, plan_path)
            data_path_effective = data_path
    except Exception as e:
        metadata["status"] = "error"
        metadata["error"] = f"precondition failure: {e}"
        metadata["finish_time"] = datetime.now(timezone.utc).isoformat()
        write_json(exp_file, metadata)
        print(json.dumps({"status": "error", "error": str(e)}))
        return 1

    try:
        from ultralytics import YOLO
    except Exception as e:
        metadata["status"] = "error"
        metadata["error"] = f"ultralytics import failed: {e}"
        metadata["finish_time"] = datetime.now(timezone.utc).isoformat()
        write_json(exp_file, metadata)
        print(json.dumps({"status": "error", "error": str(e)}))
        return 1

    try:
        model = YOLO(str(SOURCE_CKPT))
    except Exception as e:
        metadata["status"] = "error"
        metadata["error"] = f"YOLO load failed: {e}"
        metadata["finish_time"] = datetime.now(timezone.utc).isoformat()
        write_json(exp_file, metadata)
        print(json.dumps({"status": "error", "error": str(e)}))
        return 1

    model_names = getattr(model, "names", None)
    if not isinstance(model_names, dict):
        metadata["status"] = "error"
        metadata["error"] = "model.names missing"
        metadata["finish_time"] = datetime.now(timezone.utc).isoformat()
        write_json(exp_file, metadata)
        print(json.dumps({"status": "error", "error": "model.names missing"}))
        return 1

    norm_model_names = {int(k): str(v) for k, v in model_names.items()}
    if norm_model_names != NAMES:
        metadata["status"] = "error"
        metadata["error"] = f"model.names mismatch: {norm_model_names}"
        metadata["model_names"] = norm_model_names
        metadata["finish_time"] = datetime.now(timezone.utc).isoformat()
        write_json(exp_file, metadata)
        print(json.dumps({"status": "error", "error": "model.names mismatch"}))
        return 1

    metadata["model_names"] = norm_model_names

    project_dir.mkdir(parents=True, exist_ok=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    kwargs = build_train_kwargs(
        argparse.Namespace(data_resolved=data_path_effective, epochs=epochs, batch=batch, imgsz=args.imgsz),
        project_dir,
        base_name,
    )
    kwargs["epochs"] = epochs
    kwargs["batch"] = batch
    metadata["train_kwargs"] = kwargs

    metadata["status"] = "started"
    write_json(exp_file, metadata)
    start = time.time()
    status = "error"
    error_msg = None
    results_dict: Any = None
    save_dir: Path | None = None
    best_hash = None
    last_hash = None
    best_path = None
    last_path = None

    try:
        results = model.train(**kwargs)
        save_dir = Path(str(model.trainer.save_dir))
        best_path = save_dir / "weights" / "best.pt"
        last_path = save_dir / "weights" / "last.pt"
        if best_path.is_file():
            best_hash = sha256_file(best_path)
        if last_path.is_file():
            last_hash = sha256_file(last_path)
        if not best_path.is_file():
            raise RuntimeError(f"best.pt missing at {best_path}")
        try:
            results_dict = results.results_dict
        except Exception:
            results_dict = None
        status = "success"
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        status = "error"
        try:
            if model is not None and getattr(model, "trainer", None) is not None:
                save_dir = Path(str(model.trainer.save_dir))
                best_path = save_dir / "weights" / "best.pt"
                last_path = save_dir / "weights" / "last.pt"
                if best_path.is_file():
                    best_hash = sha256_file(best_path)
                if last_path.is_file():
                    last_hash = sha256_file(last_path)
        except Exception:
            pass

    elapsed = time.time() - start

    metadata["status"] = status
    metadata["finish_time"] = datetime.now(timezone.utc).isoformat()
    metadata["elapsed_seconds"] = elapsed
    metadata["save_dir"] = str(save_dir) if save_dir else None
    metadata["best_pt"] = str(best_path) if best_path else None
    metadata["last_pt"] = str(last_path) if last_path else None
    metadata["best_pt_sha256"] = best_hash
    metadata["last_pt_sha256"] = last_hash
    metadata["results_dict"] = safe_json(results_dict)
    if error_msg:
        metadata["error"] = error_msg

    write_json(exp_file, metadata)

    out = {
        "status": status,
        "experiment_metadata": str(exp_file),
        "save_dir": str(save_dir) if save_dir else None,
        "best_pt": str(best_path) if best_path else None,
        "last_pt": str(last_path) if last_path else None,
        "best_pt_sha256": best_hash,
        "last_pt_sha256": last_hash,
        "elapsed_seconds": elapsed,
        "results_dict": safe_json(results_dict),
    }
    if error_msg:
        out["error"] = error_msg
    print(json.dumps(out))
    return 0 if status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
