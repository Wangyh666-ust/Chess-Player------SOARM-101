"""Render lighting-bound probes without a detector; retain rejected visual limits.

Usage: python3 vision_eval/photo_synthetic/pilots/lighting_bounds/run.py
The config supplies base_config, presets_file, output, samples, resolution_scale, and cases
[{id, bounds, seed?, preset?}]. Existing completed cases are reused only if config matches.
"""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(PACKAGE))
from check_lighting import check
from randomized_lighting import sample_lighting


def piece_brightness(folder, report):
    """Appearance diagnostics only: weaker black-piece highlights may hide detail."""
    for sample in report["samples"]:
        record_path = next((folder / "records").glob(f"*/{sample['sample']}.json"))
        record = json.loads(record_path.read_text())
        relative = Path(record["split"]) / (record["id"] + ".png")
        rgb = np.asarray(Image.open(folder / "images" / relative).convert("RGB"))
        mask = np.asarray(Image.open(folder / "masks" / relative))
        brightness = rgb @ np.array([.2126, .7152, .0722])
        for color, white in [("black", False), ("white", True)]:
            ids = [i["instance_id"] for i in record["instances"] if i["symbol"].isupper() == white]
            region = Image.fromarray(np.isin(mask, ids).astype("uint8") * 255)
            pixels = brightness[np.asarray(region.filter(ImageFilter.MinFilter(3))) > 0]
            sample[color + "_piece_p50"] = float(np.percentile(pixels, 50)) if len(pixels) else None
            sample[color + "_piece_p95"] = float(np.percentile(pixels, 95)) if len(pixels) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    base_path = (args.config.parent / spec["base_config"]).resolve()
    presets_path = (args.config.parent / spec["presets_file"]).resolve()
    base = json.loads(base_path.read_text())
    output = (ROOT / spec["output"]).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must stay within the project")
    cases = spec["cases"]
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids) or any(not name or Path(name).name != name
                                       or name in {".", ".."} for name in ids):
        raise ValueError("Case IDs must be unique directory names")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = dict(spec, base_config=str(base_path), presets_file=str(presets_path))
    (output / "sweep_spec.json").write_text(json.dumps(snapshot, indent=2))
    lighting_tests = json.loads(presets_path.read_text())
    limits = lighting_tests["quality_limits"]
    presets = {variant["id"]: variant["overrides"] for variant in lighting_tests["variants"]}
    reports = []
    for case in cases:
        config = copy.deepcopy(base)
        if "preset" in case:
            config.update(copy.deepcopy(presets[case["preset"]]))
        # Lighting is the sole variable; use the same scene, sensor and render seed.
        config["lighting_randomization"] = case["bounds"]
        config["seed"] = case.get("seed", base["seed"])
        sample_lighting(config["lights"], config["world_strength"],
                        seed=config["seed"], bounds=case["bounds"])
        config["samples"] = spec["samples"]
        for key in ["piece_jitter_mm", "piece_yaw_jitter_degrees",
                    "camera_position_jitter_mm", "camera_target_jitter_mm"]:
            config[key] = 0
        for camera in config["cameras"].values():
            camera["resolution"] = [round(n * spec["resolution_scale"])
                                    for n in camera.get("resolution", config["resolution"])]
        folder = output / case["id"]
        config["output"] = str(folder.relative_to(ROOT))
        if folder.exists():
            if not (folder / "manifest.json").exists() or json.loads((folder / "config.json").read_text()) != config:
                raise ValueError(f"Incomplete or mismatched case: {folder}; use a new output")
        else:
            path = output / (case["id"] + ".json")
            path.write_text(json.dumps(config, indent=2))
            print("RENDER", case["id"], flush=True)
            with (output / (case["id"] + ".log")).open("w") as log:
                subprocess.run([str(ROOT / ".local-blender/start-blender.sh"),
                                "--background", "--threads", str(args.threads),
                                "--python-exit-code", "1", "--python", str(PACKAGE / "render_dataset.py"),
                                "--", "--config", str(path)], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            path.unlink()
            # Images, configs and metadata suffice for probes; avoid duplicate assets.
            (folder / "scene.blend").unlink(missing_ok=True)
        subprocess.run([sys.executable, str(PACKAGE / "check_dataset.py"), str(folder)],
                       stdout=subprocess.DEVNULL, check=True)
        try:
            report = check(folder, limits)
        except AssertionError:
            report = json.loads((folder / "lighting_qa.json").read_text())
            if report["passed"]:
                raise
        piece_brightness(folder, report)
        reports.append({**case, "seed": config["seed"], **report})
        print(case["id"], "PASS" if report["passed"] else "REJECT", flush=True)
        (output / "summary.json").write_text(json.dumps({
            "scope": "Visual lighting limits only; no detector or accuracy measurement.",
            "cases": reports,
        }, indent=2))
    font_path = Path("/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf")
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    for view in base["cameras"]:
        sheet = Image.new("RGB", (1440, 410 * ((len(cases) + 2) // 3)), "#19232d")
        draw = ImageDraw.Draw(sheet)
        for i, report in enumerate(reports):
            image_path = next((output / report["id"] / "images").glob(f"*/*_{view}.png"))
            with Image.open(image_path) as image:
                image.thumbnail((470, 350))
                x, y = (i % 3) * 480, (i // 3) * 410
                text = report["id"] + (" — PASS" if report["passed"] else " — REJECT")
                draw.text((x + 8, y + 8), text, font=font, fill="white")
                sheet.paste(image, (x + (480 - image.width) // 2, y + 45))
        sheet.save(output / f"comparison_{view}.jpg", quality=93)


if __name__ == "__main__":
    main()
