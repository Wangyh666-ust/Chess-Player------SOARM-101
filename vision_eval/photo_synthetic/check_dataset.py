"""Independent mask/label/pair QA and a contact sheet. Requires numpy + Pillow."""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def check(root):
    """Verify saved masks, annotations, and view pairs, then write a review sheet."""
    manifest = json.loads((root / "manifest.json").read_text())
    config = json.loads((root / "config.json").read_text())
    expected_ids = {position["id"] + "_" + view
                    for position in config["positions"] for view in config["cameras"]}
    actual_ids = [sample["id"] for sample in manifest["samples"]]
    assert expected_ids and len(actual_ids) == len(set(actual_ids))
    assert set(actual_ids) == expected_ids, "Missing or unexpected rendered views"
    pairs = {}
    placements = {}
    sources = {}
    labels_total = 0
    tiles = []
    for sample in manifest["samples"]:
        sample_id = sample["id"]
        split = sample["split"]
        record = json.loads(
            (root / "records" / split / (sample_id + ".json")).read_text()
        )
        mask = np.asarray(Image.open(root / "masks" / split / (sample_id + ".png")))
        image = Image.open(root / "images" / split / (sample_id + ".png")).convert(
            "RGB"
        )
        width, height = image.size
        if record.get("framing_points_m"):
            points = np.column_stack((record["framing_points_m"], np.ones(8)))
            matrix = np.asarray(record["projection_matrix"]) @ np.linalg.inv(record["camera_to_world"])
            clip = points @ matrix.T
            uv = (clip[:, :2] / clip[:, 3, None] * [1, -1] + 1) / 2
            margin = record["framing_margin_fraction"]
            assert np.all(clip[:, 3] > 0) and np.all((uv >= margin) & (uv <= 1 - margin)), (
                sample_id, "camera framing failed")
        assert np.asarray(image).max() > 30 and np.asarray(image).std() > 5, (
            sample_id,
            "blank or failed RGB render",
        )
        assert mask.shape == (height, width) and (width, height) == tuple(
            config["cameras"][record["view"]].get("resolution", config["resolution"])
        )
        assert set(np.unique(mask)) <= set(range(len(record["instances"]) + 1))
        lines = (
            (root / "labels" / split / (sample_id + ".txt"))
            .read_text()
            .strip()
            .splitlines()
        )
        expected = [i for i in record["instances"] if i["labelled"]]
        assert len(lines) == len(expected)
        draw = ImageDraw.Draw(image)
        for instance in record["instances"]:
            ys, xs = np.where(mask == instance["instance_id"])
            assert len(xs) == instance["visible_pixels"]
            if len(xs):
                actual = [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() + 1),
                    int(ys.max() + 1),
                ]
                assert actual == instance["visible_box"]
                ax, ay, bx, by = instance["amodal_box"]
                # Pixel-edge and raster sampling rounding allow a 2-pixel tolerance.
                assert (
                    actual[0] >= ax - 2
                    and actual[1] >= ay - 2
                    and actual[2] <= bx + 2
                    and actual[3] <= by + 2
                ), (sample_id, instance["square"], "projection/mask mismatch")
            assert instance["labelled"] == (len(xs) >= config["min_visible_pixels"])
        for line, instance in zip(lines, expected):
            fields = line.split()
            assert len(fields) == 5 and int(fields[0]) == instance["class_id"]
            normalized_box = np.array(list(map(float, fields[1:])))
            assert np.all((normalized_box >= 0) & (normalized_box <= 1)) and np.all(
                normalized_box[2:] > 0
            )
            x0, y0, x1, y1 = instance["amodal_box"]
            wanted = np.array(
                [
                    (x0 + x1) / 2 / width,
                    (y0 + y1) / 2 / height,
                    (x1 - x0) / width,
                    (y1 - y0) / height,
                ]
            )
            assert np.max(abs(normalized_box - wanted)) < 1e-7
            color = (255, 145, 30) if instance["symbol"].islower() else (60, 230, 100)
            draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
            draw.text(
                (x0, y0 - 12), instance["symbol"] + " " + instance["square"], fill=color
            )
        corners = [tuple(p) for p in record["corners"]]
        draw.line(corners + [corners[0]], fill=(60, 180, 255), width=2)
        # A board placement must not leak across train/validation/test splits.
        placement = record["fen"].split()[0]
        assert placement not in placements or placements[placement] == split
        placements[placement] = split
        source = record.get("source_setup_id", record["group"])
        assert source not in sources or sources[source] == (placement, split), "Source setup mismatch"
        sources[source] = (placement, split)
        # Both views must describe the same physical objects and split.
        previous = pairs.setdefault(record["group"], [])
        if previous:
            assert previous[0]["split"] == split
            for key in ("fen", "lighting", "sampled_lighting", "world_strength",
                        "world_color", "exposure"):
                assert previous[0].get(key) == record.get(key), (sample_id, key, "pair mismatch")
            assert [i["world_matrix"] for i in previous[0]["instances"]] == [
                i["world_matrix"] for i in record["instances"]
            ]
        previous.append(record)
        labels_total += len(lines)
        # Inspect every sample, but bound contact-sheet memory for large datasets.
        if len(tiles) >= 48:
            continue
        tile = Image.new("RGB", (480, 390), "#19232d")
        image.thumbnail((480, 360))
        tile.paste(image, ((480 - image.width) // 2, 30))
        ImageDraw.Draw(tile).text((10, 9), sample_id + " | " + split, fill="white")
        tiles.append(tile)
    assert all(
        {record["view"] for record in group} == set(config["cameras"])
        for group in pairs.values()
    )
    sheet = Image.new("RGB", (960, 390 * ((len(tiles) + 1) // 2)), "#19232d")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 2) * 480, (i // 2) * 390))
    sheet.save(root / "label-review.jpg", quality=93)
    report = {
        "images": len(manifest["samples"]),
        "review_images": len(tiles),
        "paired_positions": len(pairs),
        "source_positions": len(sources),
        "labels": labels_total,
        "mask_projection_and_label_checks": "passed",
        "pair_scene_consistency": "passed",
        "placement_split_leakage": False,
        "scope": "Pipeline smoke test only. Synthetic validation/test do not measure real-camera accuracy.",
    }
    (root / "qa.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    check(parser.parse_args().root)
