import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision_eval"))
import chessred_adapter as adapter  # noqa: E402

PIECE_SYMS = set("KQRBNPkqrbnp_")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_demo(path):
    text = path.read_text(encoding="utf-8")
    assert text.startswith("window.DEMO_DATA="), "demo-data prefix"
    text = text[len("window.DEMO_DATA="):].strip()
    if text.endswith(";"):
        text = text[:-1]
    return json.loads(text)


def check_split_plan(plan):
    groups = plan["groups"]
    assert len(groups) == 375, f"groups len {len(groups)}"
    counts = {"train": 0, "val": 0, "test": 0}
    placements = {}
    for g in groups:
        split = g["split"]
        assert split in counts, f"bad split {split}"
        counts[split] += 1
        assert isinstance(g["placement"], str), "placement must be str"
        assert len(g["image_seeds"]) == 8, "image_seeds len"
        placements.setdefault(split, set()).add(g["placement"])
    assert counts == {"train": 262, "val": 56, "test": 57}, counts
    ts, vs, ss = placements["train"], placements["val"], placements["test"]
    assert len(ts) == 262 and len(vs) == 56 and len(ss) == 57
    assert not (ts & vs) and not (ts & ss) and not (vs & ss)
    excluded = set(plan["excluded_placements"])
    assert not (ts & excluded) and not (vs & excluded) and not (ss & excluded)


def check_comparison(comp, expected, hash_expected, weight_path):
    datasets = comp["datasets"]
    assert set(datasets) == {"original9", "independent_test"}
    total_rows = 0
    for ds_name, ds in datasets.items():
        assert {"original", "finetuned"}.issubset(ds), ds_name
        exp = expected[ds_name]
        rowcount = exp["rowcount"]
        for model_name in ("original", "finetuned"):
            dm = ds[model_name]
            assert {"rows", "aggregate", "timing"}.issubset(dm)
            rows = dm["rows"]
            assert len(rows) == rowcount, (ds_name, model_name, len(rows))
            total_rows += len(rows)
            occ = joint = exact = 0
            for row in rows:
                assert set(row["modes"]) == {"known_corners"}
                kc = row["modes"]["known_corners"]
                board = kc["board"]
                assert isinstance(board, list) and len(board) == 64
                for sym in row["truth"]:
                    assert sym in PIECE_SYMS, sym
                for sym in board:
                    assert sym in PIECE_SYMS, sym
                recon = adapter.map_board(row["raw_detections"], row["corners"])
                assert recon == board, f"recon mismatch {row['id']}"
                r_occ = sum(1 for t in row["truth"] if t != "_")
                r_joint = sum(
                    1 for t, p in zip(row["truth"], board) if t != "_" and t == p
                )
                r_exact = int(row["truth"] == board)
                assert kc["scores"]["occupied"] == r_occ, (row["id"], "occupied")
                assert kc["scores"]["joint_correct"] == r_joint, (row["id"], "joint")
                assert kc["scores"]["exact_board"] == r_exact, (row["id"], "exact")
                occ += r_occ
                joint += r_joint
                exact += r_exact
            agg = dm["aggregate"]
            assert agg["occupied"] == occ, (ds_name, model_name, "aggregate occupied")
            assert agg["joint_correct"] == joint
            assert agg["exact_boards"] == exact
            got = (joint, occ, exact)
            assert got == exp[model_name], (ds_name, model_name, got)
    assert total_rows == 930, total_rows
    assert sha256_file(weight_path) == hash_expected, "weight hash mismatch"
    return datasets


def check_demo(demo, comp, image_root):
    assert "datasets" in demo and "training" in demo
    t = demo["training"]
    assert t["train_images"] == 2096, "training train_images"
    assert t["val_images"] == 448, "training val_images"
    assert t["test_images"] == 456, "training test_images"

    names = list(demo["datasets"].keys())
    assert len(names) == 2, names

    source_rows = {}
    for ds_name, ds in comp["datasets"].items():
        source_rows[ds_name] = {}
        for model_name in ("original", "finetuned"):
            source_rows[ds_name][model_name] = {
                r["id"]: r for r in ds[model_name]["rows"]
            }

    ft_failed_ids = {}
    for name in names:
        d = demo["datasets"][name]
        rows_orig = source_rows[name]["original"]
        rows_ft = source_rows[name]["finetuned"]
        expected_count = len(rows_orig)
        assert d["count"] == expected_count, (name, "count")
        assert d["selected_count"] == len(d["examples"]), (name, "selected_count")
        for model_name in ("original", "finetuned"):
            dsrc = comp["datasets"][name][model_name]
            assert d[model_name]["aggregate"] == dsrc["aggregate"], (
                name,
                model_name,
                "aggregate",
            )
            assert d[model_name]["timing"] == dsrc["timing"], (
                name,
                model_name,
                "timing",
            )
        failed = []
        for ex in d["examples"]:
            rid = ex["id"]
            src_o = rows_orig[rid]
            src_f = rows_ft[rid]
            assert ex["truth"] == src_o["truth"], (rid, "truth")
            assert ex["corners"] == src_o["corners"], (rid, "corners")
            for model_name, src in (("original", src_o), ("finetuned", src_f)):
                kc = src["modes"]["known_corners"]
                assert ex[model_name]["board"] == kc["board"], (rid, model_name, "board")
                assert ex[model_name]["scores"] == kc["scores"], (rid, model_name, "scores")
                assert ex[model_name]["total_ms"] == kc["total_ms"], (rid, model_name, "ms")
                assert ex[model_name]["detections"] == src["raw_detections"], (
                    rid,
                    model_name,
                    "detections",
                )
            fb = src_f["modes"]["known_corners"]["board"]
            want_flag = src_o["truth"] != fb
            assert ex["finetuned_failed"] == want_flag, (rid, "finetuned_failed")
            if want_flag:
                failed.append(rid)
            img = image_root / ex["image"]
            assert img.exists(), img
            with Image.open(img) as im:
                assert list(im.size) == [ex["width"], ex["height"]], (rid, im.size)
        ft_failed_ids[name] = failed

    demo_ft_failed = {
        e["id"]
        for e in demo["datasets"]["independent_test"]["examples"]
        if e["finetuned_failed"]
    }
    assert len(demo_ft_failed) == 14, len(demo_ft_failed)
    all_failed = set()
    for r in comp["datasets"]["independent_test"]["finetuned"]["rows"]:
        kc = r["modes"]["known_corners"]
        if kc["board"] != r["truth"]:
            all_failed.add(r["id"])
    assert len(all_failed) == 14, len(all_failed)
    assert all_failed <= set(
        e["id"] for e in demo["datasets"]["independent_test"]["examples"]
    ), "all_failed not subset of demo examples"
    assert set(ft_failed_ids["independent_test"]) == demo_ft_failed
    if "original9" in ft_failed_ids:
        assert len(ft_failed_ids["original9"]) == 0, ft_failed_ids["original9"]
    assert len(demo["datasets"]["original9"]["examples"]) == 9
    return demo_ft_failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", default=str(ROOT / "artifacts/dubrovnik-v1/comparison.json"))
    ap.add_argument("--weight", default=str(ROOT / "models/dubrovnik-yolov8m-v1.pt"))
    args = ap.parse_args()

    comp = json.load(open(args.comparison, encoding="utf-8"))
    plan = json.load(open(ROOT / "artifacts/dubrovnik-v1/split-plan.json", encoding="utf-8"))
    check_split_plan(plan)

    expected = {
        "original9": {
            "rowcount": 9,
            "original": (116, 222, 0),
            "finetuned": (222, 222, 9),
        },
        "independent_test": {
            "rowcount": 456,
            "original": (6505, 10784, 0),
            "finetuned": (10768, 10784, 442),
        },
    }
    check_comparison(
        comp,
        expected,
        "021ad9f46ad2c31856a9aa5e36b25e10c6d10be8b8005efcf9982908a4c11a9b",
        Path(args.weight),
    )

    demo = load_demo(ROOT / "docs/assets/demo-data.js")
    failed = check_demo(demo, comp, ROOT / "docs")

    print(f"OK: 930 predictions verified, {len(failed)} finetuned failures on independent_test")


if __name__ == "__main__":
    main()
