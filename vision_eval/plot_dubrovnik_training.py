from pathlib import Path
import argparse
import csv
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

EXPECTED_COLUMNS = [
    "epoch",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "train/box_loss",
    "val/box_loss",
    "train/cls_loss",
    "val/cls_loss",
]

PROBABILITY_COLUMNS = {
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
}


def resolve_inside_root(path, root):
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise SystemExit("Path must be inside ROOT: {}".format(resolved))
    return resolved


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default=str(ROOT / "dubrovnik_finetune" / "runs" / "trial_v1"),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "dubrovnik_finetune"
            / "evaluation"
            / "trial_v1"
            / "training_curve.png"
        ),
    )
    return parser.parse_args()


def read_results(csv_path):
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("results.csv has no header")
        cleaned_rows = []
        for raw_row in reader:
            row = {}
            for key, value in raw_row.items():
                if key is None:
                    continue
                row[key.strip()] = value
            cleaned_rows.append(row)
    if not cleaned_rows:
        raise SystemExit("results.csv contains no data rows")
    return cleaned_rows


def extract_series(rows):
    series = {name: [] for name in EXPECTED_COLUMNS}
    for row in rows:
        for name in EXPECTED_COLUMNS:
            if name not in row or row[name] is None:
                raise SystemExit("Missing column: {}".format(name))
            try:
                value = float(str(row[name]).strip())
            except (TypeError, ValueError):
                raise SystemExit("Non-numeric value in column {}: {!r}".format(name, row[name]))
            if not math.isfinite(value):
                raise SystemExit("Non-finite value in column {}: {!r}".format(name, row[name]))
            series[name].append(value)
    for name in EXPECTED_COLUMNS:
        if not series[name]:
            raise SystemExit("No numeric values for column: {}".format(name))
    return series


def to_percent(values):
    return [v * 100.0 for v in values]


def make_plot(series, output_path):
    epochs = series["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(epochs, to_percent(series["metrics/precision(B)"]), marker="o", markersize=2, linewidth=1.0, label="Precision")
    ax.plot(epochs, to_percent(series["metrics/recall(B)"]), marker="o", markersize=2, linewidth=1.0, label="Recall")
    ax.plot(epochs, to_percent(series["metrics/mAP50(B)"]), marker="o", markersize=2, linewidth=1.0, label="mAP50")
    ax.plot(epochs, to_percent(series["metrics/mAP50-95(B)"]), marker="o", markersize=2, linewidth=1.0, label="mAP50-95")
    ax.set_ylim(0, 100)
    ax.set_title("Validation detection metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Percent (%)")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=7, loc="best")

    ax = axes[1]
    ax.plot(epochs, series["train/box_loss"], marker="o", markersize=2, linewidth=1.0, label="Train")
    ax.plot(epochs, series["val/box_loss"], marker="o", markersize=2, linewidth=1.0, label="Val")
    ax.set_title("Box loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=7, loc="best")

    ax = axes[2]
    ax.plot(epochs, series["train/cls_loss"], marker="o", markersize=2, linewidth=1.0, label="Train")
    ax.plot(epochs, series["val/cls_loss"], marker="o", markersize=2, linewidth=1.0, label="Val")
    ax.set_title("Classification loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=7, loc="best")

    fig.suptitle("YOLOv8m Dubrovnik - validation detection metrics (not board accuracy)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = resolve_inside_root(args.run, ROOT)
    output_path = resolve_inside_root(args.output, ROOT)

    csv_path = run_dir / "results.csv"
    if not csv_path.is_file():
        raise SystemExit("Missing results.csv: {}".format(csv_path))

    rows = read_results(csv_path)
    series = extract_series(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    make_plot(series, output_path)

    print("Saved: {}".format(output_path))
    print("Epochs: {}".format(len(series["epoch"])))


if __name__ == "__main__":
    main()
