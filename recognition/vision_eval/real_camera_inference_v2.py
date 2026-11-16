import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import sys
import time

ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _as_numpy(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return x


def detections_from_result(result, names):
    boxes_obj = result.boxes
    xyxy = _as_numpy(boxes_obj.xyxy)
    cls = _as_numpy(boxes_obj.cls)
    conf = _as_numpy(boxes_obj.conf)

    import numpy as _np

    xyxy = _np.asarray(xyxy, dtype=float)
    if xyxy.size == 0:
        xyxy = xyxy.reshape(0, 4)
    if xyxy.ndim != 2 or xyxy.shape[1] != 4:
        raise ValueError("boxes must have shape (N,4), got %r" % (xyxy.shape,))

    cls_flat = _np.asarray(cls).reshape(-1)
    conf_flat = _np.asarray(conf, dtype=float).reshape(-1)

    n = xyxy.shape[0]
    if cls_flat.shape[0] != n:
        raise ValueError(
            "length mismatch: boxes=%d classes=%d" % (n, cls_flat.shape[0])
        )
    if conf_flat.shape[0] != n:
        raise ValueError(
            "length mismatch: boxes=%d conf=%d" % (n, conf_flat.shape[0])
        )
    if not _np.all(_np.isfinite(xyxy)):
        raise ValueError("non-finite box coordinate")
    if not _np.all(_np.isfinite(conf_flat)):
        raise ValueError("non-finite confidence")

    out = []
    for i in range(n):
        cid = cls_flat[i]
        try:
            cid_int = int(cid)
        except Exception:
            raise ValueError("non-integer class id: %r" % (cid,))
        if float(cid_int) != float(cid):
            raise ValueError("non-integer class id: %r" % (cid,))
        if cid_int not in names:
            raise ValueError("class id %d not present in names" % cid_int)
        box = [float(v) for v in xyxy[i]]
        out.append(
            {
                "box": box,
                "class_id": cid_int,
                "class_name": str(names[cid_int]),
                "confidence": float(conf_flat[i]),
            }
        )
    return out


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_model_factory(path):
    from ultralytics import YOLO

    return YOLO(path)


def _default_image_reader(path):
    import cv2

    return cv2.imread(path)


def _default_image_writer(path, image):
    import cv2

    ok = cv2.imwrite(path, image)
    return bool(ok)


def _validate_config(config):
    errors = []
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")

    required = [
        "models",
        "images",
        "variants",
        "output_dir",
        "max_predictions",
        "max_seconds",
    ]
    for k in required:
        if k not in config:
            errors.append("missing config key: %s" % k)
    if errors:
        raise ValueError("; ".join(errors))

    models = config["models"]
    if not isinstance(models, dict) or set(models.keys()) != {"original", "finetuned"}:
        raise ValueError("models must contain exactly 'original' and 'finetuned'")
    model_paths = {}
    for mkey, mval in models.items():
        if not isinstance(mval, dict):
            raise ValueError("model %s must be an object" % mkey)
        for f in ("path", "sha256"):
            if f not in mval or not isinstance(mval[f], str):
                raise ValueError("model %s missing string field %s" % (mkey, f))
        model_paths[mkey] = mval

    images = config["images"]
    if not isinstance(images, list) or len(images) == 0:
        raise ValueError("images must be a non-empty list")
    image_ids = set()
    for img in images:
        if not isinstance(img, dict):
            raise ValueError("each image must be an object")
        for f in ("id", "path", "sha256", "origin"):
            if f not in img or not isinstance(img[f], str):
                raise ValueError("image missing string field %s" % f)
        if not ID_RE.match(img["id"]):
            raise ValueError("invalid image id: %r" % (img["id"],))
        if img["id"] in image_ids:
            raise ValueError("duplicate image id: %r" % (img["id"],))
        image_ids.add(img["id"])

    variants = config["variants"]
    if not isinstance(variants, list) or len(variants) == 0:
        raise ValueError("variants must be a non-empty list")
    variant_ids = set()
    for v in variants:
        if not isinstance(v, dict):
            raise ValueError("each variant must be an object")
        for f in ("id", "model", "imgsz", "conf", "iou"):
            if f not in v:
                raise ValueError("variant missing field %s" % f)
        if not isinstance(v["id"], str) or not ID_RE.match(v["id"]):
            raise ValueError("invalid variant id: %r" % (v["id"],))
        if v["id"] in variant_ids:
            raise ValueError("duplicate variant id: %r" % (v["id"],))
        variant_ids.add(v["id"])
        if v["model"] not in model_paths:
            raise ValueError(
                "variant %s references unknown model %r" % (v["id"], v["model"])
            )
        if not isinstance(v["imgsz"], int) or isinstance(v["imgsz"], bool) or v["imgsz"] <= 0:
            raise ValueError("variant %s imgsz must be a positive int" % v["id"])
        for f in ("conf", "iou"):
            if not isinstance(v[f], (int, float)) or isinstance(v[f], bool):
                raise ValueError("variant %s %s must be numeric" % (v["id"], f))
            if not math.isfinite(float(v[f])):
                raise ValueError("variant %s %s must be finite" % (v["id"], f))

    output_dir = config["output_dir"]
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir must be a non-empty string")
    if os.path.exists(output_dir):
        raise ValueError("output_dir already exists: %s" % output_dir)

    max_predictions = config["max_predictions"]
    if (
        not isinstance(max_predictions, int)
        or isinstance(max_predictions, bool)
        or max_predictions <= 0
    ):
        raise ValueError("max_predictions must be a positive int")

    max_seconds = config["max_seconds"]
    if not isinstance(max_seconds, (int, float)) or isinstance(max_seconds, bool):
        raise ValueError("max_seconds must be numeric")
    if not math.isfinite(float(max_seconds)) or float(max_seconds) <= 0:
        raise ValueError("max_seconds must be positive and finite")

    total_predictions = len(images) * len(variants)
    if total_predictions > max_predictions:
        raise ValueError(
            "total predictions %d exceeds max_predictions %d"
            % (total_predictions, max_predictions)
        )

    files_to_check = []
    for mkey, mval in models.items():
        files_to_check.append((("model", mkey), mval["path"], mval["sha256"]))
    for img in images:
        files_to_check.append((("image", img["id"]), img["path"], img["sha256"]))

    for label, path, expected in files_to_check:
        if not os.path.isfile(path):
            raise ValueError("%s file does not exist: %s" % (label[0], path))
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(
                "sha256 mismatch for %s %s: expected %s got %s"
                % (label[0], label[1], expected, actual)
            )

    return total_predictions


def _environment_info():
    env = {
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "packages": {},
    }
    for name in ("numpy", "opencv-python", "opencv-python-headless", "torch", "ultralytics"):
        try:
            env["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            env["packages"][name] = None
        except Exception:
            env["packages"][name] = None
    return env


def _result_speed(result):
    speed = getattr(result, "speed", None)
    if speed is None:
        return None
    try:
        if isinstance(speed, dict):
            return {str(k): float(v) for k, v in speed.items()}
        return dict(speed)
    except Exception:
        return None


def run_suite(config, model_factory=None, image_reader=None, image_writer=None):
    if model_factory is None:
        model_factory = _default_model_factory
    if image_reader is None:
        image_reader = _default_image_reader
    if image_writer is None:
        image_writer = _default_image_writer

    total_predictions = _validate_config(config)

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=False)

    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")

    models = config["models"]
    images = config["images"]
    variants = config["variants"]

    model_cache = {}

    def get_model(model_key):
        if model_key in model_cache:
            return model_cache[model_key]
        spec = models[model_key]
        model = model_factory(spec["path"])
        names_raw = getattr(model, "names", None)
        if names_raw is None:
            raise ValueError("model %s has no names attribute" % model_key)
        if isinstance(names_raw, dict):
            names = {}
            for k, v in names_raw.items():
                try:
                    ik = int(k)
                except Exception:
                    raise ValueError("non-integer model names key: %r" % (k,))
                names[ik] = str(v)
        else:
            names = {i: str(n) for i, n in enumerate(names_raw)}
        model_cache[model_key] = (model, names)
        return model_cache[model_key]

    max_seconds = float(config["max_seconds"])
    start = time.monotonic()

    records = []

    with open(predictions_path, "w", encoding="utf-8") as pred_file:
        for variant in variants:
            model_key = variant["model"]
            model_sha = models[model_key]["sha256"]
            model_obj, names = get_model(model_key)
            for img in images:
                if time.monotonic() - start > max_seconds:
                    raise TimeoutError(
                        "max_seconds budget %.3f exceeded before prediction"
                        % max_seconds
                    )

                frame = image_reader(img["path"])
                if frame is None:
                    raise ValueError("image_reader returned None for %s" % img["path"])

                height = int(frame.shape[0]) if hasattr(frame, "shape") and len(frame.shape) >= 2 else None
                width = int(frame.shape[1]) if hasattr(frame, "shape") and len(frame.shape) >= 2 else None
                if width is None or height is None:
                    raise ValueError("image has no usable shape: %s" % img["path"])

                call_t0 = time.monotonic()
                results = model_obj.predict(
                    frame,
                    device=0,
                    imgsz=variant["imgsz"],
                    conf=variant["conf"],
                    iou=variant["iou"],
                    half=False,
                    verbose=False,
                    batch=1,
                )
                result = results[0]
                detections = detections_from_result(result, names)
                call_elapsed = time.monotonic() - call_t0

                annotated_name = "%s__%s.png" % (variant["id"], img["id"])

                counts = {}
                for det in detections:
                    cname = det["class_name"]
                    counts[cname] = counts.get(cname, 0) + 1

                record = {
                    "variant_id": variant["id"],
                    "image_id": img["id"],
                    "input_sha256": img["sha256"],
                    "model_sha256": model_sha,
                    "imgsz": variant["imgsz"],
                    "conf": variant["conf"],
                    "iou": variant["iou"],
                    "origin": img["origin"],
                    "width": width,
                    "height": height,
                    "detections": detections,
                    "counts": counts,
                    "predict_wall_seconds": float(call_elapsed),
                    "speed": _result_speed(result),
                    "annotated": annotated_name,
                }
                pred_file.write(json.dumps(record) + "\n")
                pred_file.flush()
                records.append(record)

                annotated_path = os.path.join(output_dir, annotated_name)
                plotted = result.plot()
                ok = image_writer(annotated_path, plotted)
                if not ok:
                    raise RuntimeError("image_writer returned false for %s" % annotated_path)

    elapsed = time.monotonic() - start

    summary = {
        "schema_version": 1,
        "config": config,
        "records": records,
        "prediction_count": len(records),
        "elapsed_seconds": float(elapsed),
        "environment": _environment_info(),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.flush()

    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Bounded offline YOLO diagnostic suite"
    )
    parser.add_argument("--config", required=True, help="Path to JSON config")
    args = parser.parse_args(argv)

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        summary = run_suite(config)
    except SystemExit:
        raise
    except Exception as exc:
        sys.stderr.write("error: %s: %s\n" % (type(exc).__name__, exc))
        return 1

    print("completed predictions: %d" % summary["prediction_count"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
