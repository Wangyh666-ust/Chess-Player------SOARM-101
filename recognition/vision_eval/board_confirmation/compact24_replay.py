from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from multimodel_review_trial import load_inputs, build_messages, _run_dashscope
from rook_prompt_low_trial_v2 import replace_prompt


ROOT = Path(__file__).resolve().parent
FIELD = ROOT / "field_references"
MANIFEST = FIELD / "manifest.json"
SELECTION = FIELD / "compact24_selection.json"
OUT_JPG = FIELD / "reference_compact24.jpg"
OUT_MANIFEST = FIELD / "manifest_compact24.json"

CELL_W, CELL_H = 156, 153
COLS, ROWS = 4, 6
TITLE_H = 24
CANVAS_W = CELL_W * COLS
CANVAS_H = TITLE_H + CELL_H * ROWS
IMG_MAX_W, IMG_MAX_H = 140, 118
JPEG_QUALITY = 91


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(examples: list, selection: dict) -> list:
    if len(selection) != 12 or set(selection) != {e["piece"] for e in examples}:
        raise ValueError("selection must have exactly twelve piece keys")
    all_ids = []
    selected = []
    for piece, ids in selection.items():
        if len(ids) != 2:
            raise ValueError(f"{piece} must list exactly two sample IDs")
        all_ids.extend(ids)
    if len(all_ids) != 24 or len(set(all_ids)) != 24:
        raise ValueError("selection must reference 24 unique IDs")
    by_id = {e["id"]: e for e in examples}
    for piece, ids in selection.items():
        a, b = (by_id[i] for i in ids)
        for e in (a, b):
            if e["piece"] != piece:
                raise ValueError(f"{e['id']} does not match piece {piece}")
        if (a["camera"], b["camera"]) != (1, 2):
            raise ValueError(f"{piece} requires a C1 photo followed by a C2 photo")
        if a["group"] == b["group"]:
            raise ValueError(f"{piece} C1/C2 in same group")
        selected.extend((a, b))
    for e in selected:
        data = (FIELD / e["file"]).read_bytes()
        if _sha256_bytes(data) != e["sha256"]:
            raise ValueError(f"sha256 mismatch for {e['id']}")
    return selected


def _contain(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = img.size
    scale = min(max_w / w, max_h / h, 1.0)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def build() -> None:
    source = _load_json(MANIFEST)
    examples = source["examples"]
    if source["group_split"] != "train" or len(examples) != 60:
        raise ValueError("unexpected source reference manifest")
    selection = _load_json(SELECTION)
    selected = _validate(examples, selection)

    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (245, 245, 240))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), "compact24 reference", fill=(0, 0, 0))

    for idx, e in enumerate(selected):
        row, col = divmod(idx, COLS)
        x = col * CELL_W
        y = TITLE_H + row * CELL_H
        with Image.open(FIELD / e["file"]) as raw:
            img = _contain(raw.convert("RGB"), IMG_MAX_W, IMG_MAX_H)
        canvas.paste(img, (x + (CELL_W - img.size[0]) // 2, y + 2))
        draw.text((x + 4, y + IMG_MAX_H + 4), f"{e['id']} C{e['camera']}", fill=(0, 0, 0))

    canvas.save(OUT_JPG, "JPEG", quality=JPEG_QUALITY)
    manifest = {
        "count": 24,
        "examples": selected,
        "jpeg_sha256": _sha256_file(OUT_JPG),
        "dimensions": [CANVAS_W, CANVAS_H],
        "source_manifest_sha256": _sha256_file(MANIFEST),
        "selection_sha256": _sha256_file(SELECTION),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_request(views, reference, prompt: str, reference_count: int) -> list:
    messages = build_messages(views, reference)
    messages = replace_prompt(messages, prompt)
    if reference_count == 24:
        intro = messages[0]["content"][1]
        if intro["type"] != "text" or "60 张" not in intro["text"]:
            raise ValueError("unexpected reference introduction position")
        intro["text"] = "下图是 24 张同款棋子的实拍参考，每类各有 C1、C2 一张。编号只对应参考样本。先看参考图，再看现场目标。"
    return messages


def run(prompt_path: Path, reference: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)

    views, _orig_prompt, old_atlas, old_sha = load_inputs()
    prompt = prompt_path.read_text(encoding="utf-8").strip()

    if reference == "compact":
        ref_bytes = OUT_JPG.read_bytes()
        ref_sha = _sha256_bytes(ref_bytes)
        ref_manifest = _load_json(OUT_MANIFEST)
        if ref_manifest["count"] != 24 or ref_manifest["jpeg_sha256"] != ref_sha:
            raise ValueError("compact reference manifest mismatch")
        reference_count = 24
    elif reference == "full":
        ref_bytes = old_atlas
        ref_sha = old_sha
        reference_count = 60
    else:
        raise ValueError("reference must be compact or full")

    messages = build_request(views, ref_bytes, prompt, reference_count)
    result = _run_dashscope("max", messages, list(views))

    image_blocks = sum(
        1
        for m in messages
        for b in (m.get("content") or [])
        if isinstance(b, dict) and b.get("type") == "image_url"
    )
    result.update({
        "reference_sha256": ref_sha,
        "reference_count": reference_count,
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "image_blocks": image_blocks,
    })
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--prompt", type=Path, required=True)
    run_parser.add_argument("--reference", choices=("compact", "full"), required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        build()
        return 0
    run(args.prompt, args.reference, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
