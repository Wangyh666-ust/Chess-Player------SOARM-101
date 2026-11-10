import copy
import math
import re
import sys
from pathlib import Path


def tighten_opening(frames, delta_deg):
    frames_copy = copy.deepcopy(frames)
    if not frames_copy:
        raise ValueError("frames must be nonempty")
    gc = min(float(f['g']) for f in frames_copy)
    go = max(float(f['g']) for f in frames_copy)
    assert go > gc
    assert delta_deg > 0
    newgo = min(go, gc + math.radians(delta_deg))
    for f in frames_copy:
        oldg = float(f['g'])
        f['g'] = gc + (oldg - gc) / (go - gc) * (newgo - gc)
    return frames_copy


def _patch(path):
    text = path.read_text(encoding="utf-8")
    original = text

    # 1. import addition
    imp_anchor = "from dynamics_evaluate import evaluate"
    imp_new = imp_anchor + "\nfrom opening_profile_v1 import tighten_opening"
    if "from opening_profile_v1 import tighten_opening" not in text:
        if imp_anchor not in text:
            raise RuntimeError("import anchor not found")
        text = text.replace(imp_anchor, imp_new, 1)

    # 2. SOURCE_FILES tuple
    old_tuple = 'SOURCE_FILES = ("dynamics_cli.py", "physics_world.py", "dynamics_drive.py", "dynamics_evaluate.py", "full_piece_v2.py")'
    new_tuple = 'SOURCE_FILES = ("dynamics_cli.py", "physics_world.py", "dynamics_drive.py", "dynamics_evaluate.py", "full_piece_v2.py", "opening_profile_v1.py")'
    if '"opening_profile_v1.py"' not in text:
        if old_tuple not in text:
            raise RuntimeError("SOURCE_FILES tuple not found")
        text = text.replace(old_tuple, new_tuple, 1)

    # 3. add --opening-delta-deg argument before return parse call
    if "--opening-delta-deg" not in text:
        m = re.search(r"return\s+(\w+)\.parse_args\(argv\)", text)
        if not m:
            raise RuntimeError("return parse_args(argv) pattern not found")
        var = m.group(1)
        insertion = "    {}.add_argument('--opening-delta-deg', type=float, default=None)\n".format(var)
        start = text.rfind('\n', 0, m.start()) + 1
        text = text[:start] + insertion + text[start:]

    # 4. after _load_source call insert tightening block
    if "args.opening_delta_deg is not None" not in text:
        pattern = re.compile(
            r"(\n(\s*)(\w+),\s*(\w+),\s*(\w+),\s*(\w+),\s*(\w+)\s*=\s*_load_source\(\s*args\.case\s*,\s*args\.source\s*\))"
        )
        m = pattern.search(text)
        if not m:
            raise RuntimeError("_load_source assignment not found")
        indent = m.group(2)
        if indent != "    ":
            indent = "    "
        block = (
            "\n" + indent + "if args.opening_delta_deg is not None:" +
            "\n" + indent + "    frames = tighten_opening(frames, args.opening_delta_deg)" +
            "\n" + indent + "    g = float(frames[0]['g'])" +
            "\n"
        )
        end = m.end()
        text = text[:end] + block + text[end:]

    if text != original:
        path.write_text(text, encoding="utf-8")


def main():
    here = Path(__file__).resolve().parent
    target = here / "dynamics_cli.py"
    if not target.exists():
        raise SystemExit("dynamics_cli.py not found")
    _patch(target)


if __name__ == "__main__":
    main()
