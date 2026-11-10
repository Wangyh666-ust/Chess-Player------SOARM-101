import argparse
import copy
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

PIN_AT_PREPARATION = "PIN_AT_PREPARATION"
EVIDENCE_FIELDS = ("id", "path", "sha256", "selector")


class PinError(Exception):
    pass


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _is_absolute_like(p):
    if not isinstance(p, str) or p == "":
        return True
    if p.startswith("/") or p.startswith("\\"):
        return True
    if len(p) >= 2 and p[1] == ":" and p[0].isalpha():
        return True
    if p.startswith("//") or p.startswith("\\\\"):
        return True
    return False


def _has_forbidden_component(p):
    parts = p.replace("\\", "/").split("/")
    for part in parts:
        if part in ("", "."):
            continue
        if part == "..":
            return True
        if part == ".git":
            return True
    return False


def _is_forbidden_credential(p):
    parts = p.replace("\\", "/").split("/")
    for part in parts:
        if part in ("", ".", ".."):
            continue
        low = part.lower()
        if low == "deepseek_api.txt":
            return True
        if low.endswith(".pem") or low.endswith(".key"):
            return True
    return False


def _validate_relative_path(path):
    if not isinstance(path, str) or path == "":
        raise PinError("missing or invalid relative path")
    if "\x00" in path:
        raise PinError("invalid path bytes")
    if _is_absolute_like(path):
        raise PinError("absolute path rejected: %r" % path)
    if path != path.strip() and (path.startswith(" ") or path.endswith(" ")):
        raise PinError("surrounding whitespace rejected: %r" % path)
    if _has_forbidden_component(path):
        raise PinError("traversal or .git component rejected: %r" % path)
    if _is_forbidden_credential(path):
        raise PinError("forbidden credential-like path rejected: %r" % path)
    return path


def _resolve_within_root(root, rel):
    _validate_relative_path(rel)
    if not isinstance(root, Path):
        root = Path(root)
    root_abs = Path(os.path.abspath(str(root)))
    working = root_abs
    try:
        rel_parts = rel.replace("\\", "/").split("/")
    except Exception as exc:
        raise PinError("invalid path: %r" % rel) from exc
    for part in rel_parts:
        if part in ("", "."):
            continue
        working = working / part
        try:
            st = os.lstat(str(working))
        except OSError as exc:
            raise PinError("missing path component %r for %r: %s" % (part, rel, exc)) from exc
        if stat.S_ISLNK(st.st_mode):
            raise PinError("symlink rejected in path %r" % rel)
        if stat.S_ISDIR(st.st_mode):
            continue
        if stat.S_ISREG(st.st_mode):
            break
        raise PinError("unsupported file type in path %r" % rel)
    final = root_abs / rel.replace("\\", "/")
    try:
        st = os.lstat(str(final))
    except OSError as exc:
        raise PinError("missing file %r: %s" % (rel, exc)) from exc
    if stat.S_ISLNK(st.st_mode):
        raise PinError("symlink rejected: %r" % rel)
    if not stat.S_ISREG(st.st_mode):
        raise PinError("not a regular file: %r" % rel)
    return final


def _read_pinned_file(root, rel):
    resolved = _resolve_within_root(root, rel)
    try:
        with open(str(resolved), "rb") as fh:
            data = fh.read()
    except OSError as exc:
        raise PinError("cannot read %r: %s" % (rel, exc)) from exc
    return _sha256_bytes(data)


def _check_expected(existing, actual, label):
    if not isinstance(existing, str):
        raise PinError("non-string digest at %s" % label)
    if existing == PIN_AT_PREPARATION:
        return actual
    if existing != actual:
        raise PinError(
            "digest mismatch at %s: expected %s actual %s "
            "(changes to a pinned digest require a new reviewed template)"
            % (label, existing, actual)
        )
    return existing


def _require_dict(value, label):
    if not isinstance(value, dict):
        raise PinError("required object missing at %s" % label)
    return value


def _require_list(value, label):
    if not isinstance(value, list):
        raise PinError("required list missing at %s" % label)
    return value


def pin(root, record):
    if not isinstance(record, dict):
        raise PinError("record must be a JSON object")
    root = Path(root)

    new_record = copy.deepcopy(record)

    config = _require_dict(new_record.get("configuration"), "configuration")
    factors = _require_dict(config.get("factors"), "configuration.factors")
    inputs = _require_list(factors.get("inputs"), "configuration.factors.inputs")
    source_hashes = _require_dict(factors.get("source_hashes"), "configuration.factors.source_hashes")
    evidence = _require_list(new_record.get("evidence"), "evidence")

    hashes_by_rel = {}

    for idx, item in enumerate(inputs):
        if not isinstance(item, dict):
            raise PinError("input %d is not an object" % idx)
        if "path" not in item or "sha256" not in item:
            raise PinError("input %d missing path or sha256" % idx)
        rel = item["path"]
        _validate_relative_path(rel)
        if rel in hashes_by_rel:
            raise PinError("duplicate declared input path %r" % rel)
        actual = _read_pinned_file(root, rel)
        item["sha256"] = _check_expected(item["sha256"], actual, "inputs[%d].sha256" % idx)
        hashes_by_rel[rel] = actual

    for rel, existing in list(source_hashes.items()):
        _validate_relative_path(rel)
        if rel in hashes_by_rel:
            actual = hashes_by_rel[rel]
            source_hashes[rel] = _check_expected(existing, actual, "source_hashes[%r]" % rel)
        else:
            actual = _read_pinned_file(root, rel)
            source_hashes[rel] = _check_expected(existing, actual, "source_hashes[%r]" % rel)

    for rel, actual in hashes_by_rel.items():
        if rel not in source_hashes:
            source_hashes[rel] = actual
        else:
            source_hashes[rel] = _check_expected(source_hashes[rel], actual, "source_hashes[%r]" % rel)

    for idx, entry in enumerate(evidence):
        if not isinstance(entry, dict):
            raise PinError("evidence %d is not an object" % idx)
        for field in EVIDENCE_FIELDS:
            if field not in entry:
                raise PinError("evidence %d missing %s" % (idx, field))
        rel = entry["path"]
        _validate_relative_path(rel)
        actual = _read_pinned_file(root, rel)
        entry["sha256"] = _check_expected(entry["sha256"], actual, "evidence[%d].sha256" % idx)

    return new_record


def _walk_ensure_no_symlink(path_parts, label):
    cur = Path(path_parts[0])
    st = os.lstat(str(cur))
    if stat.S_ISLNK(st.st_mode):
        raise PinError("symlink in %s" % label)
    for part in path_parts[1:]:
        cur = cur / part
        if not os.path.lexists(str(cur)):
            break
        st = os.lstat(str(cur))
        if stat.S_ISLNK(st.st_mode):
            raise PinError("symlink in %s" % label)


def _validate_output_path(root, out):
    if not isinstance(out, Path):
        out = Path(out)
    root_abs = Path(os.path.abspath(str(root)))
    out_abs = Path(os.path.abspath(str(out)))
    try:
        rel = out_abs.relative_to(root_abs)
    except ValueError:
        raise PinError("output path must be within root")
    if str(rel) in ("", "."):
        raise PinError("output path must be a file within root")
    _walk_ensure_no_symlink([str(root_abs)] + list(rel.parts), "output path")
    parent = out_abs.parent
    if not parent.is_dir():
        raise PinError("output parent directory does not exist")
    if os.path.lexists(str(out_abs)):
        raise PinError("output path already exists; refusing to overwrite")
    return out_abs


def _load_template(path):
    try:
        with open(str(path), "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise PinError("cannot read template: %s" % exc) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PinError("template is not valid UTF-8: %s" % exc) from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise PinError("template is not valid JSON: %s" % exc) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Pin declared source/input bytes for a fresh experiment record."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    try:
        root = Path(args.root)
        if not root.is_dir():
            raise PinError("root is not a directory")
        if os.path.islink(str(root)):
            raise PinError("root must not be a symlink")
        record = _load_template(Path(args.template))
        out_abs = _validate_output_path(root, Path(args.out))
        updated = pin(root, record)
        payload = json.dumps(updated, indent=2, ensure_ascii=False) + "\n"
        data = payload.encode("utf-8")
        with open(str(out_abs), "xb") as fh:
            fh.write(data)
        file_hash = _sha256_bytes(data)
    except PinError as exc:
        sys.stderr.write(json.dumps({"error": str(exc)}) + "\n")
        return 1

    sys.stdout.write(
        json.dumps({"record": str(out_abs), "sha256": file_hash}) + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
