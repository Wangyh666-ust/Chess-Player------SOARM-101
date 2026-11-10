#!/usr/bin/env python3
"""Standalone portable snapshot utility (stdlib only).

Two public operations:

  build(source, destination, files)  ->  dict
      Copy an explicitly reviewed set of relative files from ``source`` into a
      *new* ``destination`` tree:

          destination/
              manifest.json
              runtime/        (byte-for-byte copies / normalized JSON / URDF)

      JSON strings that begin with the source root are rewritten to
      ``@WORKSPACE@`` plus a forward-slash relative suffix.  URDF mesh
      ``filename`` attributes that are absolute paths under ``source`` are
      rewritten to paths relative to the destination URDF parent (same layout).

  prepare(bundle, destination)  ->  dict
      Materialize a *new* workspace from a built bundle, replacing
      ``@WORKSPACE@`` strings inside JSON files with the destination's absolute
      POSIX path.  Everything else is copied byte-for-byte.

Declared transformations (re-serialization is permitted and expected):

  * ``json_paths``       - JSON is re-serialized; only strings starting with
                           the source root are changed; all other keys, values,
                           list order, numbers, booleans and null are preserved.
  * ``urdf_mesh_paths``  - URDF is re-serialized with ElementTree; only
                           absolute ``filename`` mesh attributes under the
                           source are rewritten; mass, origin and every other
                           attribute are unchanged.

No network, no shells, no credentials, no imports from the project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

WORKSPACE_TOKEN = "@WORKSPACE@"
MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"
RUNTIME_DIRNAME = "runtime"
MATERIALIZATION_NAME = "materialization.json"

# Filename patterns that look like credentials.
_CREDENTIAL_NAMES = {"deepseek_api.txt"}
_CREDENTIAL_SUFFIXES = (".pem", ".key")

# ---------------------------------------------------------------------------
# Path validation helpers
# ---------------------------------------------------------------------------


class SnapshotError(Exception):
    """Raised for any invalid input or unsafe condition."""

    pass


def _normalize_rel_name(raw: str) -> str:
    """Validate a caller supplied relative path and return its POSIX form.

    Accepts both ``/`` and ``\\`` as separators.  Rejects absolute paths,
    drive-letter prefixes, ``..`` components, ``.git`` components and
    credential-like filenames.
    """
    if not isinstance(raw, str):
        raise SnapshotError(f"file entry is not a string: {raw!r}")
    if raw == "":
        raise SnapshotError("empty file entry")
    if "\x00" in raw:
        raise SnapshotError(f"null byte in file entry: {raw!r}")

    # Reject absolute paths.  This deliberately does not rely on the host OS
    # drive parser so that Windows-style paths are rejected on POSIX too.
    cleaned = raw.replace("\\", "/")
    if cleaned.startswith("/"):
        raise SnapshotError(f"absolute path not allowed: {raw!r}")
    if re.match(r"^[A-Za-z]:", cleaned):
        raise SnapshotError(f"drive-qualified path not allowed: {raw!r}")
    if cleaned.startswith("//"):
        raise SnapshotError(f"UNC path not allowed: {raw!r}")

    parts = [p for p in cleaned.split("/") if p not in ("",)]
    if not parts:
        raise SnapshotError(f"empty path after normalization: {raw!r}")
    for part in parts:
        if part == "..":
            raise SnapshotError(f"path traversal not allowed: {raw!r}")
        if part == ".":
            raise SnapshotError(f"dot component not allowed: {raw!r}")
        if part == ".git":
            raise SnapshotError(f".git component not allowed: {raw!r}")

    posix = "/".join(parts)
    _reject_credential_name(posix)
    return posix


def _reject_credential_name(posix_path: str) -> None:
    name = posix_path.rsplit("/", 1)[-1]
    lower = name.lower()
    if lower in _CREDENTIAL_NAMES:
        raise SnapshotError(f"credential-like filename not allowed: {posix_path!r}")
    for suffix in _CREDENTIAL_SUFFIXES:
        if lower.endswith(suffix):
            raise SnapshotError(f"credential-like filename not allowed: {posix_path!r}")


def _prepare_files_argument(files: Iterable[str]) -> list[str]:
    if isinstance(files, (str, bytes)):
        raise SnapshotError("files must be a list of relative path strings")
    normalized = {_normalize_rel_name(f) for f in files}
    return sorted(normalized)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Root prefix handling (source side)
# ---------------------------------------------------------------------------


def _source_root_variants(source: Path) -> list[str]:
    """Return absolute source root spellings (POSIX + backslash) with and
    without trailing separator, longest first so that prefixes match the
    most specific form."""
    posix = source.resolve().as_posix()
    variants = {posix, posix + "/"}
    # Backslash spelling of the same root (for Windows-style strings).
    bs = posix.replace("/", "\\")
    variants.add(bs)
    variants.add(bs + "\\")
    # Some JSON may store a trailing slash on Windows also.
    bs_slash = bs + "/"
    variants.add(bs_slash)
    posix_bs = posix + "\\"
    variants.add(posix_bs)
    return sorted(variants, key=len, reverse=True)


def _relative_suffix(rest: str) -> str:
    """Normalize a relative suffix to forward slashes."""
    if rest == "":
        return ""
    rest = rest.replace("\\", "/")
    return rest.lstrip("/") if rest.startswith("/") else rest


def _rewrite_source_string(value: str, variants: list[str]) -> str | None:
    normalized = value.replace("\\", "/")
    root = min(variants, key=len).replace("\\", "/").rstrip("/")
    if normalized == root:
        return WORKSPACE_TOKEN
    if normalized.startswith(root + "/"):
        return WORKSPACE_TOKEN + normalized[len(root):]
    return None


def _transform_json_value(value: Any, variants: list[str]) -> tuple[Any, bool]:
    if isinstance(value, str):
        new = _rewrite_source_string(value, variants)
        if new is None:
            return value, False
        return new, True
    if isinstance(value, list):
        changed = False
        out = []
        for item in value:
            nv, c = _transform_json_value(item, variants)
            out.append(nv)
            changed = changed or c
        return out, changed
    if isinstance(value, dict):
        changed = False
        out = {}
        for k, v in value.items():
            nv, cv = _transform_json_value(v, variants)
            out[k] = nv
            changed = changed or cv
        return out, changed
    return value, False


# ---------------------------------------------------------------------------
# URDF handling (source side)
# ---------------------------------------------------------------------------


def _is_absolute_mesh_path(name: str) -> bool:
    n = name.replace("\\", "/")
    if n.startswith("/"):
        return True
    if re.match(r"^[A-Za-z]:", n):
        return True
    if n.startswith("//"):
        return True
    return False


def _transform_urdf(data: bytes, source: Path, urdf_rel: str) -> bytes:
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise SnapshotError(f"invalid URDF XML in {urdf_rel!r}: {exc}") from exc

    variants = _source_root_variants(source)
    changed = False

    for mesh in root.iter("mesh"):
        fname = mesh.get("filename")
        if fname is None:
            continue
        if not _is_absolute_mesh_path(fname):
            # Relative mesh names stay exactly as supplied.
            continue
        # Normalize the absolute form to POSIX for matching.
        normalized = fname.replace("\\", "/")
        matched = None
        root_prefix = min(variants, key=len).replace("\\", "/").rstrip("/")
        if normalized == root_prefix or normalized.startswith(root_prefix + "/"):
            matched = root_prefix
        if matched is None:
            raise SnapshotError(
                f"absolute mesh path outside source in {urdf_rel!r}: {fname!r}"
            )
        rest = normalized[len(matched.replace("\\", "/")):]
        rest = _relative_suffix(rest)
        # URDF lives at runtime/<path>; meshes live at runtime/<rel>.
        # Layout is identical, so the relative path from the URDF parent to the
        # referenced mesh is preserved if we keep the original tail only when
        # the URDF is at the root (runtime/).  Otherwise compute properly.
        mesh_abs_rel = rest  # relative to source root, POSIX
        urdf_parent_rel = PurePosixPath(urdf_rel).parent
        if str(urdf_parent_rel) == ".":
            new_name = mesh_abs_rel
        else:
            # Compose a relative path from urdf parent back to the mesh.
            new_name = os.path.relpath(
                PurePosixPath("/") / mesh_abs_rel,
                PurePosixPath("/") / urdf_parent_rel,
            ).replace(os.sep, "/")
        mesh.set("filename", new_name)
        changed = True

    if not changed:
        # Avoid re-serializing untouched URDFs.
        return data
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# Bundle-internal helpers (prepare side)
# ---------------------------------------------------------------------------


def _validate_bundle_rel(posix: str) -> str:
    return _normalize_rel_name(posix)


def _workspace_suffix_ok(suffix: str) -> str:
    """Validate the suffix following @WORKSPACE@ inside a JSON string.

    Empty suffix -> token maps to destination root.
    Otherwise suffix must start with ``/``, contain no backslashes and no
    escaping ``.`` / ``..`` components.
    """
    if suffix == "":
        return ""
    if "\\" in suffix:
        raise SnapshotError(f"backslash in @WORKSPACE@ suffix: {suffix!r}")
    if not suffix.startswith("/"):
        raise SnapshotError(f"@WORKSPACE@ suffix must be empty or start with '/': {suffix!r}")
    tail = suffix[1:]
    if tail == "":
        return ""
    for part in tail.split("/"):
        if part == "":
            raise SnapshotError(f"empty component in @WORKSPACE@ suffix: {suffix!r}")
        if part in (".", ".."):
            raise SnapshotError(f"escaping component in @WORKSPACE@ suffix: {suffix!r}")
    return tail


def _rewrite_workspace_value(value: Any, workspace_posix: str) -> tuple[Any, bool]:
    if isinstance(value, str):
        if not value.startswith(WORKSPACE_TOKEN):
            return value, False
        suffix = value[len(WORKSPACE_TOKEN):]
        tail = _workspace_suffix_ok(suffix)
        new = workspace_posix if tail == "" else workspace_posix + "/" + tail
        return new, True
    if isinstance(value, list):
        changed = False
        out = []
        for item in value:
            nv, c = _rewrite_workspace_value(item, workspace_posix)
            out.append(nv)
            changed = changed or c
        return out, changed
    if isinstance(value, dict):
        changed = False
        out = {}
        for k, v in value.items():
            nv, c = _rewrite_workspace_value(v, workspace_posix)
            out[k] = nv
            changed = changed or c
        return out, changed
    return value, False


def _apply_workspace(data: bytes, workspace_posix: str) -> bytes:
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"invalid JSON in bundle: {exc}") from exc
    new, changed = _rewrite_workspace_value(parsed, workspace_posix)
    if not changed:
        return data
    return json.dumps(new, indent=2, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Filesystem checks
# ---------------------------------------------------------------------------


def _ensure_new_destination(destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise SnapshotError(f"destination already exists: {destination}")
    parent = destination.parent
    if not parent.exists():
        raise SnapshotError(f"destination parent does not exist: {parent}")


def _check_overlap(out_root: Path, input_paths: Iterable[Path]) -> None:
    out_resolved = out_root.resolve()
    for ip in input_paths:
        try:
            ir = ip.resolve()
        except OSError:
            continue
        if ir == out_resolved or out_resolved in ir.parents or ir in out_resolved.parents:
            raise SnapshotError(f"output overlaps input: {ip} vs {out_root}")


def _is_ignorable_cache(rel_posix: str) -> bool:
    return any(part == "__pycache__" for part in rel_posix.split("/"))


def _walk_regular_files(root: Path) -> list[str]:
    """Return relative POSIX paths of ordinary files under ``root``."""
    results: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            full = Path(dirpath) / name
            if full.is_symlink():
                continue
            if not full.is_file():
                continue
            rel = full.relative_to(root).as_posix()
            if _is_ignorable_cache(rel):
                continue
            results.append(rel)
    return sorted(results)


# ---------------------------------------------------------------------------
# Public: build
# ---------------------------------------------------------------------------


def build(source: Path, destination: Path, files: list[str]) -> dict:
    source = Path(source)
    destination = Path(destination)

    if not source.exists():
        raise SnapshotError(f"source does not exist: {source}")
    if source.is_symlink():
        raise SnapshotError(f"source must not be a symlink: {source}")
    if not source.is_dir():
        raise SnapshotError(f"source is not a directory: {source}")

    source_resolved = source.resolve()
    normalized = _prepare_files_argument(files)
    if not normalized:
        raise SnapshotError("no files specified")

    # Validate the entire input set before creating output.
    input_abs: list[Path] = []
    for rel in normalized:
        target = source_resolved / rel
        # Reject symlinked directories or files anywhere along the chain.
        cursor = source_resolved
        for part in rel.split("/"):
            cursor = cursor / part
            if cursor.is_symlink():
                raise SnapshotError(f"source path is a symlink: {rel}")
        if not target.exists():
            raise SnapshotError(f"missing input file: {rel}")
        if not target.is_file():
            raise SnapshotError(f"not a regular file: {rel}")
        input_abs.append(target)

    _ensure_new_destination(destination)
    _check_overlap(destination, input_abs + [source_resolved])

    destination_parent = destination.parent.resolve()
    if destination_parent in source_resolved.parents or destination_parent == source_resolved:
        raise SnapshotError("destination must not be inside the source tree")

    # Compute plan before mutating disk for stronger atomicity.
    variants = _source_root_variants(source_resolved)
    plan: list[tuple[str, bytes, str, str]] = []  # (rel, data, source_sha, transformation)
    for rel, target in zip(normalized, input_abs):
        raw = target.read_bytes()
        source_sha = _sha256_bytes(raw)
        lower = rel.lower()
        if lower.endswith(".json"):
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SnapshotError(f"invalid JSON input {rel!r}: {exc}") from exc
            new, changed = _transform_json_value(parsed, variants)
            if changed:
                out = json.dumps(new, indent=2, ensure_ascii=False).encode("utf-8")
                transformation = "json_paths"
            else:
                out = raw
                transformation = "copy"
            plan.append((rel, out, source_sha, transformation))
        elif lower.endswith(".urdf"):
            out = _transform_urdf(raw, source_resolved, rel)
            transformation = "urdf_mesh_paths" if out is not raw and out != raw else "copy"
            plan.append((rel, out, source_sha, transformation))
        else:
            plan.append((rel, raw, source_sha, "copy"))

    # Create the output tree (x-mode).  Never use exist_ok.
    destination.mkdir(mode=0o755, parents=False, exist_ok=False)
    runtime = destination / RUNTIME_DIRNAME
    runtime.mkdir(mode=0o755, exist_ok=False)

    manifest_files: list[dict] = []
    try:
        for rel, data, source_sha, transformation in plan:
            out_path = runtime / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "xb") as fh:
                fh.write(data)
            bundle_sha = _sha256_bytes(data)
            manifest_files.append(
                {
                    "path": rel,
                    "source_sha256": source_sha,
                    "bundle_sha256": bundle_sha,
                    "transformation": transformation,
                }
            )

        manifest = {
            "version": MANIFEST_VERSION,
            "scope": f"snapshot:{WORKSPACE_TOKEN}",
            "files": manifest_files,
        }
        manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
        with open(destination / MANIFEST_NAME, "xb") as fh:
            fh.write(manifest_bytes)
    except Exception:
        # Leave partial output for diagnosis, do not delete.
        raise

    manifest_sha = _sha256_bytes(manifest_bytes)
    return {
        "destination": destination.resolve().as_posix(),
        "runtime": runtime.resolve().as_posix(),
        "files": len(manifest_files),
        "manifest_sha256": manifest_sha,
    }


# ---------------------------------------------------------------------------
# Public: prepare
# ---------------------------------------------------------------------------


def _load_manifest(bundle_root: Path) -> tuple[dict, bytes]:
    manifest_path = bundle_root / MANIFEST_NAME
    if not manifest_path.exists() or not manifest_path.is_file():
        raise SnapshotError(f"manifest missing: {manifest_path}")
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"invalid manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SnapshotError("manifest must be a JSON object")
    if manifest.get("version") != MANIFEST_VERSION:
        raise SnapshotError(f"unsupported manifest version: {manifest.get('version')!r}")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise SnapshotError("manifest 'files' must be a list")
    return manifest, raw


def prepare(bundle: Path, destination: Path) -> dict:
    bundle = Path(bundle)
    destination = Path(destination)

    if not bundle.exists():
        raise SnapshotError(f"bundle does not exist: {bundle}")
    if bundle.is_symlink():
        raise SnapshotError(f"bundle must not be a symlink: {bundle}")
    if not bundle.is_dir():
        raise SnapshotError(f"bundle is not a directory: {bundle}")

    bundle_resolved = bundle.resolve()
    runtime_root = bundle_resolved / RUNTIME_DIRNAME
    if not runtime_root.is_dir():
        raise SnapshotError(f"bundle runtime directory missing: {runtime_root}")

    manifest, manifest_raw = _load_manifest(bundle_resolved)

    # Validate every manifest path and hash against runtime/.
    entries: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise SnapshotError("manifest file entry must be an object")
        rel_raw = entry.get("path")
        rel = _validate_bundle_rel(rel_raw)
        if rel in seen:
            raise SnapshotError(f"duplicate manifest path: {rel}")
        seen.add(rel)
        expected = entry.get("bundle_sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise SnapshotError(f"invalid bundle_sha256 for {rel!r}")
        transformation = entry.get("transformation")
        if transformation not in ("copy", "json_paths", "urdf_mesh_paths"):
            raise SnapshotError(f"invalid transformation for {rel!r}: {transformation!r}")
        source_sha = entry.get("source_sha256")
        if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
            raise SnapshotError(f"invalid source_sha256 for {rel!r}")

        target = runtime_root / rel
        cursor = runtime_root
        for part in rel.split("/"):
            cursor = cursor / part
            if cursor.is_symlink():
                raise SnapshotError(f"bundle path is a symlink: {rel}")
        if not target.exists():
            raise SnapshotError(f"bundle missing file: {rel}")
        if not target.is_file():
            raise SnapshotError(f"bundle entry is not a regular file: {rel}")
        actual = _sha256_file(target)
        if actual != expected:
            raise SnapshotError(
                f"bundle hash mismatch for {rel!r}: expected {expected}, got {actual}"
            )
        entries.append((rel, entry))

    # Reject unlisted ordinary files inside runtime/ (ignore __pycache__).
    on_disk = _walk_regular_files(runtime_root)
    listed = {rel for rel, _ in entries}
    unlisted = [p for p in on_disk if p not in listed and not _is_ignorable_cache(p)]
    if unlisted:
        raise SnapshotError(f"bundle contains unlisted files: {unlisted}")

    # Destination checks before any mutation.
    if destination.exists() or destination.is_symlink():
        raise SnapshotError(f"destination already exists: {destination}")
    if not destination.parent.exists():
        raise SnapshotError(f"destination parent does not exist: {destination.parent}")

    dest_resolved = destination.resolve()
    bundle_runtime_resolved = runtime_root.resolve()
    if dest_resolved == bundle_runtime_resolved:
        raise SnapshotError("destination equals bundle runtime")
    if bundle_runtime_resolved in dest_resolved.parents:
        raise SnapshotError("destination inside bundle runtime")
    if dest_resolved in bundle_resolved.parents or bundle_resolved in dest_resolved.parents:
        raise SnapshotError("destination overlaps bundle")

    input_paths = [runtime_root / rel for rel, _ in entries] + [bundle_resolved / MANIFEST_NAME]
    _check_overlap(destination, input_paths)

    workspace_posix = dest_resolved.as_posix()

    # Precompute outputs so failures during write leave partial output only.
    outputs: list[tuple[str, bytes, str]] = []
    for rel, entry in entries:
        src = runtime_root / rel
        data = src.read_bytes()
        transformation = entry["transformation"]
        if transformation == "json_paths" and rel.lower().endswith(".json"):
            data = _apply_workspace(data, workspace_posix)
        outputs.append((rel, data, transformation))

    destination.mkdir(mode=0o755, parents=False, exist_ok=False)

    count = 0
    try:
        for rel, data, _transformation in outputs:
            out_path = destination / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "xb") as fh:
                fh.write(data)
            count += 1

        # Preserve the bundle manifest verbatim at the destination root.
        with open(destination / MANIFEST_NAME, "xb") as fh:
            fh.write(manifest_raw)

        manifest_sha = _sha256_bytes(manifest_raw)
        result = {
            "workspace": workspace_posix,
            "files": count,
            "manifest_sha256": manifest_sha,
        }
        materialization = json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")
        with open(destination / MATERIALIZATION_NAME, "xb") as fh:
            fh.write(materialization)
    except Exception:
        raise

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_files_json(path: str) -> list[str]:
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"invalid files JSON: {exc}") from exc
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise SnapshotError("files JSON must be a list of strings")
    return parsed


def _cmd_build(args: argparse.Namespace) -> int:
    files = _load_files_json(args.files)
    result = build(Path(args.source), Path(args.out), files)
    print(json.dumps(result, indent=2))
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    result = prepare(Path(args.bundle), Path(args.out))
    print(json.dumps(result, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="snapshot", description="Portable snapshot utility")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build a portable snapshot bundle")
    b.add_argument("--source", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--files", required=True, help="path to JSON list of relative files")
    b.set_defaults(func=_cmd_build)

    p = sub.add_parser("prepare", help="materialize a workspace from a bundle")
    p.add_argument("--bundle", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_prepare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SnapshotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
