from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


_ID_RE = re.compile(r"^[a-z0-9_-]+$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_IDX_RE = re.compile(r"^(0|[1-9][0-9]*)$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

_CRED_NAMES = {"deepseek_api.txt"}
_CRED_SUFFIXES = (".pem", ".key")
_BANNED_COMPONENTS = {".git"}
_BACKEND_ENUM = {
    "program",
    "numeric_ik",
    "physics_simulation",
    "ik_ompl_simulation",
    "vla_simulation",
    "vla_hardware",
    "hardware_controller",
}
_HISTORICAL_KINDS = {"historical"}
_KINDS = {"historical", "planned", "executed"}
_OBS_CATEGORIES = {
    "preparation_blocked",
    "vla_execution_failed",
    "vla_execution_success",
    "program_test_passed",
    "program_test_failed",
    "physical_task_success",
    "physical_task_failed",
    "planning_check_passed",
    "historical_playback",
    "inconclusive",
    "measurement",
}
_OBS_JUDGEMENTS = {"success", "failure", "inconclusive", "not_executed"}
_ORIGINS = {"new_execution", "not_executed", "historical_replay", "unknown"}
_MECHANISMS = {
    "program",
    "numeric_ik",
    "ik_ompl",
    "vla",
    "hardware_controller",
    "unknown",
}
_PHASE_ENUM = {"start", "archive", "report"}
_DECISIONS = {"new", "reuse", "repeat", "retrospective"}
_REVIEW_STATUSES = {"pending", "verified"}
_CONCLUSIONS = {"approved", "reuse", "historical_import"}


def _err(errors: List[str], msg: str) -> None:
    errors.append(msg)


def _is_bool(v: Any) -> bool:
    return isinstance(v, bool)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        import math
        return math.isfinite(float(v))
    except (OverflowError, TypeError, ValueError):
        return False


def _nonblank(s: Any) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _loads_strict(data: bytes) -> Any:
    text = data.decode("utf-8-sig")
    rejects: List[str] = []

    def _reject_constant(name: str) -> None:
        rejects.append(name)
        return None

    def _parse_float(s: str) -> float:
        f = float(s)
        if f != f or f in (float("inf"), float("-inf")):
            raise ValueError("nonfinite number")
        return f

    def _parse_int(s: str) -> int:
        return int(s)

    def _pair_hook(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f"duplicate key: {k!r}")
            out[k] = v
        return out

    decoded = json.loads(
        text,
        object_pairs_hook=_pair_hook,
        parse_constant=_reject_constant,
        parse_float=_parse_float,
        parse_int=_parse_int,
    )
    if rejects:
        raise ValueError("nonfinite literal rejected")
    return decoded


def load_json(path: Any) -> Any:
    p = Path(path)
    try:
        if str(p).endswith(".gz"):
            with gzip.open(p, "rb") as f:
                data = f.read()
        else:
            data = p.read_bytes()
    except (OSError, EOFError):
        raise ValueError(f"cannot read JSON: {p}")
    try:
        return _loads_strict(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid JSON: {p}: {exc}")


def _normalize_rel(rel: Any) -> str:
    if not isinstance(rel, str):
        raise ValueError("path must be a string")
    if rel.strip() == "":
        raise ValueError("path must be nonblank")
    if "\x00" in rel:
        raise ValueError("NUL in path")
    s = rel.replace("\\", "/")
    if s.startswith("//"):
        raise ValueError("UNC path rejected")
    if re.match(r"^[A-Za-z]:", s):
        raise ValueError("drive letter rejected")
    if s.startswith("/"):
        raise ValueError("absolute path rejected")
    parts = s.split("/")
    cleaned: List[str] = []
    for part in parts:
        if part in ("", "."):
            if part == "" and cleaned:
                continue
            continue
        if part == "..":
            raise ValueError("'..' rejected")
        cf = part.casefold()
        if cf in _BANNED_COMPONENTS:
            raise ValueError(f"banned component: {part}")
        if cf in _CRED_NAMES or cf.endswith(_CRED_SUFFIXES):
            raise ValueError(f"credential-like path rejected: {part}")
        cleaned.append(part)
    if not cleaned:
        return "."
    return "/".join(cleaned)


def _check_components(resolved: Path, root: Path) -> None:
    for part in resolved.relative_to(root).parts if resolved != root else ():
        cf = part.casefold()
        if cf in _BANNED_COMPONENTS:
            raise ValueError(f"banned component in resolved path: {part}")
        if cf in _CRED_NAMES or cf.endswith(_CRED_SUFFIXES):
            raise ValueError(f"credential-like resolved component: {part}")


def safe_path(root: Any, rel: Any, allow_root: bool = False) -> Path:
    try:
        root_p = Path(root).resolve()
    except TypeError as exc:
        raise ValueError(f"cannot resolve root: {exc}")
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot resolve root: {exc}")
    if not root_p.exists() or not root_p.is_dir():
        raise ValueError("root is not a directory")
    norm = _normalize_rel(rel)
    if norm == ".":
        if not allow_root:
            raise ValueError("root itself not allowed")
        return root_p
    try:
        target = (root_p / norm).resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"cannot resolve path: {exc}")
    try:
        target.relative_to(root_p)
    except ValueError:
        raise ValueError("path escapes root")
    _check_components(target, root_p)
    return target


def _sha256_stream(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_factors(factors: Dict[str, Any]) -> str:
    payload = json.dumps(
        factors,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _norm_ws_rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _ledger_files(root: Path, ledger_dir: str) -> Tuple[Optional[Path], List[Path], Optional[str]]:
    try:
        ledger_root = safe_path(root, ledger_dir, allow_root=False)
    except ValueError as exc:
        return None, [], f"ledger dir unsafe: {exc}"
    if not ledger_root.exists() or not ledger_root.is_dir():
        return None, [], "ledger directory does not exist"
    try:
        entries = sorted(ledger_root.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        return None, [], f"cannot list ledger: {exc}"
    out: List[Path] = []
    for entry in entries:
        try:
            if entry.name.endswith(".json"):
                out.append(entry)
        except OSError:
            continue
    return ledger_root, out, None


def load_ledger(root: Any, ledger_dir: str = "experiment_review/ledger") -> List[Tuple[str, Dict[str, Any]]]:
    root_p = Path(root).resolve()
    ledger_root, files, err = _ledger_files(root_p, ledger_dir)
    if err is not None:
        raise ValueError(err)
    assert ledger_root is not None
    results: List[Tuple[str, Dict[str, Any]]] = []
    seen_ids: Dict[str, str] = {}
    for f in files:
        rel = _norm_ws_rel(root_p, f)
        try:
            safe_path(root_p, rel)
        except ValueError as exc:
            raise ValueError(f"ledger file unsafe {rel}: {exc}")
        try:
            raw = _read_record_bytes(f)
            decoded = _loads_strict(raw)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"ledger parse failure {rel}: {exc}")
        if not isinstance(decoded, dict):
            raise ValueError(f"ledger record not object: {rel}")
        rid = decoded.get("id")
        if isinstance(rid, str):
            if rid in seen_ids:
                raise ValueError(f"duplicate ledger id: {rid}")
            seen_ids[rid] = rel
        results.append((rel, decoded))
    return results


def _read_record_bytes(path: Path) -> bytes:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return f.read()
    with open(path, "rb") as f:
        return f.read()


def _require_keys(
    errors: List[str],
    obj: Any,
    required: Iterable[str],
    ctx: str,
) -> bool:
    if not isinstance(obj, dict):
        _err(errors, f"{ctx}: expected object")
        return False
    ok = True
    for key in required:
        if key not in obj:
            _err(errors, f"{ctx}: missing required key {key!r}")
            ok = False
    return ok


def _validate_sha(errors: List[str], value: Any, ctx: str) -> None:
    if not isinstance(value, str) or not _HEX64_RE.fullmatch(value):
        _err(errors, f"{ctx}: expected 64 lowercase hex sha256")


def _validate_string_list(
    errors: List[str],
    value: Any,
    ctx: str,
    nonempty: bool = False,
    allow_empty_strings: bool = False,
) -> None:
    if not isinstance(value, list):
        _err(errors, f"{ctx}: expected list")
        return
    if nonempty and not value:
        _err(errors, f"{ctx}: list must be nonempty")
        return
    for i, item in enumerate(value):
        if not isinstance(item, str):
            _err(errors, f"{ctx}[{i}]: expected string")
        elif not allow_empty_strings and not _nonblank(item):
            _err(errors, f"{ctx}[{i}]: must be nonblank")


def _validate_paths_exist(
    errors: List[str],
    root: Path,
    value: Any,
    ctx: str,
    allow_root: bool = False,
    require_file: bool = False,
    require_dir: bool = False,
    require_exists: bool = True,
) -> List[str]:
    norm_paths: List[str] = []
    if not isinstance(value, list):
        _err(errors, f"{ctx}: expected list")
        return norm_paths
    if not value:
        _err(errors, f"{ctx}: must be nonempty list")
        return norm_paths
    for i, item in enumerate(value):
        if not isinstance(item, str):
            _err(errors, f"{ctx}[{i}]: expected string path")
            continue
        try:
            p = safe_path(root, item, allow_root=allow_root)
        except ValueError as exc:
            _err(errors, f"{ctx}[{i}]: unsafe path: {exc}")
            continue
        if require_exists and not p.exists():
            _err(errors, f"{ctx}[{i}]: does not exist: {item}")
            continue
        if require_file and p.exists() and not p.is_file():
            _err(errors, f"{ctx}[{i}]: expected regular file: {item}")
            continue
        if require_dir and p.exists() and not p.is_dir():
            _err(errors, f"{ctx}[{i}]: expected directory: {item}")
            continue
        norm_paths.append(_norm_ws_rel(root, p))
    return norm_paths


def _validate_entrypoint(
    errors: List[str],
    root: Path,
    entry: Any,
    ctx: str,
    require_exists: bool,
) -> Optional[Path]:
    if entry is None:
        return None
    if not isinstance(entry, str):
        _err(errors, f"{ctx}: entrypoint must be string or null")
        return None
    if not entry.endswith(".py"):
        _err(errors, f"{ctx}: entrypoint must be .py")
    try:
        p = safe_path(root, entry, allow_root=False)
    except ValueError as exc:
        _err(errors, f"{ctx}: entrypoint unsafe: {exc}")
        return None
    if p.name.casefold() == "deepseek_codegen.py":
        _err(errors, f"{ctx}: deepseek_codegen.py is banned as entrypoint")
    if require_exists:
        if not p.exists():
            _err(errors, f"{ctx}: entrypoint does not exist: {entry}")
        elif not p.is_file():
            _err(errors, f"{ctx}: entrypoint not a regular file: {entry}")
    return p


def _validate_evidence_path_exists(
    errors: List[str],
    root: Path,
    rel: Any,
    ctx: str,
) -> Optional[Path]:
    if not isinstance(rel, str):
        _err(errors, f"{ctx}: path must be a string")
        return None
    try:
        p = safe_path(root, rel, allow_root=False)
    except ValueError as exc:
        _err(errors, f"{ctx}: unsafe path: {exc}")
        return None
    if not p.exists() or not p.is_file():
        _err(errors, f"{ctx}: evidence file missing or not regular: {rel}")
        return None
    return p


def _validate_pointer(errors: List[str], selector: Any, ctx: str) -> None:
    if not isinstance(selector, str):
        _err(errors, f"{ctx}: selector must be a string")
        return
    if selector == "":
        return
    if not selector.startswith("/"):
        _err(errors, f"{ctx}: selector must be RFC6901")
        return
    for token in selector.split("/")[1:]:
        i = 0
        while i < len(token):
            if token[i] == "~":
                if i + 1 >= len(token) or token[i + 1] not in ("0", "1"):
                    _err(errors, f"{ctx}: bad '~' escape in selector")
                    return
                i += 2
            else:
                i += 1


def _validate_selector_content(
    errors: List[str],
    target: Path,
    selector: str,
    ctx: str,
) -> None:
    try:
        raw = _read_record_bytes(target)
        decoded = _loads_strict(raw)
    except (OSError, EOFError, ValueError, UnicodeDecodeError) as exc:
        _err(errors, f"{ctx}: cannot decode selector target: {exc}")
        return
    if selector == "":
        return
    _walk_pointer(errors, decoded, selector, ctx)


def _walk_pointer(errors: List[str], doc: Any, selector: str, ctx: str) -> None:
    current = doc
    for token in selector.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not _IDX_RE.fullmatch(token):
                _err(errors, f"{ctx}: invalid list index in selector: {token!r}")
                return
            idx = int(token)
            if idx >= len(current):
                _err(errors, f"{ctx}: selector index out of range: {token}")
                return
            current = current[idx]
        elif isinstance(current, dict):
            if token not in current:
                _err(errors, f"{ctx}: selector key not found: {token!r}")
                return
            current = current[token]
        else:
            _err(errors, f"{ctx}: selector traverses noncontainer")
            return


def _validate_evidence(
    errors: List[str],
    root: Path,
    evidence: Any,
    ctx: str,
    require_selector_target: bool = True,
) -> Set[str]:
    ids: Set[str] = set()
    if not isinstance(evidence, list):
        _err(errors, f"{ctx}: evidence must be a list")
        return ids
    if not evidence:
        _err(errors, f"{ctx}: evidence must be nonempty")
        return ids
    for i, item in enumerate(evidence):
        ictx = f"{ctx}[{i}]"
        if not _require_keys(errors, item, ("id", "path", "sha256"), ictx):
            continue
        eid = item.get("id")
        if not _nonblank(eid):
            _err(errors, f"{ictx}.id: must be nonblank string")
        else:
            if eid in ids:
                _err(errors, f"{ictx}.id: duplicate within record: {eid!r}")
            ids.add(eid)
        p = _validate_evidence_path_exists(errors, root, item.get("path"), f"{ictx}.path")
        _validate_sha(errors, item.get("sha256"), f"{ictx}.sha256")
        if p is not None:
            try:
                actual = _sha256_stream(p)
            except OSError as exc:
                _err(errors, f"{ictx}.path: cannot hash: {exc}")
                actual = None
            if actual is not None and isinstance(item.get("sha256"), str):
                if actual != item["sha256"]:
                    _err(errors, f"{ictx}.sha256: evidence hash mismatch for {item.get('path')}")
        sel = item.get("selector")
        if "selector" in item:
            if not isinstance(sel, str):
                _err(errors, f"{ictx}.selector: must be a string when present")
            else:
                _validate_pointer(errors, sel, f"{ictx}.selector")
                if p is not None and require_selector_target:
                    _validate_selector_content(errors, p, sel, f"{ictx}.selector")
    return ids


def _validate_inputs(
    errors: List[str],
    root: Path,
    inputs: Any,
    ctx: str,
    require_match: bool,
) -> None:
    if not isinstance(inputs, list):
        _err(errors, f"{ctx}: inputs must be a list")
        return
    if not isinstance(inputs, list):
        return
    for i, item in enumerate(inputs):
        ictx = f"{ctx}[{i}]"
        if not _require_keys(errors, item, ("path", "sha256"), ictx):
            continue
        p = _validate_evidence_path_exists(errors, root, item.get("path"), f"{ictx}.path")
        _validate_sha(errors, item.get("sha256"), f"{ictx}.sha256")
        if p is not None and require_match and isinstance(item.get("sha256"), str):
            try:
                actual = _sha256_stream(p)
            except OSError as exc:
                _err(errors, f"{ictx}.path: cannot hash: {exc}")
                actual = None
            if actual is not None and actual != item["sha256"]:
                _err(errors, f"{ictx}.sha256: input hash mismatch for {item.get('path')}")


def _validate_source_hashes(
    errors: List[str],
    root: Path,
    source_hashes: Any,
    ctx: str,
    require_match: bool,
) -> None:
    if not isinstance(source_hashes, dict):
        _err(errors, f"{ctx}: source_hashes must be object")
        return
    if not source_hashes:
        _err(errors, f"{ctx}: source_hashes must be nonempty")
        return
    for rel, sha in source_hashes.items():
        if not isinstance(rel, str):
            _err(errors, f"{ctx}: key must be string path")
            continue
        try:
            p = safe_path(root, rel, allow_root=False)
        except ValueError as exc:
            _err(errors, f"{ctx}[{rel}]: unsafe path: {exc}")
            continue
        _validate_sha(errors, sha, f"{ctx}[{rel}]")
        if require_match:
            if not p.exists() or not p.is_file():
                _err(errors, f"{ctx}[{rel}]: source file missing")
                continue
            try:
                actual = _sha256_stream(p)
            except OSError as exc:
                _err(errors, f"{ctx}[{rel}]: cannot hash: {exc}")
                continue
            if isinstance(sha, str) and actual != sha:
                _err(errors, f"{ctx}[{rel}]: source hash mismatch")


def _validate_configuration(
    errors: List[str],
    root: Path,
    configuration: Any,
    kind: Any,
    phase: str,
    ctx: str,
) -> Tuple[Optional[str], bool]:
    if not _require_keys(errors, configuration, ("completeness", "factors", "unknown_fields"), ctx):
        return None, False
    cfg = configuration
    completeness = cfg.get("completeness")
    if completeness not in ("complete", "partial"):
        _err(errors, f"{ctx}.completeness: must be complete|partial")
        return None, False
    unknown = cfg.get("unknown_fields")
    _validate_string_list(errors, unknown, f"{ctx}.unknown_fields", nonempty=False)
    unknown_list = unknown if isinstance(unknown, list) else []
    unknown_ok = all(_nonblank(u) for u in unknown_list if isinstance(u, str))
    factors = cfg.get("factors")
    if not isinstance(factors, dict):
        _err(errors, f"{ctx}.factors: must be object")
        return None, False
    if not factors:
        _err(errors, f"{ctx}.factors: must be nonempty")
    partial = completeness == "partial"
    if partial:
        if kind != "historical":
            _err(errors, f"{ctx}.completeness: partial only allowed for historical")
        if len(unknown_list) < 1:
            _err(errors, f"{ctx}.unknown_fields: partial requires >=1 entry")
    else:
        if len(unknown_list) != 0:
            _err(errors, f"{ctx}.unknown_fields: complete requires empty unknown_fields")
        _validate_complete_factors(errors, root, factors, phase, f"{ctx}.factors")
    fp: Optional[str] = None
    if isinstance(factors, dict) and factors:
        try:
            fp = _canonical_factors(factors)
        except (TypeError, ValueError):
            fp = None
    if unknown_ok and partial:
        return fp, True
    return fp, not partial


def _validate_complete_factors(
    errors: List[str],
    root: Path,
    factors: Dict[str, Any],
    phase: str,
    ctx: str,
) -> None:
    required = ("task", "method", "backend", "layout", "inputs", "random_seed", "parameters", "source_hashes")
    _require_keys(errors, factors, required, ctx)
    task = factors.get("task")
    if not isinstance(task, dict):
        _err(errors, f"{ctx}.task: must be object")
    method = factors.get("method")
    if not _nonblank(method):
        _err(errors, f"{ctx}.method: must be nonblank string")
    backend = factors.get("backend")
    if backend not in _BACKEND_ENUM:
        _err(errors, f"{ctx}.backend: invalid enum value")
    layout = factors.get("layout")
    if not isinstance(layout, dict):
        _err(errors, f"{ctx}.layout: must be object")
    inputs = factors.get("inputs")
    _validate_inputs(errors, root, inputs, f"{ctx}.inputs", require_match=(phase == "start"))
    seed = factors.get("random_seed")
    if not (seed is None or _is_int(seed)):
        _err(errors, f"{ctx}.random_seed: must be int or null")
    params = factors.get("parameters")
    if not isinstance(params, dict):
        _err(errors, f"{ctx}.parameters: must be object")
    src = factors.get("source_hashes")
    _validate_source_hashes(errors, root, src, f"{ctx}.source_hashes", require_match=(phase == "start"))


def _validate_history_review(
    errors: List[str],
    root: Path,
    history: Any,
    kind: Any,
    ledger_ids: Set[str],
    decision: Any,
    phase: str,
    seen_fp: Optional[str],
    duplicate_ids_ledger: List[str],
    ctx: str,
) -> None:
    required = (
        "searched",
        "related_ids",
        "known",
        "unknown",
        "decision",
        "rationale",
        "changed_conditions",
        "repeat_of",
        "repeat_purpose",
    )
    if not _require_keys(errors, history, required, ctx):
        return
    hr = history
    _validate_paths_exist(errors, root, hr.get("searched"), f"{ctx}.searched", allow_root=True)
    related = hr.get("related_ids")
    _validate_string_list(errors, related, f"{ctx}.related_ids")
    _validate_string_list(errors, hr.get("known"), f"{ctx}.known", nonempty=True)
    _validate_string_list(errors, hr.get("unknown"), f"{ctx}.unknown", nonempty=True)
    dec = hr.get("decision")
    if dec not in _DECISIONS:
        _err(errors, f"{ctx}.decision: invalid enum")
    if kind == "historical" and dec != "retrospective":
        _err(errors, f"{ctx}.decision: historical must be retrospective")
    if dec == "retrospective" and kind != "historical":
        _err(errors, f"{ctx}.decision: retrospective only allowed for historical")
    rationale = hr.get("rationale")
    if not _nonblank(rationale):
        _err(errors, f"{ctx}.rationale: must be nonblank")
    changes = hr.get("changed_conditions")
    if not isinstance(changes, list):
        _err(errors, f"{ctx}.changed_conditions: must be list")
        changes = []
    for i, c in enumerate(changes):
        cctx = f"{ctx}.changed_conditions[{i}]"
        if not _require_keys(errors, c, ("field", "before", "after"), cctx):
            continue
        if not _nonblank(c.get("field")):
            _err(errors, f"{cctx}.field: must be nonblank")
    repeat_of = hr.get("repeat_of")
    _validate_string_list(errors, repeat_of, f"{ctx}.repeat_of")
    repeat_purpose = hr.get("repeat_purpose")
    if not isinstance(repeat_purpose, str):
        _err(errors, f"{ctx}.repeat_purpose: must be string")
    if dec == "repeat":
        if not isinstance(repeat_of, list) or not repeat_of:
            _err(errors, f"{ctx}.repeat_of: repeat requires nonempty")
        if not _nonblank(repeat_purpose):
            _err(errors, f"{ctx}.repeat_purpose: repeat requires nonblank purpose")
    if isinstance(related, list):
        for i, rid in enumerate(related):
            if isinstance(rid, str) and rid not in ledger_ids:
                _err(errors, f"{ctx}.related_ids[{i}]: unknown ledger id: {rid}")
    if isinstance(repeat_of, list):
        for i, rid in enumerate(repeat_of):
            if isinstance(rid, str) and rid not in ledger_ids:
                _err(errors, f"{ctx}.repeat_of[{i}]: unknown ledger id: {rid}")
    related_list = related if isinstance(related, list) else []
    if dec == "new" and related_list:
        has_change = False
        for c in changes:
            if isinstance(c, dict) and "before" in c and "after" in c:
                if c["before"] != c["after"]:
                    has_change = True
                    break
        if not has_change:
            _err(errors, f"{ctx}: new with related_ids requires >=1 changed_condition before!=after")
    if phase == "start" and dec in ("reuse", "retrospective"):
        _err(errors, f"{ctx}.decision: {dec} forbidden in start phase")
    if seen_fp is not None and dec in ("new", "repeat"):
        if dec == "new":
            _err(errors, f"{ctx}: new decision conflicts with existing complete config duplicates: {duplicate_ids_ledger}")
        else:
            if duplicate_ids_ledger:
                missing = [rid for rid in duplicate_ids_ledger if rid not in related_list]
                if missing:
                    _err(errors, f"{ctx}.related_ids: repeat must include duplicates: {missing}")
                if not isinstance(repeat_of, list) or not set(duplicate_ids_ledger).issubset(
                    {r for r in repeat_of if isinstance(r, str)}
                ):
                    _err(errors, f"{ctx}.repeat_of: must include duplicate ids: {duplicate_ids_ledger}")
                if not _nonblank(repeat_purpose):
                    _err(errors, f"{ctx}.repeat_purpose: repeat requires nonblank purpose")
    elif seen_fp is not None and dec not in ("new", "repeat"):
        pass


def _validate_judgement(
    errors: List[str],
    root: Path,
    judgement: Any,
    kind: Any,
    phase: str,
    ctx: str,
) -> None:
    required = (
        "unit",
        "success",
        "failure",
        "inconclusive",
        "planned_trials",
        "budget",
        "stop_conditions",
        "preregistration",
    )
    if not _require_keys(errors, judgement, required, ctx):
        return
    j = judgement
    for key in ("unit", "success", "failure", "inconclusive"):
        if not _nonblank(j.get(key)):
            _err(errors, f"{ctx}.{key}: must be nonblank")
    historical = kind == "historical"
    pt = j.get("planned_trials")
    if pt is None:
        if not historical:
            _err(errors, f"{ctx}.planned_trials: null only permitted for historical")
    else:
        if not _is_int(pt) or pt <= 0:
            _err(errors, f"{ctx}.planned_trials: must be positive int or null")
    budget = j.get("budget")
    if not isinstance(budget, dict):
        _err(errors, f"{ctx}.budget: must be object")
        budget = {}
    if not _require_keys(errors, budget, ("wall_time_seconds", "max_attempts"), f"{ctx}.budget"):
        budget = dict(budget) if isinstance(budget, dict) else {}
    wall = budget.get("wall_time_seconds")
    if wall is None:
        if not historical:
            _err(errors, f"{ctx}.budget.wall_time_seconds: null only for historical")
    else:
        if not _is_num(wall) or wall <= 0:
            _err(errors, f"{ctx}.budget.wall_time_seconds: must be positive finite or null")
    attempts = budget.get("max_attempts")
    if attempts is None:
        if not historical:
            _err(errors, f"{ctx}.budget.max_attempts: null only for historical")
    else:
        if not _is_int(attempts) or attempts <= 0:
            _err(errors, f"{ctx}.budget.max_attempts: must be positive int or null")
    if (
        isinstance(attempts, int)
        and not isinstance(attempts, bool)
        and isinstance(pt, int)
        and not isinstance(pt, bool)
        and attempts < pt
    ):
        _err(errors, f"{ctx}.budget.max_attempts: must be >= planned_trials")
    _validate_string_list(errors, j.get("stop_conditions"), f"{ctx}.stop_conditions", nonempty=True)
    prereg = j.get("preregistration")
    if prereg not in ("declared_before_execution", "retrospective_not_recorded"):
        _err(errors, f"{ctx}.preregistration: invalid enum")
    if phase == "start" and prereg != "declared_before_execution":
        _err(errors, f"{ctx}.preregistration: start requires declared_before_execution")


def _validate_execution(
    errors: List[str],
    root: Path,
    execution: Any,
    kind: Any,
    phase: str,
    ledger_root: Optional[Path],
    ctx: str,
) -> Optional[Path]:
    if not _require_keys(errors, execution, ("entrypoint", "args", "output_dir"), ctx):
        return None
    ex = execution
    require_exists = phase == "start" and kind != "historical"
    entry = ex.get("entrypoint")
    if entry is None and kind != "historical":
        _err(errors, f"{ctx}.entrypoint: null only allowed for historical")
    p = _validate_entrypoint(errors, root, entry, f"{ctx}.entrypoint", require_exists=require_exists)
    _validate_string_list(errors, ex.get("args"), f"{ctx}.args", allow_empty_strings=True)
    if phase == "start" and kind == "planned":
        pass
    out_dir = ex.get("output_dir")
    if out_dir is None and kind != "historical":
        _err(errors, f"{ctx}.output_dir: null only allowed for historical")
    if out_dir is not None:
        try:
            op = safe_path(root, out_dir, allow_root=False)
        except ValueError as exc:
            _err(errors, f"{ctx}.output_dir: unsafe: {exc}")
            op = None
        if op is not None:
            try:
                rel_check = op.relative_to(root)
            except ValueError:
                _err(errors, f"{ctx}.output_dir: escapes root")
            banned: List[str] = []
            if ledger_root is not None:
                try:
                    op.relative_to(ledger_root)
                    banned.append("ledger dir")
                except ValueError:
                    pass
            runs = (root / "experiment_review" / "runs").resolve()
            try:
                op.relative_to(runs)
                banned.append("experiment_review/runs")
            except ValueError:
                pass
            if banned:
                _err(errors, f"{ctx}.output_dir: not allowed under {banned}")
            if phase == "start" and op.exists():
                _err(errors, f"{ctx}.output_dir: already exists in start phase")
    return p


def _frontload_refs(errors: List[str], value: Any, ctx: str) -> Set[str]:
    s: Set[str] = set()
    if not isinstance(value, list):
        _err(errors, f"{ctx}: expected list")
        return s
    for i, item in enumerate(value):
        if not isinstance(item, str):
            _err(errors, f"{ctx}[{i}]: expected string")
            continue
        s.add(item)
    return s


def _validate_observation_execution(
    errors: List[str],
    root: Path,
    obs_exec: Any,
    category: Any,
    backend: Any,
    evidence_ids: Set[str],
    ctx: str,
) -> None:
    required = (
        "origin",
        "mechanism",
        "attempted",
        "confirmed",
        "physics_steps",
        "simulation_time_s",
        "policy_invocations",
    )
    if not _require_keys(errors, obs_exec, required, ctx):
        return
    oe = obs_exec
    origin = oe.get("origin")
    if origin not in _ORIGINS:
        _err(errors, f"{ctx}.origin: invalid enum")
    mech = oe.get("mechanism")
    if mech not in _MECHANISMS:
        _err(errors, f"{ctx}.mechanism: invalid enum")
    attempted = oe.get("attempted")
    confirmed = oe.get("confirmed")
    if not _is_bool(confirmed):
        _err(errors, f"{ctx}.confirmed: must be bool")
    _neutral_unknown = (
        origin == "unknown"
        and category in ("measurement", "inconclusive")
        and confirmed is False
    )
    if _neutral_unknown:
        if attempted is not None and not _is_bool(attempted):
            _err(errors, f"{ctx}.attempted: must be bool or null")
    else:
        if not _is_bool(attempted):
            _err(errors, f"{ctx}.attempted: must be bool")
    if origin == "unknown" and not _neutral_unknown:
        _err(errors, f"{ctx}.origin: unknown origin not permitted for this category")
    if origin == "unknown" and confirmed is True:
        _err(errors, f"{ctx}.confirmed: unknown origin cannot certify")
    steps = oe.get("physics_steps")
    if steps is not None and (not _is_int(steps) or steps < 0):
        _err(errors, f"{ctx}.physics_steps: must be nonnegative int or null")
    stime = oe.get("simulation_time_s")
    if stime is not None:
        if not _is_num(stime) or stime != stime or stime < 0 or stime in (float("inf"), float("-inf")):
            _err(errors, f"{ctx}.simulation_time_s: must be nonnegative finite or null")
    policy = oe.get("policy_invocations")
    if not _is_int(policy) or policy < 0:
        _err(errors, f"{ctx}.policy_invocations: must be nonnegative int")
    if category == "preparation_blocked":
        if attempted is True:
            _err(errors, f"{ctx}.attempted: blocked requires False")
        if confirmed is True:
            _err(errors, f"{ctx}.confirmed: blocked requires False")
        if origin != "not_executed":
            _err(errors, f"{ctx}.origin: blocked requires not_executed")
        if _is_int(policy) and policy != 0:
            _err(errors, f"{ctx}.policy_invocations: blocked requires 0")
        if not (steps is None or (isinstance(steps, int) and steps == 0)):
            _err(errors, f"{ctx}.physics_steps: blocked requires 0 or null")
    if confirmed is True and origin == "not_executed":
        _err(errors, f"{ctx}: confirmed true incompatible with not_executed origin")
    if _is_int(policy) and policy < 0:
        pass


def _validate_observation(
    errors: List[str],
    root: Path,
    obs_list: Any,
    kind: Any,
    phase: str,
    backend: Any,
    evidence_ids: Set[str],
    ctx: str,
) -> None:
    if not isinstance(obs_list, list):
        _err(errors, f"{ctx}: observations must be a list")
        return
    if kind == "planned" and phase == "start" and obs_list:
        _err(errors, f"{ctx}: planned start requires empty observations")
    for i, obs in enumerate(obs_list):
        octx = f"{ctx}[{i}]"
        required = (
            "category",
            "judgement",
            "scope",
            "count",
            "total",
            "evidence_ids",
            "interpretation",
            "execution",
        )
        if not _require_keys(errors, obs, required, octx):
            continue
        category = obs.get("category")
        if category not in _OBS_CATEGORIES:
            _err(errors, f"{octx}.category: invalid enum")
        jud = obs.get("judgement")
        if jud not in _OBS_JUDGEMENTS:
            _err(errors, f"{octx}.judgement: invalid enum")
        if not _nonblank(obs.get("scope")):
            _err(errors, f"{octx}.scope: must be nonblank")
        count = obs.get("count")
        total = obs.get("total")
        _scalar_measurement_nulls = (
            category == "measurement" and count is None and total is None
        )
        if _scalar_measurement_nulls:
            pass
        else:
            if not _is_int(count) or count < 0:
                _err(errors, f"{octx}.count: must be nonnegative int")
            if not _is_int(total) or total <= 0:
                _err(errors, f"{octx}.total: must be positive int")
            if _is_int(count) and _is_int(total) and count > total:
                _err(errors, f"{octx}.count: must be <= total")
        ev_ids = obs.get("evidence_ids")
        _validate_string_list(errors, ev_ids, f"{octx}.evidence_ids", nonempty=True)
        if isinstance(ev_ids, list):
            for j, eid in enumerate(ev_ids):
                if isinstance(eid, str) and eid not in evidence_ids:
                    _err(errors, f"{octx}.evidence_ids[{j}]: unknown evidence id: {eid}")
        if not _nonblank(obs.get("interpretation")):
            _err(errors, f"{octx}.interpretation: must be nonblank")
        exec_obj = obs.get("execution")
        _validate_observation_execution(errors, root, exec_obj, category, backend, evidence_ids, f"{octx}.execution")
        if isinstance(exec_obj, dict):
            origin = exec_obj.get("origin")
            mech = exec_obj.get("mechanism")
            if origin == "historical_replay":
                if category != "historical_playback":
                    _err(errors, f"{octx}: historical_replay origin only allowed for historical_playback")
            if category == "historical_playback" and origin != "historical_replay":
                _err(errors, f"{octx}: historical_playback category requires historical_replay origin")
            if category in ("program_test_passed", "program_test_failed"):
                if mech != "program":
                    _err(errors, f"{octx}: program category requires program mechanism")
                if origin != "new_execution":
                    _err(errors, f"{octx}: program category requires new_execution")
                if exec_obj.get("attempted") is not True:
                    _err(errors, f"{octx}: program category requires attempted true")
                if exec_obj.get("confirmed") is not True:
                    _err(errors, f"{octx}: program category requires confirmed true")
            if category in ("vla_execution_success", "vla_execution_failed"):
                if origin != "new_execution":
                    _err(errors, f"{octx}: vla category requires new_execution")
                if mech != "vla":
                    _err(errors, f"{octx}: vla category requires vla mechanism")
                if exec_obj.get("attempted") is not True:
                    _err(errors, f"{octx}: vla category requires attempted true")
                if exec_obj.get("confirmed") is not True:
                    _err(errors, f"{octx}: vla category requires confirmed true")
                if backend not in ("vla_simulation", "vla_hardware"):
                    _err(errors, f"{octx}: vla category requires vla backend")
                policy = exec_obj.get("policy_invocations")
                if _is_int(policy) and policy <= 0:
                    _err(errors, f"{octx}: vla category requires policy_invocations>0")
                for need in ("checkpoint_evidence_id", "action_evidence_id"):
                    if need not in exec_obj:
                        _err(errors, f"{octx}.execution: missing required key {need!r}")
                    else:
                        ref = exec_obj.get(need)
                        if not isinstance(ref, str) or ref not in evidence_ids:
                            _err(errors, f"{octx}.execution.{need}: must reference known evidence id")
            if category in ("physical_task_success", "physical_task_failed"):
                if origin != "new_execution":
                    _err(errors, f"{octx}: physical category requires new_execution")
                if exec_obj.get("attempted") is not True:
                    _err(errors, f"{octx}: physical category requires attempted true")
                if exec_obj.get("confirmed") is not True:
                    _err(errors, f"{octx}: physical category requires confirmed true")
                if backend not in ("ik_ompl_simulation", "vla_simulation", "vla_hardware", "hardware_controller", "physics_simulation"):
                    _err(errors, f"{octx}: physical category requires permitted backend")
                if mech not in ("ik_ompl", "vla", "hardware_controller", "numeric_ik"):
                    _err(errors, f"{octx}: physical category requires permitted mechanism")
                if mech == "numeric_ik" and backend != "physics_simulation":
                    _err(errors, f"{octx}: numeric_ik mechanism requires physics_simulation backend")
                if backend in ("ik_ompl_simulation", "vla_simulation", "physics_simulation"):
                    if not (_is_int(exec_obj.get("physics_steps")) and exec_obj.get("physics_steps") > 0):
                        _err(errors, f"{octx}: simulation requires physics_steps>0")
                    st = exec_obj.get("simulation_time_s")
                    if not (_is_num(st) and st > 0):
                        _err(errors, f"{octx}: simulation requires simulation_time_s>0")
                if backend in ("vla_hardware", "hardware_controller"):
                    ref = exec_obj.get("hardware_evidence_id")
                    if not isinstance(ref, str) or ref not in evidence_ids:
                        _err(errors, f"{octx}.execution.hardware_evidence_id: must reference known evidence id")
            if category in (
                "vla_execution_success",
                "physical_task_success",
                "program_test_passed",
                "planning_check_passed",
            ):
                if jud != "success":
                    _err(errors, f"{octx}.judgement: success category requires success")
            if category in ("vla_execution_failed", "program_test_failed", "physical_task_failed"):
                if jud != "failure":
                    _err(errors, f"{octx}.judgement: failure category requires failure")
            if category == "preparation_blocked":
                if jud not in ("not_executed", "failure"):
                    _err(errors, f"{octx}.judgement: blocked requires not_executed or failure")
            if category == "historical_playback":
                if jud != "not_executed":
                    _err(errors, f"{octx}.judgement: historical_playback requires not_executed")
            if category == "inconclusive":
                if jud != "inconclusive":
                    _err(errors, f"{octx}.judgement: inconclusive category requires inconclusive")
            if category == "planning_check_passed":
                if jud not in ("not_executed", "success"):
                    pass


def _validate_review(
    errors: List[str],
    root: Path,
    review: Any,
    phase: str,
    evidence_ids: Set[str],
    observations: Any,
    claims_holder: List[Any],
    ctx: str,
) -> Optional[Set[str]]:
    required = ("status", "reviewer", "independent", "checked_evidence_ids", "claims", "limitations")
    if not _require_keys(errors, review, required, ctx):
        return None
    rv = review
    status = rv.get("status")
    if status not in _REVIEW_STATUSES:
        _err(errors, f"{ctx}.status: invalid enum")
    reviewer = rv.get("reviewer")
    if status == "pending":
        if reviewer not in (None, "GPT"):
            _err(errors, f"{ctx}.reviewer: must be GPT or null while pending")
    else:
        if reviewer != "GPT":
            _err(errors, f"{ctx}.reviewer: must be 'GPT' when verified")
    indep = rv.get("independent")
    if not _is_bool(indep):
        _err(errors, f"{ctx}.independent: must be bool")
    # Explicit user prohibition on delegation can require a same-session raw
    # audit. Preserve independent=False; do not manufacture a second reviewer.
    single = rv.get("single_reviewer_exception")
    single_allowed = (
        indep is False and isinstance(single, dict)
        and single.get("mode") == "user_requested_same_session_raw_audit"
        and _nonblank(single.get("user_instruction_quote"))
        and _nonblank(single.get("limitation"))
        and isinstance(rv.get("checked_evidence_ids"), list)
        and isinstance(rv.get("limitations"), list)
        and single.get("raw_audit_evidence_id") in evidence_ids
        and single.get("raw_audit_evidence_id") in (rv.get("checked_evidence_ids") or [])
        and single.get("limitation") in (rv.get("limitations") or [])
    )
    if status == "verified" and indep is not True and not single_allowed:
        _err(errors, f"{ctx}.independent: verified requires true")
    checked = rv.get("checked_evidence_ids")
    _validate_string_list(errors, checked, f"{ctx}.checked_evidence_ids")
    claims = rv.get("claims")
    if not isinstance(claims, list):
        _err(errors, f"{ctx}.claims: must be a list")
        claims = []
    if status == "verified" and not claims:
        _err(errors, f"{ctx}.claims: verified requires nonempty")
    checked_set: Set[str] = set()
    if isinstance(checked, list):
        for i, eid in enumerate(checked):
            if isinstance(eid, str):
                if eid not in evidence_ids:
                    _err(errors, f"{ctx}.checked_evidence_ids[{i}]: unknown evidence id")
                checked_set.add(eid)
    for i, claim in enumerate(claims):
        cctx = f"{ctx}.claims[{i}]"
        if not _require_keys(errors, claim, ("text", "scope", "evidence_ids"), cctx):
            continue
        if not _nonblank(claim.get("text")):
            _err(errors, f"{cctx}.text: must be nonblank")
        if not _nonblank(claim.get("scope")):
            _err(errors, f"{cctx}.scope: must be nonblank")
        ceids = claim.get("evidence_ids")
        _validate_string_list(errors, ceids, f"{cctx}.evidence_ids", nonempty=True)
        if isinstance(ceids, list):
            for j, eid in enumerate(ceids):
                if isinstance(eid, str) and eid not in evidence_ids:
                    _err(errors, f"{cctx}.evidence_ids[{j}]: unknown evidence id: {eid}")
    _validate_string_list(errors, rv.get("limitations"), f"{ctx}.limitations", nonempty=True)
    if status == "verified" and not isinstance(checked, list):
        pass
    if status == "verified" and isinstance(checked, list) and not checked:
        _err(errors, f"{ctx}.checked_evidence_ids: verified requires nonempty")
    if status == "verified" and not isinstance(observations, list):
        pass
    if status == "verified" and isinstance(observations, list) and not observations:
        _err(errors, f"{ctx}: verified requires nonempty observations")
    return checked_set


def _validate_record(
    errors: List[str],
    root: Path,
    record: Dict[str, Any],
    phase: str,
    ledger_ids: Set[str],
    ledger_root: Optional[Path],
    seen_fp: Optional[str],
    duplicate_ids_ledger: List[str],
    require_deep: bool,
) -> None:
    ctx = "record"
    required = (
        "schema_version",
        "id",
        "title",
        "tags",
        "kind",
        "configuration",
        "evidence",
        "history_review",
        "judgement",
        "execution",
        "observations",
        "design_review",
        "review",
    )
    if not _require_keys(errors, record, required, ctx):
        return
    sv = record.get("schema_version")
    if not _is_int(sv) or sv != 1:
        _err(errors, f"{ctx}.schema_version: must be integer 1")
    rid = record.get("id")
    if not isinstance(rid, str) or not _ID_RE.fullmatch(rid):
        _err(errors, f"{ctx}.id: must match ^[a-z0-9_-]+$")
    title = record.get("title")
    if not _nonblank(title):
        _err(errors, f"{ctx}.title: must be nonblank")
    tags = record.get("tags")
    if not isinstance(tags, list) or not tags:
        _err(errors, f"{ctx}.tags: must be nonempty list")
    else:
        for i, t in enumerate(tags):
            if not _nonblank(t):
                _err(errors, f"{ctx}.tags[{i}]: must be nonblank string")
    kind = record.get("kind")
    if kind not in _KINDS:
        _err(errors, f"{ctx}.kind: invalid enum")
    configuration = record.get("configuration")
    fp, complete = _validate_configuration(errors, root, configuration, kind, phase, f"{ctx}.configuration")
    evidence_ids = _validate_evidence(errors, root, record.get("evidence"), f"{ctx}.evidence")
    _validate_history_review(
        errors,
        root,
        record.get("history_review"),
        kind,
        ledger_ids,
        record.get("history_review", {}).get("decision") if isinstance(record.get("history_review"), dict) else None,
        phase,
        seen_fp,
        duplicate_ids_ledger,
        f"{ctx}.history_review",
    )
    _validate_judgement(errors, root, record.get("judgement"), kind, phase, f"{ctx}.judgement")
    _validate_execution(errors, root, record.get("execution"), kind, phase, ledger_root, f"{ctx}.execution")
    backend = None
    if isinstance(configuration, dict) and isinstance(configuration.get("factors"), dict):
        backend = configuration["factors"].get("backend")
    _validate_observation(
        errors,
        root,
        record.get("observations"),
        kind,
        phase,
        backend,
        evidence_ids,
        f"{ctx}.observations",
    )
    dr = record.get("design_review")
    if _require_keys(errors, dr, ("reviewer", "checked_evidence_ids", "conclusion", "notes"), f"{ctx}.design_review"):
        if dr.get("reviewer") != "GPT":
            _err(errors, f"{ctx}.design_review.reviewer: must be 'GPT'")
        dchecked = dr.get("checked_evidence_ids")
        _validate_string_list(errors, dchecked, f"{ctx}.design_review.checked_evidence_ids")
        if isinstance(dchecked, list):
            for i, eid in enumerate(dchecked):
                if isinstance(eid, str) and eid not in evidence_ids:
                    _err(errors, f"{ctx}.design_review.checked_evidence_ids[{i}]: unknown evidence id")
        conc = dr.get("conclusion")
        if conc not in _CONCLUSIONS:
            _err(errors, f"{ctx}.design_review.conclusion: invalid enum")
        if conc == "approved":
            if not isinstance(dchecked, list) or not dchecked:
                _err(errors, f"{ctx}.design_review.checked_evidence_ids: approved requires >=1")
        if conc == "historical_import" and kind != "historical":
            _err(errors, f"{ctx}.design_review.conclusion: historical_import only for historical")
        if not _nonblank(dr.get("notes")):
            _err(errors, f"{ctx}.design_review.notes: must be nonblank")
    review = record.get("review")
    checked_set = _validate_review(
        errors,
        root,
        review,
        phase,
        evidence_ids,
        record.get("observations"),
        [],
        f"{ctx}.review",
    )
    if phase == "start" and isinstance(dr, dict):
        if dr.get("conclusion") != "approved":
            _err(errors, f"{ctx}.design_review.conclusion: start requires approved")
    if phase == "report" and isinstance(review, dict):
        if review.get("status") != "verified":
            _err(errors, f"{ctx}.review.status: report requires verified")
    review_verified = isinstance(review, dict) and review.get("status") == "verified"
    if review_verified and checked_set is not None and isinstance(record.get("observations"), list):
        for i, obs in enumerate(record["observations"]):
            if isinstance(obs, dict):
                evs = obs.get("evidence_ids")
                if isinstance(evs, list):
                    for eid in evs:
                        if isinstance(eid, str) and eid not in checked_set:
                            _err(errors, f"{ctx}.observations[{i}]: evidence id {eid!r} not in review.checked_evidence_ids")
    if review_verified and checked_set is not None and isinstance(review, dict):
        claims = review.get("claims")
        if isinstance(claims, list):
            for i, claim in enumerate(claims):
                if isinstance(claim, dict):
                    evs = claim.get("evidence_ids")
                    if isinstance(evs, list):
                        for eid in evs:
                            if isinstance(eid, str) and eid not in checked_set:
                                _err(errors, f"{ctx}.review.claims[{i}]: evidence id {eid!r} not in review.checked_evidence_ids")


def check(
    root: Any,
    record_path: str,
    phase: str = "archive",
    ledger_dir: str = "experiment_review/ledger",
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    result: Dict[str, Any] = {
        "ok": False,
        "errors": errors,
        "warnings": warnings,
        "fingerprint": None,
        "duplicates": [],
        "record_sha256": None,
        "record": None,
    }
    if not isinstance(phase, str) or phase not in _PHASE_ENUM:
        _err(errors, f"invalid phase: {phase!r}")
        return result
    try:
        root_p = Path(root).resolve()
    except TypeError as exc:
        _err(errors, f"cannot resolve root: {exc}")
        return result
    except (OSError, RuntimeError, ValueError) as exc:
        _err(errors, f"cannot resolve root: {exc}")
        return result
    if not root_p.exists() or not root_p.is_dir():
        _err(errors, "root is not a directory")
        return result
    if not isinstance(record_path, str):
        _err(errors, "record_path must be string")
        return result
    if not (record_path.endswith(".json") or record_path.endswith(".json.gz")):
        _err(errors, "record filename must end with .json or .json.gz")
        return result
    try:
        target = safe_path(root_p, record_path, allow_root=False)
    except ValueError as exc:
        _err(errors, f"record path unsafe: {exc}")
        return result
    if not target.exists() or not target.is_file():
        _err(errors, "record file does not exist or is not a regular file")
        return result
    try:
        raw = target.read_bytes()
    except (OSError, EOFError) as exc:
        _err(errors, f"cannot read record: {exc}")
        return result
    record_sha = _sha256_bytes(raw)
    result["record_sha256"] = record_sha
    try:
        if target.name.endswith(".gz"):
            record = _loads_strict(gzip.decompress(raw))
        else:
            record = _loads_strict(raw)
    except (ValueError, UnicodeDecodeError, EOFError, OSError) as exc:
        _err(errors, f"record JSON invalid: {exc}")
        return result
    if not isinstance(record, dict):
        _err(errors, "record must be a JSON object")
        return result
    result["record"] = record
    target_rel = _norm_ws_rel(root_p, target)

    try:
        ledger_root, ledger_files, lerr = _ledger_files(root_p, ledger_dir)
    except (OSError, ValueError) as exc:
        _err(errors, f"ledger dir error: {exc}")
        return result
    if lerr is not None:
        _err(errors, f"ledger invalid: {lerr}")
        return result

    ledger: List[Tuple[str, Dict[str, Any]]] = []
    by_id: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for fpath in ledger_files:
        try:
            fpath = safe_path(root_p, _norm_ws_rel(root_p, fpath), allow_root=False)
        except ValueError as exc:
            _err(errors, f"ledger path unsafe: {exc}")
            return result
        if not fpath.exists() or not fpath.is_file():
            _err(errors, f"ledger path missing or not regular: {fpath}")
            return result
        rel = _norm_ws_rel(root_p, fpath)
        try:
            raw_l = _read_record_bytes(fpath)
            decoded = _loads_strict(raw_l)
        except (OSError, EOFError, ValueError, UnicodeDecodeError) as exc:
            _err(errors, f"ledger parse failure {rel}: {exc}")
            return result
        if not isinstance(decoded, dict):
            _err(errors, f"ledger record not object: {rel}")
            return result
        rid = decoded.get("id")
        if not isinstance(rid, str) or not _ID_RE.fullmatch(rid):
            _err(errors, f"ledger id invalid: {rel}")
            return result
        if rid in by_id:
            _err(errors, f"duplicate ledger id: {rid}")
            return result
        by_id[rid] = (rel, decoded)
        ledger.append((rel, decoded))

    ledger_ids: Set[str] = set(by_id.keys())
    own_id = record.get("id")
    if isinstance(own_id, str):
        if own_id in by_id:
            lrel, _ = by_id[own_id]
            if lrel != target_rel:
                _err(errors, f"record id {own_id!r} already used by different path {lrel}")
                return result

    fingerprint: Optional[str] = None
    own_completeness: Optional[str] = None
    cfg = record.get("configuration")
    if isinstance(cfg, dict):
        own_completeness = cfg.get("completeness")
        if own_completeness == "complete" and isinstance(cfg.get("factors"), dict):
            try:
                fingerprint = _canonical_factors(cfg["factors"])
            except (TypeError, ValueError):
                fingerprint = None
    result["fingerprint"] = fingerprint

    duplicates: List[Dict[str, Any]] = []
    any_other_partial = any(
        rel != target_rel
        and isinstance(rec.get("configuration"), dict)
        and rec["configuration"].get("completeness") == "partial"
        for rel, rec in ledger
    )
    if fingerprint is not None:
        for rel, rec in ledger:
            if rel == target_rel:
                continue
            rcfg = rec.get("configuration")
            if not isinstance(rcfg, dict):
                continue
            rcomp = rcfg.get("completeness")
            if rcomp != "complete" or not isinstance(rcfg.get("factors"), dict):
                continue
            try:
                rfp = _canonical_factors(rcfg["factors"])
            except (TypeError, ValueError):
                continue
            if rfp == fingerprint:
                duplicates.append({
                    "id": rec.get("id"),
                    "kind": rec.get("kind"),
                    "completeness": rcomp,
                })
    result["duplicates"] = duplicates
    duplicate_ids_ledger = [d["id"] for d in duplicates if isinstance(d.get("id"), str)]

    if any_other_partial:
        warnings.append("历史配置不完整，无法排除与历史重复；GPT 必须核对适用条件")

    seen_fp: Optional[str] = None
    if phase == "start" and duplicates:
        seen_fp = fingerprint

    try:
        _validate_record(
            errors,
            root_p,
            record,
            phase,
            ledger_ids,
            ledger_root,
            seen_fp,
            duplicate_ids_ledger,
            require_deep=True,
        )
    except (ValueError, OSError, TypeError, KeyError, OverflowError, RuntimeError, EOFError) as exc:
        _err(errors, f"validation error: {exc}")

    if phase == "start":
        kind = record.get("kind")
        if kind != "planned":
            _err(errors, "start phase requires kind=planned")
        ex = record.get("execution")
        if not isinstance(ex, dict):
            _err(errors, "start phase requires execution object")
        else:
            ep = ex.get("entrypoint")
            if not isinstance(ep, str) or not ep:
                _err(errors, "start phase requires nonempty execution.entrypoint")
            cfg2 = record.get("configuration")
            sh = None
            if isinstance(cfg2, dict):
                fac = cfg2.get("factors")
                if isinstance(fac, dict):
                    sh = fac.get("source_hashes")
            if not isinstance(sh, dict):
                _err(errors, "start phase requires configuration.factors.source_hashes object")
            elif isinstance(sh, dict):
                if isinstance(ep, str) and ep:
                    if ep not in sh:
                        _err(errors, f"start phase: entrypoint {ep!r} missing from source_hashes")
                    else:
                        declared = sh.get(ep)
                        if not isinstance(declared, str):
                            _err(errors, f"start phase: source_hashes[{ep!r}] must be string")
                        else:
                            try:
                                ep_target = safe_path(root_p, ep, allow_root=False)
                            except ValueError as exc:
                                _err(errors, f"start phase: entrypoint path unsafe: {exc}")
                            else:
                                if not ep_target.exists() or not ep_target.is_file():
                                    _err(errors, "start phase: entrypoint missing or not regular")
                                else:
                                    try:
                                        actual_hash = _sha256_bytes(ep_target.read_bytes())
                                    except OSError as exc:
                                        _err(errors, f"start phase: cannot read entrypoint: {exc}")
                                    else:
                                        if actual_hash != declared:
                                            _err(errors, "start phase: entrypoint hash mismatch")

    hr = record.get("history_review")
    refs_to_validate: Set[str] = set()
    if isinstance(hr, dict):
        for key in ("related_ids", "repeat_of"):
            v = hr.get(key)
            if isinstance(v, list):
                for rid in v:
                    if isinstance(rid, str):
                        refs_to_validate.add(rid)
    if phase == "start":
        for rel, rec in ledger:
            if rel == target_rel:
                continue
            rid = rec.get("id")
            if isinstance(rid, str):
                refs_to_validate.add(rid)
    for rid in duplicate_ids_ledger:
        refs_to_validate.add(rid)

    for rid in sorted(refs_to_validate):
        if rid not in by_id:
            _err(errors, f"referenced ledger id not found: {rid!r}")
            continue
        rel, rec = by_id[rid]
        if rel == target_rel:
            continue
        rev = rec.get("review")
        ref_phase = "report" if isinstance(rev, dict) and rev.get("status") == "verified" else "archive"
        sub_errors: List[str] = []
        try:
            _validate_record(
                sub_errors,
                root_p,
                rec,
                ref_phase,
                ledger_ids,
                ledger_root,
                None,
                [],
                require_deep=False,
            )
        except (ValueError, OSError, TypeError, KeyError, OverflowError, RuntimeError, EOFError) as exc:
            sub_errors.append(f"validation error: {exc}")
        for e in sub_errors:
            _err(errors, f"ledger[{rid}]: {e}")

    if duplicates and phase in ("archive", "report"):
        warnings.append(f"complete configuration duplicates detected: {duplicate_ids_ledger}")

    try:
        current = target.read_bytes()
    except (OSError, EOFError) as exc:
        _err(errors, f"cannot re-read record: {exc}")
        current = None
    if current is not None:
        if _sha256_bytes(current) != record_sha or current != raw:
            _err(errors, "record bytes changed during validation")

    result["ok"] = len(errors) == 0
    return result
