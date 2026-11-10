"""Experiment gate: validation, history, audit, and run orchestration (no scientific conclusions)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from record_checks import check, load_ledger, safe_path, load_json

DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _emit(obj) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _fail(obj) -> int:
    _emit(obj)
    return 2


def _err(msg: str) -> int:
    _emit({"ok": False, "errors": [msg], "warnings": []})
    return 2


def _load_record_checked(root: Path, record_path: str, phase: str,
                         ledger_dir: str = "experiment_review/ledger"):
    """Reload the record via record_checks.check and return (ok, payload).

    payload: {record, record_sha256, validation}. No custom json loading and
    no handmade hash. Callers must not access payload["record"] unless ok.
    """
    try:
        result = check(root, record_path, phase=phase, ledger_dir=ledger_dir)
    except (ValueError, TypeError, OSError) as exc:
        return False, {
            "record": None,
            "record_sha256": None,
            "validation": {
                "ok": False,
                "errors": ["record check failed: %s" % type(exc).__name__],
                "warnings": [],
            },
        }
    payload = {
        "record": result.get("record"),
        "record_sha256": result.get("record_sha256"),
        "validation": result,
    }
    return bool(result.get("ok")), payload
def _result_out(result: dict, record_sha: str | None) -> dict:
    out = {
        "ok": bool(result.get("ok")),
        "errors": result.get("errors", []),
        "warnings": result.get("warnings", []),
        "fingerprint": result.get("fingerprint"),
        "duplicates": result.get("duplicates", []),
    }
    if record_sha is not None:
        out["record_sha256"] = record_sha
    return out


def cmd_check(root: Path, args) -> int:
    ok, payload = _load_record_checked(root, args.record, args.phase,
                                       ledger_dir=args.ledger)
    result = payload["validation"]
    out = _result_out(result, payload.get("record_sha256"))
    out["ok"] = ok
    _emit(out)
    return 0 if ok else 2
def _search_hits(records, query: str):
    q = query.lower()
    hits = []
    for rel, rec in records:
        try:
            blob = json.dumps(rec, ensure_ascii=False, sort_keys=True).lower()
        except (TypeError, ValueError):
            continue
        if q in blob:
            hits.append((rel, rec))
    return hits


def cmd_history(root: Path, args) -> int:
    """History command: report validated records and their evidence.

    Uses check() with phase report only when review.status == 'verified';
    otherwise archive. A failed report check never falls back to a supposedly
    verified archive success: scientific claims remain unverified unless the
    report check succeeds.
    """
    try:
        records = load_ledger(root, args.ledger)
    except (ValueError, OSError) as exc:
        return _err("ledger load failed: %s" % exc)
    query = args.query or ""
    hits = _search_hits(records, query) if query else list(records)
    results = []
    errors = []
    any_invalid = False
    for rel, rec in hits:
        review = rec.get("review") or {}
        status = review.get("status")
        phase = "report" if status == "verified" else "archive"
        res = check(root, rel, phase=phase, ledger_dir=args.ledger)
        result_ok = bool(res.get("ok"))
        review_verified = bool(status == "verified" and result_ok)
        if not result_ok:
            any_invalid = True
            errors.append({
                "id": rec.get("id"),
                "path": rel,
                "errors": list(res.get("errors", [])),
            })
        if review_verified:
            confirmed_claims = list(review.get("claims") or [])
            pending_claims = []
        else:
            confirmed_claims = []
            pending_claims = list(review.get("claims") or [])
        history_review = rec.get("history_review") or {}
        observations = list(rec.get("observations") or [])
        limitations = list(review.get("limitations") or [])
        evidence = []
        for ev in rec.get("evidence") or []:
            if not isinstance(ev, dict):
                continue
            evidence.append({
                "id": ev.get("id"),
                "archive": ev.get("path"),
                "original": ev.get("original_path"),
                "selector": ev.get("selector"),
                "sha256": ev.get("sha256"),
            })
        results.append({
            "id": rec.get("id"),
            "title": rec.get("title"),
            "record": rel,
            "configuration": rec.get("configuration"),
            "review_verified": review_verified,
            "confirmed_claims": confirmed_claims,
            "pending_claims": pending_claims,
            "open_questions": list(history_review.get("unknown") or []),
            "observations": observations,
            "limitations": limitations,
            "evidence": evidence,
            "validation": {
                "ok": result_ok,
                "errors": list(res.get("errors", [])),
                "warnings": list(res.get("warnings", [])),
            },
        })
    payload = {
        "ok": not any_invalid,
        "errors": errors,
        "results": results,
    }
    _emit(payload)
    return 2 if any_invalid else 0
def cmd_audit_ledger(root: Path, args) -> int:
    """Audit every ledger record. Uses report phase only when review.status ==
    'verified'; otherwise archive. Never reports unconditional success."""
    try:
        records = load_ledger(root, args.ledger)
    except (ValueError, OSError) as exc:
        return _err("ledger load failed: %s" % exc)
    summary = []
    total_errors = 0
    total_warnings = 0
    for rel, rec in records:
        review = rec.get("review") or {}
        status = review.get("status")
        phase = "report" if status == "verified" else "archive"
        res = check(root, rel, phase=phase, ledger_dir=args.ledger)
        result_ok = bool(res.get("ok"))
        review_verified = bool(status == "verified" and result_ok)
        errs = list(res.get("errors", []))
        warns = list(res.get("warnings", []))
        total_errors += len(errs)
        total_warnings += len(warns)
        summary.append({
            "id": rec.get("id"),
            "path": rel,
            "review_verified": review_verified,
            "ok": result_ok,
            "errors": errs,
            "warnings": warns,
        })
    ok = total_errors == 0
    _emit({
        "ok": ok,
        "records": summary,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
    })
    return 0 if ok else 2
def _terminate_tree(pid: int, notes: list) -> None:
    if not isinstance(pid, int) or pid <= 0:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
            )
            notes.append("taskkill_tree_attempted")
        except Exception as exc:  # noqa: BLE001
            notes.append("taskkill_failed:%s" % type(exc).__name__)
        if _was_alive(pid):
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    timeout=5,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=False,
                )
            except Exception:  # noqa: BLE001
                pass
    notes.append("cross_wsl_descendants_unproven")
def _was_alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", "PID eq %d" % pid],
                capture_output=True, timeout=3, shell=False,
            )
            return str(pid).encode() in (out.stdout or b"")
        else:
            os.kill(pid, 0)
            return True
    except Exception:  # noqa: BLE001
        return False


def cmd_run(root: Path, args) -> int:
    ok, payload = _load_record_checked(
        root, args.record, "start", ledger_dir=args.ledger)
    if not ok:
        return _fail(_result_out(payload.get("validation", {}),
                                 payload.get("record_sha256")))
    record = payload["record"]
    initial_hash = payload["record_sha256"]
    if record is None:
        return _fail({"ok": False,
                      "errors": ["record unavailable after validation"],
                      "warnings": []})

    try:
        record_sp = safe_path(root, args.record)
        entrypoint = record["execution"]["entrypoint"]
        pinned = safe_path(root, entrypoint, allow_root=False)
    except (KeyError, ValueError, TypeError) as exc:
        return _fail({"ok": False,
                      "errors": ["invalid record execution: %s" % exc],
                      "warnings": []})

    try:
        if _sha256_file(record_sp) != initial_hash:
            return _fail({"ok": False,
                          "errors": ["record bytes changed after validation"],
                          "warnings": []})
    except OSError as exc:
        return _fail({"ok": False,
                      "errors": ["cannot rehash record: %s" % exc],
                      "warnings": []})

    run_id = record.get("id") or "unknown"
    run_dir = safe_path(root, "experiment_review/runs/%s" % run_id)
    if run_dir.exists():
        return _fail({"ok": False,
                      "errors": ["run dir already exists"],
                      "warnings": []})

    def _canonical_fingerprint_factors(factors):
        if not isinstance(factors, dict):
            return None
        try:
            return hashlib.sha256(
                json.dumps(
                    factors,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        except (TypeError, ValueError):
            return None

    try:
        try:
            ledger_records = load_ledger(root, args.ledger)
        except (ValueError, OSError) as exc:
            return _fail({"ok": False,
                          "errors": ["pending run review failed: %s"
                                     % type(exc).__name__],
                          "warnings": []})
        ledger_ids = set()
        for _rel, _rec in ledger_records:
            if isinstance(_rec, dict) and _rec.get("id") is not None:
                ledger_ids.add(_rec.get("id"))

        runs_root = safe_path(root, "experiment_review/runs")
        if runs_root.exists():
            for child in runs_root.iterdir():
                try:
                    if not child.is_dir():
                        continue
                except OSError:
                    continue
                child_rel = "experiment_review/runs/%s" % child.name
                try:
                    child_sp = safe_path(root, child_rel)
                except (ValueError, TypeError):
                    continue
                record_file = safe_path(root, child_rel + "/record.json")
                if not record_file.exists():
                    continue
                try:
                    snapshot = load_json(record_file)
                except (ValueError, OSError) as exc:
                    return _fail({"ok": False,
                                  "errors": ["pending run review failed: %s"
                                             % type(exc).__name__],
                                  "warnings": []})
                if not isinstance(snapshot, dict):
                    return _fail({"ok": False,
                                  "errors": ["pending run review failed: malformed record snapshot"],
                                  "warnings": []})
                prior_config = snapshot.get("configuration")
                if not isinstance(prior_config, dict):
                    return _fail({"ok": False,
                                  "errors": ["pending run review failed: malformed configuration"],
                                  "warnings": []})
                if prior_config.get("completeness") != "complete":
                    continue
                prior_factors = prior_config.get("factors")
                if prior_factors is None:
                    continue
                if not isinstance(prior_factors, dict):
                    return _fail({"ok": False,
                                  "errors": ["pending run review failed: malformed factors"],
                                  "warnings": []})
                prior_fp = _canonical_fingerprint_factors(prior_factors)
                if prior_fp is None:
                    continue
                current_fp = payload["validation"].get("fingerprint")
                if prior_fp == current_fp:
                    prior_id = snapshot.get("id")
                    if prior_id not in ledger_ids:
                        return _fail({
                            "ok": False,
                            "errors": ["存在尚未入账的同配置启动记录 %s：先将该次执行入账并声明复测"
                                       % prior_id],
                            "warnings": [],
                        })
    except (ValueError, OSError, TypeError) as exc:
        return _fail({"ok": False,
                      "errors": ["pending run review failed: %s"
                                 % type(exc).__name__],
                      "warnings": []})

    args_list = record.get("execution", {}).get("args", [])
    if not isinstance(args_list, list):
        return _fail({"ok": False,
                      "errors": ["execution.args must be list"],
                      "warnings": []})

    budget = record.get("judgement", {}).get("budget", {})
    timeout_s = budget.get("wall_time_seconds") if isinstance(budget, dict) else None

    created = False
    stdout_log = stderr_log = None
    proc = None
    subprocess_started = False
    pid = None
    spawn_error = None
    termination = []
    timed_out = False
    exit_code = None
    wall_time = 0.0
    stage = "setup"

    try:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            created = True
        except FileExistsError:
            return _fail({"ok": False,
                          "errors": ["run dir race"],
                          "warnings": []})
        except OSError as exc:
            return _fail({"ok": False,
                          "errors": ["cannot create run dir: %s" % exc],
                          "warnings": []})

        (run_dir / "record.json").write_bytes(record_sp.read_bytes())
        validation_for_disk = {
            k: v for k, v in payload["validation"].items() if k != "record"
        }
        (run_dir / "validation.json").write_text(
            json.dumps(validation_for_disk, ensure_ascii=False, indent=2),
            encoding="utf-8")

        (run_dir / "started.json").write_text(json.dumps({
            "subprocess_started": False,
            "pid": None,
            "fingerprint": payload["validation"].get("fingerprint"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "argv": [sys.executable, str(pinned)] + [str(a) for a in args_list],
            "record_sha256": initial_hash,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        stdout_log = open(run_dir / "run.stdout.log", "wb")
        stderr_log = open(run_dir / "run.stderr.log", "wb")
        stage = "spawn"
    except OSError as exc:
        if stdout_log:
            try:
                stdout_log.close()
            except OSError:
                pass
        if stderr_log:
            try:
                stderr_log.close()
            except OSError:
                pass
        return _fail({"ok": False,
                      "errors": ["run setup failed: %s" % exc],
                      "warnings": []})

    child_env = dict(os.environ)
    child_env["PYTHONUTF8"] = "1"
    child_argv = [sys.executable, str(pinned)] + [str(a) for a in args_list]

    try:
        ok2, payload2 = _load_record_checked(
            root, args.record, "start", ledger_dir=args.ledger)
        rehash_ok = False
        try:
            rehash_ok = _sha256_file(record_sp) == initial_hash
        except OSError:
            rehash_ok = False
        if (not ok2 or payload2.get("record_sha256") != initial_hash
                or not rehash_ok):
            spawn_error = "pre-spawn revalidation failed"
        else:
            start = time.monotonic()
            proc = subprocess.Popen(
                child_argv,
                shell=False,
                cwd=str(root),
                stdin=subprocess.DEVNULL,
                stdout=stdout_log,
                stderr=stderr_log,
                env=child_env,
            )
            subprocess_started = True
            pid = proc.pid
            try:
                (run_dir / "started.json").write_text(json.dumps({
                    "subprocess_started": True,
                    "pid": pid,
                    "fingerprint": payload["validation"].get("fingerprint"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "argv": child_argv,
                    "record_sha256": initial_hash,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError as exc:
                raise exc
            try:
                proc.wait(timeout=(timeout_s if isinstance(timeout_s, (int, float))
                                   else None))
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_tree(pid, termination)
                try:
                    proc.wait(timeout=2)
                except Exception:  # noqa: BLE001
                    pass
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:  # noqa: BLE001
                        pass
            except KeyboardInterrupt:
                _terminate_tree(pid, termination)
                try:
                    proc.wait(timeout=2)
                except Exception:  # noqa: BLE001
                    pass
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:  # noqa: BLE001
                        pass
                spawn_error = "keyboard_interrupt"
            wall_time = time.monotonic() - start
            exit_code = proc.returncode
    except Exception as exc:  # noqa: BLE001
        spawn_error = "%s failed: %s" % (stage, type(exc).__name__)
        if proc is not None and subprocess_started:
            try:
                if proc.poll() is None:
                    _terminate_tree(pid, termination)
                    try:
                        proc.wait(timeout=2)
                    except Exception:  # noqa: BLE001
                        pass
                    if proc.poll() is None:
                        try:
                            proc.kill()
                        except Exception:  # noqa: BLE001
                            pass
            except Exception:  # noqa: BLE001
                pass
    finally:
        if stdout_log:
            try:
                stdout_log.close()
            except OSError:
                pass
        if stderr_log:
            try:
                stderr_log.close()
            except OSError:
                pass

    result = {
        "exit_code": exit_code,
        "wall_time_s": wall_time,
        "timed_out": timed_out,
        "subprocess_started": subprocess_started,
        "pid": pid,
        "task_outcome": "inconclusive",
        "needs_independent_review": True,
        "outcome": "unknown",
    }
    if spawn_error:
        result["error"] = spawn_error
    if termination:
        result["termination"] = termination

    result_ok = True
    try:
        (run_dir / "process_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        result_ok = False
        _emit({"ok": False,
               "errors": ["cannot write process_result: %s" % exc],
               "warnings": [],
               "subprocess_started": subprocess_started})

    overall_ok = (
        result_ok
        and subprocess_started
        and not timed_out
        and spawn_error is None
        and exit_code == 0
    )
    out = _result_out(payload["validation"], initial_hash)
    out["ok"] = overall_ok
    out["process_result"] = result
    out["outcome"] = "unknown"
    _emit(out)
    return 0 if overall_ok else 1
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="experiment_gate")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--ledger", default="experiment_review/ledger")
    sub = p.add_subparsers(dest="command", required=True)

    ph = sub.add_parser("history")
    ph.add_argument("--query", default="")

    pc = sub.add_parser("check")
    pc.add_argument("--record", required=True)
    pc.add_argument("--phase", required=True, choices=["archive", "start", "report"])

    sub.add_parser("audit-ledger")

    pr = sub.add_parser("run")
    pr.add_argument("--record", required=True)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    root = Path(args.root).resolve()
    try:
        if args.command == "check":
            return cmd_check(root, args)
        if args.command == "history":
            return cmd_history(root, args)
        if args.command == "audit-ledger":
            return cmd_audit_ledger(root, args)
        if args.command == "run":
            return cmd_run(root, args)
    except Exception as exc:  # noqa: BLE001
        return _err("internal error: %s" % type(exc).__name__)
    return _err("unknown command")


if __name__ == "__main__":
    sys.exit(main())
