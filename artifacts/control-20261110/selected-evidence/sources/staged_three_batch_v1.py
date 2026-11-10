"""Versioned library extracted from registered prior batch; no CLI or native work at import."""
import json,time
from pathlib import Path
import prepared_session_staged_v2 as S
E = S.E
_safe_id = S._safe_id
_write_json = S._write_json
_load_json = S._load_json
run_prepared_session = S.run_prepared_session

def run_batch(cfg, outdir):
    start_wall = time.time()
    outdir = Path(outdir)
    out_path = outdir / "result.json"
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {out_path}")
    outdir.mkdir(parents=True, exist_ok=True)

    def _err(msg, code=2):
        try:
            _write_json(out_path, {
                "mode": cfg.get("mode"), "sessions": [], "unattempted": [],
                "planned_actions": 0, "completed_actions": 0, "runner_attempted_actions": 0,
                "positive_step_actions": 0, "total_physics_steps": None, "total_sim_time_s": None,
                "known_total_physics_steps": 0, "known_completed_actions": 0,
                "known_runner_attempted_actions": 0, "accounting_complete": False,
                "wall_s": time.time() - start_wall, "error": msg,
            })
        except Exception:
            pass
        print(f"FATAL: {msg}", flush=True)
        return code

    mode = cfg.get("mode")
    if mode not in ("far", "games"):
        return _err(f"invalid mode {mode!r}")
    expected_n = 5 if mode == "far" else 3
    all_cfg = list(cfg.get("sessions") or [])
    if len(all_cfg) != expected_n:
        return _err(f"mode {mode} requires exactly {expected_n} sessions, got {len(all_cfg)}")

    try:
        budget = float(cfg["budget_s"])
    except Exception:
        return _err("budget_s missing/invalid")
    import math
    if not (budget > 0 and math.isfinite(budget)):
        return _err("budget_s must be positive finite")

    session_cfgs, ids, plan = [], [], []
    for sc in all_cfg:
        sid = _safe_id(sc.get("id"))
        if not sid or sid in ids:
            return _err(f"unsafe or duplicate session id {sid!r}")
        ids.append(sid)
        try:
            p = Path(sc["config"])
        except Exception:
            return _err(f"{sid}: missing session path")
        if not p.is_absolute():
            return _err(f"{sid}: path not absolute: {p}")
        loaded = _load_json(p)
        es = loaded.get("expected_steps")
        if es != (1 if mode == "far" else 10):
            return _err(f"{sid}: expected_steps {es!r} != {1 if mode=='far' else 10}")
        steps = loaded.get("steps")
        if not isinstance(steps, list) or len(steps) != es:
            return _err(f"{sid}: steps length != expected_steps")
        ms = loaded.get("max_steps_per_action")
        if not isinstance(ms, int) or isinstance(ms, bool) or not (1 <= ms <= 24000):
            return _err(f"{sid}: bad max_steps_per_action {ms!r}")
        session_cfgs.append((sid, sc, loaded))
    planned_actions = sum(len(l["steps"]) for _, _, l in session_cfgs)
    if planned_actions != (5 if mode == "far" else 30):
        return _err(f"planned_actions {planned_actions} != {5 if mode=='far' else 30}")

    sessions_out, unattempted = [], []
    tot_phys = tot_sim = tot_wall = None
    comp_act = run_att = pos_act = 0
    known_phys = known_comp = known_run = 0
    accounting_complete = True
    fault = None
    budget_hit = False

    for idx, (sid, sc, loaded) in enumerate(session_cfgs):
        if fault is not None:
            unattempted.append(sid); continue
        now = time.time()
        remaining = budget - (now - start_wall)
        if remaining <= 0:
            unattempted.append(sid); budget_hit = True; fault = "budget_exhausted"; continue

        sdir = outdir / sid
        if sdir.exists():
            unattempted.append(sid); fault = "program_error"; continue
        sdir.mkdir(parents=True, exist_ok=False)

        try:
            try:
                loaded["deadline_s"] = min(float(loaded.get("deadline_s", remaining)), remaining)
            except Exception:
                loaded["deadline_s"] = remaining

            rc = None; wrapper_err = None; raw = None
            try:
                rc = run_prepared_session(loaded, sdir)
            except Exception as e:
                wrapper_err = f"{type(e).__name__}: {e}"
                rc = None
            rp = sdir / "result.json"
            result_path = None; trace_paths = []
            if rp.exists():
                result_path = str(rp)
                try:
                    raw = _load_json(rp)
                    if not isinstance(raw, dict):
                        raw = None
                except Exception:
                    raw = None
            trace_paths = [str(p) for p in sdir.glob("step*/trial/trace.jsonl.gz")]

            rec = {"id": sid, "returncode": rc, "raw_classification": None, "classification": None,
                   "ok": None, "runner_attempted_actions": None, "positive_step_actions": None,
                   "completed_actions": None, "completed_action_ids": [], "total_physics_steps": None,
                   "total_sim_time_s": None, "total_wall_s": None, "result_path": result_path,
                   "error": wrapper_err, "trace_paths": trace_paths}

            if raw is None:
                accounting_complete = False
                rec["classification"] = "program_error"
                rec["error"] = wrapper_err or "missing/malformed raw session result; physics unknown"
                fault = "program_error"
            else:
                ra = raw.get("attempted_actions")
                cs = raw.get("completed_steps")
                ps = raw.get("per_step")
                ra = ra if isinstance(ra, list) else None
                cs = cs if isinstance(cs, list) else None
                ps = ps if isinstance(ps, list) else None
                if ra is None or cs is None or ps is None:
                    accounting_complete = False
                    fault = "program_error"
                rec["raw_classification"] = raw.get("classification")
                rec["ok"] = raw.get("ok") if isinstance(raw.get("ok"), bool) else None
                if ra is not None:
                    rec["runner_attempted_actions"] = len(ra); run_att += len(ra); known_run += len(ra)
                if cs is not None:
                    rec["completed_actions"] = len(cs); rec["completed_action_ids"] = list(cs)
                    comp_act += len(cs); known_comp += len(cs)
                if ps is not None:
                    pos = 0; ok_ps = True
                    for ent in ps:
                        if not isinstance(ent, dict) or not isinstance(ent.get("physics_steps_delta"), int):
                            ok_ps = False; break
                        if ent["physics_steps_delta"] > 0: pos += 1
                    if ok_ps:
                        rec["positive_step_actions"] = pos; pos_act += pos
                tp = raw.get("total_physics_steps")
                ts = raw.get("total_sim_time_s")
                tw = raw.get("total_wall_s")
                if isinstance(tp, int) and not isinstance(tp, bool):
                    rec["total_physics_steps"] = tp
                    tot_phys = tp if tot_phys is None else tot_phys + tp
                else:
                    accounting_complete = False
                if isinstance(ts, (int, float)):
                    rec["total_sim_time_s"] = float(ts)
                    tot_sim = float(ts) if tot_sim is None else tot_sim + float(ts)
                if isinstance(tw, (int, float)):
                    rec["total_wall_s"] = float(tw)
                    tot_wall = float(tw) if tot_wall is None else tot_wall + float(tw)
                rec["classification"] = "program_error" if wrapper_err else (rec["raw_classification"] or "unknown")
                planning_rejection = None
                if wrapper_err is None and rec["raw_classification"] == "program_error":
                    for step_record in (raw.get("per_step") or []):
                        result = step_record.get("results") or {}
                        reason = result.get("reason") or ""
                        tb = result.get("traceback") or ""
                        expected = reason.startswith(("RuntimeError: solve_pose fail at phase=", "RuntimeError: joint jump ", "RuntimeError: geo.check failed at phase=", "RuntimeError: route: first node jump", "RuntimeError: rebuild: open check rejected"))
                        if expected and "tilt_route" in tb and "rebuild" in tb:
                            planning_rejection = {"reason": reason, "step": step_record.get("id"), "positive_steps": step_record.get("physics_steps_delta")}
                            break
                if planning_rejection is not None:
                    rec["classification"] = "post_grasp_planning_block"
                    rec["planning_rejection"] = planning_rejection
                elif rec["classification"] == "program_error":
                    fault = "program_error"

            if tot_phys is not None and tot_phys > planned_actions * 24000:
                fault = "program_error"; rec["classification"] = "program_error"
                rec["error"] = "declared physics cap exceeded"
            if wrapper_err and fault is None:
                fault = "program_error"

            if raw is not None and fault is not None:
                # keep raw data but stop after this session
                pass

            sessions_out.append(rec)
            print(f"[{idx+1}/{len(session_cfgs)}] {sid} rc={rc} class={rec['classification']} "
                  f"att={rec['runner_attempted_actions']} comp={rec['completed_actions']}",
                  flush=True)
        except Exception as e:
            fault = "program_error"
            sessions_out.append({"id": sid, "returncode": None, "classification": "program_error",
                                 "raw_classification": None, "ok": None,
                                 "runner_attempted_actions": None, "positive_step_actions": None,
                                 "completed_actions": None, "completed_action_ids": [],
                                 "total_physics_steps": None, "total_sim_time_s": None,
                                 "total_wall_s": None, "result_path": None,
                                 "error": f"{type(e).__name__}: {e}", "trace_paths": []})
            accounting_complete = False

    summary = {
        "mode": mode, "sessions": sessions_out, "unattempted": unattempted,
        "planned_actions": planned_actions, "completed_actions": comp_act if accounting_complete else None,
        "runner_attempted_actions": run_att if accounting_complete else None, "positive_step_actions": pos_act if accounting_complete else None,
        "total_physics_steps": tot_phys if accounting_complete else None, "total_sim_time_s": tot_sim if accounting_complete else None,
        "known_total_physics_steps": known_phys if tot_phys is None else tot_phys,
        "known_completed_actions": known_comp, "known_runner_attempted_actions": known_run,
        "accounting_complete": accounting_complete, "wall_s": time.time() - start_wall,
        "error": fault,
    }
    if tot_phys is not None:
        summary["known_total_physics_steps"] = tot_phys
    _write_json(out_path, summary)

    if fault is not None or budget_hit:
        return 2 if fault == "program_error" else 1
    if not accounting_complete:
        return 1
    if comp_act == planned_actions and not unattempted and len(sessions_out) == expected_n and all(x["ok"] is True for x in sessions_out):
        return 0
    return 1


