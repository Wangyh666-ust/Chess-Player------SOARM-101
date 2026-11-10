import json
import math
import time
from pathlib import Path


def _round4(value):
    return round(float(value), 4)


def _valid_center(center):
    if not isinstance(center, dict):
        return False
    q = center.get("q_contact")
    g = center.get("gcontact_rad")
    if not isinstance(q, list) or len(q) != 5:
        return False
    if isinstance(g, bool) or not isinstance(g, (int, float)):
        return False
    if not math.isfinite(float(g)):
        return False
    for item in q:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return False
        if not math.isfinite(float(item)):
            return False
    return True


def _center_key(center):
    q = center["q_contact"]
    return (
        tuple(_round4(v) for v in q),
        _round4(center["gcontact_rad"]),
    )


def choose(rp, query_path, ik, catalogs, hovers, timeout, deadline):
    if not isinstance(catalogs, list) or not (1 <= len(catalogs) <= 28):
        raise ValueError("catalogs must be a list of length 1..28")
    if not isinstance(hovers, list) or len(hovers) != 4:
        raise ValueError("hovers must be exactly [0.01, 0.02, 0.03, 0.04]")
    expected_hovers = [0.01, 0.02, 0.03, 0.04]
    for a, b in zip(hovers, expected_hovers):
        if isinstance(a, bool) or not isinstance(a, (int, float)):
            raise ValueError("hovers must be exactly [0.01, 0.02, 0.03, 0.04]")
        if float(a) != float(b):
            raise ValueError("hovers must be exactly [0.01, 0.02, 0.03, 0.04]")

    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be a finite positive number")
        if not math.isfinite(float(timeout)) or float(timeout) <= 0:
            raise ValueError("timeout must be a finite positive number")

    if deadline is not None:
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
            raise ValueError("deadline must be a finite number")
        if not math.isfinite(float(deadline)):
            raise ValueError("deadline must be a finite number")

    start = time.time()
    limit = min(float(timeout) if timeout is not None else 60.0, 60.0)
    overall_end = start + limit
    if deadline is not None:
        overall_end = min(overall_end, float(deadline))

    query_file = Path(query_path)
    with query_file.open("r", encoding="utf-8") as handle:
        query = json.load(handle)

    counts = {
        "center_calls": 0,
        "valid_centers": 0,
        "plan_calls": 0,
    }
    tried = []
    source_attempts = []
    selected = None
    deadline_flag = False

    source_scan_end = min(start + 15.0, overall_end)
    source_scan_expired = False
    valid_sources = []
    seen = set()

    for cat in catalogs:
        if len(valid_sources) >= 4:
            break
        if time.time() >= overall_end:
            deadline_flag = True
            break
        if time.time() >= source_scan_end:
            source_scan_expired = True
            break

        qcopy = json.loads(json.dumps(query))
        qcopy["catalog_candidates"] = [cat]
        qcopy["keypoint_catalog_path"] = cat

        center_start = time.time()
        counts["center_calls"] += 1
        center = rp.center(qcopy, ik)
        center_elapsed = time.time() - center_start
        if not isinstance(center, dict):
            raise TypeError("rp.center must return a dict")

        record = {
            "catalog": cat,
            "center": center,
            "elapsed": center_elapsed,
        }
        source_attempts.append(record)

        if center.get("blocked") is not False:
            continue
        if not _valid_center(center):
            continue

        key = _center_key(center)
        if key in seen:
            continue
        seen.add(key)
        valid_sources.append({"catalog": cat, "center": center})
        counts["valid_centers"] = len(valid_sources)

    if len(valid_sources) < 4 and not source_scan_expired and not deadline_flag:
        # Scan ended naturally with fewer than four valid centers.
        pass

    tried.extend(source_attempts)

    if deadline_flag:
        return {
            "selected": None,
            "tried": tried,
            "deadline_flag": True,
            "counts": counts,
        }

    yaws = [0, 90, -90, 180]
    plan_limit = 64

    for hover in hovers:
        if selected is not None or counts["plan_calls"] >= plan_limit:
            break
        for source in valid_sources:
            if selected is not None or counts["plan_calls"] >= plan_limit:
                break
            for yaw in yaws:
                if counts["plan_calls"] >= plan_limit:
                    break
                if time.time() >= overall_end:
                    deadline_flag = True
                    break

                action = {
                    "query": str(query_path),
                    "hover_m": hover,
                    "placement_yaw_deg": yaw,
                }
                counts["plan_calls"] += 1
                plan_start = time.time()
                plan = rp.plan(action, source["center"])
                plan_elapsed = time.time() - plan_start
                if not isinstance(plan, dict):
                    raise TypeError("rp.plan must return a dict")

                ok = plan.get("ok", False) is True
                tried.append({
                    "catalog": source["catalog"],
                    "hover": hover,
                    "yaw": yaw,
                    "ok": ok,
                    "reason": plan.get("reason"),
                    "elapsed": plan_elapsed,
                })

                if time.time() >= overall_end:
                    deadline_flag = True
                    break

                if ok:
                    selected = {
                        "center": source["center"],
                        "plan": plan,
                        "catalog": source["catalog"],
                        "hover": hover,
                        "yaw": yaw,
                    }
                    break
            if deadline_flag:
                break
        if deadline_flag:
            break

    if deadline_flag:
        return {
            "selected": None,
            "tried": tried,
            "deadline_flag": True,
            "counts": counts,
        }

    return {
        "selected": selected,
        "tried": tried,
        "deadline_flag": False,
        "counts": counts,
    }
