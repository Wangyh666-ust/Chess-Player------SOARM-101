def filter_plan(planner, plan):
    if not isinstance(plan, dict):
        raise TypeError("plan must be a dict")
    if plan.get("ok") is not True:
        return plan
    context = getattr(planner, "_start_context", None)
    if context is None:
        return plan
    if not isinstance(context, (tuple, list)) or len(context) != 4:
        raise TypeError("planner._start_context must be a tuple/list of length 4")
    actual, querycfg, case, snapshot = context
    try:
        nodes = planner.bridge(actual, querycfg, case, snapshot, plan)
    except RuntimeError as exc:
        reason = "initial_bridge_preflight: " + str(exc)
        rejected = dict(plan)
        rejected["ok"] = False
        rejected["reason"] = reason
        rejected["bridge_preflight"] = {"ok": False, "reason": str(exc)}
        return rejected
    if not isinstance(nodes, list) or not nodes:
        raise TypeError("planner.bridge must return a non-empty list of nodes")
    accepted = dict(plan)
    accepted["bridge_preflight"] = {
        "ok": True,
        "nodes": len(nodes),
        "first": nodes[0],
        "last": nodes[-1],
    }
    return accepted
