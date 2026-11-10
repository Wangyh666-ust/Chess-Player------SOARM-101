REPLACEMENTS = [
    (
        '''                        expected = reason.startswith(("RuntimeError: solve_pose fail at phase=", "RuntimeError: joint jump ", "RuntimeError: geo.check failed at phase=", "RuntimeError: route: first node jump", "RuntimeError: rebuild: open check rejected"))
                        if expected and "tilt_route" in tb and "rebuild" in tb:
                            planning_rejection = {"reason": reason, "step": step_record.get("id"), "positive_steps": step_record.get("physics_steps_delta")}
                            break
''',
        '''                        expected = reason.startswith(("RuntimeError: solve_pose fail at phase=", "RuntimeError: joint jump ", "RuntimeError: geo.check failed at phase=", "RuntimeError: route: first node jump", "RuntimeError: rebuild: open check rejected"))
                        if expected and "tilt_route" in tb and "rebuild" in tb:
                            planning_rejection = {"reason": reason, "step": step_record.get("id"), "positive_steps": step_record.get("physics_steps_delta")}
                            break
                        empty_exit_release = reason.startswith("RuntimeError: empty_return_no_plan:") and "release_planner_bound_v1.py" in tb and "plan_exit" in tb
                        if empty_exit_release:
                            planning_rejection = {"reason": reason, "step": step_record.get("id"), "positive_steps": step_record.get("physics_steps_delta"), "release_started": (result.get("release_started") is True)}
                            break
''',
    ),
    (
        '''                if planning_rejection is not None:
                    rec["classification"] = "post_grasp_planning_block"
                    rec["planning_rejection"] = planning_rejection
''',
        '''                if planning_rejection is not None:
                    rec["classification"] = "post_grasp_planning_block"
                    if planning_rejection.get("release_started") is True:
                        rec["classification"] = "post_release_planning_block"
                    rec["planning_rejection"] = planning_rejection
''',
    ),
]
