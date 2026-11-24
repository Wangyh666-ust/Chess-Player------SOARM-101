"""Replay measured joint commands in an isolated PyBullet robot benchmark.

Uses the frozen runtime's robot loader, command method and step method. No board,
pieces, hardware or fitted parameters. Direct motor-degree -> URDF-radian mapping
is an explicit UNVALIDATED hypothesis, not a calibrated physical replay.
"""

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import statistics

from robot_calibration.records import ARM, read_json, read_rows, sha256, write_json


def hold_displacements(rows, values, initial):
    result = {}

    for phase in ("positive", "negative", "return"):
        held = [
            q - initial
            for row, q in zip(rows, values)
            if row["phase"] == f"cycle_1_{phase}_hold"
        ]
        result[phase] = statistics.median(held[-10:]) if held else None

    return result


def rmse(a, b):
    return math.sqrt(statistics.mean((x - y) ** 2 for x, y in zip(a, b)))


def replay(meta, rows, runtime, physics, p, gripper_rad, pose):
    """Hold each recorded command until the next send; step at 240 Hz.

    Commands activate on the first physics tick after their host send midpoint.
    Sample simulated state at the nearest completed physics tick before the real
    observation midpoint: never pretend the two clocks have infinite resolution.
    """

    world = physics.PhysicsWorld.__new__(physics.PhysicsWorld)
    world.client = p.connect(p.DIRECT)
    world.dt, world.steps, world.time = 1 / 240, 0, 0.0
    world.arm_force, world.grip_force = 1.5, 0.05
    world.friction = math.sqrt(0.4)
    measured_initial = [meta["initial_positions"][j] for j in ARM]
    initial = (
        [math.radians(q) for q in measured_initial]
        if pose == "recorded-assumed"
        else [0.0] * len(ARM)
    )
    asset_dir = runtime / "calibration/manipulation_20261109_v1/assets"

    try:
        p.setGravity(0, 0, -9.81, physicsClientId=world.client)
        p.setTimeStep(world.dt, physicsClientId=world.client)
        p.setPhysicsEngineParameter(
            numSolverIterations=150,
            contactBreakingThreshold=0.002,
            deterministicOverlappingPairs=1,
            physicsClientId=world.client,
        )

        old_assets = physics.ASSETS
        physics.ASSETS = asset_dir

        try:
            world._load_robot()
        finally:
            physics.ASSETS = old_assets

        for joint, value in zip(world.arm_joints, initial):
            info = p.getJointInfo(world.robot, joint, physicsClientId=world.client)

            if not info[8] <= value <= info[9]:
                raise ValueError("Assumed initial mapping exceeds a URDF joint limit")

        gi = p.getJointInfo(
            world.robot, world.gripper_joint, physicsClientId=world.client
        )

        if not gi[8] <= gripper_rad <= gi[9]:
            raise ValueError("Gripper reference angle exceeds URDF limit")

        world._reset_init(initial, gripper_rad)
        world.command(initial, gripper_rad)
        p.performCollisionDetection(physicsClientId=world.client)

        contacts = p.getContactPoints(
            world.robot, world.robot, physicsClientId=world.client
        )
        initial_contacts = len(contacts)
        deepest_penetration = max([0.0] + [-c[8] for c in contacts])
        names = {v: k for k, v in world.link_index.items()}
        initial_pairs = sorted(
            {(names[c[3]], names[c[4]]) for c in contacts if c[8] < 0}
        )
        origin = rows[0]["command_start_s"]
        sends = [(r["command_start_s"] + r["command_end_s"]) / 2 - origin for r in rows]
        reads = [(r["read_start_s"] + r["read_end_s"]) / 2 - origin for r in rows]

        if (
            any(t < 0 for t in sends + reads)
            or any(b <= a for a, b in zip(sends, sends[1:]))
            or any(b <= a for a, b in zip(reads, reads[1:]))
        ):
            raise ValueError("Nonmonotonic command/observation timestamps")

        qcmd = initial
        target_index = 0
        sim_values, sampled_times = [], []
        contacts_max = initial_contacts
        joint_index = world.arm_joints[ARM.index(meta["settings"]["joint"])]

        for read_time in reads:
            stop_tick = math.floor(read_time / world.dt + 1e-10)

            while world.steps < stop_tick:
                while (
                    target_index < len(rows)
                    and sends[target_index] <= world.time + 1e-12
                ):
                    qcmd = [
                        q0 + math.radians(rows[target_index]["command"][j] - real0)
                        for j, q0, real0 in zip(ARM, initial, measured_initial)
                    ]
                    target_index += 1

                world.command(qcmd, gripper_rad)
                world.step()

            angle = p.getJointState(
                world.robot, joint_index, physicsClientId=world.client
            )[0]
            axis = ARM.index(meta["settings"]["joint"])

            sim_values.append(
                measured_initial[axis] + math.degrees(angle - initial[axis])
            )
            sampled_times.append(world.time)

            contacts = p.getContactPoints(
                world.robot, world.robot, physicsClientId=world.client
            )
            contacts_max = max(contacts_max, len(contacts))
            deepest_penetration = max([deepest_penetration] + [-c[8] for c in contacts])

        if not all(math.isfinite(x) for x in sim_values):
            raise ValueError("Nonfinite simulation result")

        return (
            sim_values,
            reads,
            sampled_times,
            {
                "initial_self_contacts": initial_contacts,
                "initial_penetrating_pairs": initial_pairs,
                "max_sampled_self_contacts": contacts_max,
                "deepest_sampled_penetration_m": deepest_penetration,
                "contact_free_at_samples": contacts_max == 0,
            },
        )
    finally:
        p.disconnect(world.client)


def compare(source, output, repository, gripper_rad=0.0, pose="recorded-assumed"):
    import numpy as np
    import pybullet as p
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runtime = repository.resolve() / "simulation/runtime"
    module_path = runtime / "reachability_eval/physics_world.py"
    spec = importlib.util.spec_from_file_location("benchmark_physics", module_path)
    physics = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(physics)

    paths = sorted(source.glob("measurements/joint_response/*/metadata.json"))

    if not paths:
        raise ValueError("No joint response recordings found")

    if not math.isfinite(gripper_rad):
        raise ValueError("Gripper reference must be finite")

    output.mkdir(parents=True, exist_ok=False)

    asset_dir = runtime / "calibration/manipulation_20261109_v1/assets"

    if pose not in ("recorded-assumed", "neutral-reference"):
        raise ValueError("Unknown simulation pose mode")

    result = {
        "status": "running",
        "benchmark": "conditional isolated unloaded robot replay",
        "pose": pose,
        "suitable_for_physical_parameter_identification": False,
        "hardware_mapping_validated": False,
        "assumptions": [
            "Same joint names and positive direction assumed; physical zero/sign unvalidated",
            "recorded-assumed uses real angles as URDF angles; neutral-reference uses zero pose plus relative commands",
            "Gripper held at the chosen reference angle; jaw calibration unavailable",
            "No external support, board, pieces or table; gravity retained",
            "No parameter fitting; the current simulation controller is unchanged",
        ],
        "settings": {
            "dt_s": 1 / 240,
            "arm_force": 1.5,
            "grip_force": 0.05,
            "position_gain": 0.2,
            "velocity_gain": 1,
            "max_arm_velocity_rad_s": 0.6,
            "gripper_reference_rad": gripper_rad,
            "gravity_m_s2": -9.81,
            "contact_breaking_threshold_m": 0.002,
        },
        "versions": {
            x: importlib.metadata.version(x)
            for x in ("numpy", "pybullet", "matplotlib")
        },
        "source_sha256": {
            str(module_path.relative_to(repository.resolve())): sha256(module_path),
            "robot_windows.urdf": sha256(asset_dir / "robot_windows.urdf"),
            "robot.srdf": sha256(asset_dir / "robot.srdf"),
            "comparison_script": sha256(Path(__file__)),
        },
        "tests": [],
    }

    write_json(output / "comparison.json", result)

    completed = []

    try:
        for mp in paths:
            meta, rows = read_json(mp), read_rows(mp.with_name("samples.jsonl"))

            if meta["backend"] != "lerobot" or meta["units"]["arm"] != "deg":
                raise ValueError("Expected real arm measurements in degrees")

            if not rows or len(rows) != meta["samples"]:
                raise ValueError("Measurement count mismatch")

            sim, t, sim_t, contacts = replay(
                meta, rows, runtime, physics, p, gripper_rad, pose
            )
            joint = meta["settings"]["joint"]
            initial = meta["initial_positions"][joint]
            command = [r["command"][joint] for r in rows]
            measured = [r["measured"][joint] for r in rows]
            record = {
                "test": mp.parent.name,
                "joint": joint,
                "recording_status": meta["status"],
                "samples": len(rows),
                "measurement_sha256": sha256(mp.with_name("samples.jsonl")),
                "metadata_sha256": sha256(mp),
                "real_rmse_to_command_deg": rmse(measured, command),
                "sim_rmse_to_command_deg": rmse(sim, command),
                "sim_rmse_to_real_deg": rmse(sim, measured),
                "max_sampling_age_s": max(a - b for a, b in zip(t, sim_t)),
                "real_hold_delta_deg": hold_displacements(rows, measured, initial),
                "sim_hold_delta_deg": hold_displacements(rows, sim, initial),
                **contacts,
            }

            result["tests"].append(record)

            with (output / (mp.parent.name + ".csv")).open("x", newline="") as stream:
                writer = csv.writer(stream)

                writer.writerow(
                    [
                        "sample",
                        "phase",
                        "real_time_s",
                        "sim_time_s",
                        "command_delta_deg",
                        "real_delta_deg",
                        "sim_delta_deg",
                    ]
                )

                for i, r in enumerate(rows):
                    writer.writerow(
                        [
                            i,
                            r["phase"],
                            t[i],
                            sim_t[i],
                            command[i] - initial,
                            measured[i] - initial,
                            sim[i] - initial,
                        ]
                    )

            if meta["status"] == "completed":
                completed.append(
                    (
                        record,
                        t,
                        np.array(command) - initial,
                        np.array(measured) - initial,
                        np.array(sim) - initial,
                    )
                )

        fig, axes = plt.subplots(
            len(completed), 1, figsize=(10, 2.4 * len(completed)), squeeze=False
        )

        for ax, (record, t, cmd, measured, sim) in zip(axes.flat, completed):
            ax.plot(t, cmd, "--", color="#777777", label="Requested")
            ax.plot(t, measured, color="#1769aa", label="Real encoder")
            ax.plot(t, sim, color="#d05b20", label=f"PyBullet ({pose})")
            ax.set_title(record["joint"].replace("_", " "), loc="left")
            ax.set_ylabel("Displacement (deg)")
            ax.grid(alpha=0.2)

        axes[0, 0].legend(loc="upper right", fontsize=8)
        axes[-1, 0].set_xlabel("Host elapsed time from first command (s)")
        fig.suptitle(
            f"Real vs current simulation: {pose} diagnostic\n"
            "Different/unvalidated physical pose; no controller parameters fitted",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output / "joint_response_comparison.png", dpi=150)
        plt.close(fig)

        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "comparison.json", result)

    print(json.dumps(result["tests"], indent=2))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repository", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--gripper-reference-rad", type=float, default=0.0)
    parser.add_argument(
        "--pose",
        choices=("recorded-assumed", "neutral-reference"),
        default="recorded-assumed",
    )

    args = parser.parse_args()

    compare(
        args.source, args.output, args.repository, args.gripper_reference_rad, args.pose
    )
