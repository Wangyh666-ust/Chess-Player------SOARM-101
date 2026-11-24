"""Offline pose evidence for the 12 November SO-101 handoff.

Inspect joint definitions, render recorded poses without stepping physics, and
check the direct-angle hypothesis. This does NOT fit or validate a calibration.
No robot SDK is imported, no limits are clamped, and frozen assets are unchanged.
"""

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from robot_calibration.records import ARM, read_json, read_rows, sha256, write_json

# These are review samples, not independent evaluation episodes. Host times are
# resolved to actual recorded observations and their preceding camera frames.
REVIEW_POSES = (
    ("white-rook-1/trial-001", 3.0),
    ("white-rook-1/trial-001", 14.0),
    ("white-rook-1/trial-001", 22.5),
    ("white-rook-1/trial-001", 28.0),
    ("white-rook-1/trial-002", 13.5),
    ("white-knight-1/trial-001", 22.5),
)


def joint_definitions(path):
    return {
        j.get("name"): {
            "type": j.get("type"),
            **{
                tag: j.find(tag).attrib
                for tag in ("parent", "child", "origin", "axis", "limit")
                if j.find(tag) is not None
            },
        }
        for j in ET.parse(path).getroot().findall("joint")
    }


def out_of_limits(angles, definitions):
    """Report violations; never alter measurements to fit a model's limits."""

    return [
        joint
        for joint, degrees in angles.items()
        if not float(definitions[joint]["limit"]["lower"])
        <= math.radians(degrees)
        <= float(definitions[joint]["limit"]["upper"])
    ]


def static_pose(world, p, angles, gripper_rad):
    """Reset states, then query collisions; no motor commands or physics steps."""

    for joint, name in zip(world.arm_joints, ARM):
        p.resetJointState(
            world.robot, joint, math.radians(angles[name]), physicsClientId=world.client
        )

    p.resetJointState(
        world.robot, world.gripper_joint, gripper_rad, physicsClientId=world.client
    )
    p.performCollisionDetection(physicsClientId=world.client)

    names = {value: key for key, value in world.link_index.items()}
    pairs = {}

    for c in p.getContactPoints(world.robot, world.robot, physicsClientId=world.client):
        if c[8] < 0:
            key = " / ".join(sorted((names[c[3]], names[c[4]])))
            pairs[key] = max(pairs.get(key, 0), -1000 * c[8])

    return {
        "penetration_mm_by_pair": pairs,
        "max_penetration_mm": max(pairs.values(), default=0),
        "tool_position_m": list(
            p.getLinkState(
                world.robot,
                world.tcp_link,
                computeForwardKinematics=True,
                physicsClientId=world.client,
            )[4]
        ),
    }


def render(world, p, yaw=0):
    import numpy as np
    from PIL import Image

    view = p.computeViewMatrixFromYawPitchRoll([0.09, 0, 0.14], 0.58, yaw, -18, 0, 2)
    projection = p.computeProjectionMatrixFOV(48, 1, 0.01, 2)
    pixels = p.getCameraImage(
        600,
        600,
        view,
        projection,
        renderer=p.ER_TINY_RENDERER,
        lightDirection=[-1, -1, 2],
        physicsClientId=world.client,
    )[2]

    return Image.fromarray(
        np.asarray(pixels, dtype=np.uint8).reshape(600, 600, 4)[:, :, :3]
    )


def original_pose_probe(p, client, urdf, srdf, angles):
    """Check the same folded pose with the original collision meshes.

    Preserve all applicable SRDF exclusions and match the active loader's margin.
    Different collision geometry makes this a diagnostic, not a ground truth.
    """

    robot = p.loadURDF(
        str(urdf),
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION,
        physicsClientId=client,
    )

    try:
        infos = [
            p.getJointInfo(robot, j, physicsClientId=client)
            for j in range(p.getNumJoints(robot, physicsClientId=client))
        ]
        joints = {info[1].decode(): info[0] for info in infos}
        links = {info[12].decode(): info[0] for info in infos} | {"base_link": -1}

        for link in links.values():
            p.changeDynamics(robot, link, collisionMargin=5e-5, physicsClientId=client)

        for pair in ET.parse(srdf).getroot().findall("disable_collisions"):
            a, b = pair.get("link1"), pair.get("link2")

            if a in links and b in links:
                p.setCollisionFilterPair(
                    robot, robot, links[a], links[b], 0, physicsClientId=client
                )

        world = SimpleNamespace(
            robot=robot,
            client=client,
            link_index=links,
            arm_joints=[joints[j] for j in ARM],
            gripper_joint=joints["gripper"],
            tcp_link=links["gripper_frame_link"],
        )

        return static_pose(world, p, angles, 0)
    finally:
        p.removeBody(robot, physicsClientId=client)


def audit(source, prepared, output, repository):
    import pybullet as p
    from PIL import Image, ImageDraw

    runtime = repository / "simulation/runtime"
    loader = runtime / "reachability_eval/physics_world.py"
    assets = runtime / "calibration/manipulation_20261109_v1/assets"
    original = runtime / "reachability_eval/model/so101_new_calib.urdf"
    active = assets / "robot_windows.urdf"
    defs, base_defs = joint_definitions(active), joint_definitions(original)

    if read_json(prepared / "manifest.json")["status"] != "completed":
        raise ValueError("Use a completed preparation run")

    output.mkdir(parents=True, exist_ok=False)

    result = {
        "status": "running",
        "hardware_mapping_validated": False,
        "candidate": {
            "arm_signs": dict.fromkeys(ARM, 1),
            "arm_offsets_deg": dict.fromkeys(ARM, 0),
            "gripper_reference_rad": 0,
            "purpose": "direct-angle hypothesis only; not an operational calibration",
        },
        "assumptions": [
            "No physics steps; static inspection only",
            "Camera views for renders are illustrative, NOT registered to the physical camera",
            "Gripper fixed at 0 rad; its measured normalized opening has no validated URDF conversion",
            "Visual review cannot establish hidden joints or absolute pose accuracy",
            "Causal host-frame pairing does not correct unknown camera exposure latency",
        ],
        "source_hashes": {
            str(path.relative_to(repository)): sha256(path)
            for path in (loader, original, active, assets / "robot.srdf")
        },
        "script_sha256": sha256(Path(__file__)),
        "joint_definitions": defs,
        "original_joint_definitions": base_defs,
        "kinematic_changes": {
            j: [
                k
                for k in ("parent", "child", "origin", "axis")
                if defs[j].get(k) != base_defs[j].get(k)
            ]
            for j in (*ARM, "gripper")
        },
        "review_poses": [],
        "joint_test_poses": [],
    }

    write_json(output / "alignment.json", result)

    spec = importlib.util.spec_from_file_location("alignment_physics", loader)
    physics = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(physics)

    physics.ASSETS = assets  # Private imported module; no frozen file is changed.
    world = physics.PhysicsWorld.__new__(physics.PhysicsWorld)
    world.client, world.friction = p.connect(p.DIRECT), 0.6

    try:
        p.setPhysicsEngineParameter(
            contactBreakingThreshold=0.002, physicsClientId=world.client
        )
        world._load_robot()

        limits_count, collision_count, original_limits_count = Counter(), 0, Counter()
        ranges = {j: [math.inf, -math.inf] for j in ARM}
        trial_summaries, total = [], 0

        for path in sorted(prepared.glob("manual-recordings/*/trial-*/samples.jsonl")):
            rows = read_rows(path)
            trial = str(path.parent.relative_to(prepared / "manual-recordings"))
            raw_calibration = source / "manual-recordings" / trial / "calibration.json"
            c = read_json(raw_calibration)

            if any(
                c[j]["id"] != index + 1 or c[j]["drive_mode"] != 0
                for index, j in enumerate((*ARM, "gripper"))
            ):
                raise ValueError(
                    "This session audit expects the recorded IDs 1..6 and drive_mode=0"
                )

            invalid, collisions, deepest = Counter(), 0, 0

            for row in rows:
                q = row["arm_position_deg"]
                bad = out_of_limits(q, defs)

                limits_count.update(bad)
                invalid.update(bad)
                original_limits_count.update(out_of_limits(q, base_defs))

                for j in ARM:
                    ranges[j] = [min(ranges[j][0], q[j]), max(ranges[j][1], q[j])]

                state = static_pose(world, p, q, 0)

                # Report actual signed distances, not just a contact-manifold count.
                if state["max_penetration_mm"] > 1:
                    collisions += 1

                deepest = max(deepest, state["max_penetration_mm"])

            collision_count += collisions
            total += len(rows)

            trial_summaries.append(
                {
                    "trial": trial,
                    "samples": len(rows),
                    "calibration_sha256": sha256(raw_calibration),
                    "prepared_sha256": sha256(path),
                    "active_limit_violations": dict(invalid),
                    "samples_with_penetration_over_1mm": collisions,
                    "max_penetration_mm": deepest,
                }
            )

        result["all_recordings"] = {
            "samples": total,
            "observed_ranges_deg": ranges,
            "active_limit_violation_samples_by_joint": dict(limits_count),
            "original_limit_violation_samples_by_joint": dict(original_limits_count),
            "samples_with_penetration_over_1mm": collision_count,
            "trials": trial_summaries,
            "note": "Includes failed/setup/censored rows; gripper assumed zero; not a trajectory safety check",
        }

        for number, (trial, requested_time) in enumerate(REVIEW_POSES):
            rows = read_rows(prepared / "manual-recordings" / trial / "samples.jsonl")
            row = min(rows, key=lambda r: abs(r["time_s"] - requested_time))
            record = {
                "trial": trial,
                "requested_time_s": requested_time,
                "observation": row,
                "active_limit_violations": out_of_limits(row["arm_position_deg"], defs),
            }
            frame_images = {}

            for view in ("fixed", "wrist"):
                frame = row["frames"][view]

                if frame is None or not frame["fresh"]:
                    raise ValueError("Review pose has no fresh paired frame")

                video = source / "manual-recordings" / trial / view / "video.avi"
                target = output / f"pose_{number:02d}_{view}.png"

                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-i",
                        str(video),
                        "-vf",
                        f"select=eq(n\\,{frame['frame']})",
                        "-frames:v",
                        "1",
                        str(target),
                    ],
                    check=True,
                )

                frame_images[view] = Image.open(target).convert("RGB")
                record[view + "_video_sha256"] = sha256(video)

            record["static_diagnostic"] = static_pose(
                world, p, row["arm_position_deg"], 0
            )
            sim = render(world, p)

            sim.save(output / f"pose_{number:02d}_model.png")

            # Rotate only for readable inspection. Original extracted frames remain intact.
            fixed = frame_images["fixed"].transpose(Image.Transpose.ROTATE_90)

            fixed.thumbnail((450, 800))

            wrist = frame_images["wrist"]
            wrist.thumbnail((480, 270))

            panel = Image.new("RGB", (1050, 850), "white")

            panel.paste(fixed, (0, 45))
            panel.paste(sim, (450, 45))
            panel.paste(wrist, (510, 570))

            draw = ImageDraw.Draw(panel)

            draw.text(
                (10, 8),
                f"{trial} | observation {row['source_sample']} | t={row['time_s']:.3f} s",
                fill="black",
            )
            draw.text(
                (10, 25), "Recorded fixed view (rotated 90 degrees CCW)", fill="black"
            )
            draw.text(
                (465, 25),
                "Direct angles | illustrative camera, NOT an image overlay",
                fill="black",
            )
            draw.text(
                (510, 550),
                "Paired wrist view | simulated gripper opening NOT calibrated",
                fill="black",
            )
            panel.save(output / f"pose_{number:02d}_comparison.jpg")
            result["review_poses"].append(record)

        for path in sorted(source.glob("measurements/joint_response/*/metadata.json")):
            meta = read_json(path)
            q = {j: meta["initial_positions"][j] for j in ARM}
            # A moving-jaw angle cannot change a shoulder/fixed-jaw intersection.
            sweep = {str(g): static_pose(world, p, q, g) for g in (0, 0.3, 1.0)}

            result["joint_test_poses"].append(
                {
                    "test": path.parent.name,
                    "metadata_sha256": sha256(path),
                    "arm_position_deg": q,
                    "paired_video_available": False,
                    "gripper_sensitivity": sweep,
                }
            )

        static_pose(world, p, result["joint_test_poses"][0]["arm_position_deg"], 0)
        render(world, p).save(output / "folded_joint_test_pose.png")

        result["original_geometry_folded_pose"] = original_pose_probe(
            p,
            world.client,
            original,
            assets / "robot.srdf",
            result["joint_test_poses"][0]["arm_position_deg"],
        )
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        p.disconnect(world.client)
        write_json(output / "alignment.json", result)

    print(
        json.dumps(
            {
                "status": result["status"],
                "samples": total,
                "active_limit_violations": limits_count,
                "penetrating_samples": collision_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repository", type=Path, default=Path(__file__).resolve().parents[2]
    )

    args = parser.parse_args()

    audit(
        args.source.resolve(),
        args.prepared.resolve(),
        args.output.resolve(),
        args.repository.resolve(),
    )
