"""
Structured task context, independent of rendering and detector class IDs.

Board coordinates use the centre of a1 on the playing surface as origin,
+x toward h1 and +y toward a8. Positions are metres; yaw is radians.
The snapshot describes the board BEFORE the transfer, not the moving piece.
"""

import math

import numpy as np

PIECES = ("pawn", "knight", "bishop", "rook", "queen", "king")


def rigid_transform(value):
    """Validate a metre-valued SE(3) matrix; never silently repair calibration."""

    matrix = np.asarray(value, dtype=float)

    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 4 x 4 transform")

    rotation = matrix[:3, :3]

    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
    ):
        raise ValueError(
            "Transform must contain a proper rotation, not scale/reflection"
        )

    return matrix


def robot_from_board(camera_from_robot, camera_from_board):
    """Compose poses measured in the SAME fixed-camera frame and metric units."""

    return (
        np.linalg.inv(rigid_transform(camera_from_robot))
        @ rigid_transform(camera_from_board)
    ).tolist()


def square_index(square):
    """Map a1..h8 to 0..63, in rank-major order."""

    if not isinstance(square, str) or len(square) != 2:
        raise ValueError("Invalid square")

    if square[0] not in "abcdefgh" or square[1] not in "12345678":
        raise ValueError("Invalid square")

    return (int(square[1]) - 1) * 8 + ord(square[0]) - ord("a")


def task_context(
    board, transform, cell_size_m, source, destination, target_yaw_rad=None,
    seam_width_m=None,
):
    """
    Build numerical inputs from a complete, reviewed pre-transfer board.

    board has 64 codes: 0 empty, 1..6 white PIECES, 7..12 black PIECES.
    These are NOT the detector's class IDs: callers must map its class names.
    Square centres are nominal targets, not measured sub-square piece offsets.
    Unknown yaw is masked; [0, 0] is not a claim of zero-degree orientation.
    """

    matrix = rigid_transform(transform)
    sizes = np.asarray(cell_size_m, dtype=float)

    if sizes.shape != (2,) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError("Cell dimensions must be two positive metre values")

    if len(board) != 64 or any(
        type(code) is not int or not 0 <= code <= 12 for code in board
    ):
        raise ValueError(
            "Expected 64 known occupancy codes; unresolved squares need review"
        )

    src, dst = square_index(source), square_index(destination)

    if src == dst or board[src] == 0:
        raise ValueError(
            "Transfer needs an occupied source and a different destination"
        )

    if board[dst] != 0:
        raise ValueError("Remove a captured piece before collecting a single transfer")

    if target_yaw_rad is not None and (
        type(target_yaw_rad) not in (int, float) or not math.isfinite(target_yaw_rad)
    ):
        raise ValueError("Target yaw must be finite radians or unknown")

    centres = []

    if seam_width_m is not None and (
        type(seam_width_m) not in (int, float)
        or not math.isfinite(seam_width_m)
        or not 0 <= seam_width_m < sizes[1]
    ):
        raise ValueError("Seam must be finite, nonnegative and narrower than a cell")

    for index in (src, dst):
        board_point = [index % 8 * sizes[0], index // 8 * sizes[1], 0, 1]

        if seam_width_m is not None and index // 8 in (3, 4):
            board_point[1] += (-1 if index // 8 == 3 else 1) * seam_width_m / 4

        centres.append((matrix @ board_point)[:3].tolist())

    context = {
        "board": list(board),
        "robot_from_board": matrix.tolist(),
        "cell_size_m": sizes.tolist(),
        "source_square": source,
        "destination_square": destination,
        "source_xyz_m": centres[0],
        "destination_xyz_m": centres[1],
        "target_type": board[src],
        "target_yaw_sin_cos": (
            [math.sin(target_yaw_rad), math.cos(target_yaw_rad)]
            if target_yaw_rad is not None
            else [0.0, 0.0]
        ),
        "target_yaw_known": target_yaw_rad is not None,
    }

    if seam_width_m is not None:
        context["seam_width_m"] = seam_width_m

    return context


def simulation_context(world):
    """Use the initial square labels and board geometry; do not expose live poses."""

    origin = np.asarray(world.square_xyz("a1"))
    sizes = world.cfg.get("cell_size_m", [world.cfg["square_m"]] * 2)
    x = (np.asarray(world.square_xyz("b1")) - origin) / sizes[0]
    y = (np.asarray(world.square_xyz("a2")) - origin) / sizes[1]
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((x, y, np.cross(x, y)))
    transform[:3, 3] = origin
    board = [0] * 64

    for piece in world.case["pieces"]:
        index = square_index(piece["square"])

        if board[index]:
            raise ValueError("Duplicate occupied square")

        board[index] = PIECES.index(piece["type"]) + 1 + (0 if piece["color"] else 6)

    return task_context(
        board,
        transform,
        sizes,
        world.case["source"],
        world.case["dest"],
        seam_width_m=(world.cfg["board_model"]["dimensions"]["seam_mm"] / 1000
                      if "board_model" in world.cfg else None),
    )


def validate_context(context, task):
    """Reject inconsistent serialized geometry or task labels at the read boundary."""

    known = context["target_yaw_known"]
    yaw = context["target_yaw_sin_cos"]

    if type(known) is not bool or len(yaw) != 2 or not np.isfinite(yaw).all():
        raise ValueError("Invalid yaw encoding")

    if known and not np.isclose(sum(x * x for x in yaw), 1):
        raise ValueError("Known yaw must be a unit sine/cosine pair")

    rebuilt = task_context(
        context["board"],
        context["robot_from_board"],
        context["cell_size_m"],
        task["source_square"],
        task["destination_square"],
        math.atan2(*yaw) if known else None,
        seam_width_m=context.get("seam_width_m"),
    )

    if set(context) != set(rebuilt):
        raise ValueError("Unexpected structured context fields")

    for key, expected in rebuilt.items():
        actual = context[key]

        if isinstance(expected, list):
            matches = np.allclose(actual, expected, atol=1e-8, rtol=0)
        else:
            matches = actual == expected

        if not matches:
            raise ValueError(f"Inconsistent task context: {key}")
