"""Access the frozen reachability model and its named gripper frames."""

from pathlib import Path
import sys

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / 'simulation/runtime/reachability_eval'
URDF = MODEL_DIR / 'model/so101_new_calib.urdf'
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from core import Robot
from path_ik import pose_fk, solve_pose


def gripper_link_pose(robot, angles):
    """Return base-from-gripper_link, including for the jogger's wrist-axis model."""
    return pose_fk(robot, angles) @ np.linalg.inv(robot.joints[-1]['origin'])


def fixed_tip_point(record):
    """Read a gripper_link point in metres, accepting the two historical exports."""
    if 'point_m' in record:
        if record['frame'] != 'gripper_link':
            raise ValueError('Fixed fingertip must be expressed in gripper_link')
        point = np.asarray(record['point_m'], dtype=float)
    elif 'point_gripper_link_mm' in record:
        point = np.asarray(record['point_gripper_link_mm'], dtype=float) / 1000
    else:
        point = np.asarray(record['frozen_effective_tip_gripper_link_m'], dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Invalid fixed fingertip point')
    return np.r_[point, 1.]


def board_landmark(square, pitch_m, height_m, *, intersection=False):
    offset = 1. if intersection else .5
    return np.r_[np.asarray(pitch_m) * [ord(square[0]) - ord('a') + offset,
                                      int(square[1]) - 1 + offset], height_m, 1.]
