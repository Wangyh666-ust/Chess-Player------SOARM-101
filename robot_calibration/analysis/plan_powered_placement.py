"""Preview a single board target offline from a recorded placement-session hold."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from manipulation.calibration import arm_from_model, arm_to_model
from robot_calibration.kinematics import URDF, Robot, board_landmark, fixed_tip_point, gripper_link_pose
from robot_calibration.records import read_json, sha256

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('hold', 'profile', 'tip', 'scene', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--square', required=True)
    parser.add_argument('--height-mm', required=True, type=float)
    parser.add_argument('--height-calibration', type=Path,
                        help='Local powered-hover calibration. Interpret height as estimated physical mm.')
    args = parser.parse_args()
    if not re.fullmatch('[a-h][1-8]', args.square) or not np.isfinite(args.height_mm) or args.height_mm <= 60:
        parser.error('Use a board square and a finite modeled height above 60 mm')
    hold, profile, tip_record, scene = [read_json(p) for p in
                                        (args.hold, args.profile, args.tip, args.scene)]
    initial = hold['initial']
    board = np.asarray(hold['T_base_from_board'])
    pitch_xy = np.asarray(scene['grid']['provisional_cell_pitch_xy_m'])
    model_height = args.height_mm
    tip = fixed_tip_point(tip_record)
    robot = Robot(str(URDF))
    start = np.asarray(arm_to_model(profile, initial))
    bias = 0.
    if args.height_calibration:
        calibration = read_json(args.height_calibration)
        files = dict(profile=args.profile, tip=args.tip, scene=args.scene,
                     urdf=URDF)
        hashes = {name: sha256(path) for name, path in files.items()}
        if (calibration['schema'] != 'powered-fixed-tip-height-v1' or calibration['hashes'] != hashes
                or calibration['hardware'] != hold['hardware']
                or calibration['T_base_from_board'] != hold['T_base_from_board']
                or initial['gripper'] != calibration['gripper']):
            parser.error('Height calibration belongs to a different setup')
        limits = calibration['square_height_range_mm'].get(args.square)
        if not limits or not limits[0] <= args.height_mm <= limits[1]:
            parser.error('Physical height outside this square’s calibration range')
        for value, key in [(np.rad2deg(sum(start[1:4])), 'model_pitch_sum_range_deg'),
                           (np.rad2deg(start[4]), 'model_roll_range_deg')]:
            low, high = calibration[key]
            if not low <= value <= high:
                parser.error('Tool orientation outside the height calibration range')
        bias = calibration['encoder_minus_ruler_mm']
        model_height += bias
    goal = board_landmark(args.square, pitch_xy, model_height / 1000)[:3]
    inverse_board = np.linalg.inv(board)

    def tip_board(q):
        return (inverse_board @ gripper_link_pose(robot, q) @ tip)[:3]

    def residual(x):
        return np.r_[1000 * (tip_board(np.r_[x, start[4]]) - goal),
                     100 * (sum(x[1:4]) - sum(start[1:4]))]

    fit = least_squares(residual, start[:4], bounds=(robot.lo[:4], robot.hi[:4]))
    end = np.r_[fit.x, start[4]]
    error = float(np.linalg.norm(tip_board(end) - goal) * 1000)
    target = arm_from_model(profile, end)
    target.update({j: initial[j] for j in ('wrist_roll', 'gripper')})
    path = [dict(tip_board_m=tip_board(q).tolist(), joint_margin_deg=float(robot.margin(q)))
            for q in np.linspace(start, end, 101)]
    clearance = 1000 * min(row['tip_board_m'][2] for row in path)
    travel = max(abs(target[j] - initial[j]) for j in target)
    if not (fit.success and error < .3 and travel < 12 and clearance - bias > 60
            and min(row['joint_margin_deg'] for row in path) > 0):
        parser.error('Path exceeds placement limits. Choose a nearer hover.')
    plan = dict(hold, target=target, square=args.square, requested_height_mm=model_height,
                path=path, minimum_modeled_tip_height_mm=clearance, ik_error_mm=error)
    if args.height_calibration:
        plan.update(requested_physical_height_mm=args.height_mm, encoder_minus_ruler_mm=bias,
                    minimum_estimated_physical_height_mm=clearance - bias,
                    height_calibration_sha256=sha256(args.height_calibration))
    with args.output.open('x') as f:
        json.dump(plan, f, indent=2, allow_nan=False)
    print(f'{args.square}: modeled minimum tip height {clearance:.2f} mm, '
          f'maximum travel {travel:.2f} degrees. Review the whole-arm path before use.')


if __name__ == '__main__':
    main()
