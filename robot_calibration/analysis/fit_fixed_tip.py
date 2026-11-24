"""Fit a fixed fingertip from jogger recordings and hold out whole intersections."""

import argparse
import json
from pathlib import Path

import numpy as np

from manipulation.calibration import arm_ticks_to_degrees, arm_to_model
from manipulation.observations import rigid_transform
from robot_calibration.collection.start_registration import board_landmark, level_pose
from robot_calibration.kinematics import URDF, Robot, gripper_link_pose
from robot_calibration.records import read_json, read_rows, sha256


MARK_EVENTS = {'board_contact', 'board_manual_hover'}


def fit(runs, review, *, legacy_pitch_m=None, legacy_height_m=None):
    robot = Robot(str(URDF))
    rows, observations = [], []
    seen_reviews = set()
    identity = None
    for run in runs:
        trace = read_rows(run / 'samples.jsonl')
        hardware, profile = None, read_json(run / 'metadata.json')['profile']
        for event in trace:
            if event['event'] == 'preflight':
                hardware = event['hardware']
            if event['event'] not in MARK_EVENTS:
                continue
            number, cell = event['number'], event['lower_left_cell']
            key = f'{run.name}/{number}'
            settings = review.get(key, {})
            if key in review:
                seen_reviews.add(key)
            registration = event['registration']
            row = dict(run=run.name, number=number, intersection=cell, status=registration['status'])
            rows.append(row)
            if 'exclude' in settings:
                row.update(status='excluded', reason=settings['exclude'])
                continue
            if registration['status'] != 'board_and_base':
                row['reason'] = registration.get('reason', 'No complete board-to-base observation')
                continue
            if hardware is None:
                raise ValueError(f'Missing hardware calibration for {key}')

            mark_identity = (profile, hardware['calibration'])
            if identity is None:
                identity = mark_identity
            if identity != mark_identity:
                raise ValueError('Use one arm profile and motor calibration per tip fit')

            if 'T_base_from_board' in registration and 'target_board_m' in event:
                board = rigid_transform(registration['T_base_from_board'])
                target = np.r_[event['target_board_m'], 1.]
            elif legacy_pitch_m is not None and legacy_height_m is not None:
                board = level_pose(registration['q'], legacy_height_m)
                target = board_landmark(cell, legacy_pitch_m, event['target_height_mm'] / 1000,
                                        intersection=True)
            else:
                raise ValueError(f'{key} needs explicit legacy cell pitch and board height')

            if 'measured_board_m' in settings:
                target = np.r_[settings['measured_board_m'], 1.]
            sigma = float(settings.get('uncertainty_mm', 1.))
            if target.shape != (4,) or not np.isfinite(target).all() or not 0 < sigma < np.inf:
                raise ValueError(f'Invalid measured position or uncertainty at {key}')
            angles = arm_to_model(profile, arm_ticks_to_degrees(hardware['calibration'], event['ticks']))
            tool = gripper_link_pose(robot, angles)
            tool_point = (np.linalg.solve(tool, board @ target))[:3]
            board_from_tool = np.linalg.solve(board, tool)
            observations.append((row, tool_point, board_from_tool, target[:3], 1 / sigma**2))
            row['uncertainty_mm'] = sigma

    if set(review) - seen_reviews:
        raise ValueError(f'Unknown review marks: {sorted(set(review) - seen_reviews)}')
    if len({row['intersection'] for row, *_ in observations}) < 2:
        raise ValueError('Need usable marks at two or more different intersections')

    points = np.array([point for _, point, *_ in observations])
    weights = np.array([weight for *_, weight in observations])
    groups = np.array([row['intersection'] for row, *_ in observations])
    point = np.average(points, axis=0, weights=weights)
    for row, _, board_from_tool, target, _ in observations:
        selected = groups != row['intersection']
        held_out = np.average(points[selected], axis=0, weights=weights[selected])
        fit_error = ((board_from_tool @ np.r_[point, 1.])[:3] - target) * 1000
        check_error = ((board_from_tool @ np.r_[held_out, 1.])[:3] - target) * 1000
        row.update(fit_error_mm=float(np.linalg.norm(fit_error)),
                   held_out_xyz_mm=check_error.tolist(),
                   held_out_error_mm=float(np.linalg.norm(check_error)))
    return dict(frame='gripper_link', point_m=point.tolist(), motion_ready=False,
                validation='leave one intersection out across all heights',
                usable_marks=len(observations), measurements=rows,
                inputs=[dict(run=run.name, samples_sha256=sha256(run / 'samples.jsonl'),
                             metadata_sha256=sha256(run / 'metadata.json')) for run in runs],
                urdf_sha256=sha256(URDF))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', nargs='+', type=Path, required=True)
    parser.add_argument('--review', type=Path, help='JSON keyed by run-directory/mark number')
    parser.add_argument('--legacy-pitch-m', type=float, nargs=2)
    parser.add_argument('--legacy-board-height-m', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pitch, height = args.legacy_pitch_m, args.legacy_board_height_m
    if ((pitch is None) != (height is None)
            or (pitch is not None and (not np.isfinite([*pitch, height]).all() or min(pitch) <= 0))):
        parser.error('Legacy inputs require finite board height and positive X/Y pitch in metres')
    result = fit(args.run, read_json(args.review) if args.review else {},
                 legacy_pitch_m=pitch, legacy_height_m=height)
    result['review_sha256'] = sha256(args.review) if args.review else None
    result['legacy_geometry'] = dict(cell_pitch_m=pitch, board_height_m=height) if pitch is not None else None
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
    print(f"Fit {result['usable_marks']} marks in gripper_link. Review held-out errors before use.")


if __name__ == '__main__':
    main()
