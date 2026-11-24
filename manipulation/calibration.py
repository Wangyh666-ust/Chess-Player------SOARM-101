"""Offline use of the provisional calibration delivery; no hardware connection.

Conversions are numerical, not motion approval. Scene values describe only their
recorded capture. The five arm radians do not constitute a six-joint ACT state.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from manipulation.observations import rigid_transform, robot_from_board, task_context
from robot_calibration.records import ARM, read_json, sha256


def mapping_arrays(profile):
    if profile['joint_order'] != list(ARM):
        raise ValueError('Unexpected arm joint order')
    rows = [profile['joint_mapping']['joints'][joint] for joint in ARM]
    signs, scales, offsets = np.asarray([
        [r['sign'], r['scale'], r['offset_deg']] for r in rows
    ], dtype=float).T
    if (not np.isfinite([signs, scales, offsets]).all()
            or not np.isin(signs, [-1, 1]).all() or np.any(scales <= 0)):
        raise ValueError('Invalid joint mapping')
    return signs * scales, offsets


def arm_to_model(profile, hardware_degrees):
    """Named calibrated hardware degrees -> five model radians, in ARM order."""
    gains, offsets = mapping_arrays(profile)
    values = np.asarray([hardware_degrees[j] for j in ARM], dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError('Expected five finite arm degrees')
    return np.deg2rad(gains * values + offsets).tolist()


def arm_ticks_to_degrees(calibration, ticks):
    """Convert recorded STS3215 arm ticks to LeRobot degree coordinates."""
    degrees = {}
    for joint in ARM:
        motor = calibration[joint]
        midpoint = (motor['range_min'] + motor['range_max']) / 2
        degrees[joint] = (ticks[joint] - midpoint) * 360 / 4095
    return degrees


def encoder_act_state(profile, calibration, ticks):
    """Raw Present_Position -> five absolute model radians and physical jaw units."""
    degrees = arm_ticks_to_degrees(calibration, ticks)
    jaw = calibration['gripper']
    lo, hi = jaw['range_min'], jaw['range_max']
    if jaw['drive_mode'] != 0 or not lo <= ticks['gripper'] <= hi or hi <= lo:
        raise ValueError('Unsupported jaw calibration or out-of-range encoder reading')
    return [*arm_to_model(profile, degrees), (ticks['gripper'] - lo) * 100 / (hi - lo)]


def arm_from_model(profile, model_radians):
    """Numerical inverse only; returned degrees are not reviewed motor targets."""
    gains, offsets = mapping_arrays(profile)
    values = np.asarray(model_radians, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError('Expected five finite model radians; gripper is separate')
    return dict(zip(ARM, ((np.rad2deg(values) - offsets) / gains).tolist()))


def scene_geometry(session):
    """Check all saved transform chains and retain the explicit uniform grid.

    The delivery has unresolved seam geometry. Do not silently apply the
    simulator's separate seam convention to this provisional uniform pitch.
    """
    if session['translation_units'] != 'm':
        raise ValueError('Scene translations must be metres')
    corner = rigid_transform(session['T_base_from_board_corner'])
    shift = rigid_transform(session['T_board_corner_from_board_a1'])
    base_a1 = rigid_transform(session['T_base_from_board_a1'])
    camera_corner = rigid_transform(session['T_fixed_camera_from_board_corner'])
    camera_a1 = rigid_transform(session['T_fixed_camera_from_board_a1'])
    chain = robot_from_board(session['T_fixed_camera_from_base'], camera_a1)
    sizes = np.asarray(session['grid']['provisional_cell_pitch_xy_m'], dtype=float)
    if sizes.shape != (2,) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError('Invalid metre-valued cell pitch')
    expected_shift = np.eye(4)
    expected_shift[:2, 3] = sizes / 2
    if not all(np.allclose(a, b, rtol=0, atol=1e-8) for a, b in (
        (shift, expected_shift), (corner @ shift, base_a1),
        (camera_corner @ shift, camera_a1), (chain, base_a1),
    )):
        raise ValueError('Inconsistent board origin or camera/base transform chain')
    return base_a1.tolist(), sizes.tolist()


def context_from_session(session, board, source, destination):
    """Existing manipulation task context for offline recorded-scene inspection."""
    transform, sizes = scene_geometry(session)
    return task_context(board, transform, sizes, source, destination)


def jaw_gap_m(profile, normalized, depth_mm):
    """Passive measured-opening estimate, only at measured depths and in range."""
    candidate = profile['gripper_geometry']['post_reseat_opening_map']['candidate']
    lo, hi = candidate['normalized_domain']
    if not np.isfinite(normalized) or not lo <= normalized <= hi:
        raise ValueError('Opening outside calibrated interpolation domain')
    for row in candidate['rows']:
        if row['depth_from_extreme_fixed_tip_mm'] == depth_mm:
            return (row['slope_mm_per_normalized_unit'] * normalized
                    + row['intercept_mm']) / 1000
    raise ValueError('No measured opening map at this depth')


def fixed_tip_camera_bias(correction):
    """Explicit local empirical correction; never infer it from a joint profile."""
    if (correction['status'] != 'local_empirical_camera_plane_bias_with_pose3_excluded_check'
            or correction['frame'] != 'OpenCV fixed camera'):
        raise ValueError('Expected the local fixed-tip camera-plane correction')
    bias = np.asarray(correction['camera_bias_m'], dtype=float)
    if bias.shape != (3,) or not np.isfinite(bias).all() or bias[2] != 0:
        raise ValueError('Expected finite camera XY bias with zero depth correction')
    return bias


def corrected_fixed_tip_camera(nominal_tip_camera_m, correction):
    """Offline estimated physical tip; only the correction record's local scope."""
    point = np.asarray(nominal_tip_camera_m, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Expected finite camera-frame fingertip metres')
    return (point + fixed_tip_camera_bias(correction)).tolist()


def compensate_fixed_tip_target(desired_tip_base_m, camera_from_base, correction):
    """Offline nominal IK target, subtracting the empirical physical-tip bias.

    Requires a current camera/base transform and the same fixed fingertip,
    optics and local operating region as the correction. Not a gripper-centre
    correction, collision check, IK solution or permission to send a command.
    """
    point = np.asarray(desired_tip_base_m, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Expected finite base-frame fingertip metres')
    rotation = rigid_transform(camera_from_base)[:3, :3]
    return (point - rotation.T @ fixed_tip_camera_bias(correction)).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', required=True, type=Path)
    parser.add_argument('--capture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    paths = [args.delivery / 'profile.json', args.delivery / 'session.json', args.capture]
    profile, session, capture = map(read_json, paths)
    transform, sizes = scene_geometry(session)
    converted = []
    for sample in capture['samples']:
        hardware = sample['positions']
        model = arm_to_model(profile, hardware)
        inverse = arm_from_model(profile, model)
        converted.append(dict(image=sample['image'], arm_model_rad=model,
                              gripper_normalized=hardware['gripper'],
                              roundtrip_max_deg=max(abs(inverse[j] - hardware[j]) for j in ARM)))
    if not converted:
        raise ValueError('Capture contains no samples')
    checks = profile['gripper_geometry']['post_reseat_opening_map']['independent_check']
    errors = [jaw_gap_m(profile, checks['gripper_normalized_mean'],
                        row['depth_from_extreme_fixed_tip_mm']) * 1000 - row['measured_gap_mm']
              for row in checks['rows']]
    squares = {}
    for rank in '12345678':
        for file in 'abcdefgh':
            square = file + rank
            board = [1] * 64
            destination = 'h8' if square != 'h8' else 'a1'
            board[(int(destination[1]) - 1) * 8 + ord(destination[0]) - ord('a')] = 0
            squares[square] = context_from_session(session, board, square, destination)['source_xyz_m']
    result = dict(
        status='offline_conversion_checked_not_physical_validation', motion_ready=False,
        inputs_sha256={str(path): sha256(path) for path in paths},
        arm_joint_order=list(ARM), converted_capture=converted,
        maximum_arm_roundtrip_error_deg=max(r['roundtrip_max_deg'] for r in converted),
        scene_source=session['source'], T_base_from_board_a1=transform, cell_size_m=sizes,
        square_centres_base_m=squares, independent_gap_errors_mm=errors,
        limitations=[
            'Scene is historical and is NOT paired with the separately converted capture.',
            'Uniform pitch retained; seam and absolute height/orientation remain provisional.',
            'Gripper normalized reading is not a model angle; no complete ACT state generated.',
            'No physical accuracy, collision clearance, or current scene readiness established.',
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), samples=len(converted),
                         roundtrip_max_deg=result['maximum_arm_roundtrip_error_deg'],
                         independent_gap_errors_mm=errors)))


if __name__ == '__main__':
    main()
