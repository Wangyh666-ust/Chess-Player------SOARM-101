"""
Offline metric board pose from canonical ArUco corners and frozen intrinsics.

Board: origin at the outer grid corner by a1, +X toward h1, +Y toward a8,
+Z out of the playing surface. Camera: +X right, +Y down, +Z forward.
The returned T_camera_from_board maps board metres into camera metres.
No robot transform, square-centre assumptions or motion permission is inferred.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from robot_calibration.camera_calibration import collect_views, load_target
from robot_calibration.camera_geometry import camera_parameters, project, refine_pose, undistort_points, check_fit_target
from robot_calibration.records import read_json, sha256, write_json


def estimate_board_pose(marker_ids, corners_px, target_points_mm, calibration,
                        image_size, holdout_ids=()):
    """Estimate pose from raw canonical marker corners, excluding requested IDs.

    A detector can call this directly: corners have shape (markers, 4, 2).
    Unknown IDs are ignored and reported. Four known fitting markers are required.
    All fitting markers are retained; a bad residual is never silently discarded.
    """
    matrix, distortion, model = camera_parameters(calibration, image_size)
    ids = list(marker_ids)
    corners = np.asarray(corners_px, dtype=float)
    if len(ids) != len(set(ids)) or corners.shape != (len(ids), 4, 2):
        raise ValueError('Expected unique marker IDs and four canonical corners each')
    if not np.isfinite(corners).all():
        raise ValueError('Nonfinite image coordinates')
    if np.any(corners < 0) or np.any(corners >= np.array(image_size)):
        raise ValueError('Marker corner outside the image')

    known = [i for i, marker in enumerate(ids) if marker in target_points_mm]
    known_ids = [ids[i] for i in known]
    fit_ids = [i for i in known_ids if i not in holdout_ids]
    if len(fit_ids) < 4:
        raise ValueError('Need at least four fitting markers after reserving checks')

    objects = np.asarray([target_points_mm[i] for i in known_ids], dtype=float).reshape(-1, 3) / 1000
    pixels = corners[known].reshape(-1, 2)
    if not np.isfinite(objects).all() or not np.allclose(objects[:, 2], 0, atol=1e-9):
        raise ValueError('Expected finite, planar measured target geometry')
    fit = np.repeat([i in fit_ids for i in known_ids], 4)
    if np.linalg.matrix_rank(objects[fit, :2] - objects[fit, :2].mean(axis=0)) < 2:
        raise ValueError('Fitting coordinates do not span a board plane')

    normalized = undistort_points(pixels[fit], matrix, distortion, model)
    if not np.isfinite(normalized).all() or np.any(np.abs(normalized) > 1e5):
        raise ValueError('Lens inversion failed at a marker')

    count, rotations, translations, _ = cv2.solvePnPGeneric(
        objects[fit], normalized, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    if not count:
        raise ValueError('Planar pose initialization failed')
    candidates = []
    rejected = []

    for rotation, translation in zip(rotations, translations):
        pose = np.r_[rotation.ravel(), translation.ravel()]
        try:
            pose = refine_pose(objects[fit], pixels[fit], pose, matrix, distortion,
                               model, translation_step=1e-6)
            rotation_matrix = cv2.Rodrigues(pose[:3])[0]
            camera_points = objects @ rotation_matrix.T + pose[3:]
            camera_in_board = -rotation_matrix.T @ pose[3:]
            if np.any(camera_points[:, 2] <= 0) or camera_in_board[2] <= 0:
                raise ValueError('Pose places target behind camera or camera behind printed face')
            errors = np.linalg.norm(project(objects, pose, matrix, distortion, model) - pixels, axis=1)
            if not np.isfinite(errors).all():
                raise ValueError('Nonfinite reprojection residual')
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = rotation_matrix, pose[3:]
            candidates.append({'T_camera_from_board': transform.tolist(),
                               'rvec_rad': pose[:3].tolist(), 'translation_m': pose[3:].tolist(),
                               'fit_rms_px': float(np.sqrt(np.mean(errors[fit] ** 2)))})
        except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
            rejected.append(str(error))

    if not candidates:
        raise ValueError('No physically consistent pose: ' + '; '.join(rejected))
    candidates.sort(key=lambda c: c['fit_rms_px'])
    best = candidates[0]
    pose = np.r_[best['rvec_rad'], best['translation_m']]
    errors = np.linalg.norm(project(objects, pose, matrix, distortion, model) - pixels, axis=1)
    per_marker = [{'id': marker, 'used_for_pose': marker in fit_ids,
                   'rms_px': float(np.sqrt(np.mean(errors[4*i:4*i+4] ** 2)))}
                  for i, marker in enumerate(known_ids)]
    held_out = [m for m in per_marker if not m['used_for_pose']]

    return {'status': 'estimated_not_metric_validated', 'motion_ready': False,
            **best, 'translation_units': 'm', 'rotation_units': 'rad',
            'fit_marker_ids': fit_ids, 'holdout_marker_ids': [m['id'] for m in held_out],
            'missing_holdout_ids': sorted(set(holdout_ids) - set(known_ids)),
            'unknown_ids': sorted(set(ids) - set(known_ids)), 'marker_errors': per_marker,
            'holdout_rms_px': float(np.sqrt(np.mean([m['rms_px'] ** 2 for m in held_out]))) if held_out else None,
            'pose_candidates': candidates, 'rejected_candidates': rejected,
            'fit_image_hull_fraction': float(cv2.contourArea(cv2.convexHull(
                pixels[fit].astype('float32'))) / (image_size[0] * image_size[1]))}


def draw_overlay(frame, result, points_mm, calibration):
    """Draw projected measured markers and 30 mm board axes on the raw image."""
    matrix, distortion, model = camera_parameters(calibration, (frame.shape[1], frame.shape[0]))
    pose = np.r_[result['rvec_rad'], result['translation_m']]
    for marker in result['marker_errors']:
        points = np.asarray(points_mm[marker['id']], dtype=float) / 1000
        pixels = project(points, pose, matrix, distortion, model)
        colour = (0, 255, 0) if marker['used_for_pose'] else (255, 0, 255)
        cv2.polylines(frame, [np.int32(pixels)], True, colour, 2)
        label = f"{marker['id']} {marker['rms_px']:.1f}px"
        cv2.putText(frame, label, tuple(np.int32(pixels.mean(axis=0))),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, colour, 1)

    axes = np.array([[0., 0, 0], [.03, 0, 0], [0, .03, 0], [0, 0, .03]])
    pixels = project(axes, pose, matrix, distortion, model)
    for i, (label, colour) in enumerate([('X', (0, 0, 255)), ('Y', (0, 255, 0)),
                                        ('Z', (255, 0, 0))], start=1):
        cv2.line(frame, tuple(np.int32(pixels[0])), tuple(np.int32(pixels[i])), colour, 2)
        cv2.putText(frame, label, tuple(np.int32(pixels[i])), cv2.FONT_HERSHEY_SIMPLEX, .7, colour, 2)
    cv2.putText(frame, 'Green: pose markers | Magenta: reserved checks | Axes: 30mm',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    return frame


def main():
    """Detect saved photos through the existing calibration adapter, then estimate poses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--target', type=Path, default=Path('robot_calibration/configs/board_target.json'))
    parser.add_argument('--holdout-ids', type=int, nargs='+', default=[8, 9])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    candidate = read_json(args.candidate)
    target, points = load_target(args.target)
    if set(args.holdout_ids) - set(points):
        raise ValueError('Reserved marker ID not in measured target')
    check_fit_target(candidate, args.target, points)
    size, views = collect_views(args.images, target, points)
    camera_parameters(candidate, size)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'observations.json', views)
    results = []

    for index, view in enumerate(views):
        result = {'image': view['path'], 'sha256': view['sha256']}
        try:
            result.update(estimate_board_pose(view['marker_ids'],
                          np.array(view['image_points_px']).reshape(-1, 4, 2), points,
                          candidate, size, args.holdout_ids))
            frame = cv2.imread(view['path'])
            frame = draw_overlay(frame, result, points, candidate)
            cv2.imwrite(str(args.output / f'pose_{index:02d}.jpg'), frame)
        except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
            result.update(status='failed', motion_ready=False, error=str(error))
        results.append(result)

    write_json(args.output / 'board_poses.json', {
        'camera': candidate['camera'], 'model': candidate['model'],
        'candidate_sha256': sha256(args.candidate), 'target_sha256': sha256(args.target),
        'board_frame': target['coordinate_frame'],
        'camera_frame': {'x': 'image right', 'y': 'image down', 'z': 'forward'},
        'transform_convention': 'p_camera_m = T_camera_from_board @ p_board_m (homogeneous)',
        'holdout_requested': args.holdout_ids, 'views': results,
        'limitations': ['Lens candidates and ruler target are not independent metric ground truth.',
                        'Planar alternative solutions are retained; no automatic ambiguity approval.',
                        'No robot/base transform, square centres or motion readiness is supplied.']})
    print(f"{candidate['camera']}: {sum(r['status'] != 'failed' for r in results)}/{len(results)} poses estimated")


if __name__ == '__main__':
    main()
