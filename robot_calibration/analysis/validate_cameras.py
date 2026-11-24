"""Audit saved camera fits without installing or approving calibration values."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from robot_calibration.camera_calibration import fit_intrinsics, validate_intrinsics
from robot_calibration.records import read_json, sha256, write_json


def project_view(view, matrix, distortion, excluded_marker=None):
    """Estimate pose, optionally withholding one whole marker from pose fitting."""
    objects = np.float64(view['object_points_mm'])
    pixels = np.float64(view['image_points_px'])
    mask = np.ones(len(objects), dtype=bool)
    if excluded_marker is not None:
        mask[4 * excluded_marker:4 * excluded_marker + 4] = False

    normalized = cv2.undistortPointsIter(
        pixels[mask, None], matrix, distortion, None, None,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10))
    ok, rotation, translation = cv2.solvePnP(
        objects[mask], normalized, np.eye(3), None)
    if not ok:
        raise ValueError('Board pose could not be fitted')

    rotation, translation = cv2.solvePnPRefineLM(
        objects[mask], pixels[mask], matrix, distortion, rotation, translation)
    projected, _ = cv2.projectPoints(objects, rotation, translation, matrix, distortion)
    return projected.reshape(-1, 2)


def audit_view(view, matrix, distortion, output):
    """Save corner overlays and residuals, retaining every detected marker."""
    pixels = np.float64(view['image_points_px'])
    projected = project_view(view, matrix, distortion)
    errors = np.linalg.norm(projected - pixels, axis=1)
    frame = cv2.imread(view['path'])
    markers = []

    for index, marker_id in enumerate(view['marker_ids']):
        section = slice(index * 4, index * 4 + 4)
        excluded_projection = project_view(view, matrix, distortion, index)
        withheld_errors = np.linalg.norm(excluded_projection[section] - pixels[section], axis=1)
        markers.append({'id': marker_id,
                        'rms_px': float(np.sqrt(np.mean(errors[section] ** 2))),
                        'withheld_from_pose_rms_px': float(np.sqrt(np.mean(withheld_errors ** 2)))})
        position = tuple(np.int32(pixels[section].mean(axis=0)))
        cv2.putText(frame, str(marker_id), position, cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (0, 255, 255), 2)

    for observed, predicted in zip(pixels, projected):
        a, b = tuple(np.int32(observed)), tuple(np.int32(predicted))
        cv2.circle(frame, a, 4, (0, 255, 0), 1)
        cv2.drawMarker(frame, b, (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
        cv2.line(frame, a, b, (0, 255, 255), 1)

    cv2.putText(frame, 'Green: detected | Red: predicted | Yellow: ID/residual',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    cv2.imwrite(str(output), frame)
    return {'path': view['path'], 'rms_px': float(np.sqrt(np.mean(errors ** 2))),
            'max_px': float(errors.max()), 'markers': markers}


def distortion_check(matrix, distortion, size):
    """Check inverse/forward consistency on a pixel grid, including image edges.

    This numerical screen detects problematic extrapolation; it is not a proof
    of a globally invertible lens model or independent optical accuracy.
    """
    x, y = np.meshgrid(np.linspace(0, size[0] - 1, 33),
                       np.linspace(0, size[1] - 1, 25))
    pixels = np.column_stack([x.ravel(), y.ravel()])
    normalized = cv2.undistortPointsIter(
        pixels[:, None], matrix, distortion, None, None,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10))[:, 0]
    rays = np.column_stack([normalized, np.ones(len(normalized))])
    projected, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), matrix, distortion)
    errors = np.linalg.norm(projected[:, 0] - pixels, axis=1)
    return {'grid_points': len(pixels), 'roundtrip_max_px': float(errors.max()),
            'roundtrip_over_1px': int(np.count_nonzero(errors > 1))}


def audit_camera(directory, output):
    """Check source integrity, original fit residuals and leave-one-view-out stability."""
    report = read_json(directory / 'intrinsics.json')
    views = read_json(directory / 'observations.json')
    views = [v for v in views if v['usable']]
    if sha256(Path(report['target_path'])) != report['target_sha256']:
        # Metadata may have changed without altering coordinates. Compare those explicitly.
        target = read_json(Path(report['target_path']))
        corners = {m['id']: m['corners_mm'] for m in target['markers']}
        for view in views:
            expected = [point for i in view['marker_ids'] for point in corners[i]]
            if not np.allclose(expected, view['object_points_mm'], atol=1e-5, rtol=0):
                raise ValueError('Target geometry changed since fitting')

    for view in views:
        if sha256(Path(view['path'])) != view['sha256']:
            raise ValueError(f"Source image changed: {view['path']}")

    matrix = np.array(report['camera_matrix'])
    distortion = np.array(report['distortion'])
    size = tuple(report['image_size_wh'])
    output.mkdir(parents=True, exist_ok=False)
    residuals = [audit_view(v, matrix, distortion, output / f'corners_{i:02d}.jpg')
                 for i, v in enumerate(views)]
    folds = []

    for index, view in enumerate(views):
        training = views[:index] + views[index + 1:]
        try:
            rms, fitted_matrix, fitted_distortion = fit_intrinsics(training, size)
            folds.append({'excluded_image': view['path'], 'training_rms_px': rms,
                          'camera_matrix': fitted_matrix.tolist(),
                          'distortion': fitted_distortion.flatten().tolist(),
                          **validate_intrinsics(view, fitted_matrix, fitted_distortion)})
        except (cv2.error, ValueError) as error:
            folds.append({'excluded_image': view['path'], 'error': str(error)})

    successful = [f for f in folds if 'error' not in f]
    matrices = np.array([f['camera_matrix'] for f in successful])
    summary = {}
    if successful:
        for name, row, col in [('fx', 0, 0), ('fy', 1, 1), ('cx', 0, 2), ('cy', 1, 2)]:
            values = matrices[:, row, col]
            summary[name + '_range_px'] = [float(values.min()), float(values.max())]
        summary['held_out_rms_range_px'] = [min(f['rms_px'] for f in successful),
                                             max(f['rms_px'] for f in successful)]

    result = {'camera': report['camera'], 'status': 'diagnostic_only_not_approved',
              'source_fit_sha256': sha256(directory / 'intrinsics.json'),
              'current_target_sha256': sha256(Path(report['target_path'])),
              'original_fit_views': residuals, 'leave_one_view_out': folds,
              'stability_summary': summary,
              'distortion_screen': distortion_check(matrix, distortion, size),
              'limitations': [
                  'Cross-validation reuses this capture session, not an independent final test.',
                  'Marker exclusion is from pose only; lens fitting may have used that marker.',
                  'No independent millimetre accuracy or robot-to-board validation.',
                  'No automatic acceptance threshold has been established.',
              ]}
    write_json(output / 'validation.json', result)
    print(report['camera'], summary, result['distortion_screen'])


def main():
    """Run against one saved fit directory and write fresh local diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit_camera(args.fit, args.output)


if __name__ == '__main__':
    main()
