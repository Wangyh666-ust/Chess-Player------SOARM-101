"""Shared lens projection and pixel-space pose refinement; no hardware access."""

from pathlib import Path

import cv2
import numpy as np

from robot_calibration.records import read_json, sha256

CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-9)


def camera_parameters(calibration, image_size):
    """Validate the lens convention and exact unrotated image resolution."""
    model = calibration['model']
    if model == 'opencv_pinhole':
        model = 'brown5'
    if model not in {'brown5', 'radial1', 'radial2', 'none', 'fisheye2', 'fisheye4'}:
        raise ValueError(f'Unsupported lens model: {model}')
    if list(image_size) != calibration['image_size_wh']:
        raise ValueError('Resolution mismatch; do not resize/rotate inputs silently')

    matrix = np.asarray(calibration['camera_matrix'], dtype=float)
    distortion = np.asarray(calibration['distortion'], dtype=float)
    count = 4 if model.startswith('fisheye') else 5
    if matrix.shape != (3, 3) or distortion.shape != (count,):
        raise ValueError('Invalid camera matrix or distortion coefficient count')
    if not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError('Lens values must be finite')
    if min(matrix[0, 0], matrix[1, 1]) <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError('Invalid focal length or homogeneous camera matrix')
    if not np.allclose([matrix[0, 1], matrix[1, 0]], 0):
        raise ValueError('Only zero-skew camera models are supported')
    return matrix, distortion, model


def check_fit_target(candidate, target_path, points):
    """Use self-contained provenance or check legacy archived fitting observations."""
    digest = candidate.get('target_sha256')
    if digest is not None:
        if sha256(target_path) != digest:
            raise ValueError('Measured target changed since camera fitting')
        if 'source_image_sha256' in candidate:
            return candidate['source_image_sha256']
    source = candidate.get('source_observations')
    if source is None:
        return []
    views = read_json(Path(source))
    for view in views:
        if not set(view['marker_ids']).issubset(points):
            raise ValueError('Measured target changed since camera fitting')
        expected = [point for marker in view['marker_ids'] for point in points[marker]]
        if not np.allclose(expected, view['object_points_mm'], atol=1e-5, rtol=0):
            raise ValueError('Measured target changed since camera fitting')
    return [view['sha256'] for view in views]


def undistort_points(pixels, matrix, distortion, model):
    points = np.asarray(pixels, dtype=float).reshape(-1, 1, 2)
    if model.startswith('fisheye'):
        return cv2.fisheye.undistortPoints(points, matrix, distortion, criteria=CRITERIA)
    return cv2.undistortPointsIter(points, matrix, distortion, None, None, CRITERIA)


def write_lens_preview(source, output, matrix, distortion, model):
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f'Cannot read preview image: {source}')
    if model.startswith('fisheye'):
        corrected = cv2.fisheye.undistortImage(image, matrix, distortion, Knew=matrix)
    else:
        corrected = cv2.undistort(image, matrix, distortion)
    panel = cv2.resize(np.hstack([image, corrected]),
                       (1280, round(640 * image.shape[0] / image.shape[1])))
    if not cv2.imwrite(str(output), panel):
        raise OSError(f'Cannot write preview: {output}')


def project(objects, pose, matrix, distortion, model):
    """Project measured coordinates using the appropriate distortion convention."""
    function = cv2.fisheye.projectPoints if model.startswith('fisheye') else cv2.projectPoints
    return function(objects.reshape(-1, 1, 3), pose[:3], pose[3:],
                    matrix, distortion)[0].reshape(-1, 2)


def solve_pose(view, matrix, distortion, model):
    """Fit six pose parameters in pixel space, keeping the lens fixed.

    Both lens families use the same damped least-squares refinement so that
    normalized-coordinate PnP does not bias the pixel-error comparison.
    """
    objects = np.float64(view['object_points_mm'])
    pixels = np.float64(view['image_points_px'])
    normalized = undistort_points(pixels, matrix, distortion, model)
    ok, rotation, translation = cv2.solvePnP(objects, normalized, np.eye(3), None)
    if not ok:
        raise ValueError('Pose initialization failed')
    pose = np.concatenate([rotation.ravel(), translation.ravel()])
    return refine_pose(objects, pixels, pose, matrix, distortion, model)


def refine_pose(objects, pixels, pose, matrix, distortion, model, translation_step=1e-3):
    """Refine a supplied pose in original pixels; preserve the object-coordinate units."""
    damping = 1e-3

    for _ in range(60):
        residual = (project(objects, pose, matrix, distortion, model) - pixels).ravel()
        columns = []
        for j in range(6):
            step = 1e-6 if j < 3 else translation_step
            shifted = pose.copy()
            shifted[j] += step
            changed = (project(objects, shifted, matrix, distortion, model) - pixels).ravel()
            columns.append((changed - residual) / step)
        jacobian = np.column_stack(columns)
        normal = jacobian.T @ jacobian
        delta = np.linalg.solve(normal + damping * np.diag(np.maximum(np.diag(normal), 1e-9)),
                                -jacobian.T @ residual)
        candidate = pose + delta
        next_residual = (project(objects, candidate, matrix, distortion, model) - pixels).ravel()
        if next_residual @ next_residual < residual @ residual:
            pose = candidate
            damping = max(damping / 3, 1e-12)
            if np.linalg.norm(delta) < 1e-7:
                break
        else:
            damping *= 10

    if not np.isfinite(pose).all():
        raise ValueError('Nonfinite board pose')
    return pose


def pose_error(view, matrix, distortion, model):
    """Return pixel RMS after fitting only board pose with a fixed lens."""
    objects = np.float64(view['object_points_mm'])
    pixels = np.float64(view['image_points_px'])
    pose = solve_pose(view, matrix, distortion, model)
    errors = np.linalg.norm(project(objects, pose, matrix, distortion, model) - pixels, axis=1)
    if not np.isfinite(errors).all():
        raise ValueError('Nonfinite pose error')
    return float(np.sqrt(np.mean(errors ** 2)))
