"""Fit provisional camera intrinsics from photos of the measured ArUco board.

Only fixed/wrist images are accepted. No hardware is opened and no recognition
configuration is changed. Validation images never contribute to the lens fit.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from robot_calibration.camera_geometry import write_lens_preview
from robot_calibration.records import read_json, sha256, write_json


def load_target(path):
    """Load measured canonical corners; refuse unverified or malformed geometry."""
    target = read_json(path)
    if target['units'] != 'mm':
        raise ValueError('Target coordinates must be in mm')

    points = {}
    for marker in target['markers']:
        corners = np.asarray(marker['corners_mm'], dtype=np.float32)
        if not marker['corner_order_verified'] or corners.shape != (4, 3):
            raise ValueError('Each marker needs four verified canonical corners')
        if not np.isfinite(corners).all() or np.any(corners[:, 2] != 0):
            raise ValueError('Expected finite planar target coordinates')
        if marker['id'] in points:
            raise ValueError('Duplicate target marker ID')

        points[marker['id']] = corners

    return target, points


def marker_detector(target, *, adaptive_max=None):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, target['marker_dictionary']))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if adaptive_max is not None:
        parameters.adaptiveThreshWinSizeMax = adaptive_max
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def collect_views(directory, target, points):
    """Detect raw pixel corners in original images, without legacy radial warping."""
    detector = marker_detector(target)
    views = []
    size = None

    for path in sorted(directory.rglob('*')):
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue

        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f'Cannot read {path}')
        current_size = (frame.shape[1], frame.shape[0])
        if size is not None and current_size != size:
            raise ValueError('Do not mix camera resolutions in one calibration')
        size = current_size
        corners, ids, _ = detector.detectMarkers(frame)
        matched = {} if ids is None else {
            int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)
            if int(i) in points}
        if ids is not None and len(set(ids.flatten())) != len(ids):
            raise ValueError(f'Duplicate decoded marker in {path}')

        marker_ids = sorted(matched)
        views.append({
            'path': str(path.resolve()), 'sha256': sha256(path),
            'marker_ids': marker_ids, 'usable': len(marker_ids) >= 4,
            'object_points_mm': [p.tolist() for i in marker_ids for p in points[i]],
            'image_points_px': [p.tolist() for i in marker_ids for p in matched[i]],
        })

    if not views:
        raise ValueError('No input images found')
    return size, views


def fit_intrinsics(views, size):
    """Fit the five-coefficient OpenCV pinhole model on training views only."""
    objects = [np.float32(v['object_points_mm']) for v in views]
    pixels = [np.float32(v['image_points_px']) for v in views]
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        objects, pixels, size, None, None)

    if not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError('Nonfinite calibration')
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError('Invalid focal length')
    return rms, matrix, distortion


def validate_intrinsics(view, matrix, distortion):
    """Fit only board pose in a held-out view and measure pixel residuals."""
    objects = np.float32(view['object_points_mm'])
    pixels = np.float32(view['image_points_px'])
    ok, rotation, translation = cv2.solvePnP(objects, pixels, matrix, distortion)
    if not ok:
        raise ValueError(f"Pose fit failed: {view['path']}")

    projected, _ = cv2.projectPoints(objects, rotation, translation, matrix, distortion)
    errors = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
    return {'rms_px': float(np.sqrt(np.mean(errors ** 2))),
            'max_px': float(errors.max())}


def main():
    """Write a candidate fit, auditable observations and undistorted previews."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', choices=['fixed', 'wrist'], required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--target', type=Path,
                        default=Path('robot_calibration/configs/board_target.json'))
    parser.add_argument('--validation', nargs='+', required=True,
                        help='Exact image filenames reserved before fitting')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    target, points = load_target(args.target)
    size, views = collect_views(args.images, target, points)
    write_json(args.output / 'observations.json', views)
    held_out = set(args.validation)
    available = [Path(v['path']).name for v in views]
    if len(set(available)) != len(available):
        raise ValueError('Image basenames must be unique')
    if not held_out.issubset(available):
        raise ValueError('Validation filenames not found')

    train = [v for v in views if v['usable'] and Path(v['path']).name not in held_out]
    validation = [v for v in views if v['usable'] and Path(v['path']).name in held_out]
    if len(train) < 4 or len(validation) != len(held_out):
        raise ValueError('Need >=4 training views and all validation views with >=4 markers')

    rms, matrix, distortion = fit_intrinsics(train, size)
    results = []
    for index, view in enumerate(train + validation):
        results.append({'path': view['path'],
                        'split': 'training' if index < len(train) else 'validation',
                        **validate_intrinsics(view, matrix, distortion)})
        write_lens_preview(view['path'], args.output / f'preview_{index:02d}.jpg',
                           matrix, distortion, 'brown5')

    report = {
        'status': 'candidate_not_approved', 'motion_ready': False,
        'camera': args.camera, 'image_size_wh': size,
        'model': 'brown5', 'distortion_order': ['k1', 'k2', 'p1', 'p2', 'k3'],
        'camera_matrix': matrix.tolist(), 'distortion': distortion.flatten().tolist(),
        'training_rms_px': rms, 'views': results,
        'source_image_sha256': [view['sha256'] for view in views],
        'target_path': str(args.target.resolve()), 'target_sha256': sha256(args.target),
        'opencv_version': cv2.__version__,
        'limitations': [
            'Marker ruler precision is approximately +/-0.5 mm, not a statistical sigma.',
            'Held-out images test intrinsics; each board pose is fitted in its own image.',
            'Focus/crop/device identity and independent metric accuracy are unverified.',
            'No lens monotonicity or repeatability approval; inspect previews before use.',
            'Do not combine this correction with the old recognition radial correction.',
        ],
    }
    write_json(args.output / 'intrinsics.json', report)
    print(f'{args.camera}: {len(train)} training, {len(validation)} validation; RMS {rms:.3f}px')
    for result in results[len(train):]:
        print(f"  held out {Path(result['path']).name}: {result['rms_px']:.3f}px")


if __name__ == '__main__':
    main()
