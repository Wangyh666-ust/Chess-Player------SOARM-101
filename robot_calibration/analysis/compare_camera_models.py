"""Compare lens models on identical photo folds; never publish an approved fit."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from robot_calibration.analysis.validate_cameras import distortion_check
from robot_calibration.records import read_json, sha256, write_json

from robot_calibration.camera_geometry import CRITERIA, pose_error, write_lens_preview


def fit_model(views, size, model):
    """Fit a named model, retaining every supplied marker and image."""
    objects = [np.float64(v['object_points_mm']).reshape(-1, 1, 3) for v in views]
    pixels = [np.float64(v['image_points_px']).reshape(-1, 1, 2) for v in views]

    if model.startswith('fisheye'):
        flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
                 cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW)
        if model == 'fisheye2':
            flags |= cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4
        result = cv2.fisheye.calibrate(objects, pixels, size, None, None,
                                       flags=flags, criteria=CRITERIA)
    else:
        flags = 0
        if model != 'brown5':
            flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
        if model in {'radial1', 'none'}:
            flags |= cv2.CALIB_FIX_K2
        if model == 'none':
            flags |= cv2.CALIB_FIX_K1
        result = cv2.calibrateCamera([o.astype('float32') for o in objects],
                                     [p.astype('float32') for p in pixels], size,
                                     None, None, flags=flags, criteria=CRITERIA)

    rms, matrix, distortion = result[:3]
    if not np.isfinite(rms) or not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError('Nonfinite lens fit')
    if min(matrix[0, 0], matrix[1, 1]) <= 0:
        raise ValueError('Invalid focal length')
    return float(rms), matrix, distortion



def screen(matrix, distortion, size, model):
    """Screen numerical inversion across the full frame for both lens families."""
    if not model.startswith('fisheye'):
        return distortion_check(matrix, distortion, size)
    x, y = np.meshgrid(np.linspace(0, size[0] - 1, 33), np.linspace(0, size[1] - 1, 25))
    pixels = np.column_stack([x.ravel(), y.ravel()]).reshape(-1, 1, 2)
    normalized = cv2.fisheye.undistortPoints(pixels, matrix, distortion, criteria=CRITERIA)
    restored = cv2.fisheye.distortPoints(normalized, matrix, distortion)
    errors = np.linalg.norm(restored[:, 0] - pixels[:, 0], axis=1)
    return {'grid_points': len(pixels), 'roundtrip_max_px': float(errors.max()),
            'roundtrip_over_1px': int(np.count_nonzero(errors > 1))}


def compare(directory, output):
    """Save all-data candidates and leave-one-photo-out results, including failures."""
    original = read_json(directory / 'intrinsics.json')
    views = [v for v in read_json(directory / 'observations.json') if v['usable']]
    for view in views:
        if sha256(Path(view['path'])) != view['sha256']:
            raise ValueError('Source image changed')
    size = tuple(original['image_size_wh'])
    models = ['brown5', 'radial2', 'radial1', 'none'] if original['camera'] == 'fixed' else [
        'brown5', 'fisheye2', 'fisheye4']
    output.mkdir(parents=True, exist_ok=False)
    results = []

    for model in models:
        record = {'model': model, 'folds': []}
        try:
            rms, matrix, distortion = fit_model(views, size, model)
            record.update(training_rms_px=rms, camera_matrix=matrix.tolist(),
                          distortion=distortion.ravel().tolist(),
                          distortion_screen=screen(matrix, distortion, size, model))
            for index, view in enumerate(views):
                fold = {'excluded_image': view['path']}
                try:
                    _, k, d = fit_model(views[:index] + views[index + 1:], size, model)
                    fold.update(rms_px=pose_error(view, k, d, model), camera_matrix=k.tolist(),
                                distortion_screen=screen(k, d, size, model))
                except (cv2.error, ValueError, np.linalg.LinAlgError) as error:
                    fold['error'] = str(error)
                record['folds'].append(fold)

            successful = [f for f in record['folds'] if 'error' not in f]
            record['failed_folds'] = len(views) - len(successful)
            if successful:
                errors = [f['rms_px'] for f in successful]
                matrices = np.array([f['camera_matrix'] for f in successful])
                record['median_rms_px'] = float(np.median(errors))
                record['max_rms_px'] = max(errors)
                record['parameter_ranges_px'] = {
                    name: [float(matrices[:, r, c].min()), float(matrices[:, r, c].max())]
                    for name, r, c in [('fx', 0, 0), ('fy', 1, 1), ('cx', 0, 2), ('cy', 1, 2)]}

            view = views[min(6, len(views) - 1)] if original['camera'] == 'wrist' else views[0]
            write_lens_preview(view['path'], output / f'{model}.jpg', matrix, distortion, model)
            write_json(output / f'{model}-candidate.json', {
                'status': 'candidate_not_approved', 'motion_ready': False, 'model': model,
                'camera': original['camera'], 'image_size_wh': size,
                'camera_matrix': record['camera_matrix'], 'distortion': record['distortion'],
                'target_sha256': original['target_sha256'],
                'source_image_sha256': [v['sha256'] for v in views],
            })
        except (cv2.error, ValueError, np.linalg.LinAlgError) as error:
            record['error'] = str(error)
        results.append(record)
        print(original['camera'], model, {k: v for k, v in record.items()
              if k in ['median_rms_px', 'max_rms_px', 'failed_folds', 'distortion_screen', 'error']}, flush=True)
        write_json(output / 'comparison.json', {
            'status': 'development_comparison_not_approved', 'camera': original['camera'],
            'image_size_wh': size, 'source_fit_sha256': sha256(directory / 'intrinsics.json'),
            'observations_sha256': sha256(directory / 'observations.json'),
            'opencv_version': cv2.__version__, 'optimizer_criteria': list(CRITERIA),
            'results': results,
            'limitations': ['Same-session model selection; fresh validation images still needed.',
                            'All-data candidates include former validation photos; not independent test fits.',
                            'No physical accuracy or motion readiness established.']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    compare(arguments.fit, arguments.output)
