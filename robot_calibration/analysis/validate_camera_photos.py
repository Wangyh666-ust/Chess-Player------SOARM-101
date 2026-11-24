"""Evaluate fresh photos against a frozen lens candidate; never refit intrinsics."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from robot_calibration.camera_calibration import collect_views, load_target
from robot_calibration.camera_geometry import camera_parameters, check_fit_target, pose_error, write_lens_preview
from robot_calibration.records import read_json, sha256, write_json


def validate(candidate_path, images, target_path, output):
    """Check resolution, geometry and image independence before fitting board poses."""
    candidate = read_json(candidate_path)
    target, points = load_target(target_path)
    old_hashes = set(check_fit_target(candidate, target_path, points))
    if not old_hashes:
        raise ValueError('Lens candidate has no recorded fitting image hashes')
    size, views = collect_views(images, target, points)
    hashes = [v['sha256'] for v in views]
    if len(set(hashes)) != len(hashes) or old_hashes.intersection(hashes):
        raise ValueError('Duplicate images or fitting images found in validation set')

    matrix, distortion, model = camera_parameters(candidate, size)

    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'observations.json', views)
    results = []

    for index, view in enumerate(views):
        result = {'path': view['path'], 'sha256': view['sha256'],
                  'markers': view['marker_ids'], 'usable': view['usable']}
        if view['usable']:
            try:
                result['rms_px'] = pose_error(view, matrix, distortion, model)
            except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
                result['error'] = str(error)
        else:
            result['error'] = 'Fewer than four decoded known markers'

        write_lens_preview(view['path'], output / f'preview_{index:02d}.jpg',
                           matrix, distortion, model)
        results.append(result)

    errors = [r['rms_px'] for r in results if 'rms_px' in r]
    report = {'status': 'fresh_photo_evaluation_not_motion_approval',
              'candidate_path': str(candidate_path.resolve()),
              'candidate_sha256': sha256(candidate_path), 'target_sha256': sha256(target_path),
              'camera': candidate['camera'], 'model': candidate['model'],
              'intrinsics_refitted': False, 'image_size_wh': size, 'views': results,
              'summary': {'total': len(results), 'evaluated': len(errors),
                          'median_rms_px': float(np.median(errors)) if errors else None,
                          'max_rms_px': max(errors) if errors else None},
              'limitations': ['Only board pose is fitted per view; lens values remain fixed.',
                              'Image hashes cannot verify unchanged focus, crop or device settings.',
                              'No predefined acceptance tolerance or independent metric truth.']}
    write_json(output / 'validation.json', report)
    print(candidate['camera'], report['summary'])
    for r in results:
        print(Path(r['path']).name, len(r['markers']), r.get('rms_px', r.get('error')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--target', type=Path, default=Path('robot_calibration/configs/board_target.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    validate(args.candidate, args.images, args.target, args.output)
