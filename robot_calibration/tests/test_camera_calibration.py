"""Offline checks for measured-target validation and held-out reprojection."""

import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, 'Run with the camera environment for OpenCV checks')
class CameraCalibrationTests(unittest.TestCase):
    def test_known_lens_on_separate_pose(self):
        """Recover a synthetic lens and predict a pose excluded from lens fitting."""
        from robot_calibration.camera_calibration import (
            fit_intrinsics, load_target, validate_intrinsics)

        _, markers = load_target(Path('robot_calibration/configs/board_target.json'))
        objects = np.concatenate(list(markers.values()))
        matrix = np.array([[800., 0, 640], [0, 810, 360], [0, 0, 1]])
        distortion = np.array([-.25, .08, .001, -.002, -.01])
        views = []

        for i in range(7):
            rotation = np.array([-.3 + i * .1, .25 - i * .07, i * .13])
            translation = np.array([-100. + i * 5, -90. + i * 3, 420. + i * 20])
            pixels, _ = cv2.projectPoints(objects, rotation, translation, matrix, distortion)
            views.append({'object_points_mm': objects.tolist(),
                          'image_points_px': pixels.reshape(-1, 2).tolist(),
                          'path': 'synthetic'})

        _, fitted_matrix, fitted_distortion = fit_intrinsics(views[:6], (1280, 720))
        result = validate_intrinsics(views[6], fitted_matrix, fitted_distortion)
        self.assertLess(result['rms_px'], .01)
        np.testing.assert_allclose(fitted_matrix, matrix, atol=.1)


@unittest.skipIf(cv2 is None, 'Run with the camera environment for OpenCV checks')
class CameraValidationTests(unittest.TestCase):
    def test_withheld_marker_does_not_change_pose(self):
        """A displaced withheld marker must remain visible as prediction error."""
        from robot_calibration.analysis.validate_cameras import project_view, distortion_check
        from robot_calibration.camera_calibration import load_target

        _, markers = load_target(Path('robot_calibration/configs/board_target.json'))
        objects = np.concatenate(list(markers.values()))
        matrix = np.array([[800., 0, 640], [0, 810, 360], [0, 0, 1]])
        distortion = np.zeros(5)
        pixels, _ = cv2.projectPoints(objects, np.array([.2, -.3, .1]),
                                     np.array([-100., -100., 500.]), matrix, distortion)
        pixels = pixels.reshape(-1, 2).astype(float)
        clean = pixels.copy()
        pixels[:4, 0] += 20
        view = {'object_points_mm': objects.tolist(), 'image_points_px': pixels.tolist()}
        projected = project_view(view, matrix, distortion, excluded_marker=0)

        np.testing.assert_allclose(projected, clean, atol=.001)
        self.assertGreater(np.linalg.norm(projected[:4] - pixels[:4], axis=1).mean(), 19.9)
        self.assertLess(distortion_check(matrix, distortion, (1280, 720))['roundtrip_max_px'], 1e-8)


@unittest.skipIf(cv2 is None, 'Run with the camera environment for OpenCV checks')
class CameraModelTests(unittest.TestCase):
    def test_fisheye_fit_predicts_separate_pose(self):
        """Check model conventions and pose refinement on a known wide-angle lens."""
        from robot_calibration.analysis.compare_camera_models import fit_model, pose_error, screen
        from robot_calibration.camera_calibration import load_target

        _, markers = load_target(Path('robot_calibration/configs/board_target.json'))
        objects = np.float64(np.concatenate(list(markers.values())))
        matrix = np.array([[700., 0, 640], [0, 705, 360], [0, 0, 1]])
        distortion = np.array([-.1, -.003, 0., 0.])
        views = []

        for i in range(9):
            rotation = np.array([-.4 + i * .1, .35 - i * .08, i * .15])
            translation = np.array([-120. + i * 12, -100. + i * 5, 280. + i * 15])
            pixels, _ = cv2.fisheye.projectPoints(objects[:, None], rotation, translation,
                                                  matrix, distortion)
            views.append({'object_points_mm': objects.tolist(),
                          'image_points_px': pixels[:, 0].tolist()})

        _, fitted, coefficients = fit_model(views[:8], (1280, 720), 'fisheye2')
        np.testing.assert_allclose(fitted, matrix, atol=.1)
        self.assertLess(pose_error(views[8], fitted, coefficients, 'fisheye2'), .01)
        self.assertEqual(screen(fitted, coefficients, (1280, 720), 'fisheye2')['roundtrip_over_1px'], 0)


@unittest.skipIf(cv2 is None, 'Run with the camera environment for OpenCV checks')
class FreshPhotoTests(unittest.TestCase):
    def test_rejects_reused_training_photo(self):
        """Validation must reject reused input before writing a report."""
        import tempfile
        from unittest.mock import patch
        from robot_calibration.records import write_json
        from robot_calibration.analysis.validate_camera_photos import validate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            view = {'sha256': 'same-image', 'marker_ids': [0],
                    'object_points_mm': [[0., 0., 0.]] * 4}
            write_json(root / 'old.json', [view])
            write_json(root / 'candidate.json', {
                'source_observations': str(root / 'old.json'), 'image_size_wh': [1280, 720]})
            with patch('robot_calibration.analysis.validate_camera_photos.load_target',
                       return_value=({}, {0: np.zeros((4, 3))})), patch(
                       'robot_calibration.analysis.validate_camera_photos.collect_views',
                       return_value=((1280, 720), [view])):
                with self.assertRaisesRegex(ValueError, 'Duplicate images'):
                    validate(root / 'candidate.json', root, root / 'target.json', root / 'output')
            self.assertFalse((root / 'output').exists())


if __name__ == '__main__':
    unittest.main()
