"""Known-transform tests for units, both lens families and reserved markers."""

import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, 'Requires the camera environment')
class BoardPoseTests(unittest.TestCase):
    def setUp(self):
        from robot_calibration.camera_calibration import load_target
        _, self.points = load_target(Path('robot_calibration/configs/board_target.json'))
        self.ids = list(self.points)
        self.objects = np.concatenate(list(self.points.values())).astype(float) / 1000
        self.rotation = cv2.Rodrigues(np.array([.2, -.15, .1]))[0] @ np.diag([1., -1., -1.])
        self.rvec = cv2.Rodrigues(self.rotation)[0].ravel()
        self.translation = np.array([-.1, .1, .5])
        self.matrix = np.array([[800., 0, 640], [0, 805., 360], [0, 0, 1.]])

    def sample(self, model):
        from robot_calibration.camera_geometry import project
        distortion = np.array([-.1, -.003, 0., 0.]) if model == 'fisheye2' else np.array([-.1, .02, .001, -.001, 0.])
        calibration = {'model': model, 'camera_matrix': self.matrix.tolist(),
                       'distortion': distortion.tolist(), 'image_size_wh': [1280, 720]}
        pixels = project(self.objects, np.r_[self.rvec, self.translation],
                         self.matrix, distortion, model).reshape(-1, 4, 2)
        return calibration, pixels

    def test_pose_units_axes_and_both_models(self):
        from robot_calibration.board_pose import estimate_board_pose
        for model in ['brown5', 'fisheye2']:
            with self.subTest(model=model):
                calibration, pixels = self.sample(model)
                result = estimate_board_pose(self.ids, pixels, self.points, calibration,
                                             (1280, 720), [8, 9])
                transform = np.array(result['T_camera_from_board'])
                np.testing.assert_allclose(transform[:3, :3], self.rotation, atol=1e-6)
                np.testing.assert_allclose(transform[:3, 3], self.translation, atol=1e-7)
                self.assertLess(result['holdout_rms_px'], .001)
                self.assertFalse(result['motion_ready'])

    def test_reserved_marker_cannot_pull_pose(self):
        from robot_calibration.board_pose import estimate_board_pose
        calibration, pixels = self.sample('fisheye2')
        pixels[self.ids.index(8), :, 0] += 20
        result = estimate_board_pose(self.ids, pixels, self.points, calibration,
                                     (1280, 720), [8, 9])
        np.testing.assert_allclose(result['translation_m'], self.translation, atol=1e-7)
        self.assertGreater(result['holdout_rms_px'], 14)

    def test_rejects_invalid_inputs(self):
        from robot_calibration.board_pose import estimate_board_pose
        calibration, pixels = self.sample('brown5')
        with self.assertRaisesRegex(ValueError, 'Resolution mismatch'):
            estimate_board_pose(self.ids, pixels, self.points, calibration, (640, 360))
        with self.assertRaisesRegex(ValueError, 'four fitting markers'):
            estimate_board_pose(self.ids[:3], pixels[:3], self.points, calibration, (1280, 720))
        with self.assertRaisesRegex(ValueError, 'unique marker IDs'):
            estimate_board_pose([0] * len(self.ids), pixels, self.points, calibration, (1280, 720))
        calibration['distortion'] = [0.] * 4
        with self.assertRaisesRegex(ValueError, 'coefficient count'):
            estimate_board_pose(self.ids, pixels, self.points, calibration, (1280, 720))


if __name__ == '__main__':
    unittest.main()
