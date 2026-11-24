import unittest
from pathlib import Path
from unittest.mock import Mock

try:
    import numpy as np
    import cv2
    from robot_calibration.collection import start_registration as reg
except ModuleNotFoundError as exc:
    if exc.name.split('.')[0] not in ('cv2', 'numpy', 'scipy'):
        raise
    raise unittest.SkipTest('Registration needs OpenCV, NumPy and SciPy')


class StartupRegistrationTest(unittest.TestCase):
    def setUp(self):
        # Inject decoded corners so projection and transform tests do not depend
        # on local teaching videos.
        self.reg = reg.Registration.__new__(reg.Registration)
        self.reg.detector = Mock()
        self.reg.anchors = {1, 4, 5, 8}
        self.reg.height_m = reg.BENCH_BOARD_HEIGHT_M
        self.reg.lens = dict(model='brown5', image_size_wh=[1920, 1080],
                             camera_matrix=[[1200, 0, 960], [0, 1200, 540], [0, 0, 1]],
                             distortion=[0.] * 5)
        _, self.reg.board = reg.load_target(Path('robot_calibration/configs/board_target.json'))
        corners = np.array([[0, 0, 0], [.019, 0, 0], [.019, .019, 0], [0, .019, 0]])
        self.reg.geometry = {20: corners + [.03, -.03, .016],
                             22: corners + [-.02, -.02, .04]}
        self.frame = np.zeros((1080, 1920), np.uint8)

    def observe(self, board_in_base, camera_shift=0.):
        camera = np.eye(4)
        camera[:3, :3] = cv2.Rodrigues(np.array([2.9, .1, .02]))[0]
        camera[:3, 3] = [-.05 + camera_shift, .1, .9]
        geometry = {i: np.asarray(x)/1000 for i, x in self.reg.board.items()}
        for i, xyz in self.reg.geometry.items():
            geom = np.c_[xyz, np.ones(4)] @ np.linalg.inv(board_in_base).T
            geometry[i] = geom[:, :3]
        k = np.array(self.reg.lens['camera_matrix'], dtype=float)
        corners = []
        for xyz in geometry.values():
            xyz = xyz @ camera[:3, :3].T + camera[:3, 3]
            uv = xyz @ k.T
            corners.append((uv[:, :2] / uv[:, 2:]).astype(np.float32)[None])
        self.reg.detector.detectMarkers.return_value = corners, np.array(list(geometry))[:, None], []
        return self.reg.measure(self.frame)

    def test_detect_base_change_independent_of_camera_shift(self):
        old = reg.level_pose([.115, .102, -1.59])
        new = reg.level_pose([.117, .101, -1.58])
        a, b = self.observe(old), self.observe(new, camera_shift=.012)
        for key, file_x in [('e4', 4.5), ('f4', 5.5)]:
            shift, yaw = reg.displacement(a, b, key)
            expected = ((new-old) @ [file_x*.202/8, 3.5*.198/8, 0, 1])[:3]
            np.testing.assert_allclose(shift, expected, atol=1e-5)
            self.assertAlmostEqual(yaw, .01, places=4)

    def test_missing_marker_aborts(self):
        self.reg.detector.detectMarkers.return_value = [], np.array([[1], [4], [20]]), []
        with self.assertRaisesRegex(ValueError, r'20, 22'):
            self.reg.measure(self.frame)
