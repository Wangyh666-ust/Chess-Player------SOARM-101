"""Offline fixed-tip fitter checks for new, self-contained jogger observations."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import numpy as np
    from robot_calibration.analysis.fit_fixed_tip import fit
    from robot_calibration.kinematics import URDF, Robot, gripper_link_pose, pose_fk
except ModuleNotFoundError as exc:
    raise unittest.SkipTest('Fixed-tip fitting requires the offline model environment') from exc

from robot_calibration.records import ARM, JOINTS


class FixedTipFitTest(unittest.TestCase):
    def test_explicit_gripper_frame_and_held_out_intersections(self):
        robot = Robot(str(URDF))
        profile = dict(joint_order=list(ARM), joint_mapping=dict(joints={
            joint: dict(sign=1, scale=1, offset_deg=0) for joint in ARM
        }))
        hardware = dict(calibration={joint: dict(range_min=0, range_max=4094, drive_mode=0)
                                     for joint in JOINTS})
        board = np.eye(4)
        tip = np.array([-.004, -.005, -.110, 1.])
        events = [dict(event='preflight', hardware=hardware)]
        for number, (cell, pan) in enumerate([('d4', -120), ('d4', 0), ('g7', 150)], 1):
            ticks = dict.fromkeys(JOINTS, 2047) | {'shoulder_pan': 2047 + pan}
            angles = [pan * 2 * np.pi / 4095, 0., 0., 0., 0.]
            tool = gripper_link_pose(robot, angles)
            np.testing.assert_allclose(tool @ robot.joints[-1]['origin'], pose_fk(robot, angles))
            events.append(dict(event='board_manual_hover', number=number, lower_left_cell=cell,
                               ticks=ticks, target_board_m=(tool @ tip)[:3].tolist(),
                               registration=dict(status='board_and_base', T_base_from_board=board.tolist())))
        events.append(dict(event='board_contact', number=4, lower_left_cell='a1',
                           registration=dict(status='partial')))

        with TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'metadata.json').write_text(json.dumps(dict(profile=profile)))
            (run / 'samples.jsonl').write_text('\n'.join(map(json.dumps, events)))
            result = fit([run], {})
        self.assertEqual(result['frame'], 'gripper_link')
        self.assertEqual(result['usable_marks'], 3)
        self.assertFalse(result['motion_ready'])
        np.testing.assert_allclose(result['point_m'], tip[:3], atol=1e-12)
        for row in result['measurements'][:3]:
            self.assertLess(row['held_out_error_mm'], 1e-8)
        self.assertEqual(result['measurements'][-1]['status'], 'partial')
