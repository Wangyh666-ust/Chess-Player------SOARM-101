"""Offline calibration boundary checks; synthetic cases are not hardware evidence."""

import copy
import unittest

import numpy as np

from manipulation.calibration import (
    arm_to_model, arm_from_model, encoder_act_state, context_from_session, jaw_gap_m, scene_geometry,
    corrected_fixed_tip_camera, compensate_fixed_tip_target,
)
from robot_calibration.records import ARM


class CalibrationTests(unittest.TestCase):
    def test_local_correction_target_sign_and_rotated_frame(self):
        correction = dict(status='local_empirical_camera_plane_bias_with_pose3_excluded_check',
                          frame='OpenCV fixed camera', camera_bias_m=[-.006, .004, 0])
        transform = np.array([[0., -1, 0, .3], [1, 0, 0, -.1],
                              [0, 0, 1, .2], [0, 0, 0, 1]])
        desired = np.array([.1, .2, .3])
        target = compensate_fixed_tip_target(desired, transform, correction)
        nominal_camera = (transform @ np.r_[target, 1])[:3]
        corrected = corrected_fixed_tip_camera(nominal_camera, correction)
        np.testing.assert_allclose(corrected, (transform @ np.r_[desired, 1])[:3], atol=1e-12)
        np.testing.assert_allclose(target, [.096, .194, .3])
        correction['camera_bias_m'][2] = .01
        with self.assertRaises(ValueError):
            compensate_fixed_tip_target(desired, transform, correction)

    def test_mapping_sign_scale_offset_order_and_separate_gripper(self):
        profile = dict(joint_order=list(ARM), joint_mapping=dict(joints={
            j: dict(sign=-1, scale=2, offset_deg=10) for j in ARM}))
        hardware = dict(zip(ARM, [10, 20, 30, 40, 50]))
        hardware['gripper'] = 99
        model = arm_to_model(profile, hardware)
        np.testing.assert_allclose(np.rad2deg(model), [-10, -30, -50, -70, -90])
        np.testing.assert_allclose(list(arm_from_model(profile, model).values()), [10, 20, 30, 40, 50])
        for bad in ([0] * 6, [0, 0, np.nan, 0, 0]):
            with self.assertRaises(ValueError):
                arm_from_model(profile, bad)
        profile['joint_mapping']['joints'][ARM[0]]['scale'] = 0
        with self.assertRaises(ValueError):
            arm_to_model(profile, hardware)

    def test_encoder_state_uses_calibrated_midpoint_at_both_start_poses(self):
        profile = dict(joint_order=list(ARM), joint_mapping=dict(joints={
            j: dict(sign=1, scale=1, offset_deg=0) for j in ARM}))
        calibration = {j: dict(range_min=839, range_max=3245) for j in ARM}
        calibration['gripper'] = dict(range_min=2047, range_max=3092, drive_mode=0)
        ticks = dict.fromkeys(ARM, 2042) | {'gripper': 2086}
        for shoulder in (1759, 1772):
            ticks['shoulder_lift'] = shoulder
            state = encoder_act_state(profile, calibration, ticks)
            self.assertAlmostEqual(state[1], (shoulder - 2042) * 2*np.pi / 4095)
            self.assertAlmostEqual(state[-1], 39 * 100 / 1045)

    def test_rotated_scene_origin_and_axis_conversion(self):
        base_corner = np.array([[0., -1, 0, .3], [1, 0, 0, -.1],
                                [0, 0, 1, .02], [0, 0, 0, 1]])
        camera_base = np.array([[1., 0, 0, .1], [0, 0, -1, .5],
                                [0, 1, 0, .8], [0, 0, 0, 1]])
        shift = np.eye(4)
        shift[:2, 3] = [.01, .015]
        session = dict(translation_units='m', grid=dict(provisional_cell_pitch_xy_m=[.02, .03]),
                       T_base_from_board_corner=base_corner,
                       T_board_corner_from_board_a1=shift,
                       T_base_from_board_a1=base_corner @ shift,
                       T_fixed_camera_from_base=camera_base,
                       T_fixed_camera_from_board_corner=camera_base @ base_corner,
                       T_fixed_camera_from_board_a1=camera_base @ base_corner @ shift)
        board = [0] * 64
        board[0] = 1
        context = context_from_session(session, board, 'a1', 'h8')
        np.testing.assert_allclose(context['source_xyz_m'], [.285, -.09, .02])
        np.testing.assert_allclose(context['destination_xyz_m'], [.075, .05, .02])
        bad = copy.deepcopy(session)
        bad['T_base_from_board_a1'][0, 3] += .01
        with self.assertRaisesRegex(ValueError, 'Inconsistent'):
            scene_geometry(bad)

    def test_aperture_rejects_extrapolation_and_unmeasured_depth(self):
        profile = dict(gripper_geometry=dict(post_reseat_opening_map=dict(candidate=dict(
            normalized_domain=[10, 30], rows=[dict(depth_from_extreme_fixed_tip_mm=5,
                slope_mm_per_normalized_unit=2, intercept_mm=1)]))))
        self.assertAlmostEqual(jaw_gap_m(profile, 20, 5), .041)
        for opening, depth in [(9, 5), (31, 5), (float('nan'), 5), (20, 6.5)]:
            with self.assertRaises(ValueError):
                jaw_gap_m(profile, opening, depth)
