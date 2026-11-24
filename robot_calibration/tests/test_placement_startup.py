"""Offline verification of the powered placement startup transition."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from robot_calibration.collection.placement_guards import JOINTS, StartupTorqueTimeout, check_feedback, seed_and_verify
from robot_calibration.collection.powered_placement import check_next, validate_plan


class PlacementStartupTests(unittest.TestCase):
    def test_first_and_continuation_share_path_checks(self):
        positions = dict.fromkeys(JOINTS, 0)
        plan = dict(initial=positions, target=positions, square='e4',
                    requested_height_mm=62.5, ik_error_mm=.1,
                    minimum_modeled_tip_height_mm=62.5,
                    path=[dict(tip_board_m=[0., 0., .0625], joint_margin_deg=5.)] * 101,
                    start_raw=positions, hardware={}, T_base_from_board=[])
        with self.assertRaisesRegex(ValueError, 'Unknown first'):
            validate_plan(plan, 'a' * 64)
        validate_plan(plan, 'a' * 64, reviewed_digest='a' * 64)
        check_next(plan, {}, positions, dict, plan)
        bad = plan | {'path': [dict(tip_board_m=[0., 0., .04], joint_margin_deg=5.)] * 101}
        with self.assertRaisesRegex(ValueError, 'Preview exceeds clearance'):
            validate_plan(bad, 'a' * 64, reviewed_digest='a' * 64)
        with self.assertRaisesRegex(ValueError, 'Preview exceeds clearance'):
            check_next(bad, {}, positions, dict, plan)

    def verify(self, flags, error=None, drift=0, passive_drift=False, goal_errors=(),
               write_error=None, enable_if_all_off=False):
        bus = Mock()
        bus.calibration = {joint: SimpleNamespace(id=i) for i, joint in enumerate(JOINTS)}
        bus._unnormalize.side_effect = lambda values: values
        bus.sync_write.side_effect = write_error
        observed = []
        passive_positions = 0
        def registers(name, **kwargs):
            nonlocal passive_positions
            seeded = bus.sync_write.called
            if not seeded and name == 'Present_Position':
                passive_positions += 1
                if passive_drift and passive_positions == 3:
                    return dict.fromkeys(JOINTS, 1003)
            if seeded and name == 'Torque_Enable':
                values = flags[min(len(observed), len(flags)-1)]
                observed.append(values)
                if isinstance(values, Exception):
                    raise values
                return dict(zip(JOINTS, values))
            values = {'Lock': 1, 'Present_Position': 1000 + (drift if seeded else 0),
                      'Present_Temperature': 25, 'Torque_Enable': 0, 'Status': 0, 'Operating_Mode': 0}
            return dict.fromkeys(JOINTS, values[name])
        bus.sync_read.side_effect = registers
        failures = iter(goal_errors)
        def addressed(name, joint, **kwargs):
            if name == 'Goal_Position' and joint == 'shoulder_lift' and bus.sync_write.called:
                failure = next(failures, None)
                if failure:
                    raise failure
            return 1 if name == 'Lock' else 1000
        bus.read.side_effect = addressed
        arm = SimpleNamespace(bus=bus, motion_cancelled=False, recovery=object(), enabled=False)
        def run():
            seed_and_verify(arm, dict.fromkeys(JOINTS, 1000), JOINTS, lambda *args: None,
                            enable_if_all_off=enable_if_all_off)
        if error:
            with self.assertRaises(error):
                run()
        else:
            run()
        if passive_drift:
            bus.sync_write.assert_not_called()
            self.assertFalse(arm.enabled)
            return observed
        bus.sync_write.assert_called_once_with('Goal_Position', dict.fromkeys(JOINTS, 1000),
                                               normalize=False, num_retry=0)
        self.assertTrue(arm.enabled)
        self.assertTrue(all(call.args[0] == 'Lock' for call in bus.write.call_args_list))
        self.assertEqual(bus.enable_torque.call_count, int(enable_if_all_off))
        return observed

    def test_failed_goal_write_still_requires_recovery(self):
        self.verify([(1,) * 6], TimeoutError, write_error=TimeoutError('Write timeout'))

    def test_collector_can_enable_after_two_verified_off_samples(self):
        off, on = (0,) * 6, (1,) * 6
        self.assertEqual(self.verify([off, off, on, on], enable_if_all_off=True),
                         [off, off, on, on])

    def test_passive_drift_stops_before_energizing_write(self):
        self.verify([(1,)*6], RuntimeError, passive_drift=True)

    def test_elbow_only_sample_retries_without_another_write(self):
        on, mixed = (1,)*6, (0,0,1,0,0,0)
        self.assertEqual(self.verify([mixed, on, on]), [mixed, on, on])

    def test_mixed_after_one_all_on_resets_confirmation(self):
        on, mixed = (1,)*6, (1,1,1,1,1,0)
        self.assertEqual(self.verify([on, mixed, on, on]), [on, mixed, on, on])

    def test_all_off_then_two_all_on_passes(self):
        off, on = (0,)*6, (1,)*6
        self.assertEqual(self.verify([off, on, on]), [off, on, on])

    def test_persistent_mixed_times_out(self):
        self.verify([(0,0,1,0,0,0)], StartupTorqueTimeout)

    def test_all_off_times_out(self):
        self.verify([(0,)*6], StartupTorqueTimeout)

    def test_drift_during_transition_stops_immediately(self):
        self.assertEqual(len(self.verify([(1,)*6], RuntimeError, drift=3)), 1)

    def test_torque_checksum_requires_two_fresh_confirmations(self):
        on = (1,) * 6
        bad = ConnectionError('motor 5: Checksum error')
        self.assertEqual(self.verify([on, bad, on, on]), [on, bad, on, on])

    def test_missing_torque_after_seed_waits_for_two_complete_reads(self):
        on = (1,) * 6
        timeout = ConnectionError('motor 6: Operation timed out')
        flags = [on, timeout, on[:-1], on, on]
        self.assertEqual(self.verify(flags), flags)

    def test_motion_feedback_waits_for_gripper_reply(self):
        bus = Mock()
        positions = dict.fromkeys(JOINTS, 1000)
        torque = iter([ConnectionError('motor 6: Operation timed out'), dict.fromkeys(JOINTS, 1)])
        def read(name, **kwargs):
            if name == 'Torque_Enable':
                sample = next(torque)
                if isinstance(sample, Exception):
                    raise sample
                return sample
            if name in ('Present_Position', 'Goal_Position'):
                return positions
            return dict.fromkeys(JOINTS, 0)
        bus.sync_read.side_effect = read
        row = check_feedback(bus, positions, positions, dict.fromkeys(JOINTS, (900, 1100)))
        self.assertEqual(row['Torque_Enable'], dict.fromkeys(JOINTS, 1))
        bus.sync_write.assert_not_called()
        bus.write.assert_not_called()

    def test_repeated_checksum_failure_stops(self):
        bad = ConnectionError('motor 5: Checksum error')
        self.assertEqual(self.verify([bad, bad], ConnectionError), [bad, bad])

    def test_addressed_goal_checksum_recovers_once(self):
        on = (1,) * 6
        error = ConnectionError('motor ID 2 after 1 tries. Checksum error')
        self.assertEqual(self.verify([on], goal_errors=[error]), [on, on, on])
        self.verify([on], ConnectionError, goal_errors=[error, error])
