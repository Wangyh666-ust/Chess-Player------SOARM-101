"""Offline regression checks: no SDK imports or serial access."""
import unittest
from robot_calibration.collection.stop_recovery import ARM, JOINTS, hold_measured_pose, release_supported


class Bus:
    def __init__(self):
        self.actual = dict.fromkeys(JOINTS, 2000)
        self.goal = self.actual | {'elbow_flex': 1977, 'gripper': 1900}
        self.torque = dict.fromkeys(JOINTS, 1)
        self.fault = 0
        self.fail = None
        self.writes = []
        self.releases = 0

    def sync_read(self, name, normalize=False):
        if name == self.fail:
            raise OSError('Lost reply')
        return {'Present_Position': self.actual, 'Goal_Position': self.goal,
                'Torque_Enable': self.torque, 'Status': dict.fromkeys(JOINTS, self.fault),
                'Operating_Mode': dict.fromkeys(JOINTS, 0)}[name].copy()

    def sync_write(self, name, goal, normalize=False):
        self.writes.append(goal.copy())
        if self.fail == 'partial_write':
            self.goal['shoulder_pan'] = goal['shoulder_pan']
            raise OSError('Partial write')
        if self.fail != 'mismatch':
            self.goal = goal.copy()

    def disable_torque(self):
        self.releases += 1
        self.torque = dict.fromkeys(JOINTS, 0)


class StopPolicyTests(unittest.TestCase):
    def stop(self, bus, **kwargs):
        return hold_measured_pose(bus, dict.fromkeys(JOINTS, (1800, 2200)), .5, **kwargs)

    def test_tracking_failure_latches_actual_and_keeps_grip(self):
        bus = Bus()
        result = self.stop(bus)
        self.assertEqual(result['state'], 'hold_goal_verified_requires_supervision')
        self.assertEqual({j: bus.goal[j] for j in ARM}, {j: bus.actual[j] for j in ARM})
        self.assertEqual(bus.goal['gripper'], 1900)
        self.assertEqual(bus.releases, 0)
        self.assertFalse(result['motion_ready'])

    def test_uncertain_stops_never_release_or_retry(self):
        for failure in ('Present_Position', 'partial_write', 'mismatch', 'fault', 'torque', 'bounds', 'missing'):
            with self.subTest(failure=failure):
                bus = Bus()
                bus.fail = failure
                if failure == 'fault': bus.fault = 8
                if failure == 'torque': bus.torque['elbow_flex'] = 0
                if failure == 'bounds': bus.actual['elbow_flex'] = 2300
                if failure == 'missing': del bus.actual['elbow_flex']
                result = self.stop(bus)
                self.assertEqual(result['state'], 'stop_unverified')
                self.assertEqual(bus.releases, 0)
                self.assertLessEqual(len(bus.writes), 1)
                if failure not in ('partial_write', 'mismatch'):
                    self.assertEqual(len(bus.writes), 0)

    def test_stale_read_cannot_command_hold(self):
        bus = Bus()
        ticks = iter((0, 1))
        self.assertEqual(self.stop(bus, clock=lambda: next(ticks))['state'], 'stop_unverified')
        self.assertEqual(bus.writes, [])
        self.assertEqual(bus.releases, 0)

    def test_release_requires_explicit_support(self):
        bus = Bus()
        with self.assertRaises(RuntimeError):
            release_supported(bus, mechanically_supported=False)
        self.assertEqual(bus.releases, 0)
        self.assertEqual(release_supported(bus, mechanically_supported=True)['state'],
                         'released_with_confirmed_support')




from robot_calibration.collection.stop_recovery import StopSupervisor


class Clock:
    now = 0.

    def __call__(self):
        return self.now


class SupervisedBus(Bus):
    def __init__(self):
        super().__init__()
        self.temperature = 30
        self.disconnections = []
        self.bad_release = False

    def sync_read(self, name, normalize=False):
        if name == 'Present_Temperature':
            return dict.fromkeys(JOINTS, self.temperature)
        return super().sync_read(name, normalize)

    def disable_torque(self):
        if self.bad_release:
            raise OSError('Release packet failed')
        super().disable_torque()

    def disconnect(self, disable_torque=True):
        self.disconnections.append(disable_torque)


class SupervisorTests(unittest.TestCase):
    def test_isolated_gripper_spike_requires_three_clean_confirmations(self):
        from unittest.mock import patch
        for samples, expected in [([48, 38, 38, 38], 'holding'),
                                  ([48, 38, 45], 'intervention_required'),
                                  ([60], 'intervention_required'),
                                  ([48, -1], 'intervention_required'),
                                  ([48, None], 'intervention_required')]:
            with self.subTest(samples=samples):
                s, bus, _, _ = self.setup_session()
                s.max_temperature = 45
                original = bus.sync_read
                temperatures = iter(samples)
                def read(name, normalize=False):
                    if name != 'Present_Temperature':
                        return original(name, normalize)
                    value = next(temperatures)
                    if value is None:
                        raise OSError('Lost temperature reply')
                    return dict.fromkeys(JOINTS, 35) | {'gripper': value}
                bus.sync_read = read
                with patch('robot_calibration.collection.stop_recovery.time.sleep'):
                    s.begin('completed')
                self.assertEqual(s.state, expected)
                self.assertEqual(len(bus.writes), 1)
                self.assertEqual(bus.releases, 0)
                if expected == 'holding':
                    feedback = next(e for e in s.events if e['event'] == 'feedback')
                    self.assertEqual(feedback['readings']['Present_Temperature']['gripper'], 48)
                    self.assertEqual(len(feedback['readings']['Temperature_Confirmation']), 3)

    def setup_session(self):
        bus, clock, cancelled = SupervisedBus(), Clock(), []

        def cancel():
            self.assertEqual(bus.writes, [])
            cancelled.append(True)

        supervisor = StopSupervisor(bus, cancel, dict.fromkeys(JOINTS, (1800, 2200)),
                                    3, 60, .5, 1., 10., clock)
        return supervisor, bus, clock, cancelled

    def test_temperature_confirmation_preserves_deadline_and_arm_guard(self):
        from unittest.mock import patch
        for failure in ('deadline', 'arm_hot', 'missing_joint'):
            with self.subTest(failure=failure):
                s, bus, clock, _ = self.setup_session()
                s.max_temperature = 45
                original = bus.sync_read
                count = 0
                def read(name, normalize=False):
                    nonlocal count
                    if name != 'Present_Temperature':
                        return original(name, normalize)
                    count += 1
                    values = dict.fromkeys(JOINTS, 35)
                    values['gripper'] = 48 if count == 1 else 38
                    if count > 1:
                        if failure == 'deadline': clock.now += 1
                        if failure == 'arm_hot': values['wrist_flex'] = 45
                        if failure == 'missing_joint': del values['wrist_flex']
                    return values
                bus.sync_read = read
                with patch('robot_calibration.collection.stop_recovery.time.sleep'):
                    s.begin('completed')
                self.assertEqual(s.state, 'intervention_required')
                self.assertEqual(len(bus.writes), 1)
                self.assertEqual(bus.releases, 0)

    def test_full_tracking_failure_recovery(self):
        s, bus, clock, cancelled = self.setup_session()
        s.begin('tracking failure')
        self.assertEqual(cancelled, [True])
        self.assertEqual(s.state, 'holding')
        self.assertEqual(bus.goal['elbow_flex'], 2000)
        self.assertEqual(bus.goal['gripper'], 1900)

        for _ in range(3):
            clock.now += .2
            s.poll()
        self.assertEqual(len(bus.writes), 1)
        with self.assertRaises(RuntimeError):
            s.disconnect()
        with self.assertRaises(RuntimeError):
            s.release(mechanically_supported=False)
        with self.assertRaises(RuntimeError):
            s.begin('resume')
        self.assertEqual(bus.releases, 0)

        s.release(mechanically_supported=True)
        s.disconnect()
        self.assertEqual(s.state, 'disconnected')
        self.assertEqual(bus.disconnections, [False])
        self.assertEqual(bus.releases, 1)

    def test_faults_latch_without_release_or_refresh(self):
        for failure in ('drift', 'overheat', 'communication', 'torque', 'goal', 'status',
                        'poll_timeout', 'hold_timeout'):
            with self.subTest(failure=failure):
                s, bus, clock, _ = self.setup_session()
                s.begin('completed')
                if failure == 'drift': bus.actual['elbow_flex'] += 4
                if failure == 'overheat': bus.temperature = 60
                if failure == 'communication': bus.fail = 'Present_Position'
                if failure == 'torque': bus.torque['elbow_flex'] = 0
                if failure == 'goal': bus.goal['elbow_flex'] += 1
                if failure == 'status': bus.fault = 8
                if failure == 'poll_timeout': clock.now = 2
                if failure == 'hold_timeout': clock.now = 11
                s.poll()
                self.assertEqual(s.state, 'intervention_required')
                self.assertEqual(len(bus.writes), 1)
                self.assertEqual(bus.releases, 0)
                self.assertEqual(bus.disconnections, [])

                bus.actual = dict.fromkeys(JOINTS, 2000)
                bus.goal = s.goal.copy()
                bus.temperature, bus.fail, bus.fault = 30, None, 0
                bus.torque = dict.fromkeys(JOINTS, 1)
                s.poll()
                self.assertEqual(s.state, 'intervention_required')
                s.release(mechanically_supported=True)
                s.disconnect()
                self.assertEqual(s.state, 'disconnected')

    def test_partial_hold_write_and_failed_release(self):
        s, bus, _, _ = self.setup_session()
        bus.fail = 'partial_write'
        s.begin('operator stop')
        self.assertEqual(s.state, 'intervention_required')
        self.assertEqual(len(bus.writes), 1)
        self.assertEqual(bus.releases, 0)
        bus.bad_release = True
        s.release(mechanically_supported=True)
        self.assertEqual(s.state, 'intervention_required')
        with self.assertRaises(RuntimeError):
            s.disconnect()
        bus.bad_release = False
        s.release(mechanically_supported=True)
        s.disconnect()

    def test_cancellation_failure_cannot_latch_hold(self):
        s, bus, _, _ = self.setup_session()

        def fail():
            raise RuntimeError('Trajectory owner unavailable')

        s.cancel_trajectory = fail
        s.begin('tracking failure')
        self.assertEqual(s.state, 'intervention_required')
        self.assertEqual(bus.writes, [])
        self.assertEqual(bus.releases, 0)

    def test_interrupt_keeps_hold_and_requires_supported_release(self):
        s, bus, _, cancelled = self.setup_session()
        s.interrupted()
        self.assertEqual(cancelled, [True])
        self.assertEqual(s.state, 'holding')
        s.interrupted()
        self.assertEqual(s.state, 'intervention_required')
        self.assertEqual(bus.releases, 0)
        self.assertEqual(len(bus.writes), 1)
        s.release(mechanically_supported=True)
        s.disconnect()

    def test_slow_feedback_latches_intervention(self):
        s, bus, clock, _ = self.setup_session()
        s.begin('tracking failure')
        original = bus.sync_read

        def slow(name, normalize=False):
            clock.now += .2
            return original(name, normalize)

        bus.sync_read = slow
        s.poll()
        self.assertEqual(s.state, 'intervention_required')
        self.assertEqual(bus.releases, 0)
        self.assertEqual(len(bus.writes), 1)

    def test_invalid_limits_fail_before_bus_access(self):
        for value in (0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                StopSupervisor(SupervisedBus(), lambda: None, dict.fromkeys(JOINTS, (1800, 2200)),
                               3, 60, value, 1., 10.)



class CollectorRecoveryTests(unittest.TestCase):
    """Exercise the real collector lifecycle with injected motors and operator."""

    def test_collector_completion_failure_and_partial_enable(self):
        import json
        from pathlib import Path
        import tempfile
        from robot_calibration.collection import collect_robot_calibration as c
        from robot_calibration.collection import stop_recovery as r

        for failure in (None, 'send', 'interrupt', 'enable'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                bus = SupervisedBus()
                bus.torque = dict.fromkeys(JOINTS, 0)
                calls = []

                class Operator:
                    def notify(self, text):
                        pass

                    def event(self, timeout):
                        calls.append('operator_release')
                        self_test.assertEqual(bus.releases, 0)
                        self_test.assertEqual(bus.disconnections, [])
                        with self_test.assertRaises(RuntimeError):
                            arm.send(c.MockArm().read())
                        return 'supported-release'

                    def close(self):
                        calls.append('console_closed')

                class Arm(c.LeRobotArm):
                    def __init__(self):
                        self.bus, self.enabled = bus, False

                    def connect(self):
                        return {'fake': True}

                    def read(self):
                        return c.MockArm().read()

                    def prepare_recovery(self, path, limits, output):
                        self.operator = Operator()
                        self.motion_cancelled = False
                        self.recovery_poll_s = .001
                        self.recovery = r.StopSupervisor(bus, self.cancel_motion,
                            dict.fromkeys(JOINTS, (1800, 2200)), 3, 60, 1, 1, 10)

                    def enable(self, initial):
                        self.enabled = True
                        bus.torque = dict.fromkeys(JOINTS, 1)
                        if failure == 'enable':
                            bus.torque['elbow_flex'] = 0
                            raise OSError('partial enable')

                    def send(self, command):
                        if self.motion_cancelled:
                            return super().send(command)
                        if failure == 'interrupt':
                            raise KeyboardInterrupt('stop')
                        if failure == 'send':
                            raise OSError('lost command')
                        return super().send(command)

                self_test = self
                arm = Arm()
                args = c.parse_args(['--output', str(Path(directory)/'run'), '--mode', 'joint_response'])
                if failure:
                    with self.assertRaises(KeyboardInterrupt if failure == 'interrupt' else OSError):
                        c.run(args, arm, None, [('hold', 0)])
                else:
                    c.run(args, arm, None, [('hold', 0)])
                self.assertEqual(bus.releases, 1)
                self.assertEqual(bus.disconnections, [False])
                self.assertEqual(calls, ['operator_release', 'console_closed'])
                meta = json.loads((args.output/'metadata.json').read_text())
                self.assertFalse(meta['recovery']['motion_ready'])
                self.assertEqual(meta['recovery']['state'], 'released')

    def test_recovery_loop_interrupts_and_failed_release_stay_supervised(self):
        from robot_calibration.collection.stop_recovery import supervise
        s, bus, clock, _ = SupervisorTests().setup_session()
        events = iter(('stop', 'input-closed', KeyboardInterrupt(), 'supported-release',
                       'power-off-supported'))

        class Operator:
            def notify(self, message):
                pass

            def event(self, timeout):
                event = next(events)
                if isinstance(event, BaseException):
                    raise event
                if event == 'supported-release':
                    bus.bad_release = True
                return event

        result = supervise(s, 'failure', Operator(), .01, sleep=lambda _: None)
        self.assertEqual(result['last_event']['event'], 'power_removed_by_operator')
        self.assertEqual(bus.disconnections, [])
        self.assertEqual(bus.releases, 0)
        self.assertEqual(len(bus.writes), 1)

    def test_example_cannot_enable_and_logger_failure_latches(self):
        from pathlib import Path
        from robot_calibration.collection.stop_recovery import load_config
        with self.assertRaises(ValueError):
            load_config(Path(__file__).resolve().parents[1]/'configs/recovery.example.json')
        s, bus, _, _ = SupervisorTests().setup_session()
        def fail(row):
            raise OSError('disk full')
        s.emit = fail
        s.begin('completed')
        self.assertTrue(s.logging_failed)
        self.assertEqual(s.state, 'intervention_required')
        self.assertEqual(bus.releases, 0)

    def test_hardware_preparation_converts_bounds_and_rejects_bad_config(self):
        import json
        from pathlib import Path
        from types import SimpleNamespace
        import tempfile
        from unittest.mock import patch
        from robot_calibration.collection import collect_robot_calibration as c
        from robot_calibration.collection import stop_recovery as r

        bus = SupervisedBus()
        bus.torque = dict.fromkeys(JOINTS, 0)
        bus.calibration = {j: SimpleNamespace(id=i, range_min=1000, range_max=3000)
                           for i, j in enumerate(JOINTS)}
        bus._unnormalize = lambda values: {i: int(2000 + 10*v) for i, v in values.items()}
        arm = c.LeRobotArm.__new__(c.LeRobotArm)
        arm.bus, arm.enabled = bus, False
        config = dict(drift_ticks=3, max_temperature=60, read_deadline_s=.5,
                      poll_deadline_s=1, hold_deadline_s=10, poll_s=.1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output/'config.json'
            path.write_text(json.dumps(config))
            with patch.object(r, 'TerminalOperator') as operator:
                arm.prepare_recovery(path, dict.fromkeys(JOINTS, (-10, 10)), output)
                self.assertEqual(arm.recovery.bounds, dict.fromkeys(JOINTS, (1900, 2100)))
                self.assertEqual(bus.writes, [])
                arm.close()
                operator.return_value.close.assert_called_once()
            self.assertEqual(bus.disconnections, [False])
            self.assertIn('prepared', (output/'recovery.jsonl').read_text())

    def test_pre_enable_failure_only_disconnects_passively(self):
        from pathlib import Path
        import tempfile
        from robot_calibration.collection import collect_robot_calibration as c

        class Arm(c.LeRobotArm):
            def __init__(self):
                self.bus, self.enabled = SupervisedBus(), False

            def connect(self):
                raise OSError('handshake failed')

        with tempfile.TemporaryDirectory() as directory:
            arm = Arm()
            args = c.parse_args(['--output', str(Path(directory)/'run')])
            with self.assertRaises(OSError):
                c.run(args, arm, None, [('read', 0)])
            self.assertEqual(arm.bus.disconnections, [False])
            self.assertEqual(arm.bus.releases, 0)
            self.assertEqual(arm.bus.writes, [])



if __name__ == '__main__':
    unittest.main()
