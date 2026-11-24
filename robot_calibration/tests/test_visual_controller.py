"""Hardware-free input lifecycle and real URDF IK checks."""
import threading
import unittest
try:
    import numpy as np
    from robot_calibration.collection.collect_robot_calibration import MockArm
    from robot_calibration.collection.visual_controller import JogController, WristIK, ARM, JOINTS
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f'Controller tests need numpy/scipy: {exc}')


class VisualControllerTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.profile = dict(joint_order=list(ARM), joint_mapping=dict(joints={
            j: dict(sign=1, scale=1, offset_deg=0) for j in ARM}))
        self.arm = MockArm()
        self.arm.enable(self.arm.read())
        self.events = []
        self.controller = JogController(self.arm, {j: [-80, 80] for j in ARM} | {'gripper': [0, 100]},
            WristIK(self.profile), lambda event, **data: self.events.append((event, data)), lambda: self.now)

    def advance(self, heartbeat=True):
        self.now += .1
        if heartbeat:
            self.controller.pulse()
        self.controller.tick()

    def test_tap_hold_release_and_immediate_second_tap(self):
        c = self.controller
        c.press('joint', 'shoulder_pan', 1, .25)
        c.tick()
        self.assertEqual(c.goal['shoulder_pan'], .25)
        for ticks in [1]*3 + [2]*7:
            previous = c.goal['shoulder_pan']
            self.advance()
            self.assertAlmostEqual(c.goal['shoulder_pan'] - previous, ticks*360/4095)
        c.release()
        held = dict(c.goal)
        for _ in range(10):
            self.advance()
        self.assertEqual(c.goal, held)
        c.press('joint', 'shoulder_pan', -1, .25)
        c.tick()
        self.assertAlmostEqual(c.goal['shoulder_pan'], held['shoulder_pan'] - .25)

    def test_panel_mouse_keyboard_modes_and_focus(self):
        import tkinter as tk
        from unittest.mock import patch
        from robot_calibration.collection.visual_controller import panel
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f'No display: {exc}')
        errors = []
        def exercise():
            try:
                root.update()
                def descendants(widget):
                    return [item for child in widget.winfo_children()
                            for item in [child, *descendants(child)]]
                def find(text):
                    return next(w for w in descendants(root)
                                if 'text' in w.keys() and w.cget('text') == text)
                plus = find('+')
                plus.event_generate('<ButtonPress-1>', x=10, y=10)
                self.controller.tick()
                plus.event_generate('<ButtonRelease-1>', x=10, y=10)
                self.assertEqual(self.controller.goal['shoulder_pan'], .25)
                self.assertIsNone(self.controller.input)
                find('Wrist mode').invoke()
                up = find('Up +Z')
                up.event_generate('<ButtonPress-1>', x=10, y=10)
                self.assertEqual(self.controller.input['mode'], 'wrist')
                root.event_generate('<FocusOut>')
                self.assertIsNone(self.controller.input)
                find('Joint mode').invoke()
                plus = find('+')
                plus.focus_force()
                root.update()
                plus.event_generate('<KeyPress-space>')
                self.controller.tick()
                plus.event_generate('<KeyRelease-space>')
                self.assertEqual(self.controller.goal['shoulder_pan'], .5)
                root.update()
                plus.event_generate('<KeyPress-w>')
                self.controller.tick()
                plus.event_generate('<KeyRelease-w>')
                plus.event_generate('<KeyPress-w>')  # X11 repeat pair
                self.controller.tick()
                self.assertEqual(self.controller.goal['shoulder_lift'], .25)
                self.advance()
                plus.event_generate('<KeyRelease-w>')
                root.update()
                self.advance()
                self.assertEqual(self.controller.goal['shoulder_lift'], .5)
                self.assertIsNone(self.controller.input)
                plus.event_generate('<KeyPress-w>')
                plus.event_generate('<KeyPress-Escape>')
                plus.event_generate('<KeyRelease-w>')
                plus.event_generate('<KeyPress-w>')  # Still held; ignore repeat.
                self.advance()
                self.assertIsNone(self.controller.input)
                self.assertEqual(self.controller.goal['shoulder_lift'], .5)
                plus.event_generate('<KeyRelease-w>')
                root.update()
                plus.event_generate('<KeyPress-w>')
                self.controller.tick()
                self.assertAlmostEqual(self.controller.goal['shoulder_lift'], .75)
                plus.event_generate('<KeyRelease-w>')
                root.update()
                for box, key, joint in zip(
                        (w for w in descendants(root) if w.winfo_class() == 'TSpinbox'),
                        ('w', 'y'), ('shoulder_lift', 'gripper')):
                    box.focus_force()
                    root.update()
                    box.delete(0, 'end')
                    for char in ('0', 'period', '2', '5'):
                        box.event_generate(f'<KeyPress-{char}>')
                    before = self.controller.goal[joint]
                    box.event_generate(f'<KeyPress-{key}>')
                    self.controller.tick()
                    box.event_generate(f'<KeyRelease-{key}>')
                    root.update()
                    self.assertEqual(box.get(), '0.25')
                    self.assertEqual(self.controller.goal[joint], before + .25)
                find('STOP — End session').invoke()
                self.assertTrue(self.controller.exit)
            except BaseException as exc:
                errors.append(exc)
            finally:
                root.destroy()
        root.after(100, exercise)
        with patch('tkinter.Tk', return_value=root):
            panel(self.controller)
        if errors:
            raise errors[0]

    def test_short_click_is_exactly_one_increment(self):
        c = self.controller
        c.press('joint', 'shoulder_pan', 1, .25)
        c.release(finish_tap=True)
        c.tick()
        for _ in range(10):
            self.advance()
        self.assertEqual(c.goal['shoulder_pan'], .25)
        self.assertIsNone(c.input)

    def test_stale_input_no_replay_on_heartbeat(self):
        c = self.controller
        c.press('joint', 'wrist_roll', 1, .5)
        c.tick()
        self.now += .3
        c.tick()
        held = dict(c.goal)
        self.advance()
        self.assertEqual(c.goal, held)
        self.assertIsNone(c.input)
        self.assertTrue(self.arm.enabled)

    def test_pending_release_focus_and_stop_cancel(self):
        c = self.controller
        for reason in ('Released', 'Focus lost', 'Disconnected'):
            c.press('joint', 'elbow_flex', 1, .5)
            c.release(reason)
            c.tick()
            self.assertEqual(c.goal['elbow_flex'], 0)
        c.stop()
        c.press('joint', 'elbow_flex', 1, .5)
        c.tick()
        self.assertEqual(c.goal['elbow_flex'], 0)

    def test_release_while_ik_running(self):
        c = self.controller
        begun, resume = threading.Event(), threading.Event()
        def slow(*args):
            begun.set()
            resume.wait(2)
            return dict(c.goal, shoulder_pan=.5)
        c.ik.move = slow
        c.press('wrist', 0, 1, .5)
        worker = threading.Thread(target=c.tick)
        worker.start()
        self.assertTrue(begun.wait(2))
        c.release()
        resume.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(c.goal['shoulder_pan'], 0)

    def test_gripper_units_limits_and_invalid_input(self):
        c = self.controller
        c.press('joint', 'gripper', -1, .5)
        c.tick()
        self.assertEqual(c.goal['gripper'], 49.5)
        c.limits['gripper'] = [49.5, 100]
        self.advance()
        self.assertEqual(c.goal['gripper'], 49.5)
        self.assertIn('Rejected', c.status)
        for step in (float('nan'), float('inf'), -1, 2):
            with self.assertRaises(ValueError):
                c.press('joint', 'gripper', 1, step)

    def test_ik_from_measured_pose_and_infeasible(self):
        c = self.controller
        measured = dict(c.measured)
        pose = c.ik.pose(measured)
        accepted = 0
        for axis in range(3):
            for sign in (-1, 1):
                try:
                    target = c.ik.move(measured, axis, sign * .5, pose[:3, :3])
                except ValueError:
                    continue
                delta = (c.ik.pose(target)[:3, 3] - pose[:3, 3]) * 1000
                expected = np.zeros(3)
                expected[axis] = sign * .5
                self.assertLess(np.linalg.norm(delta - expected), .1)
                accepted += 1
        self.assertGreaterEqual(accepted, 2)
        with self.assertRaises(ValueError):
            c.ik.move(measured, 0, 10000, pose[:3, :3])
        # The IK receives the measurement, not the previous commanded pose.
        c.goal['shoulder_pan'] = .2
        seen = []
        def capture(measurement, *args):
            seen.append(dict(measurement))
            return dict(c.goal)
        c.ik.move = capture
        c.press('wrist', 0, 1, .5)
        c.tick()
        self.assertEqual(seen[0], c.measured)

    def test_real_worker_hold_cadence_and_release(self):
        import time
        c = self.controller
        c.clock = time.monotonic
        c.heartbeat = time.monotonic()
        worker = threading.Thread(target=c.run)
        c.press('joint', 'shoulder_pan', 1, .5)
        worker.start()
        try:
            until = time.monotonic() + .36
            while time.monotonic() < until:
                c.pulse()
                time.sleep(.02)
            c.release()
            held = dict(c.goal)
            time.sleep(.15)
            self.assertEqual(c.goal, held)
            self.assertGreaterEqual(c.goal['shoulder_pan'], .5 + 1.99*360/4095)
            self.assertLessEqual(c.goal['shoulder_pan'], .5 + 3.01*360/4095)
        finally:
            c.stop()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(c.fault)

    def test_all_joint_feedback_guards_keep_default_placement_behavior(self):
        from unittest.mock import Mock
        from robot_calibration.collection.placement_guards import check_feedback
        bus = Mock()
        start = dict.fromkeys(JOINTS, 1000)
        goal = dict(start, wrist_roll=1010, gripper=1010)
        actual = dict(goal)
        def read(name, **kwargs):
            return dict({'Present_Position': actual, 'Goal_Position': goal,
                'Torque_Enable': dict.fromkeys(JOINTS, 1),
                'Status': dict.fromkeys(JOINTS, 0),
                'Present_Temperature': dict.fromkeys(JOINTS, 25),
                'Present_Load': dict.fromkeys(JOINTS, 0),
                'Present_Current': dict.fromkeys(JOINTS, 0)}[name])
        bus.sync_read.side_effect = read
        bounds = dict.fromkeys(JOINTS, (900, 1100))
        with self.assertRaisesRegex(RuntimeError, 'drift'):
            check_feedback(bus, start, goal, bounds)
        check_feedback(bus, start, goal, bounds, active=JOINTS, tracking_ticks=8)
        actual['gripper'] -= 9
        with self.assertRaisesRegex(RuntimeError, 'tracking'):
            check_feedback(bus, start, goal, bounds, active=JOINTS, tracking_ticks=8)

    def test_gripper_settling_pauses_writes_and_times_out(self):
        from unittest.mock import Mock
        from types import SimpleNamespace
        from robot_calibration.collection.collect_robot_calibration import LeRobotArm
        c = self.controller
        arm = Mock(spec=LeRobotArm)
        c.arm = arm
        bus = arm.bus = Mock()
        bus.calibration = {j: SimpleNamespace(id=i) for i, j in enumerate(JOINTS)}
        arm.normalized_positions.side_effect = lambda values: LeRobotArm.normalized_positions(arm, values)
        bus._normalize.side_effect = lambda d: {i: x / 10 for i, x in d.items()}
        c.raw = lambda values: {j: round(v * 10) for j, v in values.items()}
        arm.recovery = SimpleNamespace(bounds=dict.fromkeys(JOINTS, (-1000, 1000)))
        actual = c.raw(c.goal)
        def read(name, **kwargs):
            if name == 'Present_Position':
                return dict(actual)
            if name == 'Goal_Position':
                return c.raw(c.goal)
            return dict.fromkeys(JOINTS, int(name in ('Torque_Enable', 'Lock')))
        bus.sync_read.side_effect = read
        c.press('joint', 'gripper', -1, 1)
        c.tick()
        self.now = .0065
        actual['gripper'] -= 1  # Nine ticks behind the new goal.
        c.tick()
        arm.send.assert_called_once()
        self.now = .12
        actual['gripper'] -= 8
        c.release()
        c.tick()
        self.assertIsNone(c.gripper_from)
        arm.send.assert_called_once()
        c.press('joint', 'gripper', -1, 1)
        c.tick()
        self.now += .31
        c.tick()
        self.assertTrue(c.close_blocked)
        self.assertEqual(arm.send.call_count, 2)
        held = c.goal['gripper']
        c.press('joint', 'shoulder_pan', 1, .5)
        c.tick()
        self.assertEqual(c.goal['gripper'], held)
        c.press('joint', 'gripper', 1, 1)
        c.tick()
        self.assertGreater(c.goal['gripper'], held)
        c.tick()  # Reopening is confirmed by the next encoder sample.
        self.assertFalse(c.close_blocked)
        c.press('joint', 'gripper', 1, 1)
        c.tick()
        self.now += .31
        with self.assertRaisesRegex(RuntimeError, '300 ms'):
            c.tick()

    def test_stale_during_ik_discards_target(self):
        c = self.controller
        def slow(*args):
            self.now += .3
            return dict(c.goal, shoulder_pan=.5)
        c.ik.move = slow
        c.press('wrist', 0, 1, .5)
        c.tick()
        self.assertEqual(c.goal['shoulder_pan'], 0)

    def test_feedback_failure_latches_and_prevents_writes(self):
        c = self.controller
        def failed():
            raise ConnectionError('lost bus')
        c.feedback = failed
        c.press('joint', 'shoulder_pan', 1, .5)
        c.run()
        self.assertTrue(c.fault)
        self.assertFalse(any(e == 'command' for e, _ in self.events))


if __name__ == '__main__':
    unittest.main()
