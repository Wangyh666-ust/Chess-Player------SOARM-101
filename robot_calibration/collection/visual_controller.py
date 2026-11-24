"""Local Tk jog panel. Mock by default; hardware requires a reviewed session.

One worker owns the adapter. Input is a replaceable lease, never a trajectory
queue. Releasing cancels even an IK calculation that has not yet been sent.
"""
import argparse
import json
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.calibration import arm_from_model, arm_to_model
from robot_calibration.records import ARM, JOINTS, read_json, sha256
from robot_calibration.kinematics import URDF, Robot, gripper_link_pose, pose_fk, solve_pose
from .collect_robot_calibration import (LeRobotArm, MockArm, check_positions, read_limits, require_unowned_port)
from . import placement_guards as guards

PERIOD = .1
LEASE = .25


class WristIK:
    def __init__(self, profile):
        self.profile = profile
        self.robot = Robot(str(URDF))
        # Use the existing FK/IK at the wrist-roll axis, not the fingertip.
        self.robot.joints[-1] = dict(self.robot.joints[-1], origin=np.eye(4))

    def pose(self, measured):
        return gripper_link_pose(self.robot, np.asarray(arm_to_model(self.profile, measured)))

    def move(self, measured, axis, distance_mm, orientation):
        seed = arm_to_model(self.profile, measured)
        target = self.pose(measured)[:3, 3].copy()
        target[axis] += distance_mm / 1000
        fit = solve_pose(self.robot, target, Rotation.from_matrix(orientation).as_quat(), seed)
        # solve_pose's downward-tool condition is for picking, not jogging.
        if (fit['error_mm'] > min(.3, abs(distance_mm) * .2)
                or fit['rot_error_deg'] > 3 or fit['margin_deg'] < 5):
            raise ValueError('Wrist move infeasible: position/orientation/joint margin')
        result = arm_from_model(self.profile, fit['q'])
        result['gripper'] = measured['gripper']
        return result


class PanelRecoveryOperator(guards.PowerOperator):
    """Bridge the local acknowledgement to the serial owner's recovery loop."""
    def __init__(self, wrapped):
        super().__init__(wrapped)
        self.awaiting = threading.Event()
        self.confirmed = threading.Event()

    def event(self, timeout):
        self.awaiting.set()
        if self.confirmed.is_set():
            self.awaiting.clear()
            self.confirmed.clear()
            return 'power-off-supported'
        return super().event(timeout)


def confirmed_pose_ticks(bus):
    ticks = bus.sync_read('Present_Position', normalize=False, num_retry=0)
    if set(ticks) != set(JOINTS) or any(type(value) is not int for value in ticks.values()):
        raise RuntimeError(f'Incomplete confirmed-pose encoder readings: {ticks}')
    return ticks


class JogController:
    def __init__(self, arm, limits, ik, emit=lambda *a, **kw: None, clock=time.monotonic, *, elbow_tracking_ticks=8):
        self.elbow_tracking_ticks = elbow_tracking_ticks
        self.arm, self.limits, self.ik, self.emit, self.clock = arm, limits, ik, emit, clock
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.input = None
        self.generation = 0
        self.heartbeat = clock()
        self.measured = arm.read()
        check_positions(self.measured, limits)
        self.goal = dict(self.measured)
        self.status = 'Holding — ready'
        self.fault = False
        self.disconnected = threading.Event()
        self.exit = False
        self.arm_from = {}
        self.arm_settling = False
        self.pan_ticks = None
        self.gripper_ticks = None
        self.gripper_from = None
        self.close_blocked = False
        self.gripper_closing = False
        self.gripper_deadline = 0.0
        self.last_send = -float('inf')
        self.last_feedback = clock()

    def press(self, mode, axis, direction, step, *, calibrated_tap=False):
        minimum = .25 if mode == 'joint' and axis in ARM and axis not in ('shoulder_pan', 'shoulder_lift') and not calibrated_tap else .05
        if (mode not in ('joint', 'wrist') or direction not in (-1, 1)
                or not np.isfinite(step) or not minimum <= step <= 1
                or (mode == 'joint' and axis not in JOINTS)
                or (mode == 'wrist' and axis not in (0, 1, 2))):
            raise ValueError(f'Select an increment from {minimum} to 1')
        with self.lock:
            if self.fault or self.exit:
                return
            self.generation += 1
            self.heartbeat = self.clock()
            self.input = dict(mode=mode, axis=axis, direction=direction, step=step,
                              first=True, pan_repeats=0, orientation=self.ik.pose(self.measured)[:3, :3])
            self.wake.set()

    def release(self, reason='Released — holding', finish_tap=False):
        with self.lock:
            if self.exit:
                return
            if finish_tap and self.input is not None and self.input['first']:
                # A completed short click is one tap, not a held-input backlog.
                self.input['single'] = True
                self.wake.set()
                return
            self.input = None
            self.generation += 1
            if not self.fault:
                self.status = reason
            self.wake.set()

    def pulse(self):
        with self.lock:
            self.heartbeat = self.clock()

    def stop(self, reason='STOP — supervised recovery'):
        with self.lock:
            self.release(reason)
            self.exit = True

    def snapshot(self):
        with self.lock:
            return dict(measured=dict(self.measured), goal=dict(self.goal), status=self.status,
                        fault=self.fault, exit=self.exit,
                        wrist_mm=(self.ik.pose(self.measured)[:3, 3] * 1000).tolist())

    def raw(self, values):
        return self.arm.raw_positions(values)

    def apply_raw_arm_goal(self, requested, current):
        """Latch a corrected goal and verify SDK quantization before writing."""
        changed = {joint: value for joint, value in requested.items() if value != current[joint]}
        if not changed:
            return None
        if not changed.keys() <= set(ARM):
            raise ValueError('Raw arm correction cannot move the gripper')
        angles = self.arm.normalized_positions(changed)
        # LeRobot truncates on write. Aim inside the desired register bin.
        goal = self.goal | {joint: angle + .25 * 360 / 4095 for joint, angle in angles.items()}
        check_positions(goal, self.limits)
        if self.raw(goal) != requested:
            raise RuntimeError('Encoder correction failed conversion to the requested register')
        self.arm.send(goal)
        self.goal = goal
        self.last_send = time.monotonic()
        return goal

    def feedback(self):
        if isinstance(self.arm, MockArm):
            # MockArm advances only on send; emulate settling between samples.
            for j in JOINTS:
                self.arm.position[j] += .8 * (self.arm.goal[j] - self.arm.position[j])
            measured = self.arm.read()
        else:
            expected = self.raw(self.goal)
            if self.clock() - self.last_send >= PERIOD:
                self.arm_from.clear()
            row = guards.check_feedback(self.arm.bus, expected, expected,
                                        self.arm.recovery.bounds,
                                        lambda data: self.emit('feedback', registers=data),
                                        tracking_ticks=self.elbow_tracking_ticks, active=JOINTS, gripper_from=self.gripper_from, arm_from=self.arm_from)
            self.pan_ticks = row['Present_Position']['shoulder_pan']
            self.gripper_ticks = row['Present_Position']['gripper']
            measured = self.arm.normalized_positions(row['Present_Position'])
            self.arm_settling = any(abs(row['Present_Position'][j] - expected[j]) > 8 for j in self.arm_from)
            if self.gripper_from is not None:
                error = abs(expected['gripper'] - row['Present_Position']['gripper'])
                if error <= 8:
                    if not self.close_blocked or not self.gripper_closing:
                        self.gripper_from = None
                        self.close_blocked = False
                elif self.gripper_closing and measured['gripper'] > self.goal['gripper']:
                    if self.clock() >= self.gripper_deadline and not self.close_blocked:
                        self.close_blocked = True
                        self.emit('gripper_close_blocked', tracking_error_ticks=error)
                elif not self.close_blocked and self.clock() >= self.gripper_deadline:
                    raise RuntimeError(f'Gripper still {error} ticks from goal after 300 ms')
            for name, required in (('Operating_Mode', 0), ('Lock', 1)):
                values = self.arm.bus.sync_read(name, normalize=False, num_retry=0)
                if set(values) != set(JOINTS) or any(type(v) is not int or v != required for v in values.values()):
                    raise RuntimeError(f'{name} changed')
        check_positions(measured, self.limits)
        with self.lock:
            self.measured = measured
        self.last_feedback = self.clock()

    def tick(self):
        begun = self.clock()
        self.feedback()
        if self.clock() - begun > .5:
            raise RuntimeError('Complete feedback deadline exceeded')
        with self.lock:
            if self.clock() - self.heartbeat > LEASE:
                self.release('Input stale — holding; press again')
            request, generation = self.input, self.generation
            if request is None or self.exit or self.arm_settling or (self.gripper_from is not None and not self.close_blocked):
                return
            if self.close_blocked and request['axis'] == 'gripper' and request['direction'] < 0:
                self.release('Gripper close blocked — holding grip; open to reset')
                return
            # First tap has no repeat delay; holds start at the next feedback slot.
            if not request['first'] and self.clock() - self.last_send < PERIOD - .001:
                return
            measured = dict(self.measured)
            goal = dict(self.goal)
        amount = request['step'] if request['first'] else min(request['step'], .5)
        pan_hold = request['axis'] == 'shoulder_pan' and not request['first']
        if pan_hold:
            # Cumulative servo goals: one encoder tick initially, then two per slot.
            tick_deg = 360 / 4095
            amount = tick_deg * (1 if request['pan_repeats'] < 3 else 2)
            ahead = (goal['shoulder_pan'] - measured['shoulder_pan']) * request['direction']
            if ahead + amount > 6 * tick_deg:
                return  # Wait for feedback; never accumulate missed steps.
        amount *= request['direction']
        try:
            if request['mode'] == 'joint':
                target = dict(goal)
                target[request['axis']] += amount
            else:
                target = self.ik.move(measured, request['axis'], amount, request['orientation'])
                target['gripper'] = goal['gripper']
            check_positions(target, self.limits)
            # At most one degree/unit per tap; holds <= 5 degrees/units per second.
            cap = 1 if request['first'] else .5
            if max(abs(target[j] - goal[j]) for j in JOINTS) > cap + 1e-8:
                raise ValueError('Joint speed bound: select a smaller wrist increment')
        except ValueError as exc:
            self.release(f'Rejected — {exc}')
            self.emit('rejected', reason=str(exc))
            return
        with self.lock:
            # Recheck after feedback/IK. No obsolete command may leave this worker.
            if (generation != self.generation or self.exit
                    or self.clock() - self.heartbeat > LEASE
                    or self.clock() - self.last_feedback > PERIOD):
                return
            if isinstance(self.arm, LeRobotArm) and target['gripper'] != goal['gripper']:
                self.gripper_closing = target['gripper'] < goal['gripper']
                if not self.close_blocked:
                    self.gripper_from = self.raw(goal)['gripper']
                elif target['gripper'] >= measured['gripper']:
                    self.gripper_from = self.raw(measured)['gripper']
                    self.close_blocked = False
                self.gripper_deadline = self.clock() + .3
            if isinstance(self.arm, LeRobotArm):
                previous = self.raw(goal)
                self.arm_from = {j: previous[j] for j in ARM if target[j] != goal[j]}
            self.arm.send(target)
            self.goal = target
            if pan_hold:
                request['pan_repeats'] += 1
            request['first'] = False
            self.last_send = self.clock()
            self.status = f"Moving — {request['mode']}"
            if request.get('single'):
                self.input = None
                self.status = 'Tap complete — holding'
            self.emit('command', command=target)

    def run(self):
        deadline = self.clock()
        try:
            while not self.exit:
                self.wake.wait(max(0, deadline - self.clock()))
                self.wake.clear()
                if self.clock() - deadline > .2:
                    raise RuntimeError('Feedback schedule deadline exceeded')
                if isinstance(self.arm, LeRobotArm):
                    if self.arm.operator.event(0) is not None:
                        self.stop('Terminal input — supervised recovery')
                tick_started = self.clock()
                self.tick()
                deadline = tick_started + PERIOD
        except BaseException as exc:
            with self.lock:
                self.fault = True
                self.input = None
                self.status = f'FAULT — {exc}; use owner terminal'
                self.exit = True
            self.emit('fault', error=repr(exc))
        finally:
            if isinstance(self.arm, LeRobotArm):
                if self.arm.enabled:
                    self.arm.operator = PanelRecoveryOperator(self.arm.operator)
                    self.arm.recover(self.status)
                self.arm.close()
                self.disconnected.set()
            else:
                self.arm.close()


def panel(controller, record_e4=False, validation_reset=False, *, setup=False, allow_gripped=False, f4_placements=False, low_hover=False, board_contacts=False, board_23=False, board_start=1, repeat_60=False):
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    root.title('SO-ARM101 • Local visual controller')
    root.geometry('760x710' if validation_reset or f4_placements or low_hover or board_contacts else '760x650')
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill='both', expand=True)
    title = 'MOCK — no hardware' if isinstance(controller.arm, MockArm) else 'HARDWARE — operator supervised'
    ttk.Label(frame, text=title, font=('', 18, 'bold')).pack(anchor='w')
    ttk.Label(frame, text='Wrist point: wrist_roll axis / gripper_link origin (not fingertip).\n'
              'Base frame: forward +X, left +Y, up +Z; directions follow the model base axes.').pack(anchor='w', pady=10)
    status = tk.StringVar()
    ttk.Label(frame, textvariable=status, wraplength=710).pack(anchor='w')
    measurements = tk.StringVar()
    ttk.Label(frame, textvariable=measurements, font=('monospace', 11)).pack(anchor='w', pady=10)
    mode = tk.StringVar(value='joint')
    step = tk.StringVar(value='0.25')
    grip_step = tk.StringVar(value='0.5')
    toolbar = ttk.Frame(frame)
    toolbar.pack(fill='x')
    for name in ('joint', 'wrist'):
        ttk.Radiobutton(toolbar, text=name.title() + ' mode', variable=mode, value=name,
                        command=lambda: (controller.release('Mode changed — holding'), show_mode())).pack(side='left')
    ttk.Label(toolbar, text='  Per tap (degrees / mm):').pack(side='left')
    angle_box = ttk.Spinbox(toolbar, from_=.05, to=1, increment=.05, textvariable=step, width=6)
    angle_box.pack(side='left')
    controls = ttk.Frame(frame)
    controls.pack(fill='x', pady=12)
    pressed = set()
    key_release_job = [None]

    def release(event=None, finish_tap=False):
        if key_release_job[0] is not None:
            root.after_cancel(key_release_job[0])
            key_release_job[0] = None
        pressed.clear()
        controller.release(finish_tap=finish_tap)

    def button(parent, label, selected_mode, axis, direction, value):
        widget = ttk.Button(parent, text=label)
        widget.pack(side='left', padx=5, pady=3, fill='x', expand=True)
        def down(event):
            key = getattr(event, 'keysym', None) or 'pointer'
            if key_release_job[0] is not None:
                root.after_cancel(key_release_job[0])
                key_release_job[0] = None
            if key in pressed:
                return 'break'
            pressed.add(key)
            try:
                controller.press(selected_mode, axis, direction, float(value.get()))
            except ValueError as exc:
                controller.release(str(exc))
            return 'break'
        widget.bind('<ButtonPress-1>', down)
        widget.bind('<KeyPress-space>', down)
        def key_up(event):
            # X11 auto-repeat emits release/press pairs; coalesce those in idle.
            key_release_job[0] = root.after_idle(lambda: release(finish_tap=True))
            return 'break'
        widget.bind('<KeyRelease-space>', key_up)
        widget.bind('<Leave>', release)
        return widget

    # Upper row increases; the key directly below decreases.
    key_pairs = dict(zip(JOINTS, zip('asdfgh', 'qwerty')))
    def shortcut_down(event):
        key = event.keysym.lower()
        if mode.get() != 'joint':
            return
        if event.state & (0x4 | 0x8 | 0x40):  # Control, Alt, or Super shortcut
            return
        for axis, pair in key_pairs.items():
            if key not in pair:
                continue
            if key in pressed:
                if key_release_job[0] is not None:
                    root.after_cancel(key_release_job[0])
                    key_release_job[0] = None
                return 'break'
            if pressed:
                return 'break'
            pressed.add(key)
            try:
                value = grip_step if axis == 'gripper' else step
                controller.press('joint', axis, 1 if key == pair[1] else -1, float(value.get()))
            except ValueError as exc:
                controller.release(str(exc))
            return 'break'

    def shortcut_up(event):
        if event.keysym.lower() in pressed and event.keysym.lower() in 'asdfghqwerty':
            if key_release_job[0] is None:
                key_release_job[0] = root.after_idle(lambda: release(finish_tap=True))
            return 'break'

    root.bind('<KeyPress>', shortcut_down)
    root.bind('<KeyRelease>', shortcut_up)
    def show_mode():
        angle_box.configure(from_=.25 if mode.get() == 'joint' else .05)
        if mode.get() == 'joint':
            try:
                if float(step.get()) < .25:
                    step.set('0.25')
            except ValueError:
                step.set('0.25')
        for child in controls.winfo_children():
            child.destroy()
        if mode.get() == 'joint':
            for joint in ARM:
                row = ttk.Frame(controls)
                row.pack(fill='x')
                ttk.Label(row, text=f'{joint}  [{key_pairs[joint][0].upper()}− / {key_pairs[joint][1].upper()}+]', width=27).pack(side='left')
                button(row, '−', 'joint', joint, -1, step)
                button(row, '+', 'joint', joint, 1, step)
        else:
            for axis, negative, positive in ((2, 'Down −Z', 'Up +Z'), (1, 'Right −Y', 'Left +Y'),
                                               (0, 'Back −X', 'Forward +X')):
                row = ttk.Frame(controls)
                row.pack(fill='x')
                button(row, negative, 'wrist', axis, -1, step)
                button(row, positive, 'wrist', axis, 1, step)
    show_mode()
    grip = ttk.Frame(frame)
    grip.pack(fill='x')
    ttk.Label(grip, text='Gripper: calibrated 0–100\nJoint keys: H close / Y open\nStep in normalized units').pack(side='left')
    gripper_box = ttk.Spinbox(grip, from_=.05, to=1, increment=.05, textvariable=grip_step, width=6)
    gripper_box.pack(side='left')
    # Handle jog letters before Tk's spinbox class inserts them as text.
    for box in (angle_box, gripper_box):
        box.bind('<KeyPress>', shortcut_down)
    button(grip, 'Close −', 'joint', 'gripper', -1, grip_step)
    button(grip, 'Open +', 'joint', 'gripper', 1, grip_step)
    finish_bar = ttk.Frame(frame)
    finish_bar.pack(fill='x', pady=10)
    start_mode = None
    if setup:
        root.title('Position arm for supervised test')
        def finish_setup(mode):
            nonlocal start_mode
            controller.release('Setup complete — holding')
            start_mode = mode
            root.destroy()
        ttk.Button(finish_bar, text='Start test from this pose',
                   command=lambda: finish_setup('next')).pack(side='left', expand=True)
        if allow_gripped:
            ttk.Button(finish_bar, text='Start already gripping',
                       command=lambda: finish_setup('already_gripped')).pack(side='left', expand=True)
    elif board_contacts:
        # File/rank of the lower-left cell at each shared intersection.
        points = ['d4', 'd2', 'b2', 'a1', 'f2', 'g1', 'f4',
                  'f6', 'g7', 'd6', 'b6', 'a7', 'b4']
        points = [(cell, 0) for cell in points]
        if board_23:
            points.extend((cell, height) for height in (60, 92)
                          for cell in ('d4', 'a1', 'g1', 'g7', 'a7'))
        if repeat_60:
            points = [('d4', 60), ('g7', 60)] * 2
        index = board_start - 1
        last_save = -1.
        label = tk.StringVar()
        def prompt():
            cell, height = points[index]
            f, r = cell
            g, t = chr(ord(f)+1), int(r)+1
            return f'{index+1}/{len(points)}   {f}{r}, {g}{r}, {f}{t}, {g}{t}   •   {height} mm above board'
        label.set(prompt())
        bar = ttk.Frame(frame)
        bar.pack(before=controls, fill='x')
        ttk.Label(bar, textvariable=label).pack(side='left')
        def save_contact():
            nonlocal index, last_save
            with controller.lock:
                if controller.exit or controller.fault or index >= len(points) or time.monotonic()-last_save < 1:
                    return
                controller.release()
                cell, height = points[index]
                data = dict(number=index+1, lower_left_cell=cell, intersection=prompt(),
                            landmark='fixed fingertip', target_height_mm=height)
                if height == 0:
                    data['surface_height_mm'] = 0
                if not controller.emit('board_manual_hover' if height else 'board_contact', **data):
                    label.set(prompt() + ' — wait for fresh camera and joint feedback, then click again')
                    return
                last_save = time.monotonic()
                index += 1
            if index == len(points):
                button.configure(state='disabled')
                label.set(f'{len(points)}/{len(points)} saved. Lift clear manually, then press STOP.')
            else:
                label.set(prompt())
        button = ttk.Button(bar, text='Save measurement and advance', command=save_contact)
        button.pack(side='right')
    elif f4_placements or low_hover:
        root.title('Manual f4 low-hover release' if low_hover else 'Manual f4 placements')
        bar = ttk.Frame(frame)
        bar.pack(before=controls, fill='x')
        phase = tk.StringVar()
        ttk.Label(bar, textvariable=phase).pack(anchor='w')
        trial = 1
        stage = 'setup'

        def mark_f4(next_stage):
            nonlocal trial, stage
            if (stage, next_stage) not in (('setup', 'descent_start'), ('descent_start', 'flat_gripped'),
                                            ('flat_gripped', 'released_flat')):
                return
            with controller.lock:
                if controller.exit or controller.fault:
                    return
                controller.release()
                controller.emit('manual_f4_placement', trial=trial,
                                split='validation' if trial == 4 else 'train', stage=next_stage)
            stage = next_stage
            if stage == 'released_flat' and trial < 4:
                trial += 1
                stage = 'setup'
            start.configure(state='normal' if stage == 'setup' else 'disabled')
            released.configure(state='normal' if stage == 'flat_gripped' else 'disabled')
            update_prompt()

        def update_prompt():
            instruction = {'setup': 'Jog to a low hover holding the knight over f4, then click Start descent.',
                           'descent_start': 'Lower manually. Press Enter when flat and still gripped.',
                           'flat_gripped': 'Open manually, then click Released flat if the knight stayed flat.',
                           'released_flat': 'All four placements marked. Press STOP to finish.'}[stage]
            phase.set(f'Trial {trial}/4 ({"validation" if trial == 4 else "train"}) — {instruction}')
        if low_hover:
            phase.set('Jog to your release height above f4. Press Enter while still holding the knight.')
            def mark_hover(outcome=None):
                nonlocal stage
                expected = 'setup' if outcome is None else 'hover_marked'
                with controller.lock:
                    if stage != expected or controller.exit or controller.fault:
                        return
                    controller.release()
                    stage = 'hover_marked' if outcome is None else 'landing_observed'
                    controller.emit('manual_f4_low_hover', stage=stage, outcome=outcome)
                for widget in buttons:
                    widget.configure(state='normal' if outcome is None else 'disabled')
                phase.set('Open manually and select the observed landing.' if outcome is None
                          else 'Outcome recorded. Press STOP to finish.')
            buttons = []
            for label, value in [('Stayed upright on f4', 'upright_on_f4'),
                                 ('Tipped or landed off f4', 'tipped_or_off_f4')]:
                widget = ttk.Button(bar, text=label, state='disabled', command=lambda v=value: mark_hover(v))
                widget.pack(side='left')
                buttons.append(widget)
        else:
            start = ttk.Button(bar, text='Start descent', command=lambda: mark_f4('descent_start'))
            released = ttk.Button(bar, text='Released flat', state='disabled', command=lambda: mark_f4('released_flat'))
            start.pack(side='left')
            released.pack(side='left')
            update_prompt()
        # Consume Enter before Tk can activate a focused button (including STOP).
        def mark_flat(event):
            if low_hover:
                mark_hover()
            else:
                mark_f4('flat_gripped')
            return 'break'
        for key in ('<Return>', '<KP_Enter>'):
            root.bind_class('F4FlatMarker', key, mark_flat)
        def bind_marker(event):
            tags = event.widget.bindtags()
            if 'F4FlatMarker' not in tags:
                event.widget.bindtags(('F4FlatMarker', *tags))
        root.bind('<Map>', bind_marker)
    elif validation_reset:
        phase_bar = ttk.Frame(frame)
        phase_bar.pack(before=controls, fill='x')
        phase = tk.StringVar(value='Setup: jog to the pickup start, then mark ready.')
        ttk.Label(phase_bar, textvariable=phase).pack(anchor='w')
        def mark(stage):
            with controller.lock:
                if controller.exit or controller.fault:
                    return
                controller.release()
                if stage == 'e4_released_flat':
                    controller.emit('placement_reset_confirmed', stage=stage)
                else:
                    controller.emit('operator_confirmed', stage=stage)
            placed.configure(state='normal' if stage == 'e4_clear' else 'disabled')
            ready.configure(state='disabled')
            active = stage == 'e4_pickup_start'
            for widget in (clear, empty):
                widget.configure(state='normal' if active else 'disabled')
            phase.set({
                'e4_pickup_start': 'Pickup: mark the first lift outcome.',
                'e4_clear': 'Lower, open when flat, then mark Released flat if the knight stays flat on e4.',
                'e4_empty': 'Reset the gripper, then press STOP to end.',
                'e4_released_flat': 'Placement marked. Reset the gripper, then press STOP to end.',
            }[stage])
        ready = ttk.Button(phase_bar, text='Ready on e4', command=lambda: mark('e4_pickup_start'))
        clear = ttk.Button(phase_bar, text='Knight clear', state='disabled', command=lambda: mark('e4_clear'))
        empty = ttk.Button(phase_bar, text='Empty lift', state='disabled', command=lambda: mark('e4_empty'))
        placed = ttk.Button(phase_bar, text='Released flat on e4', state='disabled',
                            command=lambda: mark('e4_released_flat'))
        for widget in (ready, clear, empty, placed):
            widget.pack(side='left')
    elif record_e4:
        def mark_clear():
            with controller.lock:
                if controller.exit or controller.fault:
                    return
                controller.stop('Operator observed knight clear of e4')
                controller.emit('operator_confirmed', stage='e4_clear')
        ttk.Button(finish_bar, text='Knight visibly clear — finish recording',
                   command=mark_clear).pack(side='left', fill='x', expand=True)
    tk.Button(finish_bar, text='STOP — End session', bg='#b02020', fg='white', font=('', 16, 'bold'),
              command=controller.stop).pack(side='left', fill='x', expand=True)
    ttk.Label(frame, text='Hold: 10 Hz; ≤5°/s, ≤5 mm/s, ≤5 gripper units/s.\n'
              'Release/focus loss cancels pending increments; torque stays on. Esc pauses. Stop ends the session.\n'
              'Short clicks send one tap; no held-motion queue. Focus loss cancels unsent taps.').pack(anchor='w')
    root.bind_all('<ButtonRelease-1>', lambda event: release(event, True))
    root.bind('<FocusOut>', release)
    # Keep held keys marked down so auto-repeat cannot restart a cancelled jog.
    root.bind('<Escape>', lambda event: controller.release('Paused — press a jog control to continue'))
    def close_window():
        controller.stop('Window closed — recovery')
        if isinstance(controller.arm, MockArm) or controller.disconnected.is_set():
            root.destroy()

    root.protocol('WM_DELETE_WINDOW', close_window)
    def acknowledge():
        from tkinter import messagebox
        operator = getattr(controller.arm, 'operator', None)
        if isinstance(operator, PanelRecoveryOperator) and operator.awaiting.is_set():
            if messagebox.askyesno('Finish recovery',
                    'Have you supported the arm and any held piece and physically switched off motor power?',
                    default='no', parent=root):
                operator.confirmed.set()
    confirmation = ttk.Button(frame, text='Confirm physical power is off…', command=acknowledge)
    def refresh():
        if controller.disconnected.is_set() or (setup and controller.exit):
            root.destroy()
            return
        operator = getattr(controller.arm, 'operator', None)
        if isinstance(operator, PanelRecoveryOperator) and operator.awaiting.is_set():
            confirmation.pack(fill='x')
        controller.pulse()
        view = controller.snapshot()
        status.set(f"Active mode: {mode.get()} | {view['status']}")
        measurements.set('\n'.join(f'{j:16} {view["measured"][j]:8.3f} {"units" if j == "gripper" else "deg"}'
                                   for j in JOINTS) + '\nWrist XYZ (mm): ' + ', '.join(f'{x:.2f}' for x in view['wrist_mm']))
        root.after(25, refresh)
    refresh()
    root.mainloop()
    return start_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('mock', 'lerobot'), default='mock')
    parser.add_argument('--session', type=Path, help='Reviewed hardware/initial, profile and limits paths')
    recording = parser.add_mutually_exclusive_group()
    recording.add_argument('--repeat-60', action='store_true', help='Jog two visits each to the center and g7/h7/g8/h8 intersections at 60 mm')
    recording.add_argument('--board-23', action='store_true', help='Manually mark 13 board contacts and ten elevated intersections in one session')
    recording.add_argument('--board-contacts', action='store_true', help='Record 13 board intersections in one manual jog session')
    recording.add_argument('--record-f4-low-hover', action='store_true', help='Mark one manual release height and landing on f4')
    recording.add_argument('--record-f4-placements', action='store_true', help='Record four manual f4 placements. Trial 4 is held out.')
    recording.add_argument('--record-e4-pickup', action='store_true', help='Record manual e4 pickup with fixed/wrist video and mark visible clearance')
    parser.add_argument('--validation-reset', action='store_true', help='Mark the scored pickup interval and keep jogging for manual reset')
    parser.add_argument('--fixed-camera', help='Fixed camera device or index')
    parser.add_argument('--wrist-camera', help='Wrist camera device or index')
    parser.add_argument('--wrist-resolution', type=int, nargs=2, default=(1280, 720),
                        metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--registration', type=Path, help='Level-board calibration inputs for landmark recording')
    parser.add_argument('--port')
    parser.add_argument('--robot-id')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-board-23', type=Path, help='Continue after the last saved mark in a previous board-23 directory')
    parser.add_argument('--resume-repeat-60', type=Path, help='Continue a four-mark 60 mm repeat session')
    args = parser.parse_args()
    board_start = 1
    resume = args.resume_repeat_60 or args.resume_board_23
    if resume:
        if args.resume_repeat_60 and not args.repeat_60:
            parser.error('--resume-repeat-60 requires --repeat-60')
        if args.resume_board_23 and not args.board_23:
            parser.error('--resume-board-23 requires --board-23')
        previous = json.loads((resume/'metadata.json').read_text())
        with (resume/'samples.jsonl').open() as source:
            rows = [json.loads(line) for line in source if line.endswith('\n')]
        operation = 'repeat_60' if args.repeat_60 else 'board_23'
        if not any(r.get('operation') == operation for r in rows):
            parser.error(f'Previous recording is not a {operation} session')
        numbers = [r['number'] for r in rows if r['event'] in ('board_contact','board_manual_hover')]
        first = previous.get('board_start', 1)
        if numbers != list(range(first, first+len(numbers))):
            parser.error('Previous measurement numbering is discontinuous')
        board_start = first + len(numbers)
        if board_start > (4 if args.repeat_60 else 23):
            parser.error('All measurements are already saved')
    args.board_contacts = args.board_contacts or args.board_23 or args.repeat_60
    if args.backend == 'lerobot' and not all((args.session, args.port, args.robot_id)):
        parser.error('Hardware needs --session, --port, --robot-id in a local terminal')
    recording_enabled = (args.board_contacts or args.record_e4_pickup
                         or args.record_f4_placements or args.record_f4_low_hover)
    if recording_enabled and args.backend != 'lerobot':
        parser.error('Camera recording needs the hardware backend')
    if recording_enabled and not (args.fixed_camera and args.wrist_camera):
        parser.error('Camera recording requires --fixed-camera and --wrist-camera')
    if min(args.wrist_resolution) <= 0:
        parser.error('Wrist resolution must be positive')
    if args.board_contacts and args.registration is None:
        parser.error('Landmark recording requires --registration')
    if args.validation_reset and not args.record_e4_pickup:
        parser.error('--validation-reset requires --record-e4-pickup')
    from .start_registration import Registration, board_landmark, capture_registration
    registration = Registration(args.registration) if args.registration is not None else None
    args.output.mkdir(parents=True, exist_ok=False)
    trace = (args.output / 'samples.jsonl').open('x', buffering=1)
    cameras = None
    trace_lock = threading.Lock()
    last_feedback = None
    def emit(event, **data):
        nonlocal last_feedback
        if cameras is not None and event == 'feedback':
            cameras.check()
        with trace_lock:
            now = time.monotonic()
            if args.board_contacts or args.record_f4_placements or args.record_f4_low_hover:
                if event == 'feedback':
                    last_feedback = dict(feedback_monotonic=now, ticks=data['registers']['Present_Position'].copy())
                elif event in ('board_contact', 'board_manual_hover'):
                    if last_feedback is None or now-last_feedback['feedback_monotonic'] > .3:
                        return False
                    sample = cameras.cameras['fixed'].latest_sample
                    if sample is None or now - (origin + sample[1]) > .3:
                        return False
                    index, frame_time, frame = sample
                    observation = registration.measure(frame, manual=True)
                    data.update(last_feedback, fixed_frame=index, frame_time_s=frame_time,
                                registration=observation,
                                target_board_m=board_landmark(data['lower_left_cell'], registration.pitch_m,
                                                              data['target_height_mm'] / 1000,
                                                              intersection=True)[:3].tolist())
                    message = f"Saved {data['number']}: {observation['status']}, markers {observation['ids']}"
                    print(message, flush=True)
                elif event in ('manual_f4_placement', 'manual_f4_low_hover') and last_feedback:
                    data.update(last_feedback)
            trace.write(json.dumps(dict(event=event, time_monotonic=now, **data), allow_nan=False) + '\n')
            return True
    arm = None
    worker = None
    try:
        if args.backend == 'mock':
            profile = dict(joint_order=list(ARM), joint_mapping=dict(joints={
                j: dict(sign=1, scale=1, offset_deg=0) for j in ARM}))
            limits = {j: [-80, 80] for j in ARM} | {'gripper': [0, 100]}
            arm = MockArm()
            arm.connect()
            arm.enable(arm.read())
        else:
            session = read_json(args.session)
            profile = read_json(Path(session['profile']))
            limits = read_limits(Path(session['limits']))
            # Load model and verify the mapping before touching the serial bus.
            WristIK(profile).pose(session['initial'])
            require_unowned_port(args.port)
            import tkinter as tk
            from tkinter import messagebox
            dialog = tk.Tk()
            dialog.withdraw()
            try:
                ranges = '\n'.join(f'{j}: {lo:.2f} to {hi:.2f} '
                                   + ('units' if j == 'gripper' else 'degrees')
                                   for j, (lo, hi) in limits.items())
                ready = messagebox.askokcancel('Start manual jog',
                    f'Confirm the arm is supported, jog path clear and power switch accessible. OK starts the session.\n\n{ranges}',
                    parent=dialog, default='cancel')
            finally:
                dialog.destroy()
            if not ready:
                return
            arm = LeRobotArm(args.port, args.robot_id)
            hardware = arm.connect()
            if hardware != session['hardware']:
                raise RuntimeError('Hardware/calibration/gains differ from reviewed passive snapshot')
            raw = confirmed_pose_ticks(arm.bus)
            check_positions(arm.read(), limits)
            arm.prepare_recovery(Path(session['recovery']), limits, args.output)
            arm.operator.discard_pending()
            arm.operator = guards.PowerOperator(arm.operator)
            initial = arm.read()
            check_positions(initial, limits)
            emit('preflight', session=session, hardware=hardware)
            if recording_enabled:
                from .powered_placement import StereoRecorder
                origin = time.monotonic()
                fixed_size = tuple(registration.lens['image_size_wh']) if registration else (1920, 1080)
                cameras = StereoRecorder(args.fixed_camera, args.wrist_camera, side_name='wrist',
                                         side_resolution=tuple(args.wrist_resolution), fixed_resolution=fixed_size)
                emit('camera_setup', origin_monotonic=origin, fixed=args.fixed_camera, wrist=args.wrist_camera,
                     operation='repeat_60' if args.repeat_60 else 'board_23' if args.board_23 else 'board_contacts' if args.board_contacts else 'manual_f4_low_hover' if args.record_f4_low_hover else 'manual_f4_placements' if args.record_f4_placements else 'manual_e4_pickup', start_ticks=raw)
                cameras.start(args.output, origin)
                if (args.record_f4_placements or args.record_f4_low_hover) and registration:
                    index, frame_time, frame = cameras.cameras['fixed'].latest_sample
                    observation = registration.measure(frame, manual=True)
                    emit('registration_preview', frame=index, frame_time_s=frame_time,
                         observation=observation)
                if args.record_e4_pickup:
                    capture_registration(cameras.cameras['fixed'], [], args.output)
                cameras.check()
                initial = arm.read()
                check_positions(initial, limits)
            guards.seed_and_verify(arm, initial, JOINTS, lambda event, values: emit(event, values=values))
        controller = JogController(arm, limits, WristIK(profile), emit)
        if args.backend == 'lerobot':
            controller.goal = dict(initial)  # Seeded goal is distinct from measured feedback.
        (args.output / 'metadata.json').write_text(json.dumps(dict(backend=args.backend,
            physical_validation=False, profile=profile, limits=limits, automatic_release=False,
            **({'trial_splits': ['train', 'train', 'train', 'validation']} if args.record_f4_placements else {}),
            **({'board_start':board_start, 'continued_from':str(resume.resolve()) if resume else None} if args.board_23 or args.repeat_60 else {}),
            source_sha256=sha256(Path(__file__)),
            git_revision=subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                        text=True).stdout.strip()), indent=2))
        worker = threading.Thread(target=controller.run, name='single-arm-owner')
        worker.start()
        signal.signal(signal.SIGTERM, lambda *_: controller.stop('Termination — recovery'))
        try:
            panel(controller, args.record_e4_pickup, args.validation_reset, f4_placements=args.record_f4_placements, low_hover=args.record_f4_low_hover, board_contacts=args.board_contacts, board_23=args.board_23, board_start=board_start, repeat_60=args.repeat_60)
        finally:
            controller.stop('Panel closed — recovery')
            worker.join()  # Hardware recovery waits for the operator in this terminal.
    finally:
        if worker is None and arm is not None:
            if isinstance(arm, LeRobotArm):
                if arm.enabled:
                    arm.recover('Startup failed')
                if arm.bus.is_connected:
                    arm.close()
            else:
                arm.close()
        try:
            if cameras is not None:
                emit('camera_recording_finished', recordings=cameras.close())
        finally:
            trace.close()


if __name__ == '__main__':
    main()
