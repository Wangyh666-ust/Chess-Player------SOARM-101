"""Supervised recovery for the collector; never release torque in cleanup.

A held goal is not proof of physical holding. Mechanical support must protect
against power loss. All control thresholds are explicit local configuration.
"""
import json
import math
import os
import select
import termios
from collections import deque
from pathlib import Path
import time

from robot_calibration.records import ARM, JOINTS
from .placement_guards import validate_temperatures


def hold_measured_pose(bus, bounds, max_read_s, clock=time.monotonic):
    """Replace failed arm goals with fresh raw positions; never release torque.

    Keep the existing gripper goal so this stop does not deliberately release a
    piece. On any uncertainty report that the stop is unverified; do not retry,
    resume the trajectory, enable torque, or fall back to automatic torque-off.
    Partial writes may leave mixed goals, so failure needs operator intervention.
    """
    result = {'state': 'stop_unverified', 'motion_ready': False}

    def read(name):
        values = bus.sync_read(name, normalize=False)
        if set(values) != set(JOINTS) or any(type(v) is not int for v in values.values()):
            raise RuntimeError('Incomplete or invalid raw register read')
        return values

    try:
        if max_read_s <= 0 or set(bounds) != set(JOINTS):
            raise ValueError('Explicit bounds and read deadline required')

        started = clock()
        if any(read('Status').values()):
            raise RuntimeError('Motor fault: supervised intervention required')
        if any(v != 1 for v in read('Torque_Enable').values()):
            raise RuntimeError('Torque state changed; do not re-enable automatically')
        if any(read('Operating_Mode').values()):
            raise RuntimeError('Expected position mode')

        goal = read('Goal_Position')
        actual = read('Present_Position')
        for joint in JOINTS:
            low, high = bounds[joint]
            if not low <= actual[joint] <= high:
                raise RuntimeError('Measured position outside reviewed envelope')
        if not bounds['gripper'][0] <= goal['gripper'] <= bounds['gripper'][1]:
            raise RuntimeError('Gripper goal outside reviewed envelope')
        if clock() - started > max_read_s:
            raise RuntimeError('Stale stop observation')

        goal.update({joint: actual[joint] for joint in ARM})
        result['requested_hold_raw'] = goal.copy()
        started = clock()
        bus.sync_write('Goal_Position', goal, normalize=False)
        if read('Goal_Position') != goal:
            raise RuntimeError('Hold goal readback mismatch')
        if any(v != 1 for v in read('Torque_Enable').values()) or any(read('Status').values()):
            raise RuntimeError('Torque or fault state changed during stop')
        if clock() - started > max_read_s:
            raise RuntimeError('Hold verification deadline exceeded')

        result['state'] = 'hold_goal_verified_requires_supervision'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'

    return result


def release_supported(bus, *, mechanically_supported):
    """Explicit torque release after physical support; never a timed cleanup."""
    if mechanically_supported is not True:
        raise RuntimeError('Mechanical support must be confirmed before torque release')

    bus.disable_torque()
    state = bus.sync_read('Torque_Enable', normalize=False)
    if set(state) != set(JOINTS) or any(v != 0 for v in state.values()):
        raise RuntimeError('Torque release unverified')
    return {'state': 'released_with_confirmed_support', 'motion_ready': False}


class StopSupervisor:
    """Latch trajectory cancellation, monitor a hold, and supervise release.

    All thresholds are explicit, reviewed inputs in raw register units/seconds.
    Tests use synthetic values; they are not hardware operating limits.
    Faults are latched: a later good reading cannot automatically restore a hold.
    """

    def __init__(self, bus, cancel_trajectory, bounds, drift_ticks, max_temperature,
                 read_deadline_s, poll_deadline_s, hold_deadline_s, clock=time.monotonic, emit=None):
        self.emit = emit
        self.logging_failed = False
        self.bus = bus
        self.cancel_trajectory = cancel_trajectory
        self.bounds = dict(bounds)
        self.clock = clock

        for value in (drift_ticks, max_temperature, read_deadline_s,
                      poll_deadline_s, hold_deadline_s):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Explicit positive finite supervision limits required')
        if set(bounds) != set(JOINTS):
            raise ValueError('Bounds required for every joint')
        for low, high in bounds.values():
            if type(low) is not int or type(high) is not int or low >= high:
                raise ValueError('Ordered raw integer bounds required')

        self.drift_ticks = drift_ticks
        self.max_temperature = max_temperature
        self.read_deadline_s = read_deadline_s
        self.poll_deadline_s = poll_deadline_s
        self.hold_deadline_s = hold_deadline_s
        self.state = 'not_started'
        self.events = deque(maxlen=2048)  # Bounded recent history during a prolonged recovery.
        self.goal = None
        self.started = None
        self.last_poll = None

    def record(self, event, **details):
        """Keep an offline audit trail without relying on a fallible file logger."""
        row = dict(event=event, state=self.state, time_s=self.clock(), **details)
        self.events.append(row)
        if self.emit:
            try:
                self.emit(row)
            except Exception:
                self.logging_failed = True
        return row

    def require_intervention(self, reason):
        """Latch uncertainty; neither power off nor resume a failed trajectory."""
        self.state = 'intervention_required'
        return self.record('intervention_required', reason=reason, motion_ready=False)

    def begin(self, reason):
        """Use for completion, operator stop or tracking failure; cancel FIRST."""
        if self.state != 'not_started':
            raise RuntimeError('Stop already latched; cannot start or resume again')

        self.state = 'stopping'
        self.started = self.last_poll = self.clock()
        try:
            self.cancel_trajectory()
        except Exception as exc:
            return self.require_intervention(f'Trajectory cancellation failed: {exc}')

        self.record('trajectory_cancelled', reason=reason)
        result = hold_measured_pose(self.bus, self.bounds, self.read_deadline_s, self.clock)
        self.record('hold_attempt', result=result)
        if result['state'] != 'hold_goal_verified_requires_supervision':
            return self.require_intervention(result.get('error', 'Hold unverified'))

        self.goal = result['requested_hold_raw'].copy()
        self.state = 'holding'
        self.last_poll = self.clock()
        return self.poll()

    def poll(self):
        """Check feedback without refreshing targets or concealing drift.

        Expired timers request intervention; they NEVER trigger torque release.
        In a latched fault state, polling records feedback for the operator but
        cannot restore motion or clear the fault.
        """
        if self.state in ('not_started', 'released', 'disconnected'):
            raise RuntimeError('No active supervised stop')

        started = self.clock()
        issues = ['Recovery log failed'] if self.logging_failed else []
        if started - self.last_poll > self.poll_deadline_s:
            issues.append('Supervisor poll deadline missed')
        if started - self.started > self.hold_deadline_s:
            issues.append('Supervised hold deadline exceeded; obtain physical support')

        readings = {}
        try:
            for name in ('Present_Position', 'Goal_Position', 'Torque_Enable',
                         'Status', 'Operating_Mode', 'Present_Temperature'):
                values = self.bus.sync_read(name, normalize=False)
                if set(values) != set(JOINTS) or any(type(v) is not int for v in values.values()):
                    raise RuntimeError(f'Invalid {name} feedback')
                readings[name] = values

            if self.clock() - started > self.read_deadline_s:
                issues.append('Feedback deadline exceeded')
            if any(readings['Status'].values()):
                issues.append('Motor fault')
            if any(readings['Operating_Mode'].values()):
                issues.append('Position mode changed')
            if any(v != 1 for v in readings['Torque_Enable'].values()):
                issues.append('Torque changed; never re-enable automatically')
            def confirm_temperature():
                values = self.bus.sync_read('Present_Temperature', normalize=False)
                readings.setdefault('Temperature_Confirmation', []).append(values.copy())
                if set(values) != set(JOINTS) or any(type(v) is not int or v < 0 for v in values.values()):
                    raise RuntimeError('Invalid temperature confirmation')
                if self.clock() - started > self.read_deadline_s:
                    raise RuntimeError('Temperature confirmation deadline exceeded')
                if any(v >= self.max_temperature for v in values.values()):
                    raise RuntimeError('Temperature threshold confirmed')
                return values['gripper']

            validate_temperatures(readings['Present_Temperature'], confirm_temperature,
                                  maximum=self.max_temperature, deadline_s=self.read_deadline_s,
                                  clock=self.clock)
            if self.goal is None or readings['Goal_Position'] != self.goal:
                issues.append('Hold goal unverified or changed')

            for joint, value in readings['Present_Position'].items():
                low, high = self.bounds[joint]
                if not low <= value <= high:
                    issues.append(f'{joint} outside reviewed bounds')
                if self.goal and joint in ARM and abs(value - self.goal[joint]) > self.drift_ticks:
                    issues.append(f'{joint} moved away from latched hold')
        except Exception as exc:
            issues.append(f'Feedback unavailable: {exc}')

        self.last_poll = self.clock()
        self.record('feedback', readings=readings, issues=issues)
        if issues:
            return self.require_intervention('; '.join(issues))
        return self.record('feedback_received', motion_ready=False)

    def interrupted(self):
        """Handle an operator stop request without treating it as torque release.

        A future live entry point must route its signal to this method instead
        of exiting through unconditional cleanup. Process kill/power loss cannot
        be handled by this software.
        """
        if self.state == 'not_started':
            return self.begin('operator interruption')
        if self.state in ('released', 'disconnected'):
            return self.record('interrupted_after_release')
        return self.require_intervention('Operator interruption; supervised recovery required')

    def release(self, *, mechanically_supported):
        """Operator action only; recheck support for each release attempt."""
        if self.state not in ('holding', 'intervention_required'):
            raise RuntimeError('Release requires an active stop')
        if mechanically_supported is not True:
            raise RuntimeError('Explicit mechanical support confirmation required')

        try:
            result = release_supported(self.bus, mechanically_supported=True)
        except Exception as exc:
            return self.require_intervention(f'Release unverified: {exc}')

        self.state = 'released'
        return self.record('release_verified', result=result)

    def confirm_power_removed(self, *, mechanically_supported, power_removed):
        """Operator-confirmed fallback when unpowered motors cannot reply.

        This is an operator assertion, not a measured torque-off verification.
        It never writes a motor command and never restores readiness.
        """
        if self.state not in ('holding', 'intervention_required'):
            raise RuntimeError('No active recovery')
        if mechanically_supported is not True or power_removed is not True:
            raise RuntimeError('Physical support and removed motor power must be confirmed')
        self.state = 'released'
        return self.record('power_removed_by_operator', telemetry_verified=False)

    def disconnect(self):
        """Disconnect only after verified release; never implicit SDK torque-off."""
        if self.state != 'released':
            raise RuntimeError('Cannot abandon an unverified or energized stop')
        try:
            self.bus.disconnect(disable_torque=False)
        except Exception as exc:
            return self.record('disconnect_failed', error=str(exc))
        self.state = 'disconnected'
        return self.record('disconnected')


CONFIG_KEYS = ('drift_ticks', 'max_temperature', 'read_deadline_s',
               'poll_deadline_s', 'hold_deadline_s', 'poll_s')


def load_config(path):
    """Require reviewed finite local thresholds, never executable example defaults."""
    config = json.loads(Path(path).read_text())
    if set(config) != set(CONFIG_KEYS):
        raise ValueError(f'Recovery config must contain exactly {CONFIG_KEYS}')
    for value in config.values():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Fill every recovery threshold with a positive finite value')
    if not config['poll_s'] < config['poll_deadline_s']:
        raise ValueError('Recovery polling must fit the poll deadline')
    return config


class TerminalOperator:
    """Nonblocking local terminal input, separate from pipes feeding the collector."""

    def __init__(self):
        self.fd = os.open('/dev/tty', os.O_RDWR | os.O_NONBLOCK)
        self.buffer = b''

    def discard_pending(self):
        """Require a fresh release instruction after recovery starts."""
        self.buffer = b''
        termios.tcflush(self.fd, termios.TCIFLUSH)

    def notify(self, message):
        try:
            os.write(self.fd, (message + '\n').encode())
        except OSError:
            pass  # Lost terminal cannot end an energized recovery.

    def event(self, timeout):
        if b'\n' not in self.buffer:
            readable, _, _ = select.select([self.fd], [], [], timeout)
            if not readable:
                return None
            chunk = os.read(self.fd, 4096)
            if not chunk:
                return 'input-closed'
            self.buffer += chunk
        line, separator, self.buffer = self.buffer.partition(b'\n')
        if not separator:
            self.buffer = line
            return None
        return line.decode(errors='replace').strip()

    def close(self):
        os.close(self.fd)


def supervise(supervisor, reason, operator, poll_s, clock=time.monotonic, sleep=time.sleep):
    """Own recovery until supported release or explicit physical power removal.

    Interrupts and logging/input failures do not trigger disconnect/torque-off.
    The caller disconnects only after this function reports a released state.
    """
    try:
        supervisor.begin(reason)
        if isinstance(operator, TerminalOperator):
            operator.discard_pending()
        operator.notify('Trajectory stopped. Secure the arm AND any held piece on mechanical support.\n'
                        'Then type supported-release to disable torque.\n'
                        'If communication is lost: physically remove motor power with the arm supported,\n'
                        'then type power-off-supported. Ctrl+C does NOT release torque.')
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        supervisor.require_intervention(f'Stop setup interrupted: {type(exc).__name__}: {exc}')

    previous_notice = None
    while supervisor.state != 'released':
        cycle = clock()
        try:
            supervisor.poll()
            if supervisor.state != previous_notice:
                operator.notify(f'Recovery state: {supervisor.state}; motion remains cancelled.')
                if supervisor.state == 'intervention_required':
                    operator.notify(str(supervisor.events[-1].get('reason', 'Inspect recovery.jsonl.')))
                previous_notice = supervisor.state
            event = operator.event(max(0., poll_s - (clock() - cycle)))
            if event == 'supported-release':
                supervisor.release(mechanically_supported=True)
            elif event == 'power-off-supported':
                supervisor.confirm_power_removed(mechanically_supported=True, power_removed=True)
            elif event in ('stop', 'input-closed'):
                supervisor.require_intervention(f'Operator event: {event}')
            elif event:
                operator.notify('Use supported-release only after mechanical support, or power-off-supported '
                                'only after support and physical motor-power removal.')
        except (Exception, KeyboardInterrupt, SystemExit) as exc:
            supervisor.require_intervention(f'Recovery interrupted: {type(exc).__name__}: {exc}')
        if supervisor.state != 'released':
            try:
                sleep(max(0., poll_s - (clock() - cycle)))
            except (KeyboardInterrupt, SystemExit):
                supervisor.interrupted()

    return {'state': supervisor.state, 'last_event': supervisor.events[-1],
            'logging_failed': supervisor.logging_failed, 'motion_ready': False}
