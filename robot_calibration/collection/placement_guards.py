"""Guarded startup and feedback checks for powered placement."""
import math
import time
from robot_calibration.records import JOINTS

ELBOW = 'elbow_flex'
ACTIVE = ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex')
TRACKING_TICKS = 26
OTHER_DRIFT_TICKS = 2

def _convert_positions(bus, values, convert):
    ids = {joint: bus.calibration[joint].id for joint in values}
    converted = convert({ids[joint]: value for joint, value in values.items()})
    return {joint: converted[motor_id] for joint, motor_id in ids.items()}


def raw_positions(bus, values):
    return _convert_positions(bus, values, bus._unnormalize)


def normalized_positions(bus, values):
    return _convert_positions(bus, values, bus._normalize)


def validate_temperatures(temperatures, read_gripper, *, maximum=45,
                          deadline_s=.15, clock=time.monotonic, sleep=time.sleep):
    if set(temperatures) != set(JOINTS) or any(type(v) is not int or v < 0 for v in temperatures.values()):
        raise RuntimeError('Invalid temperature feedback')
    if any(v >= maximum for j, v in temperatures.items() if j != 'gripper'):
        raise RuntimeError('Arm temperature ceiling reached')
    value = temperatures['gripper']
    if value < maximum:
        return
    if value >= 60:
        raise RuntimeError('Gripper temperature high; immediate stop')
    begun = clock()
    for _ in range(3):
        sleep(.02)  # No goal writes while corroborating the sensor value.
        value = read_gripper()
        if type(value) is not int or not 0 <= value < maximum:
            raise RuntimeError('Gripper temperature ceiling confirmed or invalid feedback')
        if clock() - begun > deadline_s:
            raise RuntimeError('Temperature corroboration deadline exceeded')


def validate_motion_temperatures(temperatures, bus, evidence, clock=time.monotonic, sleep=time.sleep):
    begun = clock()

    def read_gripper():
        value = bus.read('Present_Temperature', 'gripper', normalize=False, num_retry=0)
        evidence.append({'elapsed_s': clock() - begun, 'temperature': value})
        return value

    validate_temperatures(temperatures, read_gripper, clock=clock, sleep=sleep)


class MissingTorqueReply(ConnectionError):
    """A motor did not report its torque flag in this sample."""


def read_torque(bus, joints):
    flags = bus.sync_read('Torque_Enable', normalize=False, num_retry=0)
    if set(flags) < set(joints):
        raise MissingTorqueReply(f'Missing torque replies from {sorted(set(joints) - set(flags))}')
    if set(flags) != set(joints) or any(type(v) is not int or v not in (0, 1) for v in flags.values()):
        raise RuntimeError('Invalid torque feedback')
    return flags


def torque_reply_missing(exc):
    return isinstance(exc, (MissingTorqueReply, TimeoutError)) or 'operation timed out' in str(exc).lower()


def check_feedback(bus, start, expected, bounds, emit=None, tracking_ticks=TRACKING_TICKS, active=ACTIVE, gripper_from=None, arm_from=None):
    """Keep raw tracking, travel, non-elbow, goal, fault and timing checks."""
    begun = time.monotonic()
    while True:
        try:
            torque = read_torque(bus, JOINTS)
            break
        except (ConnectionError, TimeoutError) as exc:
            if not torque_reply_missing(exc) or time.monotonic() - begun >= .5:
                raise
            time.sleep(.02)
    registers = {'Torque_Enable': torque}
    registers.update({name: bus.sync_read(name, normalize=False) for name in
                      ('Present_Position', 'Goal_Position', 'Status', 'Present_Temperature',
                       'Present_Load', 'Present_Current')})
    if time.monotonic() - begun > .5:
        raise RuntimeError('Feedback deadline exceeded')
    if any(set(values) != set(JOINTS) or any(type(v) is not int for v in values.values())
           for values in registers.values()):
        raise RuntimeError('Incomplete or invalid raw feedback')
    if emit:
        emit(registers)  # Preserve even the observation that trips a guard.
    actual = registers['Present_Position']
    if registers['Goal_Position'] != expected:
        raise RuntimeError('Goal readback changed unexpectedly')
    if any(v != 1 for v in registers['Torque_Enable'].values()) or any(registers['Status'].values()):
        raise RuntimeError('Torque changed or motor fault')
    confirmation=[]
    try:
        validate_motion_temperatures(registers['Present_Temperature'], bus, confirmation)
    finally:
        if confirmation:
            registers['Temperature_Confirmation']=confirmation
            if emit:emit(registers)
    if time.monotonic()-begun>.5:
        raise RuntimeError('Complete feedback deadline exceeded')
    if any(not bounds[j][0] <= actual[j] <= bounds[j][1] for j in JOINTS):
        raise RuntimeError('Actual-position envelope exceeded')
    if any(abs(actual[j] - start[j]) > OTHER_DRIFT_TICKS for j in JOINTS if j not in active):
        raise RuntimeError('Non-elbow joint drift')
    # During a manual command's first feedback period, a moving shaft can
    # still be between the preceding and new goals. Keep the same tolerance.
    def tracking_error(j):
        previous = (arm_from or {}).get(j, expected[j])
        lo, hi = sorted((previous, expected[j]))
        return max(lo - actual[j], actual[j] - hi, 0)

    if tracking_error(ELBOW) > tracking_ticks:
        raise RuntimeError(f'{tracking_ticks}-tick elbow tracking guard exceeded')
    if gripper_from is not None:
        lo, hi = sorted((gripper_from, expected['gripper']))
        if not lo - 8 <= actual['gripper'] <= hi + 8:
            raise RuntimeError('Gripper outside commanded settling path')
    if any(tracking_error(j)>8 for j in active if j!=ELBOW
           and (j != 'gripper' or gripper_from is None)):
        raise RuntimeError('8-tick active non-elbow tracking guard exceeded')
    return registers


def build_commands(start, targets):
    commands=[(dict(start),False)]*5
    previous=dict(start)
    for target in targets:
        steps=math.ceil(max(abs(target[j]-previous[j]) for j in JOINTS)/.025)
        if steps:
            commands.extend(({j:previous[j]+(target[j]-previous[j])*i/steps for j in JOINTS},False) for i in range(1,steps+1))
        commands.extend([(dict(target),False)]*9+[(dict(target),True)])
        previous=dict(target)
    return commands

class StartupTorqueTimeout(RuntimeError):
    """The supported arm did not establish a stable powered hold in time."""


def seed_and_verify(arm, initial, joints, log, *, enable_if_all_off=False):
    bus = arm.bus
    def read(name):
        values = bus.sync_read(name, normalize=False, num_retry=0)
        if set(values) != set(joints) or any(type(x) is not int for x in values.values()):
            raise RuntimeError(f'Incomplete {name} feedback')
        return values
    def read_goals(deadline, expected=None):
        values = {}
        for joint in joints:
            value = bus.read('Goal_Position', joint, normalize=False, num_retry=0)
            if type(value) is not int or time.monotonic() >= deadline:
                raise RuntimeError(f'Invalid or late goal read from {joint}')
            if expected is not None and value != expected[joint]:
                raise RuntimeError(f'Goal changed during seed for {joint}')
            values[joint] = value
        return values
    if getattr(arm, 'motion_cancelled', False) or getattr(arm, 'recovery', None) is None:
        raise RuntimeError('Supervised recovery must be ready')
    if any(read('Torque_Enable').values()) or any(read('Operating_Mode').values()):
        raise RuntimeError('Expected torque off and position mode')
    log('passive_goal_readback', read_goals(time.monotonic()+.5))
    for joint in joints:
        start=time.monotonic()
        bus.write('Lock',joint,1,normalize=False,num_retry=0)
        if bus.read('Lock',joint,normalize=False,num_retry=0)!=1 or time.monotonic()-start>.5:
            raise RuntimeError(f'Lock verification failed for {joint}')
    pos=read('Present_Position')
    initial_raw = raw_positions(bus, {joint: initial[joint] for joint in joints})
    if any(abs(pos[j]-initial_raw[j])>2 for j in joints) or any(read('Status').values()) or any(read('Torque_Enable').values()):
        raise RuntimeError('Start pose, fault or torque changed before seed')
    # Confirm a stationary, fully responsive bus before the single energizing write.
    # These samples precede the write. The first post-write torque read is immediate.
    for sample_index in range(3):
        sample_start = time.monotonic()
        pos = read('Present_Position')
        if (any(abs(pos[j] - initial_raw[j]) > 2 for j in joints)
                or any(read('Torque_Enable').values())
                or any(read('Status').values())
                or any(read('Operating_Mode').values())
                or any(value != 1 for value in read('Lock').values())):
            raise RuntimeError('Unstable passive startup state')
        if time.monotonic() - sample_start >= .5:
            raise RuntimeError('Passive startup read deadline exceeded')
        log('passive_start_verified', pos)
        if sample_index < 2:
            time.sleep(.02)
    # Use the same quantized initial goals expected by the ramp.
    log('before_seed',pos)
    arm.enabled=True
    started=time.monotonic()
    bus.sync_write('Goal_Position',initial_raw,normalize=False,num_retry=0)
    previous_all_on = False
    previous_all_off = False
    stayed_off = True
    repeated_read = False
    while time.monotonic()-started<.5:
        register = 'Torque_Enable'
        try:
            torque = read_torque(bus, joints)
            log('torque_readback_after_seed', torque)
            if any(value not in (0, 1) for value in torque.values()):
                raise RuntimeError('Invalid startup torque flag')
            if len(set(torque.values())) != 1:
                log('mixed_torque_pending', torque)
            register = 'Goal_Position'
            read_goals(started + .5, expected=initial_raw)
        except (ConnectionError, TimeoutError) as exc:
            message = str(exc).lower()
            retryable = (('checksum error' in message and register in ('Torque_Enable', 'Goal_Position'))
                         or (register == 'Goal_Position' and 'operation timed out' in message))
            missing_torque = register == 'Torque_Enable' and torque_reply_missing(exc)
            if not missing_torque:
                if repeated_read or not retryable:
                    raise
                repeated_read = True
            previous_all_on = previous_all_off = False
            stayed_off = False
            log('startup_read_discarded', {'register': register, 'error': str(exc)})
            if any(read('Status').values()) or any(v != 1 for v in read('Lock').values()):
                raise RuntimeError('Fault or lock changed after failed startup read') from exc
            position = read('Present_Position')
            if any(abs(position[j] - pos[j]) > 2 for j in joints):
                raise RuntimeError('Drift after failed startup read') from exc
            if missing_torque:
                time.sleep(.02)
            continue
        if any(read('Status').values()) or any(v!=1 for v in read('Lock').values()):
            raise RuntimeError('Lock or status changed during seed')
        actual=read('Present_Position')
        if any(abs(actual[j]-pos[j])>2 for j in joints):
            raise RuntimeError('Drift during seed')
        temperatures = read('Present_Temperature')
        confirmation = []
        try:
            validate_motion_temperatures(temperatures, bus, confirmation)
        finally:
            log('startup_checks', dict(positions=actual, temperatures=temperatures,
                                       temperature_confirmation=confirmation))
        all_off = not any(torque.values())
        stayed_off = stayed_off and all_off
        if enable_if_all_off and stayed_off and previous_all_off:
            if time.monotonic() - started >= .5:
                raise StartupTorqueTimeout('Startup deadline exceeded before torque enable')
            log('explicit_torque_enable', torque)
            bus.enable_torque()
            stayed_off = False
        previous_all_off = all_off
        all_on = all(torque.values())
        if all_on and previous_all_on:
            if not all(v==1 for v in read('Lock').values()) or time.monotonic()-started>=.5:
                raise RuntimeError('Final lock or timing verification failed')
            log('all_powered_verified',actual)
            return
        previous_all_on = all_on
        time.sleep(.02)
    raise StartupTorqueTimeout('Torque did not stabilize within the startup deadline')

class PowerOperator:
    def __init__(self, wrapped):
        self.wrapped = wrapped
    def event(self, timeout):
        event = self.wrapped.event(timeout)
        if event == 'supported-release':
            self.notify('Software release disabled. Operator controls power; confirm power-off-supported only after support and physical power removal.')
            return None
        return event
    def notify(self, message):
        self.wrapped.notify(message.replace('Then type supported-release to disable torque.', 'Software torque release is disabled.'))
    def close(self):
        self.wrapped.close()
