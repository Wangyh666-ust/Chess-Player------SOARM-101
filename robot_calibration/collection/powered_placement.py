"""Record powered fingertip measurements, one reviewed path at a time."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import signal
import time

from robot_calibration.collection.collect_robot_calibration import LeRobotArm, JOINTS, require_unowned_port
from robot_calibration.collection import placement_guards as guards
from robot_calibration.collection.pickup_trial import VideoRecorder


def chunked_commands(initial, target):
    distance = max(abs(target[j] - initial[j]) for j in JOINTS)
    count = max(1, math.ceil(distance))  # At most one degree per moving joint per checkpoint.
    checkpoints = [
        {j: initial[j] + (target[j] - initial[j]) * i / count for j in JOINTS}
        for i in range(1, count)
    ]
    return guards.build_commands(initial, [*checkpoints, target])


def check_path(plan):
    initial, target = plan['initial'], plan['target']
    if set(initial) != set(JOINTS) or set(target) != set(JOINTS):
        raise ValueError('Incomplete joints')
    if any(not math.isfinite(v) for v in (*initial.values(), *target.values())):
        raise ValueError('Invalid joint coordinate')
    if (not re.fullmatch('[a-h][1-8]', plan['square'])
            or not math.isfinite(plan['requested_height_mm']) or plan['requested_height_mm'] <= 60):
        raise ValueError('Invalid target square or height')
    if (any(target[j] != initial[j] for j in ('gripper', 'wrist_roll'))
            or max(abs(target[j] - initial[j]) for j in JOINTS) >= 12):
        raise ValueError('Path exceeds existing step limits')
    if not 0 <= plan['ik_error_mm'] < .3 or len(plan['path']) != 101:
        raise ValueError('Missing path verification')
    if not math.isfinite(plan['minimum_modeled_tip_height_mm']) or plan['minimum_modeled_tip_height_mm'] <= 60:
        raise ValueError('Offline path screen failed')
    for point in plan['path']:
        xyz, margin = point['tip_board_m'], point['joint_margin_deg']
        if len(xyz) != 3 or not all(map(math.isfinite, xyz)) or xyz[2] <= .06 or not 0 < margin < math.inf:
            raise ValueError('Preview exceeds clearance or joint limits')


def validate_plan(plan, plan_digest=None, *, measure_start=False, reviewed_digest=None):
    check_path(plan)
    reviewed_placement = plan_digest in {
        '757d387964d9cbc14588e7d5aaf0c94e2232a9652ceadb4ed3605c9132b2abbc',
        'a040185fcd54bafc79513f3e95ff6d71f1487df592323c6d6033dec7b7137fda',
        '3c0105ae7a8637511ded21f8711f355504b279976ec4730f9d89821e8fc37c62',
        'e82543cfcc7f12b7e62e91c954f672b6988c3a162cd8c582592d548ba0f4fec4',
        '2dce810c9ae7b6ae889bceff7a67833c72ddfd188bf18df7ce0ef9ee15fb5f8d',
        '4c308158a508b776138d846edd17b8c771423f364c6e0d224c5794602efe62d9',
        '7f06f11832880887cc1891186f4fbb1f924a365d79403958fc6a30f72db9e290',
        'e419e1c1ae5a95bcc27cb3ccc8cf8429919f39942329ecdb0e7258f361949b36',
    }
    reviewed_stationary = measure_start and plan_digest == '0c3bed905e7920c92d8a81b62bc83807b809cf7e8309842ab8a69072c6ba597c'
    if not (reviewed_placement or reviewed_stationary
            or (reviewed_digest is not None and plan_digest == reviewed_digest)):
        raise ValueError('Unknown first placement plan')



def check_next(plan, hardware, expected, raw, original):
    """A separately reviewed path starts from the exact currently held goal."""
    check_path(plan)
    if plan['hardware'] != hardware or plan['T_base_from_board'] != original['T_base_from_board']:
        raise ValueError('Calibration or board frame changed')
    if plan['start_raw'] != expected or raw(plan['initial']) != expected:
        raise ValueError('Plan must start at the held goal')


def check_continuation_start(plan, feedback, held_goal):
    """Keep the previous goal latched until the first checked trajectory write."""
    start = plan['start_raw']
    if feedback['Goal_Position'] != held_goal:
        raise RuntimeError('Held goal changed before continuation')
    if plan.get('held_goal_raw', held_goal) != held_goal:
        raise ValueError('Measured plan belongs to another held goal')
    if start != held_goal and 'held_goal_raw' not in plan:
        raise ValueError('Measured start needs its held goal')
    for joint in JOINTS:
        actual = feedback['Present_Position'][joint]
        if abs(actual - start[joint]) > 2:
            raise RuntimeError('Measured pose differs from reviewed start')
        if joint not in guards.ACTIVE and start[joint] != held_goal[joint]:
            raise ValueError('Continuation would move a stationary joint')
        if abs(actual - held_goal[joint]) > 8 or abs(start[joint] - held_goal[joint]) > 8:
            raise RuntimeError('Continuation exceeds settled tracking limit')


class StereoRecorder:
    def __init__(self, fixed, side, *, side_name='angetube', side_resolution=(1920, 1080),
                 fixed_resolution=(1920, 1080)):
        self.cameras = {'fixed': VideoRecorder(fixed, 15, fixed_resolution),
                        side_name: VideoRecorder(side, 15, side_resolution)}

    def start(self, output, origin):
        for name, camera in self.cameras.items():
            directory = output / name
            directory.mkdir()
            camera.start(directory, origin)

    def check(self):
        for camera in self.cameras.values():
            camera.check()

    def close(self):
        result = {}
        errors = []
        for name, camera in self.cameras.items():
            try:
                result[name] = camera.close()
            except Exception as exc:
                errors.append(f'{name}: {exc}')
        if errors:
            raise RuntimeError(', '.join(errors))
        return result


class StationaryRecoveryInput:
    """Annotate a verified, mechanically supported hold while cancellation stays latched."""

    def __init__(self, operator, supervisor, video):
        self.operator = operator
        self.supervisor = supervisor
        self.video = video
        self.recorded = False

    def notify(self, message):
        self.operator.notify(message)
        if message == 'Recovery state: holding; motion remains cancelled.':
            self.operator.notify('Supported stationary hold. Optional ruler entry: supported-height HEIGHT UNCERTAINTY (mm).')

    def close(self):
        self.operator.close()

    def event(self, timeout):
        line = self.operator.event(timeout)
        if not line or not line.startswith('supported-height '):
            return line
        try:
            _, height, uncertainty = line.split()
            height, uncertainty = float(height), float(uncertainty)
            if not 0 <= height < math.inf or not 0 < uncertainty < math.inf:
                raise ValueError
        except ValueError:
            self.notify('Enter supported-height followed by height and ruler uncertainty in mm.')
            return None
        supervisor = self.supervisor
        if self.recorded or supervisor.state != 'holding':
            self.notify('Reading unavailable outside a verified hold or already recorded.')
            return None
        self.video.check()
        supervisor.poll()
        if supervisor.state == 'holding' and not supervisor.logging_failed:
            feedback = next(row for row in reversed(supervisor.events) if row['event'] == 'feedback')
            supervisor.record('supported_stationary_measurement', height_mm=height,
                              ruler_uncertainty_mm=uncertainty, registers=feedback['readings'],
                              feedback_time_s=feedback['time_s'], load_condition='mechanically_supported')
            self.recorded = True
            self.notify('Supported height recorded. Motion remains cancelled.')
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Run the reviewed plan on hardware')
    parser.add_argument('--measure-start', action='store_true',
                        help='Hold the reviewed starting pose for the first observation')
    parser.add_argument('--port', help='Follower serial device')
    parser.add_argument('--robot-id', help='Saved follower calibration ID')
    parser.add_argument('--camera', help='Fixed calibration camera device')
    parser.add_argument('--side-camera', help='Record a second camera and use visual hold review')
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--reviewed-plan-sha256', help='SHA-256 of the first plan reviewed by the operator')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    plan_bytes = args.plan.read_bytes()
    plan_digest = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes)
    if args.execute:
        validate_plan(plan, plan_digest, measure_start=args.measure_start,
                      reviewed_digest=args.reviewed_plan_sha256)
    else:
        check_path(plan)
    start_height = 1000 * plan['path'][0]['tip_board_m'][2]
    if not args.execute and args.measure_start:
        print(f"Stationary preview: modeled start height {start_height:.2f} mm. No devices opened.")
        return
    if not args.execute:
        print(f"Preview: {plan['square']} at {plan['requested_height_mm']} mm. No devices opened.")
        return
    if not all((args.port,args.robot_id,args.camera,args.output)):
        parser.error('--execute requires --port, --robot-id, --camera and --output')
    try:
        require_unowned_port(args.port)
    except (OSError, RuntimeError) as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True,exist_ok=False)
    report = dict(status='preflight', motion_ready=False, plan=plan,
                  plan_sha256=plan_digest,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  gain_writes=False, automatic_return=False, automatic_release=False)
    arm = LeRobotArm(args.port, args.robot_id)
    startup_torque_timeout = False
    video = None
    def save():
        (args.output/'metadata.json').write_text(json.dumps(report, indent=2)+'\n')
    raw = arm.raw_positions
    try:
        report['hardware'] = arm.connect()
        if report['hardware'] != plan['hardware']:
            raise RuntimeError('Calibration, gains or passive state differs from reviewed diagnostic; no writes')
        start = arm.bus.sync_read('Present_Position', normalize=False, num_retry=0)
        if any(abs(start[j]-plan['start_raw'][j])>2 for j in JOINTS):
            raise RuntimeError('Start pose changed; replan from actual encoders before motion')
        check_next(plan, report['hardware'], plan['start_raw'], raw, plan)
        initial = arm.read()
        target = plan['target'] | {j:initial[j] for j in ('wrist_roll','gripper')}
        if args.measure_start:
            target = initial.copy()
            plan = dict(plan, requested_height_mm=start_height)
            report['first_pose'] = 'stationary_start'
        endpoint = raw(target)
        bounds = {j:(min(start[j],endpoint[j])-8,max(start[j],endpoint[j])+8) if j in guards.ACTIVE else (start[j]-2,start[j]+2) for j in JOINTS}
        limits = {}
        for j,(lo,hi) in bounds.items():
            c = arm.bus.calibration[j]
            if not c.range_min <= lo < hi <= c.range_max:
                raise RuntimeError('Envelope exceeds saved calibration')
            limits[j] = [arm.normalized_positions({j: v})[j] for v in (lo, hi)]
        config = args.output/'recovery-config.json'
        config.write_text(json.dumps(dict(drift_ticks=2,max_temperature=45,read_deadline_s=.5,poll_deadline_s=.5,hold_deadline_s=60,poll_s=.1)))
        arm.prepare_recovery(config, limits, args.output)
        arm.operator.discard_pending()
        arm.operator = guards.PowerOperator(arm.operator)
        origin = time.monotonic()
        report.update(origin_monotonic=origin, start_raw=start, target_raw=endpoint, bounds=bounds)
        save()
        video = (StereoRecorder(args.camera, args.side_camera) if args.side_camera else
                 VideoRecorder(args.camera,15, fixed_resolution=(1920,1080)))
        video.start(args.output,origin)
        with (args.output/'samples.jsonl').open('x',buffering=1) as trace:
            def emit(event, **data):
                trace.write(json.dumps(dict(event=event,time_monotonic=time.monotonic(),elapsed_s=time.monotonic()-origin,**data))+'\n')
            # Passive video/encoder baseline before the energizing goal write.
            for _ in range(10):
                video.check()
                emit('passive',positions=arm.bus.sync_read('Present_Position',normalize=False,num_retry=0))
                time.sleep(.1)
            guards.seed_and_verify(arm,initial,JOINTS,lambda event,values:emit(event,values=values))
            expected = raw(initial)
            original=plan
            pose=0
            while True:
                pose+=1
                measured=False
                emit('pose_started',pose=pose,square=plan['square'],height_mm=plan['requested_height_mm'])
                begun = time.monotonic()
                stationary_start = args.measure_start and pose == 1
                commands = () if stationary_start else chunked_commands(initial, target)
                for index,(command,arrival) in enumerate(commands):
                    deadline = begun+index*.1
                    time.sleep(max(0,deadline-time.monotonic()))
                    if time.monotonic()-deadline>.2 or time.monotonic()-begun>55:
                        raise RuntimeError('Motion schedule deadline exceeded')
                    video.check()
                    if arm.operator.event(0) is not None:
                        raise RuntimeError('Operator input cancelled motion')
                    feedback = guards.check_feedback(arm.bus,start,expected,bounds,lambda data:emit('feedback',registers=data),tracking_ticks=8 if arrival or (pose > 1 and index == 0) else 26)
                    if pose > 1 and index == 0:
                        check_continuation_start(plan, feedback, expected)
                    if arrival:
                        emit('checkpoint_verified', expected_raw=expected)
                    arm.send(command)
                    expected=raw(command)
                    emit('command',command=command,command_raw=expected)
                guards.check_feedback(arm.bus,start,expected,bounds,lambda data:emit('arrival',registers=data),tracking_ticks=8)
                hold = dict(initial=target, start_raw=expected, hardware=report['hardware'],
                            T_base_from_board=plan['T_base_from_board'])
                with (args.output / f'hold-{pose}.json').open('x') as file:
                    json.dump(hold, file, indent=2)
                report['status']='target_reached_powered_measurement_hold'; save()
                if stationary_start:
                    print('Stationary hold verified. Starting height is a model estimate.', flush=True)
                if args.side_camera:
                    print('Recording powered hold. Enter snapshot for a measured path start, next PATH, or stop.', flush=True)
                else:
                    print(f'REACHED {plan["square"]} at modeled {plan["requested_height_mm"]} mm. Enter height and ruler uncertainty in mm, e.g. 70.4 0.5. Type stop to end.',flush=True)
                hold_start=time.monotonic()
                while time.monotonic()-hold_start<180:
                    video.check()
                    feedback=guards.check_feedback(arm.bus,start,expected,bounds,lambda data:emit('hold',registers=data),tracking_ticks=8)
                    line = arm.operator.event(0)
                    if line is None:
                        time.sleep(.1)
                        continue
                    if line in ('stop', 'input-closed'):
                        report['status'] = 'operator_finished' if line == 'stop' else 'terminal_closed'
                        return
                    if line == 'snapshot':
                        positions = feedback['Present_Position']
                        normalized = arm.normalized_positions(positions)
                        snapshot = dict(hold, initial=normalized, start_raw=positions,
                                        held_goal_raw=dict(expected), elapsed_s=time.monotonic()-origin)
                        name = f'measured-{pose}-{time.monotonic_ns()}.json'
                        with (args.output / name).open('x') as file:
                            json.dump(snapshot, file, indent=2)
                        print(f'Measured start saved to {name}', flush=True)
                        continue
                    if not measured and not args.side_camera:
                        try:
                            height, uncertainty = map(float,line.split())
                            if not math.isfinite(height) or height < 0 or not 0 < uncertainty < math.inf:
                                raise ValueError
                        except ValueError:
                            print('Enter two numbers in mm, or stop.',flush=True)
                            continue
                        reading = dict(height_mm=height,ruler_uncertainty_mm=uncertainty)
                        emit('operator_measurement_received',pose=pose,square=plan['square'],
                             measurement=reading,registers=feedback)
                        measured=True
                        report['status']='measured_waiting_for_reviewed_target'
                        save()
                        print('Saved. Enter next PATH to a reviewed plan, or stop.',flush=True)
                        continue
                    if line.startswith('next ') and time.monotonic()-hold_start < 180:
                        try:
                            following=json.loads(Path(line[5:].strip()).read_text())
                        except (ValueError,OSError) as exc:
                            print(f'Could not load plan: {exc}',flush=True)
                            continue
                        next_start = following['start_raw']
                        check_next(following, report['hardware'], next_start, raw, original)
                        check_continuation_start(following, feedback, expected)
                        initial=following['initial']
                        target=following['target']
                        endpoint=raw(target)
                        proposed={j:(min(next_start[j],endpoint[j])-8,max(next_start[j],endpoint[j])+8)
                                  if j in guards.ACTIVE else (expected[j]-2,expected[j]+2) for j in JOINTS}
                        for j,(lo,hi) in proposed.items():
                            c=arm.bus.calibration[j]
                            if not c.range_min<=lo<hi<=c.range_max:
                                raise ValueError('New path outside calibration')
                        bounds=proposed
                        arm.recovery.bounds=dict(bounds)
                        start=dict(next_start)
                        plan=following
                        emit('accepted_next_path',pose=pose+1,plan=following,bounds=bounds)
                        break
                    time.sleep(.1)
                else:
                    raise RuntimeError('Powered hold deadline reached')
    except BaseException as exc:
        startup_torque_timeout = isinstance(exc, guards.StartupTorqueTimeout)
        report.update(status='stopped',error=repr(exc))
    finally:
        # Recovery must run even if writing metadata or printing failed.
        try:
            if arm.enabled:
                if args.measure_start and startup_torque_timeout:
                    arm.operator = StationaryRecoveryInput(arm.operator, arm.recovery, video)
                report['recovery']=arm.recover(report.get('error',report['status']))
            if arm.bus.is_connected:
                arm.close()
        finally:
            if video is not None:
                try: report['video']=video.close()
                except Exception as exc: report['video_error']=repr(exc)
            save()
            print(report['status'], report.get('error',''),flush=True)
    if report.get('error'):
        raise SystemExit(1)

if __name__=='__main__':
    def stop(*_):
        raise KeyboardInterrupt('Operator stop')
    signal.signal(signal.SIGTERM,stop)
    main()
