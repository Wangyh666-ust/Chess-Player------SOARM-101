"""Retain closure preload; force feedback adjusts position, never effort ceiling.

The implicit position motor supplies pressure and a finite velocity request.
Raw unilateral loss switches to a zero-velocity brake. No state resets,
extra joint damping, altered forces, or displacement/tilt task vetoes.
"""
import numpy as np
from pressure_center_control_v1 import PressureGrip as PreviousGrip, SETTINGS as OLD

SETTINGS = dict(OLD, holding_motor_max_velocity_rad_s=.1,
    force_position_gain_rad_per_N_s=.02, reference_rate_rad_s=.01,
    control='retained_position_preload_force_feedback;raw_loss_velocity_brake')


class PressureGrip(PreviousGrip):
    def __init__(self, *args):
        super().__init__(*args)
        self.pressure_command = None
        self.was_braking = False
        self.last_preload = 0.

    def hold(self, actual, forces, dt, supported=False):
        self._filter(forces, dt)
        weak, strong = min(self.filtered), max(self.filtered)
        error = self.target - weak
        if strong > self.high:
            error = self.high - strong
        bilateral = min(forces) > .01
        if self.pressure_command is None:
            # Preserve the real closure command before geometry switches to
            # measured angle; resetting to actual would erase existing preload.
            self.pressure_command = float(self.command)
        braking = not bilateral or actual <= self.limit
        if not braking:
            if self.was_braking:
                # Do not resume a stale position target across lost contact.
                self.pressure_command = actual - self.last_preload
            rate = float(np.clip(SETTINGS['force_position_gain_rad_per_N_s']*error,
                                 -SETTINGS['reference_rate_rad_s'], SETTINGS['reference_rate_rad_s']))
            self.pressure_command = float(np.clip(self.pressure_command-rate*dt, self.limit, self.opened))
            self.last_preload = max(0., actual-self.pressure_command)
        self.was_braking = braking
        self.command = float(actual)  # Checked geometry uses measured joint state.
        return dict(ok=True, command=self.command, motor_effort_cap_Nm=.05,
            motor_position_rad=self.pressure_command, target_velocity_rad_s=0.,
            motor_max_velocity_rad_s=SETTINGS['holding_motor_max_velocity_rad_s'],
            braking=braking, bilateral=bilateral, control=SETTINGS['control'],
            pressure_error_N=error, contact_loss_signal=not bilateral,
            force_cap_warning=strong>.8, pause_signal=weak<.8*self.target)


def apply_motor(p, w, decision):
    common=dict(force=decision['motor_effort_cap_Nm'], velocityGain=1., physicsClientId=w.client)
    if decision['braking']:
        p.setJointMotorControl2(w.robot,w.gripper_joint,p.VELOCITY_CONTROL,targetVelocity=0.,**common)
    else:
        p.setJointMotorControl2(w.robot,w.gripper_joint,p.POSITION_CONTROL,
            targetPosition=decision['motor_position_rad'],targetVelocity=0.,positionGain=.2,
            maxVelocity=decision['motor_max_velocity_rad_s'],**common)


def transition_check(records, target):
    """One source-held commissioning check, not a new airborne protection gate."""
    rows=[r for r in records if r['phase']=='close_hold']
    if not rows:
        return dict(passed=False,reason='missing_holding_evidence')
    end=rows[-1]['after_time'];tail=[r for r in rows if r['after_time']>=end-.5]
    med=np.median([r['after_forces_N'] for r in tail],axis=0).tolist()
    return dict(passed=min(med)>=.8*target and max(med)<=.8,
        target_N=target,initial_forces_N=rows[0]['forces_N'],last_half_second_median_N=med,
        observed_s=end-rows[0]['time'],samples=len(rows),
        peak_speed_rad_s=max(abs(r['after_velocity_rad_s']) for r in rows),
        scope='source-supported pressure commissioning;not dynamic-grasp success')
