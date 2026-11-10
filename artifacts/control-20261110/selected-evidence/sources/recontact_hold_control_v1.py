"""One bounded, low-speed contact recovery after braking; original pressure PI.

Limits govern actuator behavior, never declare a physical task failure. No
blind closing once the last measured piece clearance is unavailable or large.
"""
import math
from effort_hold_control_v1 import PressureGrip as Previous, apply_motor, SETTINGS as OLD
from preload_hold_control_v1 import transition_check

SETTINGS = dict(OLD, loss_recontact_speed_rad_s=.1, loss_recontact_time_s=.05,
                loss_recontact_travel_rad=.005, loss_near_gap_mm=.2,
                control='original_force_PI;one_brake_tick;bounded_near_contact_recovery')


class PressureGrip(Previous):
    def __init__(self, *args):
        super().__init__(*args)
        self.jaw_gaps_mm = None
        self.loss_start = None
        self.loss_elapsed = 0.
        self.loss_finished = False

    def hold(self, actual, forces, dt, supported=False):
        d = super().hold(actual, forces, dt, supported)
        phase = 'pressure_hold'
        if d['bilateral']:
            self.loss_start = None
            self.loss_elapsed = 0.
            self.loss_finished = False
        elif self.loss_start is None:
            self.loss_start = float(actual)
            self.loss_elapsed = 0.
            phase = 'initial_brake'
        else:
            self.loss_elapsed += dt
            gaps = self.jaw_gaps_mm
            near = (gaps is not None and len(gaps) == 2
                    and all(math.isfinite(x) for x in gaps)
                    and max(gaps) <= SETTINGS['loss_near_gap_mm'])
            travel = max(0., self.loss_start - actual)
            if (self.loss_elapsed > SETTINGS['loss_recontact_time_s'] + 1e-10
                    or travel >= SETTINGS['loss_recontact_travel_rad']
                    or actual <= self.limit or not near):
                self.loss_finished = True
            if not self.loss_finished and max(forces) < self.high:
                speed = min(SETTINGS['loss_recontact_speed_rad_s'],
                            (SETTINGS['loss_recontact_travel_rad'] - travel) / dt,
                            (actual - self.limit) / dt)
                d.update(target_velocity_rad_s=-max(0., speed), braking=False)
                phase = 'slow_recontact'
            else:
                phase = 'brake_until_contact_returns'
        return dict(d, control=SETTINGS['control'], recontact_phase=phase,
                    recontact_elapsed_s=self.loss_elapsed,
                    recontact_travel_rad=0. if self.loss_start is None else max(0., self.loss_start-actual),
                    recontact_gap_mm=self.jaw_gaps_mm,
                    recontact_exhausted=self.loss_finished)
