"""Pure board-supported seating control; post-confirmation guards are unchanged.

Radians decrease to close. Total rotation is never reset or hidden. A bounded
seating allowance is separate from recent motion and from airborne stability.
"""
from settled_pressure_control_v1 import SettledTargetGrip

SEATING = dict(close_total_rotation_deg=6., stable_relative_rotation_deg=.2,
               stable_relative_shift_mm=.1, persistent_motion_s=.4,
               persistent_rotation_window_deg=.5, persistent_shift_window_mm=.15,
               observation_window_s=.2, persistent_force_fraction=.8)


class CenteredTargetGrip(SettledTargetGrip):
    def __init__(self, opened, limit, target_force_N):
        super().__init__(opened, limit, target_force_N)
        self.cfg.update(SEATING)
        self.motion_streak = 0.

    def close(self, actual, velocity, forces, dt, shift_mm, rotation_deg,
              window_shift_mm=0., window_rotation_deg=0., window_duration_s=0.):
        decision = super().close(actual, velocity, forces, dt, shift_mm,
                                 rotation_deg, window_shift_mm, window_rotation_deg)
        if not decision['ok']:
            if decision.get('reason') == 'piece_moved_during_close':
                decision['reason'] = ('close_translation_limit' if shift_mm > self.cfg['close_total_shift_mm']
                                      else 'close_seating_rotation_limit')
            return decision
        full_window = window_duration_s >= self.cfg['observation_window_s']-1e-9
        moving = (window_rotation_deg > self.cfg['persistent_rotation_window_deg']
                  or window_shift_mm > self.cfg['persistent_shift_window_mm'])
        near_pressure = min(self.filtered) >= self.cfg['persistent_force_fraction']*self.target
        self.motion_streak = self.motion_streak+dt if full_window and near_pressure and moving else 0.
        decision.update(window_complete=full_window, motion_streak_s=self.motion_streak,
                        seating_rotation_limit_deg=self.cfg['close_total_rotation_deg'],
                        window_rotation_deg=window_rotation_deg, window_shift_mm=window_shift_mm,
                        window_duration_s=window_duration_s)
        if self.motion_streak >= self.cfg['persistent_motion_s']-1e-9:
            self.mode = 'closing'
            return dict(decision, ok=False, ready=False, reason='persistent_motion_near_pressure')
        # A full observation window must exist before accumulating stability.
        if not full_window:
            self.streak=0.; self.mode='closing'; decision['ready']=False
        decision['stability_class'] = ('pressure_and_pose_stable' if decision.get('ready') else
                                      'ongoing_motion' if moving and full_window else 'seating_or_pressure_build')
        return decision
