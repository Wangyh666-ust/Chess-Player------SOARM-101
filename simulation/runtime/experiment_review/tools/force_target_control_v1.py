"""Pure per-region target pressure control; decreasing radians closes gripper."""
import math
PARAMETERS=dict(close_speed_rad_s=.05,close_timeout_s=6.,stable_contact_s=.2,
    contact_min_N=.10,target_band_fraction=.05,force_cap_N=.8,
    stable_velocity_deg_s=2.,stable_relative_shift_mm=.1,stable_relative_rotation_deg=.5,
    close_total_shift_mm=.5,close_total_rotation_deg=2.,hold_rate_deg_s=1.5,
    hold_preload_cap_deg=5.,hold_update_s=1/60,force_filter_s=.04,
    trial_lift_mm=3.,trial_hold_s=.3,trial_shift_limit_mm=.5,
    trial_rotation_limit_deg=2.,trial_tilt_growth_limit_deg=.5)

class TargetGrip:
    def __init__(self,opened,limit,target_force_N,parameters=None):
        self.cfg=dict(PARAMETERS,**(parameters or {}));self.opened=float(opened);self.limit=float(limit)
        self.target=float(target_force_N)
        if not all(math.isfinite(x) for x in (self.opened,self.limit,self.target)) or not self.limit<self.opened:
            raise ValueError('invalid_angle_bounds')
        if not .1<self.target<self.cfg['force_cap_N']/(1+self.cfg['target_band_fraction']):
            raise ValueError('reference_target_outside_reviewed_force_cap')
        self.low=self.target*(1-self.cfg['target_band_fraction']);self.high=self.target*(1+self.cfg['target_band_fraction'])
        self.command=self.opened;self.mode='closing';self.elapsed=self.streak=self.hold_elapsed=0.
        self.filtered=None
    def _forces(self,forces,dt):
        if len(forces)!=2 or dt<=0 or not all(math.isfinite(float(x)) and x>=0 for x in forces):
            raise ValueError('invalid_feedback')
        if max(forces)>self.cfg['force_cap_N']:return 'contact_force_cap'
        alpha=dt/(self.cfg['force_filter_s']+dt)
        self.filtered=list(forces) if self.filtered is None else [a+alpha*(b-a) for a,b in zip(self.filtered,forces)]
        return None
    def close(self,actual,velocity,forces,dt,shift_mm,rotation_deg,window_shift_mm=0.,window_rotation_deg=0.):
        if not all(math.isfinite(x) for x in (actual,velocity,dt,shift_mm,rotation_deg,window_shift_mm,window_rotation_deg)):
            return dict(ok=False,reason='invalid_feedback')
        try:why=self._forces(forces,dt)
        except ValueError:return dict(ok=False,reason='invalid_feedback')
        self.elapsed+=dt
        if why:return dict(ok=False,reason=why)
        if shift_mm>self.cfg['close_total_shift_mm'] or rotation_deg>self.cfg['close_total_rotation_deg']:
            return dict(ok=False,reason='piece_moved_during_close')
        if self.elapsed>self.cfg['close_timeout_s']:return dict(ok=False,reason='pressure_timeout')
        weak,strong=min(self.filtered),max(self.filtered)
        if weak<self.target and strong<self.high:
            self.command=max(self.limit,self.command-self.cfg['close_speed_rad_s']*dt)
        elif strong>self.high:
            self.command=min(self.opened,self.command+self.cfg['close_speed_rad_s']*dt)
        stable=(min(forces)>=self.low and max(forces)<=self.high and weak>=self.low and strong<=self.high
            and abs(math.degrees(velocity))<=self.cfg['stable_velocity_deg_s']
            and window_shift_mm<=self.cfg['stable_relative_shift_mm']
            and window_rotation_deg<=self.cfg['stable_relative_rotation_deg'])
        self.streak=self.streak+dt if stable else 0.
        if self.streak+1e-10>=self.cfg['stable_contact_s']:
            self.mode='holding';return dict(ok=True,ready=True,command=self.command)
        if self.command<=self.limit+1e-12 and weak<self.low:
            return dict(ok=False,reason='closure_limit_before_target_pressure')
        return dict(ok=True,ready=False,command=self.command)
    def hold(self,actual,forces,dt,supported=False):
        if self.mode!='holding':raise RuntimeError('holding requires pressure confirmation')
        if not math.isfinite(actual):return dict(ok=False,reason='invalid_feedback')
        try:why=self._forces(forces,dt)
        except ValueError:return dict(ok=False,reason='invalid_feedback')
        if why:return dict(ok=False,reason=why)
        self.hold_elapsed+=dt
        if min(forces)<=.01:return dict(ok=True,command=self.command,updated=False,contact_loss_signal=True)
        if self.hold_elapsed+1e-10<self.cfg['hold_update_s'] or supported:
            return dict(ok=True,command=self.command,updated=False)
        step=math.radians(self.cfg['hold_rate_deg_s'])*self.hold_elapsed;self.hold_elapsed=0.
        if max(self.filtered)>self.high:self.command+=step
        elif min(self.filtered)<self.low:self.command-=step
        self.command=min(self.opened,actual,max(self.limit,actual-math.radians(self.cfg['hold_preload_cap_deg']),self.command))
        return dict(ok=True,command=self.command,updated=True)
