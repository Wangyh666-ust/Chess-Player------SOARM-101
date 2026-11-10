"""Pure controls: observed arm settling and slower contact-pressure ramp."""
import math
from force_target_control_v1 import TargetGrip

SETTLE = dict(timeout_s=1., stable_s=.1, joint_error_deg=.02,
              joint_speed_deg_s=.5, tcp_speed_mm_s=.1, tcp_rotation_deg_s=.2)
PRESSURE = dict(close_total_shift_mm=1.2, close_timeout_s=8.,
                free_speed_rad_s=.05, contact_speed_rad_s=.01, near_target_speed_rad_s=.002,
                near_target_fraction=.8, slow_contact_N=.01)


class ArmSettler:
    def __init__(self, config=None):
        self.cfg=dict(SETTLE, **(config or {})); self.streak=0.; self.last=None; self.start=None;self.previous_stable=False

    def update(self, time_s, q, command_q, velocities, tcp_pos, tcp_quat):
        if self.start is None: self.start=time_s
        dt=0 if self.last is None else time_s-self.last[0]
        if dt<0: raise ValueError('nonmonotonic_settle_time')
        err=math.degrees(max(abs(a-b) for a,b in zip(q,command_q)))
        speed=math.degrees(max(abs(v) for v in velocities))
        linear=angular=float('inf')
        if dt>0:
            linear=1000*math.sqrt(sum((a-b)**2 for a,b in zip(tcp_pos,self.last[1])))/dt
            norm=math.sqrt(sum(a*a for a in tcp_quat)*sum(b*b for b in self.last[2]))
            dot=abs(sum(a*b for a,b in zip(tcp_quat,self.last[2])))/norm
            angular=math.degrees(2*math.acos(min(1.,max(0.,dot))))/dt
        stable=(err<=self.cfg['joint_error_deg'] and speed<=self.cfg['joint_speed_deg_s']
                and linear<=self.cfg['tcp_speed_mm_s'] and angular<=self.cfg['tcp_rotation_deg_s'])
        self.streak=self.streak+dt if stable and self.previous_stable else 0.
        self.previous_stable=stable
        self.last=(time_s,list(tcp_pos),list(tcp_quat))
        ready=self.streak>=self.cfg['stable_s']-1e-9
        return dict(ok=ready or time_s-self.start<self.cfg['timeout_s'], ready=ready,
                    elapsed_s=time_s-self.start,stable_s=self.streak,joint_error_deg=err,
                    joint_speed_deg_s=speed,tcp_speed_mm_s=linear,tcp_rotation_deg_s=angular)


class SettledTargetGrip(TargetGrip):
    def __init__(self, opened, limit, target_force_N):
        super().__init__(opened,limit,target_force_N,
                         {k:PRESSURE[k] for k in ('close_total_shift_mm','close_timeout_s')})
        self.contact_latched=False

    def close(self,actual,velocity,forces,dt,shift_mm,rotation_deg,window_shift_mm=0.,window_rotation_deg=0.):
        self.contact_latched=self.contact_latched or max(forces)>=PRESSURE['slow_contact_N']
        weak=min(self.filtered) if self.filtered is not None else min(forces)
        speed=PRESSURE['free_speed_rad_s']; stage='free_close'
        if self.contact_latched:
            speed=PRESSURE['contact_speed_rad_s']; stage='contact_pressure'
            if weak>=PRESSURE['near_target_fraction']*self.target:
                speed=PRESSURE['near_target_speed_rad_s']; stage='near_target_pressure'
        self.cfg['close_speed_rad_s']=speed
        decision=super().close(actual,velocity,forces,dt,shift_mm,rotation_deg,window_shift_mm,window_rotation_deg)
        decision.update(stage=stage,command_speed_rad_s=speed,contact_latched=self.contact_latched)
        return decision
