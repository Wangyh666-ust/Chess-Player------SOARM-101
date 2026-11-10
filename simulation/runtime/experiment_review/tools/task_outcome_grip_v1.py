"""Simulation-only task checks; pressure values guide control, not task vetoes."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from material_grip_control_v1 import MaterialReference, MaterialTargetGrip, relative
from settled_pressure_control_v1 import PRESSURE


class TaskGrip(MaterialTargetGrip):
    def _filter(self, forces, dt):
        if len(forces)!=2 or dt<=0 or not all(math.isfinite(float(f)) and f>=0 for f in forces):
            raise ValueError('invalid_feedback')
        alpha=dt/(self.cfg['force_filter_s']+dt)
        self.filtered=list(forces) if self.filtered is None else [a+alpha*(b-a) for a,b in zip(self.filtered,forces)]

    def close(self,actual,velocity,forces,dt,shift_mm,rotation_deg,*window):
        if not all(math.isfinite(x) for x in (actual,velocity,shift_mm,rotation_deg)):
            return dict(ok=False,reason='invalid_feedback')
        self._filter(forces,dt);self.elapsed+=dt
        weak,strong=min(self.filtered),max(self.filtered)
        rate=PRESSURE['free_speed_rad_s'] if max(forces)<.01 else PRESSURE['contact_speed_rad_s']
        if weak>=PRESSURE['near_target_fraction']*self.target:rate=PRESSURE['near_target_speed_rad_s']
        if weak>=self.target or self.elapsed>=self.cfg['close_timeout_s'] or self.command<=self.limit+1e-12:
            self.mode='holding';self.streak=0.
            return dict(ok=True,ready=True,command=self.command,pressure_target_reached=weak>=self.target,
                        confirmation='bounded_closure_complete;dynamic_pickup_checked_separately')
        if strong>self.high:self.command=min(self.opened,self.command+rate*dt)
        else:self.command=max(self.limit,self.command-rate*dt)
        return dict(ok=True,ready=False,command=self.command)

    def hold(self,actual,forces,dt,supported=False):
        self._filter(forces,dt);self.hold_elapsed+=dt
        if self.hold_elapsed>=self.cfg['hold_update_s'] and not supported:
            step=math.radians(self.cfg['hold_rate_deg_s'])*self.hold_elapsed;self.hold_elapsed=0.
            if max(self.filtered)>self.high:self.command+=step
            elif min(forces)>.01 and min(self.filtered)<self.low:self.command-=step
            self.command=min(self.opened,actual,max(self.limit,actual-math.radians(self.cfg['hold_preload_cap_deg']),self.command))
        return dict(ok=True,command=self.command,contact_loss_signal=min(forces)<=.01,
                    force_cap_warning=max(forces)>self.cfg['force_cap_N'])


class TaskReference(MaterialReference):
    def __init__(self,obs):
        try:
            super().__init__(obs);self.reference_kind='force_weighted_material_point'
        except ValueError:
            self.material=np.zeros(3);self.pos,self.rot=relative(obs);self.center=self.pos.copy()
            self.reference_time=float(obs['time']);self.last_time=self.reference_time
            self.last_center=self.center.copy();self.reference_kind='COM_fallback;no_bilateral_confirmation'


def separation_drop(separated,relative_down_m_s,dt,previous_s,released=False):
    # Geometry and uncommanded falling must agree. This is a 50ms debounce of
    # actual separation, not pressure/pose stability or accumulated slip.
    duration=previous_s+dt if separated and relative_down_m_s>.02 and not released else 0.
    return duration,duration>=.05-1e-9


class TaskObserver:
    def __init__(self,w,planner):
        self.w=w;self.pl=planner;self.falling_s=0.;self.dropped=False;self.airborne_steps=0
        self.pl.task_records=[]

    def sample(self,case,phase,obs,forces,material,released):
        import pybullet as p
        w=self.w;cid=w.client;body=w.bodies[case['target_key']]
        metrics=material.measure(obs)
        gaps=[]
        for link in (6,7):
            points=p.getClosestPoints(w.robot,body,.03,linkIndexA=link,physicsClientId=cid)
            gaps.append(min((cp[8] for cp in points),default=.03))
        link=p.getLinkState(w.robot,w.tcp_link,computeLinkVelocity=1,computeForwardKinematics=1,physicsClientId=cid)
        lv,_=p.getBaseVelocity(body,physicsClientId=cid)
        relative_down=float(link[6][2]-lv[2])
        board=p.getContactPoints(body,w.board,physicsClientId=cid)
        supported=any(cp[9]>0 for cp in board)
        moving=phase in ('lift','trial_lift_hold','lift_hold','transfer','lower','support_hold','settle_before_open')
        separated=min(gaps)>.0002 and max(forces)<=.01
        self.falling_s,drop=separation_drop(separated,relative_down,w.dt,self.falling_s,released or not moving or supported)
        if moving and not released and not supported and min(gaps)<=.0002:self.airborne_steps+=1
        self.dropped=self.dropped or drop
        self.pl.task_records.append(dict(metrics,event='task_observation',phase=phase,
            jaw_gap_mm=[1000*x for x in gaps],forces_N=list(forces),relative_down_m_s=relative_down,
            fall_duration_s=self.falling_s,dropped=drop,supported=supported,released=released,
            tilt_deg=float(np.degrees(np.arccos(np.clip(Rotation.from_quat(obs['target_quat']).as_matrix()[2,2],-1,1))))))
        return drop


class TaskTail:
    def __init__(self,planner,material):self.pl=planner;self.material=material;self.last=-1.
    def begin(self,obs):self.pl.grip_records.append(dict(event='task_tail_reference',time=obs['time']))
    def check(self,w,case,selected,phase,commandq,queue,forces,supported):
        if w.time-self.last<w.dt*3.9:return None
        self.last=w.time;obs=w.observe(phase,commandq,selected['g_closed'])
        self.pl.grip_records.append(dict(self.material.measure(obs),event='diagnostic_only_grip',
            phase=phase,forces_N=list(forces),supported=supported))
        return None
