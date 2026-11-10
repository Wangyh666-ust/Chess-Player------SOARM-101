"""Fixed material-point slip and bounded pressure hold; SI angles, mm metrics.

Pure numerical classes only. No worlds, motor calls or planner execution.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from centered_pressure_control_v1 import CenteredTargetGrip

SETTINGS=dict(shift_limit_mm=.5,rotation_limit_deg=2.,low_force_fraction=.8,
    saturated_fraction=.95,opening_velocity_deg_s=2.,point_velocity_mm_s=2.,
    recovery_timeout_s=.3,recovery_stable_s=.05,recovery_max=2,
    hold_rate_deg_s=.5)


def relative(obs):
    tcp=Rotation.from_quat(obs['tcp_quat']).as_matrix()
    return tcp.T@(np.asarray(obs['target_pos'])-obs['tcp_pos']), tcp.T@Rotation.from_quat(obs['target_quat']).as_matrix()


class MaterialReference:
    def __init__(self,obs):
        points=[]
        for link in (6,7):
            cs=[c for c in obs['robot_piece_contact_vectors']
                if c['kind_a']=='robot' and c['piece_b']==obs['target_key']
                and c['link_a']==link and c['normal_force']>1e-8]
            if not cs:raise ValueError('material_reference_missing_bilateral_contact')
            points.append(np.average([c['position_on_b'] for c in cs],axis=0,
                                     weights=[c['normal_force'] for c in cs]))
        self.material=Rotation.from_quat(obs['target_quat']).inv().apply(np.mean(points,axis=0)-obs['target_pos'])
        self.pos,self.rot=relative(obs)
        self.center=self.pos+self.rot@self.material
        self.reference_time=float(obs['time']);self.last_time=self.reference_time;self.last_center=self.center.copy()

    def measure(self,obs):
        pos,rot=relative(obs);center=pos+rot@self.material
        delta=(center-self.center)*1000;elapsed=float(obs['time'])-self.last_time
        velocity=float(np.linalg.norm(center-self.last_center)*1000/elapsed) if elapsed>1e-10 else 0.
        world_delta=Rotation.from_quat(obs['tcp_quat']).apply(delta)
        metrics=dict(time=float(obs['time']),shift_mm=float(np.linalg.norm(delta)),
            rotation_deg=float(np.degrees(np.linalg.norm(Rotation.from_matrix(self.rot.T@rot).as_rotvec()))),
            bottom_shift_mm=float(np.linalg.norm(pos-self.pos)*1000),delta_mm=delta.tolist(),
            relative_world_down_mm=float(-world_delta[2]),point_velocity_mm_s=velocity,
            reference_time=self.reference_time,material_point_mm=(self.material*1000).tolist())
        if elapsed>1e-10:self.last_time=float(obs['time']);self.last_center=center.copy()
        return metrics


def warning(metrics,forces,velocity_rad_s,torque_Nm,target_N,cap_Nm):
    if min(forces)<SETTINGS['low_force_fraction']*target_N:return 'bilateral_force_low'
    if abs(torque_Nm)>=SETTINGS['saturated_fraction']*cap_Nm and math.degrees(velocity_rad_s)>SETTINGS['opening_velocity_deg_s']:
        return 'saturated_motor_opening'
    if metrics['point_velocity_mm_s']>SETTINGS['point_velocity_mm_s']:return 'contact_point_motion_fast'
    return None


class MaterialTargetGrip(CenteredTargetGrip):
    def __init__(self,*args):
        super().__init__(*args);self.cfg['hold_rate_deg_s']=SETTINGS['hold_rate_deg_s']
        self.in_contact_recovery=False

    def hold(self,actual,forces,dt,supported=False):
        # An interval with absent contact must not accumulate a later angle jump.
        self.hold_elapsed=min(self.hold_elapsed,self.cfg['hold_update_s'])
        previous=self.command
        result=super().hold(actual,forces,dt,supported)
        if supported or result.get('contact_loss_signal'):self.hold_elapsed=0.
        # During recovery, preserve existing preload without chasing lower angles.
        # Opening to relieve excessive pressure remains possible; no force/cap increase.
        if self.in_contact_recovery and result.get('ok') and result['command']<previous:
            self.command=previous;result['command']=previous;result['tightening_frozen']=True
        return result
