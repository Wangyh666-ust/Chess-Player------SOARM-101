"""Pure pressure control and measured contact geometry. No physics/worlds.

Closure still uses the proven slow position sweep. Holding uses negative motor
torque for closure, with bilateral force feedback, unchanged .05 Nm saturation.
Force bands trigger control/pause; they are not physical-success criteria.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from task_outcome_grip_v1 import TaskGrip

SETTINGS = dict(kp_Nm_per_N=.04, ki_Nm_per_N_s=.04,
                damping_Nm_s_per_rad=.001, torque_cap_Nm=.05,
                pause_fraction=.8, pause_s=.25, pause_cooldown_s=.5,
                max_recenter_actions=2, recenter_step_mm=.5)


class PressureGrip(TaskGrip):
    def __init__(self, *args):
        super().__init__(*args)
        self.integral = None
        self.torque = 0.
        self.velocity = 0.
        self.measured_torque = 0.
        self.last_pause = -float('inf')
        self.recenters = 0

    def hold(self, actual, forces, dt, supported=False):
        self._filter(forces, dt)
        if self.integral is None:
            # Last native POSITION_CONTROL effort is measured before switching.
            self.integral = min(.05, max(0., -self.measured_torque))
        weak, strong = min(self.filtered), max(self.filtered)
        error = self.target - weak
        if strong > self.high:
            error = self.high - strong
        bilateral = min(forces) > .01
        # Missing contact: keep pressure, don't integrate toward more torque.
        if bilateral:
            self.integral = float(np.clip(self.integral + SETTINGS['ki_Nm_per_N_s']*error*dt, 0., .05))
            magnitude = self.integral + SETTINGS['kp_Nm_per_N']*error
            self.torque = -float(np.clip(magnitude, 0., .05))
        elif self.torque == 0.:
            self.torque = -self.integral
        tau = float(np.clip(self.torque-SETTINGS['damping_Nm_s_per_rad']*self.velocity, -.05, 0.))
        if actual <= self.limit and tau < 0:
            tau = 0.  # Maximum closure range; no force increase or success claim.
        self.command = float(actual)  # Geometry uses measured angle, not a fictitious pressure angle.
        return dict(ok=True, command=self.command, motor_torque_Nm=tau,
                    control='bilateral_force_PI_torque_hold', bilateral=bilateral,
                    pressure_error_N=error, integral_Nm=self.integral,
                    contact_loss_signal=not bilateral, force_cap_warning=strong>.8,
                    pause_signal=weak<SETTINGS['pause_fraction']*self.target)


def contact_geometry(obs):
    """Solver-weighted points/normals; observed link6 moving, link7 fixed.

    Robot contact arrays themselves determine grouping; no force-array ordering
    is used. Position_on_b and normals are world-frame native solver data.
    """
    points=[]; normals=[]; forces=[]; distances=[]
    for link in (6, 7):
        cs=[c for c in obs.get('robot_piece_contact_vectors', [])
            if c['kind_a']=='robot' and c['piece_b']==obs['target_key']
            and c['link_a']==link and c['normal_force']>1e-8]
        if not cs:return dict(bilateral=False)
        weights=[c['normal_force'] for c in cs]
        points.append(np.average([c['position_on_b'] for c in cs], axis=0, weights=weights))
        n=np.average([c['contact_normal_on_b'] for c in cs], axis=0, weights=weights)
        normals.append(n/np.linalg.norm(n)); forces.append(sum(weights))
        distances.append(min(c['distance'] for c in cs)*1000)
    midpoint=np.mean(points, axis=0)
    # Piece axis runs through bottom_pose's XY at the measured contact height.
    R=Rotation.from_quat(obs['target_quat']).as_matrix()
    p0=np.asarray(obs['target_pos']); local=R.T@(midpoint-p0)
    axis=p0+R[:,2]*local[2]
    grip_axis=normals[0]-normals[1];grip_axis/=np.linalg.norm(grip_axis)
    delta=axis-midpoint
    tangent=delta-grip_axis*np.dot(delta,grip_axis)
    tangent[2]=0.  # Approximately vertical pickup; this is a lateral correction.
    return dict(bilateral=True, points_m=[x.tolist() for x in points],
                normals=[x.tolist() for x in normals], forces_N=forces,
                distances_mm=distances, midpoint_m=midpoint.tolist(),
                axis_m=axis.tolist(), tangent_to_axis_mm=(1000*tangent).tolist(),
                opposition_deg=math.degrees(math.acos(float(np.clip(-np.dot(*normals),-1,1)))))


def apply_torque(p, w, decision):
    """Override only the gripper's default position motor for this holding tick."""
    p.setJointMotorControl2(w.robot, w.gripper_joint, p.VELOCITY_CONTROL,
                           targetVelocity=0., force=0., physicsClientId=w.client)
    p.setJointMotorControl2(w.robot, w.gripper_joint, p.TORQUE_CONTROL,
                           force=decision['motor_torque_Nm'], physicsClientId=w.client)
