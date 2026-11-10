"""Torque-demand-limited velocity actuator, explicit pressure-delivery telemetry.

The old .01rad/s setpoint starved contact force before the effort limit engaged.
A .5rad/s normal target gives the original force PI effort request authority;
raw loss commands zero velocity at the same .05Nm ceiling. Closure remains the
original slow sweep. One240Hz tick of free closure is at most .12deg nominal.
"""
from pressure_center_control_v1 import PressureGrip as PreviousGrip,SETTINGS as OLD
from preload_hold_control_v1 import transition_check
SETTINGS=dict(OLD,governor_velocity_rad_s=.5,control='original_force_PI_effort_limited_velocity;raw_loss_brake')


class PressureGrip(PreviousGrip):
    def hold(self,actual,forces,dt,supported=False):
        d=super().hold(actual,forces,dt,supported)
        brake=not d['bilateral'] or actual<=self.limit
        cap=.05 if brake else abs(d['motor_torque_Nm'])
        return dict(d,control=SETTINGS['control'],braking=brake,
            pressure_request_Nm=d['motor_torque_Nm'],drive_torque_Nm=0.,
            governor_effort_cap_Nm=cap,motor_effort_cap_Nm=cap,
            target_velocity_rad_s=0. if brake else -.5)


def apply_motor(p,w,d):
    p.setJointMotorControl2(w.robot,w.gripper_joint,p.VELOCITY_CONTROL,
        targetVelocity=d['target_velocity_rad_s'],velocityGain=1.,
        force=d['motor_effort_cap_Nm'],physicsClientId=w.client)
