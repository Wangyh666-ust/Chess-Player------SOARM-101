import math
import numpy as np
import pybullet


def _norm(v):
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _ceil_div(a, b):
    return int(math.ceil(a / b))


def run(world, frames, preload_deg=2., speed=.025, no_grip=False):
    obs = []
    dt = world.dt
    preload = math.radians(preload_deg)
    g_fixed = frames[0]['g'] if no_grip else None

    lo, hi = pybullet.getJointInfo(world.robot, world.gripper_joint,
                                   physicsClientId=world.client)[8:10]

    def clamp_g(g):
        return min(max(float(g), float(lo)), float(hi))

    cmds = []
    n = len(frames)
    i = 0
    while i < n:
        ph = frames[i]['phase']
        j = i
        while j + 1 < n and frames[j + 1]['phase'] == ph:
            j += 1
        start, end = i, j
        span = end - start
        for k in range(i, j + 1):
            fr = frames[k]
            frac = (k - start) / span if span > 0 else 1.0
            g = fr['g'] if no_grip else float(fr['g'])
            if ph == 'close':
                g = g - preload * frac
            elif ph in ('lift', 'transfer', 'transfer_via', 'lower'):
                g = g - preload
            elif ph == 'open':
                g = g - preload * (1.0 - frac)
            if no_grip:
                g = g_fixed
            q = [float(x) for x in fr['q']]
            cmds.append({'q': q, 'g': clamp_g(g), 'phase': ph,
                         'tcp': [float(x) for x in fr['tcp_pos']]})
        i = j + 1

    orig = [{'phase': frames[k]['phase']} for k in range(n)]

    def hold(q, g, phase, duration):
        N = max(1, _ceil_div(duration, dt))
        for k in range(1, N + 1):
            if world.time > 90.0:
                raise RuntimeError('time limit exceeded')
            world.command(list(q), float(g))
            world.step()
            if k % 4 == 0 or k == N:
                obs.append(world.observe(phase, list(q), float(g)))

    def emit(prev, cur, N, phase, min_dur):
        for k in range(1, N + 1):
            if world.time > 90.0:
                raise RuntimeError('time limit exceeded')
            f = k / N
            q = [prev['q'][a] + (cur['q'][a] - prev['q'][a]) * f
                 for a in range(5)]
            g = prev['g'] + (cur['g'] - prev['g']) * f
            world.command(q, g)
            world.step()
            if k % 4 == 0 or k == N:
                obs.append(world.observe(phase, q, g))
        return 1

    hold(cmds[0]['q'], cmds[0]['g'], 'settle_start', .5)

    phase_count = {}
    for c in cmds:
        phase_count[c['phase']] = phase_count.get(c['phase'], 0) + 1

    pending_first = {p: False for p in phase_count}
    done_hold_close = False
    done_hold_lift = False

    for idx in range(len(cmds) - 1):
        if world.time > 90.0:
            raise RuntimeError('time limit exceeded')
        a = cmds[idx]
        b = cmds[idx + 1]
        ph = b['phase']
        dq = max(abs(b['q'][m] - a['q'][m]) for m in range(5))
        dg = abs(b['g'] - a['g'])
        dtc = _norm([b['tcp'][m] - a['tcp'][m] for m in range(3)])
        dur = max(dq / .3, dg / .5, dtc / speed, dt)
        if ph in ('close', 'open'):
            dur = max(dur, .6 / max(1, phase_count[ph]))
        N = max(1, _ceil_div(dur, dt))
        if not emit(a, b, N, ph, dur):
            break
        pending_first[ph] = True
        if ph == 'close':
            is_last = (idx + 1 == len(cmds) - 1 or cmds[idx + 2]['phase'] != 'close')
            if is_last and not done_hold_close:
                hold(b['q'], b['g'], 'close_hold', .4)
                done_hold_close = True
        if ph == 'lift':
            is_last = (idx + 1 == len(cmds) - 1 or cmds[idx + 2]['phase'] != 'lift')
            if is_last and not done_hold_lift:
                hold(b['q'], b['g'], 'lift_hold', .3)
                done_hold_lift = True

    last = cmds[-1]
    hold(last['q'], last['g'], 'settle_end', 1.0)
    return obs
