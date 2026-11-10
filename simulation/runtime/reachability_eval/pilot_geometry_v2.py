import os
import math
import pybullet as p
from path_ik import solve_pose

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "model", "so101_new_calib.urdf")

CONFIG = {
    "reserve_neighbor_mm": 0.5,
    "hard_penetration_tolerance_mm": 0.05,
    "target_support_penetration_mm": 0.1,
    "max_center_mm": 2.0,
    "closure_angle_step_deg": 2.0,
}

CLEAR_BUF = 0.002
HARD_TOL = -0.00005
NEIGH_TOL = 0.0005
SURF_TOL = -0.0001


def _axis_min_gap(aabbA, aabbB):
    best = -1e9
    for i in range(3):
        lo_a, hi_a = aabbA[0][i], aabbA[1][i]
        lo_b, hi_b = aabbB[0][i], aabbB[1][i]
        g = max(lo_b - hi_a, lo_a - hi_b)
        if g > best:
            best = g
    return best


def _mancp(cps):
    if not cps:
        return None
    return sorted(cps, key=lambda c: c[8])[0]


class Geometry:
    def __init__(self, w):
        self.w = w
        self.case = None
        self.snapshot = None
        self._parse_limits()

    def _parse_limits(self):
        import xml.etree.ElementTree as ET
        tree = ET.parse(URDF)
        root = tree.getroot()
        lo = hi = None
        for j in root.findall("joint"):
            if j.get("name") == "gripper":
                lim = j.find("limit")
                if lim is None:
                    raise RuntimeError("gripper joint limit missing")
                lo = float(lim.get("lower"))
                hi = float(lim.get("upper"))
                break
        if lo is None or hi is None:
            raise RuntimeError("gripper joint limits missing in urdf")
        self.lo = lo
        self.hi = hi

    def reset(self, case, snapshot, start_q, start_g):
        self.w.set_case(case, start_q, start_g, poses=snapshot)
        self.case = case
        self.snapshot = snapshot

    def restore_target(self):
        key = self.case["target_key"]
        pose = self.snapshot[key]
        self.w.set_query_object(key, pose["pos"], pose["quat"])

    def jaw(self, q, g, which):
        w = self.w
        w.set_query_robot(q, g)
        link = w.link_index["fixed_jaw_collision" if which == "fixed" else "moving_jaw_so101_v1_link"]
        tgt = w.bodies[self.case["target_key"]]
        cp = _mancp(p.getClosestPoints(w.robot, tgt, distance=0.05,
                                       linkIndexA=link, physicsClientId=w.client))
        return cp

    def _robot_aabbs(self):
        w = self.w
        n = p.getNumJoints(w.robot, physicsClientId=w.client)
        out = {}
        for li in [-1] + list(range(n)):
            out[li] = p.getAABB(w.robot, linkIndex=li, physicsClientId=w.client)
        return out

    def check(self, q, g, phase, targetpose=None):
        w = self.w
        cid = w.client
        key = self.case["target_key"]
        if targetpose is not None:
            w.set_query_object(key, targetpose["pos"], targetpose["quat"])
        else:
            self.restore_target()
        w.set_query_robot(q, g)
        tgt = w.bodies[key]
        n = p.getNumJoints(w.robot, physicsClientId=cid)
        links = [-1] + list(range(n))
        aabbs = self._robot_aabbs()
        reasons = []
        clearance = CLEAR_BUF

        other_keys = [k for k in w.bodies.keys() if k != key]
        bodies = [(k, w.bodies[k]) for k in w.bodies]
        for nm in ("board", "tray"):
            b = getattr(w, nm, None)
            if b is not None:
                bodies.append((nm, b))

        allow_phase = phase in ("close", "lift", "transfer", "lower", "open", "retreat")
        finger_links = (w.link_index["fixed_jaw_collision"],
                        w.link_index["moving_jaw_so101_v1_link"],
                        w.link_index["gripper_frame_link"])

        for k, body in bodies:
            for li in links:
                if _axis_min_gap(aabbs[li], p.getAABB(body, physicsClientId=cid)) > CLEAR_BUF:
                    continue
                cp = _mancp(p.getClosestPoints(w.robot, body, distance=CLEAR_BUF,
                                               linkIndexA=li, physicsClientId=cid))
                if cp is None:
                    continue
                d = cp[8]
                if k == key:
                    if li in finger_links:
                        if allow_phase:
                            continue
                        if d < HARD_TOL:
                            reasons.append("phase=%s target link=%d d=%.5f" % (phase, li, d))
                    else:
                        if d < HARD_TOL:
                            reasons.append("phase=%s target link=%d d=%.5f" % (phase, li, d))
                elif k in ("board", "tray"):
                    if d < HARD_TOL:
                        reasons.append("phase=%s %s link=%d d=%.5f" % (phase, k, li, d))
                else:
                    if d < NEIGH_TOL:
                        reasons.append("phase=%s neighbor=%s link=%d d=%.5f" % (phase, k, li, d))
                    clearance = min(clearance, max(0.0, d))

        excl = w.excluded_pairs
        for i in range(len(links)):
            for j in range(i + 1, len(links)):
                a, b = links[i], links[j]
                if tuple(sorted((a, b))) in excl:
                    continue
                if _axis_min_gap(aabbs[a], aabbs[b]) > CLEAR_BUF:
                    continue
                cp = _mancp(p.getClosestPoints(w.robot, w.robot, distance=CLEAR_BUF,
                                               linkIndexA=a, linkIndexB=b, physicsClientId=cid))
                if cp is None:
                    continue
                if cp[8] < HARD_TOL:
                    reasons.append("phase=%s self a=%d b=%d d=%.5f" % (phase, a, b, cp[8]))

        if targetpose is not None and phase in ("lift", "transfer", "lower"):
            taabb = p.getAABB(tgt, physicsClientId=cid)
            for k, body in bodies:
                if k == key:
                    continue
                if _axis_min_gap(taabb, p.getAABB(body, physicsClientId=cid)) > CLEAR_BUF + NEIGH_TOL:
                    continue
                cp = _mancp(p.getClosestPoints(tgt, body, distance=CLEAR_BUF, physicsClientId=cid))
                if cp is None:
                    continue
                d = cp[8]
                if k in ("board", "tray"):
                    if d < SURF_TOL:
                        reasons.append("carried phase=%s %s d=%.5f" % (phase, k, d))
                else:
                    if d < NEIGH_TOL:
                        reasons.append("carried phase=%s %s d=%.5f" % (phase, k, d))

        if reasons:
            return {"ok": False, "reason": "; ".join(reasons), "clearance_mm": clearance * 1000.0}
        return {"ok": True, "reason": "", "clearance_mm": clearance * 1000.0}

    def contact_grasp(self, q, pos, quat, seed, kin, nominal_g):
        w = self.w
        cid = w.client
        key = self.case["target_key"]
        tgt = w.bodies[key]
        fli = w.link_index["fixed_jaw_collision"]
        mli = w.link_index["moving_jaw_so101_v1_link"]
        max_shift = CONFIG["max_center_mm"] / 1000.0
        total_shift = 0.0
        for _ in range(4):
            w.set_query_robot(q, nominal_g)
            fp = _mancp(p.getClosestPoints(w.robot, tgt, distance=0.05,
                                           linkIndexA=fli, physicsClientId=cid))
            if fp is None:
                return {"ok": False, "reason": "no fixed cp"}
            d = fp[8]
            nvec = fp[7]
            nx, ny = nvec[0], nvec[1]
            mag = math.hypot(nx, ny)
            if mag < 0.5 or abs(nvec[2]) > 0.8:
                return {"ok": False, "reason": "fixed jaw normal invalid"}
            err = d - 0.0001
            if abs(err) < 0.00005:
                break
            denom = nx * nx + ny * ny
            if denom < 1e-9:
                return {"ok": False, "reason": "degenerate normal"}
            cx = -err * nx / denom
            cy = -err * ny / denom
            total_shift += math.hypot(cx, cy)
            if total_shift > max_shift:
                return {"ok": False, "reason": "centering shift exceeds 2mm"}
            pos = (pos[0] + cx, pos[1] + cy, pos[2])
            sol = solve_pose(kin, pos, quat, seed)
            if sol is None or not sol.get("pass"):
                return {"ok": False, "reason": "ik fail"}
            q = sol["q"]
        w.set_query_robot(q, nominal_g)
        fp = _mancp(p.getClosestPoints(w.robot, tgt, distance=0.05,
                                       linkIndexA=fli, physicsClientId=cid))
        if fp is None:
            return {"ok": False, "reason": "no fixed cp after solve"}
        if abs(fp[8] - 0.0001) > 0.0002:
            return {"ok": False, "reason": "fixed cp dist off %.5f" % fp[8]}
        step = math.radians(CONFIG["closure_angle_step_deg"])
        ang = self.lo
        prev_d = None
        neg_found = False
        root = None
        while ang <= self.hi:
            w.set_query_robot(q, ang)
            mcps = _mancp(p.getClosestPoints(w.robot, tgt, distance=0.05,
                                             linkIndexA=mli, physicsClientId=cid))
            dcur = mcps[8] if mcps is not None else 1.0
            if dcur <= 0.0001:
                neg_found = True
            if prev_d is not None and prev_d <= 0.0001 <= dcur:
                a, b = ang - step, ang
                for _ in range(10):
                    mid = 0.5 * (a + b)
                    w.set_query_robot(q, mid)
                    mc = _mancp(p.getClosestPoints(w.robot, tgt, distance=0.05,
                                                   linkIndexA=mli, physicsClientId=cid))
                    dm = mc[8] if mc is not None else 1.0
                    if dm <= 0.0001:
                        a = mid
                    else:
                        b = mid
                root = 0.5 * (a + b)
                break
            prev_d = dcur
            ang += step
        if root is None or not neg_found:
            return {"ok": False, "reason": "no upward closure crossing"}
        w.set_query_robot(q, root)
        mp = _mancp(p.getClosestPoints(w.robot, tgt, distance=0.05,
                                       linkIndexA=mli, physicsClientId=cid))
        if mp is None:
            return {"ok": False, "reason": "no moving cp at root"}
        fn = fp[7]
        mn = mp[7]
        dot = fn[0] * mn[0] + fn[1] * mn[1] + fn[2] * mn[2]
        if dot > -0.2:
            return {"ok": False, "reason": "normals not opposing dot=%.3f" % dot}
        if abs(mn[2]) > 0.8:
            return {"ok": False, "reason": "moving normal z too big"}
        return {
            "ok": True,
            "q": q,
            "pos": pos,
            "centering_mm": total_shift * 1000.0,
            "gcontact": root,
            "normals": (fn, mn),
        }

    def opening(self, q, gcontact, gap_mm):
        w = self.w
        cid = w.client
        tgt = w.bodies[self.case["target_key"]]
        mli = w.link_index["moving_jaw_so101_v1_link"]
        target = gap_mm / 1000.0
        step = math.radians(CONFIG["closure_angle_step_deg"])
        ang = gcontact
        prev = None
        crossings = None
        while ang <= self.hi:
            w.set_query_robot(q, ang)
            mc = _mancp(p.getClosestPoints(w.robot, tgt, distance=0.05,
                                           linkIndexA=mli, physicsClientId=cid))
            d = mc[8] if mc is not None else 1.0
            if prev is not None and prev < target <= d:
                crossings = (ang - step, ang)
                break
            prev = d
            ang += step
        if crossings is None:
            return None
        a, b = crossings
        for _ in range(10):
            mid = 0.5 * (a + b)
            w.set_query_robot(q, mid)
            mc = _mancp(p.getClosestPoints(w.robot, tgt, distance=0.05,
                                           linkIndexA=mli, physicsClientId=cid))
            dm = mc[8] if mc is not None else 1.0
            if dm < target:
                a = mid
            else:
                b = mid
        return 0.5 * (a + b)
