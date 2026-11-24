"""Differential, level-board registration for the taught rank-four poses."""
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from manipulation.calibration import arm_from_model, arm_ticks_to_degrees, arm_to_model
from robot_calibration.board_pose import estimate_board_pose
from robot_calibration.camera_calibration import load_target, marker_detector
from robot_calibration.camera_geometry import camera_parameters, check_fit_target, undistort_points
from robot_calibration.records import ARM, read_json as load, write_json
from robot_calibration.kinematics import board_landmark, fixed_tip_point, gripper_link_pose

ROOT = Path('runs/robot-calibration')
BENCH_BOARD_HEIGHT_M = .0173
BENCH_CELL_PITCH_M = (.202 / 8, .198 / 8)
REFERENCES = {
    'e4': ('f4-e4-f4-1795320957796506596', 400),
    'e4_return': ('e4-f4-e4-1795325443461197397', 'registration.json'),
    'e4_manual': ('manual-e4-five-1795344969385300689/trial-4', 'registration.json'),
    'e4_local': ('e4-recenter-1795321285716310432', 504),
    'f4': ('f4-place-1795320452161752189', 48),
}
MARKERS = {
    20: ('minimum-pickup-20261118/post-reseat-B-marker20-localization.json', 'marker20_corners_base_m'),
    22: ('powered-placement-20261120/marker22-resecured-20261120-154606/remount-candidate.json', 'corners_base_m'),
}


def level_pose(q, height_m=BENCH_BOARD_HEIGHT_M):
    x, y, yaw = q
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, height_m], [0, 0, 0, 1]])


class Registration:
    def __init__(self, config_path=None):
        if config_path is None:
            self.lens = load(ROOT / 'fixed-camera-investigation-20261117/candidate.json')
            target, self.board = load_target(Path('robot_calibration/configs/board_target.json'))
            self.detector = marker_detector(target, adaptive_max=53)
            self.geometry = {i: np.array(load(ROOT / path)[key]) for i, (path, key) in MARKERS.items()}
            self.height_m = BENCH_BOARD_HEIGHT_M
            self.pitch_m = BENCH_CELL_PITCH_M
            self.anchors = {1, 4, 5, 8}
            return

        config_path = Path(config_path)
        config = load(config_path)
        target_path = config_path.parent / config['board_target']
        self.lens = load(config_path.parent / config['fixed_lens'])
        target, self.board = load_target(target_path)
        self.detector = marker_detector(target, adaptive_max=53)
        check_fit_target(self.lens, target_path, self.board)
        camera_parameters(self.lens, self.lens['image_size_wh'])
        self.geometry = {int(i): np.asarray(corners, dtype=float)
                         for i, corners in config['base_marker_corners_m'].items()}
        self.anchors = set(config['board_anchor_ids'])
        self.height_m = config['board_height_m']
        self.pitch_m = np.asarray(config['cell_pitch_m'], dtype=float)
        if (len(self.geometry) < 2 or set(self.geometry).intersection(self.board)
                or any(c.shape != (4, 3) or not np.isfinite(c).all() for c in self.geometry.values())
                or len(self.anchors) < 4 or not self.anchors < self.board.keys()
                or not isinstance(self.height_m, (int, float)) or not np.isfinite(self.height_m)
                or self.pitch_m.shape != (2,) or not np.isfinite(self.pitch_m).all()
                or np.any(self.pitch_m <= 0)):
            raise ValueError('Invalid base survey, board anchors, height or cell pitch')

    def measure(self, frame, *, manual=False):
        corners, ids, _ = self.detector.detectMarkers(frame)
        ids = [] if ids is None else ids.ravel().tolist()
        if not manual and not set(self.geometry).issubset(ids):
            raise ValueError(f'Need base markers {sorted(self.geometry)}; visible IDs {ids}')
        points = [c[0] for c in corners]
        known = sorted(set(ids) & self.board.keys())
        anchors = self.anchors
        marker_data = dict(ids=ids, corners_px=[p.tolist() for p in points])
        if manual:
            if len(known) < 4:
                return dict(marker_data, status='partial', reason='Fewer than four board markers')
            # Reserve an interior marker when possible; fit the outer markers.
            centers = np.array([points[ids.index(i)].mean(axis=0) for i in known])
            holdout = [known[np.argmin(np.linalg.norm(centers-centers.mean(axis=0), axis=1))]] if len(known)>4 else []
        else:
            if not anchors.issubset(known) or len(known) <= len(anchors):
                raise ValueError(f'Need board anchors {sorted(anchors)} plus an independent board marker')
            holdout = sorted(set(known)-anchors)
        try:
            board = estimate_board_pose(ids, points, self.board, self.lens, frame.shape[1::-1], holdout)
            if max(board['fit_rms_px'], board['holdout_rms_px'] or 0) > 3:
                raise ValueError('Board reprojection exceeds 3 px')
        except (ValueError, cv2.error) as exc:
            if manual:
                return dict(marker_data, status='partial', reason=str(exc))
            raise
        if manual and not set(self.geometry).issubset(ids):
            return dict(marker_data, board=board, status='board_only')
        camera = np.array(board['T_camera_from_board'])
        k, d, model = camera_parameters(self.lens, frame.shape[1::-1])
        # Intersect each undistorted ray with its surveyed corner height.
        # This uses both markers' separation to resolve yaw.
        board_from_camera = np.linalg.inv(camera)
        observed = []
        surveyed = []
        for marker, geometry in self.geometry.items():
            rays = undistort_points(points[ids.index(marker)], k, d, model)
            rays = np.c_[rays.reshape(4, 2), np.ones(4)] @ board_from_camera[:3, :3].T
            origin = board_from_camera[:3, 3]
            depth = (geometry[:, 2] - self.height_m - origin[2]) / rays[:, 2]
            if not np.isfinite(depth).all() or min(depth) <= 0:
                if manual:
                    return dict(marker_data, board=board, status='board_only', reason='Invalid marker-plane intersection')
                raise ValueError('Invalid marker-plane intersection')
            observed.extend((origin + rays * depth[:, None])[:, :2])
            surveyed.extend(geometry[:, :2])
        xy, base = np.array(observed), np.array(surveyed)
        u, _, vt = np.linalg.svd((xy-xy.mean(0)).T @ (base-base.mean(0)))
        rotation = vt.T @ np.diag([1., np.linalg.det(vt.T @ u.T)]) @ u.T
        translation = base.mean(0) - rotation @ xy.mean(0)
        residuals = xy @ rotation.T + translation - base
        if np.max(np.linalg.norm(residuals, axis=1)) > .004:
            if manual:
                return dict(marker_data, board=board, status='board_only', reason='Base marker residual exceeds 4 mm')
            raise ValueError('Marker corners inconsistent with level, surveyed base')
        q = [*translation, np.arctan2(rotation[1, 0], rotation[0, 0])]
        return dict(board=board, q=q, T_base_from_board=level_pose(q, self.height_m).tolist(),
                    residuals_m=residuals.tolist(), ids=ids,
                    **({'status': 'board_and_base', 'corners_px': marker_data['corners_px']} if manual else {}))

    def reference(self, key):
        run, index = REFERENCES[key]
        if index == 'registration.json':
            return median_fit(load(ROOT / run / index)['observations'])
        video = cv2.VideoCapture(str(ROOT / run / 'video.avi'))
        try:
            video.set(cv2.CAP_PROP_POS_FRAMES, index)
            samples = []
            for _ in range(100):
                ok, frame = video.read()
                if not ok:
                    break
                try:
                    samples.append(self.measure(frame))
                except ValueError:
                    continue
                if len(samples) == 7:
                    return median_fit(samples)
            raise ValueError(f'Incomplete teaching registration in {run} at frame {index}')
        finally:
            video.release()


def median_fit(samples):
    return dict(q=np.median([s['q'] for s in samples], axis=0).tolist(),
                residuals_m=np.median([s['residuals_m'] for s in samples], axis=0).tolist())


def displacement(reference, current, square):
    point = board_landmark(square, BENCH_CELL_PITCH_M, 0.)
    old, new = reference['q'], current['q']
    changes = np.array(current['residuals_m']) - reference['residuals_m']
    if np.max(np.linalg.norm(changes, axis=1)) > .0015:
        raise ValueError('Relative base marker geometry changed')
    shift = ((level_pose(new) - level_pose(old)) @ point)[:3]
    return shift, float(np.arctan2(np.sin(new[2]-old[2]), np.cos(new[2]-old[2])))


def capture_registration(video, keys, output, check_session=None):
    reg = Registration()
    refs = {key: reg.reference(key) for key in keys}
    surveyed = None if keys else reg.reference('e4_return')
    deadline = time.monotonic() + 12
    samples = []
    current = video.latest_sample
    last_index = current[0] if current is not None else None
    error = 'No fresh frame'
    while len(samples) < 14 and time.monotonic() < deadline:
        if check_session is not None:
            check_session()
        video.check()
        latest = video.latest_sample
        if latest is None or latest[0] == last_index:
            time.sleep(.02)
            continue
        last_index, frame_time_s, frame = latest
        try:
            sample = reg.measure(frame)
            sample.update(frame=last_index, frame_time_s=frame_time_s)
            if surveyed is not None:
                displacement(surveyed, sample, 'd4')
            for key in keys:
                displacement(refs[key], sample, key[:2])
        except (ValueError, RuntimeError) as exc:
            error = str(exc)
            continue
        samples.append(sample)
    if len(samples) != 14:
        raise ValueError(f'Startup registration incomplete: {error}')
    result = {}
    if not keys:
        # Absolute planning uses the current board transform. Compare independent
        # halves at the destination and retain the surveyed marker-geometry check.
        first, second = median_fit(samples[:7]), median_fit(samples[7:])
        shift, yaw = displacement(first, second, 'd4')
        if np.linalg.norm(shift) > .001 or abs(yaw) > np.deg2rad(.4):
            raise ValueError('Unstable live board registration')
    for key in keys:
        first = displacement(refs[key], median_fit(samples[:7]), key[:2])
        second = displacement(refs[key], median_fit(samples[7:]), key[:2])
        if np.linalg.norm(first[0]-second[0]) > .001 or abs(first[1]-second[1]) > np.deg2rad(.4):
            raise ValueError(f'Unstable board registration at {key}')
        shift, yaw = displacement(refs[key], median_fit(samples), key[:2])
        if np.linalg.norm(shift) > .005 or abs(yaw) > np.deg2rad(2):
            raise ValueError(f'{key}: setup change exceeds local 5 mm / 2 degree correction range')
        # Held-out frames from unchanged teaching sessions shift the estimated
        # centers by up to 0.8 mm. Do not turn that noise into arm motion.
        applied = shift if np.linalg.norm(shift) > .001 else np.zeros(3)
        result[key] = dict(shift_m=shift.tolist(), applied_shift_m=applied.tolist(), yaw_rad=float(yaw))
    write_json(output / 'registration.json', dict(references={k: REFERENCES[k] for k in keys},
               reference_fits=refs, observations=samples, changes=result))
    return result


def joint_offsets(ik, hardware, square, measured_pan, change):
    # The wrist IK model ends at gripper_link (before the moving jaw).
    # Preserve the taught tool pitch and roll while displacing the fixed pad.
    hover = (measured_pan, 1769, 2480, 2769, 2027) if square == 'e4' else (
        measured_pan, 1772, 2480, 2753, 2027)
    calibration = hardware['calibration']
    tick_deg = 360 / 4095
    degrees = arm_ticks_to_degrees(calibration, dict(zip(ARM, hover)))
    q = np.array(arm_to_model(ik.profile, degrees))
    tip_record = load(ROOT / 'calibration-delivery-20261119/post-reseat-three-contact-tip.json')
    tip = fixed_tip_point(tip_record)

    def position(angles):
        return (gripper_link_pose(ik.robot, angles) @ tip)[:3]

    shift = np.array(change['applied_shift_m'])
    goal = position(q) + shift

    def residual(x):
        tip_error_mm = 1000 * (position(np.r_[x, q[4]]) - goal)
        pitch_error = sum(x[1:4]) - sum(q[1:4])
        return np.r_[tip_error_mm, 100 * pitch_error]

    fit = least_squares(residual, q[:4], bounds=(ik.robot.lo[:4], ik.robot.hi[:4]))
    angles = arm_from_model(ik.profile, np.r_[fit.x, q[4]])
    offsets = {j: round((angles[j] - degrees[j]) / tick_deg) for j in ARM}
    if not fit.success or max(abs(v) for v in offsets.values()) > 35:
        raise ValueError(f'Outside local taught-pose IK range at {square}')
    # The same differential is applied over the short pickup/set-down stroke.
    # Check its quantized displacement at both ends in the actual model.
    for shoulder in (hover[1], 1773 if square == 'e4' else 1775):
        before = dict(degrees)
        before['shoulder_lift'] += (shoulder - hover[1]) * tick_deg
        after = {j: before[j] + offsets[j] * tick_deg for j in ARM}
        delta = position(arm_to_model(ik.profile, after)) - position(arm_to_model(ik.profile, before))
        if np.linalg.norm(delta - shift) > .0005:
            raise ValueError(f'Quantized correction misses {square} by more than 0.5 mm in the model')
    return offsets
