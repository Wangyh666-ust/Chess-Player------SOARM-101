"""Sample reproducible, board-facing perspective cameras without importing Blender.

The reference location supplies the nominal orbit. We aim at the board volume,
reset photographic lens shifts, and reject views that crop that volume. Blender
checks the saved framing points again with its actual projection before rendering.
"""
import copy
import math
import random

import numpy as np


def interval(value, name, positive=False):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
            or value[0] > value[1] or (positive and value[0] <= 0)):
        raise ValueError(f"Invalid camera bounds for {name}: {value}")
    return value


def framing_points(dimensions, tallest_piece_mm):
    """Conservative box containing the whole board and any standing piece."""
    half_x, half_y = dimensions['width_mm'] / 2000, dimensions['depth_mm'] / 2000
    top = (dimensions['thickness_mm'] + tallest_piece_mm + 2) / 1000
    return [[x, y, z] for x in [-half_x, half_x] for y in [-half_y, half_y] for z in [0, top]]


def project_frame(points, spec):
    """Return normalized image coordinates and depth for a centred horizontal sensor."""
    matrix = np.asarray(spec['matrix_world'])
    local = (np.asarray(points) - matrix[:3, 3]) @ matrix[:3, :3]
    depth = -local[:, 2]
    width, height = spec['resolution']
    scale = spec['lens_mm'] / spec['sensor_width_mm']
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = np.column_stack((0.5 + local[:, 0] / depth * scale,
                              0.5 - local[:, 1] / depth * scale * width / height))
    return uv, depth


def sample_camera(reference, bounds, points, *, seed):
    """Sample inside every bound; never zoom out beyond bounds to force acceptance."""
    if type(seed) is not int or type(bounds.get('enabled')) is not bool:
        raise ValueError('Camera seed must be integer and enabled must be boolean')
    if not bounds['enabled']:
        # Explicit opt-out preserves the saved reference, including its crop/shifts.
        return copy.deepcopy(reference)
    expected = {'enabled', 'azimuth_offset_degrees', 'elevation_offset_degrees',
                'distance_scale', 'target_offset_mm', 'roll_degrees', 'lens_scale',
                'margin_fraction', 'max_attempts'}
    if set(bounds) != expected:
        raise ValueError(f'Camera settings must contain exactly {sorted(expected)}')
    for key in ['azimuth_offset_degrees', 'elevation_offset_degrees', 'roll_degrees',
                'distance_scale', 'lens_scale']:
        interval(bounds[key], key, key in {'distance_scale', 'lens_scale'})
    offsets = bounds['target_offset_mm']
    if not isinstance(offsets, list) or len(offsets) != 3:
        raise ValueError('target_offset_mm needs X, Y and Z intervals')
    for value in offsets:
        interval(value, 'target_offset_mm')
    margin, attempts = bounds['margin_fraction'], bounds['max_attempts']
    if type(margin) not in (int, float) or not math.isfinite(margin) or not 0 <= margin < .5:
        raise ValueError('margin_fraction must be between 0 and 0.5 (exclusive)')
    if type(attempts) is not int or attempts < 1:
        raise ValueError('max_attempts must be a positive integer')
    points = np.asarray(points, dtype=float)
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    nominal = np.asarray(reference.get('matrix_world', np.eye(4)))[:3, 3] if 'matrix_world' in reference else np.asarray(reference['location_m'])
    delta = nominal - centre
    distance = float(np.linalg.norm(delta))
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError('Reference camera must differ from board centre')
    azimuth = math.atan2(delta[1], delta[0])
    elevation = math.asin(delta[2] / distance)
    rng = random.Random(seed)
    for attempt in range(1, attempts + 1):
        az_delta = rng.uniform(*bounds['azimuth_offset_degrees'])
        el_delta = rng.uniform(*bounds['elevation_offset_degrees'])
        az, el = azimuth + math.radians(az_delta), elevation + math.radians(el_delta)
        distance_scale = rng.uniform(*bounds['distance_scale'])
        target_offset = np.array([rng.uniform(*axis) for axis in offsets])
        roll = rng.uniform(*bounds['roll_degrees'])
        lens_scale = rng.uniform(*bounds['lens_scale'])
        # Stay above the board and avoid the look-at singularity at the pole.
        if not 0 < el < math.radians(89.5):
            continue
        location = centre + distance * distance_scale * np.array([
            math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        target = centre + target_offset / 1000
        forward = target - location
        norm = np.linalg.norm(forward)
        if norm < 1e-9:
            continue
        forward /= norm
        right = np.cross(forward, [0, 0, 1])
        norm = np.linalg.norm(right)
        if norm < 1e-9:
            continue
        right /= norm
        up = np.cross(right, forward)
        cosine, sine = math.cos(math.radians(roll)), math.sin(math.radians(roll))
        matrix = np.eye(4)
        matrix[:3, :3] = np.column_stack((cosine * right + sine * up,
                                         -sine * right + cosine * up, -forward))
        matrix[:3, 3] = location
        spec = {key: copy.deepcopy(reference[key])
                for key in ['resolution', 'sensor_width_mm', 'lens_mm']}
        spec.update(location_m=location.tolist(), target_m=target.tolist(),
                    matrix_world=matrix.tolist(), shift_x=0, shift_y=0,
                    lens_mm=reference['lens_mm'] * lens_scale)
        uv, depth = project_frame(points, spec)
        if np.all(depth > .001) and np.all(uv >= margin) and np.all(uv <= 1 - margin):
            spec['framing_points_m'] = points.tolist()
            spec['framing_margin_fraction'] = margin
            spec['sampling'] = dict(seed=seed, attempt=attempt,
                                    azimuth_offset_degrees=az_delta,
                                    elevation_offset_degrees=el_delta,
                                    distance_scale=distance_scale,
                                    target_offset_mm=target_offset.tolist(),
                                    roll_degrees=roll, lens_scale=lens_scale)
            return spec
    raise ValueError(f'No camera fits the board within {attempts} attempts; widen distance/lens bounds or reduce the margin')
