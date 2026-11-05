"""Seeded, bounded variations of an existing diffuse lighting preset.

Pure Python: usable by a dataset planner without importing Blender. Coordinates
are metres relative to the board centre at (0, 0, 0); angles are degrees.
This preserves source geometry, not a guarantee against camera-dependent glare.
"""

import copy
import math
import random


DEFAULT_BOUNDS = {
    # Visual sweep across both cameras and seven presets; recheck new camera poses.
    # Direction is asymmetric because the two cameras see different reflections.
    "azimuth_degrees": [-30.0, 90.0],
    "power_scale": [0.8, 1.0],
    "per_light_power_scale": [0.65, 1.0],
    "world_scale": [1.0, 1.0],
}


def _numbers(values, length, name):
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in values):
        raise ValueError(f"{name} must contain finite numbers")
    return list(values)


def _range(values, name, nonnegative=True):
    low, high = _numbers(values, 2, name)
    if low > high or (nonnegative and low < 0):
        raise ValueError(f"Invalid bounds for {name}: {values}")
    return low, high


def sample_lighting(lights, world_strength, *, seed, bounds=None):
    """Return new light specs, world strength and replay metadata; never mutate inputs.

    ``lights`` uses the existing renderer's location_m/target_m/power_w schema.
    ``world_strength`` is a number or [minimum, maximum]. Bounds override defaults.
    A common power scale changes fixtures and room fill together; independent
    per-light scales vary their balance. world_scale adjusts fill separately.
    All sources and targets rotate together around Z, preserving distances,
    height, diffuser dimensions, and horizontal/vertical source orientations.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if bounds is None:
        bounds = {}
    if not isinstance(bounds, dict):
        raise ValueError("bounds must be an object")
    unknown = bounds.keys() - DEFAULT_BOUNDS.keys()
    if unknown:
        raise ValueError(f"Unknown lighting bounds: {sorted(unknown)}")
    limits = {
        name: _range(bounds.get(name, default), name, name != "azimuth_degrees")
        for name, default in DEFAULT_BOUNDS.items()
    }
    world_range = _range(
        world_strength if isinstance(world_strength, (list, tuple))
        else [world_strength, world_strength], "world_strength"
    )
    if not isinstance(lights, (list, tuple)) or not lights:
        raise ValueError("An explicit non-empty diffuse lights list is required")
    result = copy.deepcopy(list(lights))
    for light in result:
        if not isinstance(light, dict):
            raise ValueError("Each light must be an object")
        if light.get("shape", "SQUARE") not in {"SQUARE", "RECTANGLE", "DISK", "ELLIPSE"}:
            raise ValueError("Only diffuse area-light shapes are supported")
        location = _numbers(light.get("location_m"), 3, "location_m")
        target = _numbers(light.get("target_m", [0, 0, 0]), 3, "target_m")
        if location == target:
            raise ValueError("A light's target must differ from its location")
        power, size, size_y = _numbers(
            [light.get("power_w"), light.get("size_m"),
             light.get("size_y_m", light.get("size_m"))], 3, "power and diffuser sizes"
        )
        if power < 0 or size <= 0 or size_y <= 0:
            raise ValueError("Power must be nonnegative and diffuser sizes positive")
        color = _numbers(light.get("color"), 3, "color")
        _numbers([light.get("rig_rotation_degrees", 0)], 1, "rig_rotation_degrees")
        if any(v < 0 or v > 1 for v in color):
            raise ValueError("Light color components must be in [0, 1]")

    rng = random.Random(seed)
    angle = rng.uniform(*limits["azimuth_degrees"])
    power_scale = rng.uniform(*limits["power_scale"])
    world_scale = rng.uniform(*limits["world_scale"])
    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))

    def rotate(point):
        x, y, z = point
        return [cosine * x - sine * y, sine * x + cosine * y, z]

    individual_scales = []
    for light in result:
        scale = rng.uniform(*limits["per_light_power_scale"])
        individual_scales.append(scale)
        light["location_m"] = rotate(light["location_m"])
        light["target_m"] = rotate(light.get("target_m", [0, 0, 0]))
        # Preserve the rectangle's in-plane rotation too, not just its normal.
        light["rig_rotation_degrees"] = light.get("rig_rotation_degrees", 0) + angle
        light["power_w"] *= power_scale * scale
    world = rng.uniform(*world_range) * power_scale * world_scale
    if not math.isfinite(world) or any(not math.isfinite(l["power_w"]) for l in result):
        raise ValueError("Lighting power overflow; reduce the configured bounds")
    return {
        "lights": result,
        "world_strength": world,
        "parameters": {
            "seed": seed,
            "bounds": {name: list(value) for name, value in limits.items()},
            "azimuth_degrees": angle,
            "power_scale": power_scale,
            "per_light_power_scales": individual_scales,
            "world_scale": world_scale,
        },
    }
