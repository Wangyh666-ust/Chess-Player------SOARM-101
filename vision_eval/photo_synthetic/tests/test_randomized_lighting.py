"""Run with: python3 vision_eval/photo_synthetic/tests/test_randomized_lighting.py"""

import copy
import json
import math
from pathlib import Path
import sys
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from randomized_lighting import sample_lighting


class LightingTests(unittest.TestCase):
    def test_blender_diffuser_axes(self):
        """Also runnable with Blender --background --python-exit-code 1 --python THIS_FILE."""
        try:
            import bpy
            from mathutils import Quaternion, Vector
            from reference_style import set_lights
        except ImportError:
            self.skipTest("Run inside Blender to check the actual diffuser axes")
        presets = json.loads((PACKAGE / "configs/lighting_presets.json").read_text())
        for preset in presets["variants"]:
            config = preset["overrides"]
            objects = []
            for _ in config["lights"]:
                data = bpy.data.lights.new("Sampler test", "AREA")
                objects.append(bpy.data.objects.new("Sampler test", data))
            try:
                set_lights(objects, config["lights"])
                baseline = [obj.rotation_euler.to_quaternion() for obj in objects]
                for angle in [-30, 0, 45, 90]:
                    sample = sample_lighting(config["lights"], config["world_strength"], seed=3,
                                             bounds={"azimuth_degrees": [angle, angle]})
                    set_lights(objects, sample["lights"])
                    rotation = Quaternion((0, 0, 1), math.radians(angle))
                    for obj, original, spec in zip(objects, baseline, sample["lights"]):
                        for axis in [(1, 0, 0), (0, 1, 0), (0, 0, -1)]:
                            expected = rotation @ original @ Vector(axis)
                            actual = obj.rotation_euler.to_quaternion() @ Vector(axis)
                            self.assertLess((expected - actual).length, 1e-5)
                        self.assertAlmostEqual(obj.data.energy, spec["power_w"], places=4)
            finally:
                for obj in objects:
                    data = obj.data
                    bpy.data.objects.remove(obj)
                    bpy.data.lights.remove(data)

    def test_presets_preserve_geometry_and_replay(self):
        presets = json.loads((PACKAGE / "configs/lighting_presets.json").read_text())
        for preset in presets["variants"]:
            config = preset["overrides"]
            original = copy.deepcopy(config)
            for seed in range(50):
                result = sample_lighting(config["lights"], config["world_strength"], seed=seed)
                self.assertEqual(result, sample_lighting(config["lights"], config["world_strength"], seed=seed))
                self.assertNotEqual(result, sample_lighting(config["lights"], config["world_strength"], seed=seed + 1))
                parameters = result["parameters"]
                limits = parameters["bounds"]
                self.assertTrue(limits["azimuth_degrees"][0] <= parameters["azimuth_degrees"] <= limits["azimuth_degrees"][1])
                self.assertTrue(limits["power_scale"][0] <= parameters["power_scale"] <= limits["power_scale"][1])
                for before, after in zip(config["lights"], result["lights"]):
                    for key in ["shape", "size_m", "size_y_m", "color"]:
                        self.assertEqual(before.get(key), after.get(key))
                    self.assertEqual(before["location_m"][2], after["location_m"][2])
                    self.assertAlmostEqual(math.dist(before["location_m"], [0, 0, 0]),
                                           math.dist(after["location_m"], [0, 0, 0]))
                    self.assertAlmostEqual(math.dist(before["location_m"], before["target_m"]),
                                           math.dist(after["location_m"], after["target_m"]))
                    minimum = limits["power_scale"][0] * limits["per_light_power_scale"][0]
                    maximum = limits["power_scale"][1] * limits["per_light_power_scale"][1]
                    self.assertTrue(minimum <= after["power_w"] / before["power_w"] <= maximum)
                self.assertEqual(config, original)

    def test_exact_bounds_and_invalid_input(self):
        lights = [{"location_m": [1, 0, 3], "target_m": [1, 0, 0],
                   "power_w": 40, "shape": "RECTANGLE", "size_m": 1.2,
                   "size_y_m": .3, "color": [1, 1, 1]}]
        bounds = {"azimuth_degrees": [90, 90], "power_scale": [2, 2],
                  "per_light_power_scale": [.5, .5], "world_scale": [.25, .25]}
        result = sample_lighting(lights, .5, seed=42, bounds=bounds)
        light = result["lights"][0]
        self.assertAlmostEqual(light["location_m"][0], 0)
        self.assertAlmostEqual(light["location_m"][1], 1)
        self.assertEqual(light["power_w"], 40)
        self.assertEqual(light["rig_rotation_degrees"], 90)
        self.assertEqual(result["world_strength"], .25)
        for bad in [{"power_scale": [-1, 1]}, {"power_scale": [2, 1]},
                    {"azimuth_degrees": [0, float("nan")]}, {"typo": [0, 1]},
                    {"world_scale": [True, 1]}, {"power_scale": [1]}]:
            with self.assertRaises(ValueError):
                sample_lighting(lights, .5, seed=1, bounds=bad)
        for key, value in [("size_m", 0), ("power_w", -1), ("shape", "SPOT"),
                           ("location_m", [1, 0, 0]), ("color", [1, 2, 1])]:
            bad = copy.deepcopy(lights)
            bad[0][key] = value
            with self.assertRaises(ValueError):
                sample_lighting(bad, .5, seed=1)
        with self.assertRaises(ValueError):
            sample_lighting([], .5, seed=1)


if __name__ == "__main__":
    unittest.main(argv=[__file__])
