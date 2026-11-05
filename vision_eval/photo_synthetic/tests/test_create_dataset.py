"""Planner and failure-state checks; the renderer is exercised separately in Blender."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from create_dataset import ROOT, prepare_dataset, run_dataset, sample_piece_transforms
from randomized_cameras import project_frame, sample_camera
import chess
import numpy as np
from create_setups import create_setups


class DatasetTests(unittest.TestCase):
    def setUp(self):
        setups = json.loads((PACKAGE / 'configs/setups.json').read_text())
        setups.update(mode='random', games=2, count=3, max_plies=24)
        self.base = json.loads((PACKAGE / 'configs/scene.json').read_text())
        self.source = create_setups(setups)
        self.settings = json.loads((PACKAGE / 'configs/dataset.json').read_text())
        self.presets = json.loads((PACKAGE / 'configs/lighting_presets.json').read_text())

    def test_plan_replay_and_preserved_setups(self):
        before = copy.deepcopy(self.source)
        plan = prepare_dataset(self.source, self.settings, self.presets, self.base)
        self.assertEqual(plan, prepare_dataset(self.source, self.settings, self.presets, self.base))
        self.assertEqual(self.source, before)
        for old, new in zip(self.source['positions'], plan['positions']):
            self.assertEqual(old, {k: v for k, v in new.items() if k not in {'lighting_setup', 'piece_transforms', 'piece_seed', 'camera_setup', 'source_setup_id', 'variant_index'}})
            lighting = new['lighting_setup']
            variant = next(v for v in self.presets['variants'] if v['id'] == lighting['preset_id'])
            for original, sampled in zip(variant['overrides']['lights'], lighting['lights']):
                for key in ('size_m', 'size_y_m', 'shape'):
                    self.assertEqual(original.get(key), sampled.get(key))
        self.assertNotEqual(plan['positions'][0]['lighting_setup'], plan['positions'][1]['lighting_setup'])
        self.assertEqual(plan['cameras'], self.base['cameras'])

    def test_four_variants_preserve_source_and_vary_physical_scene(self):
        settings = {**self.settings, 'variants_per_setup': 4}
        plan = prepare_dataset(self.source, settings, self.presets, self.base)
        self.assertEqual(len(plan['positions']), 4 * len(self.source['positions']))
        self.assertEqual(plan, prepare_dataset(self.source, settings, self.presets, self.base))
        self.assertEqual(len({p['id'] for p in plan['positions']}), len(plan['positions']))
        for source in self.source['positions']:
            variants = [p for p in plan['positions'] if p['source_setup_id'] == source['id']]
            self.assertEqual([p['variant_index'] for p in variants], list(range(4)))
            for entry in variants:
                for key in ('fen', 'game_id', 'split', 'ply'):
                    self.assertEqual(entry[key], source[key])
            for key in ('piece_transforms', 'camera_setup', 'lighting_setup'):
                self.assertEqual(len({json.dumps(p[key], sort_keys=True) for p in variants}), 4)
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                prepare_dataset(self.source, {**settings, 'variants_per_setup': value}, self.presets, self.base)

    def test_bad_inputs_fail_before_rendering(self):
        with self.assertRaises(ValueError):
            prepare_dataset(self.source, {**self.settings, 'lighting_presets': ['typo']}, self.presets, self.base)
        bad = copy.deepcopy(self.source)
        bad['positions'][0]['id'] = '../escape'
        with self.assertRaises(ValueError):
            prepare_dataset(bad, self.settings, self.presets, self.base)
        bad = copy.deepcopy(self.source)
        bad['positions'][0]['fen'] = '8/8/8/8/8/8/8/8 w - - 0 1'
        with self.assertRaises(ValueError):
            prepare_dataset(bad, self.settings, self.presets, self.base)
        bad = copy.deepcopy(self.source)
        bad['positions'][0]['ply'] += 1
        with self.assertRaises(ValueError):
            prepare_dataset(bad, self.settings, self.presets, self.base)

    def test_physical_randomization_and_independent_streams(self):
        plan = prepare_dataset(self.source, self.settings, self.presets, self.base)
        changed = copy.deepcopy(self.settings)
        changed['camera_randomization']['fixed']['roll_degrees'] = [0, 0]
        other = prepare_dataset(self.source, changed, self.presets, self.base)
        dimensions = {kind: json.loads((ROOT / asset['file']).with_name('dimensions.json').read_text())
                      for kind, asset in self.base['assets'].items()}
        half_cell = dimensions['board']['square_mm'] / 2
        for entry, alternate in zip(plan['positions'], other['positions']):
            self.assertEqual(entry['piece_transforms'], alternate['piece_transforms'])
            self.assertEqual(entry['lighting_setup'], alternate['lighting_setup'])
            self.assertNotEqual(entry['camera_setup']['fixed'], alternate['camera_setup']['fixed'])
            board = chess.Board(entry['fen'])
            for square, pose in entry['piece_transforms'].items():
                kind = chess.piece_name(board.piece_at(chess.parse_square(square)).piece_type)
                radius = dimensions[kind]['base_diameter_mm'] / 2
                self.assertTrue(all(abs(v) + radius <= half_cell for v in pose['offset_mm']))
                self.assertTrue(0 <= pose['yaw_degrees'] < 360)
            for view, camera in entry['camera_setup'].items():
                uv, depth = project_frame(camera['framing_points_m'], camera)
                margin = camera['framing_margin_fraction']
                self.assertTrue(np.all(depth > 0) and np.all((uv >= margin) & (uv <= 1 - margin)))
                rotation = np.asarray(camera['matrix_world'])[:3, :3]
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                for key in ['azimuth_offset_degrees', 'elevation_offset_degrees', 'distance_scale',
                            'roll_degrees', 'lens_scale']:
                    low, high = self.settings['camera_randomization'][view][key]
                    self.assertTrue(low <= camera['sampling'][key] <= high)
        legacy = copy.deepcopy(self.source)
        legacy['positions'][0]['piece_transforms'] = {'a1': {'offset_mm': [999, 999]}}
        self.assertEqual(plan, prepare_dataset(legacy, self.settings, self.presets, self.base))

    def test_camera_rejection_disable_and_piece_limits(self):
        plan = prepare_dataset(self.source, self.settings, self.presets, self.base)
        frame = plan['positions'][0]['camera_setup']['fixed']['framing_points_m']
        reference = self.base['cameras']['fixed']
        disabled = sample_camera(reference, {'enabled': False}, frame, seed=1)
        self.assertEqual(disabled, reference)
        impossible = {**self.settings['camera_randomization']['fixed'],
                      'distance_scale': [.01, .01], 'max_attempts': 2}
        with self.assertRaises(ValueError):
            sample_camera(reference, impossible, frame, seed=1)
        for value in [[2, 1], [-1, 1], [1, float('nan')]]:
            bad = {**impossible, 'distance_scale': value}
            with self.assertRaises(ValueError):
                sample_camera(reference, bad, frame, seed=1)
        bad = copy.deepcopy(self.settings)
        bad['piece_randomization']['cell_margin_mm'] = 20
        with self.assertRaises(ValueError):
            prepare_dataset(self.source, bad, self.presets, self.base)

        class UpperBoundRandom:
            def uniform(self, low, high):
                return high

        diameters = {chess.piece_name(kind): 16 for kind in chess.PIECE_TYPES}
        poses = sample_piece_transforms(chess.Board(), UpperBoundRandom(),
                                       self.settings['piece_randomization'], 26, diameters)
        self.assertEqual(poses['a1']['offset_mm'], [5, 5])
        self.assertEqual(poses['a1']['yaw_degrees'], 270)

    def test_failed_qa_does_not_publish_training_entry_point(self):
        # Tiny fake render output isolates the runner's failure policy, not image quality.
        with tempfile.TemporaryDirectory(prefix='.dataset-test-', dir=ROOT) as temporary:
            parent = Path(temporary)
            source_path = parent / 'setups.json'
            source_path.write_text(json.dumps(self.source))
            output = parent / 'run'

            def fake_render(*args, **kwargs):
                data = output / 'data'
                data.mkdir()
                (data / 'dataset.yaml').write_text('names: []\n')

            with patch('create_dataset.subprocess.run', side_effect=fake_render), \
                 patch('check_dataset.check') as dataset_check, \
                 patch('check_lighting.check', side_effect=AssertionError('glare')):
                with self.assertRaises(RuntimeError):
                    run_dataset(source_path, PACKAGE / 'configs/dataset.json', output=str(output))
            dataset_check.assert_called_once()
            self.assertFalse((output / 'data/dataset.yaml').exists())
            self.assertTrue((output / 'data/dataset.unvalidated.yaml').exists())
            state = json.loads((output / 'run.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertIn('glare', state['check_errors']['lighting'])
            with self.assertRaises(ValueError):
                run_dataset(source_path, PACKAGE / 'configs/dataset.json', output=str(output))


if __name__ == '__main__':
    unittest.main()
