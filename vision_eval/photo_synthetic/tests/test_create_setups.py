"""Run with .local-blender/venv/bin/python; also tests Stockfish when installed."""
import copy
from collections import Counter
import json
from pathlib import Path
import sys
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from create_setups import ROOT, chess, create_setups


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads((PACKAGE / 'configs/setups.json').read_text())
        self.settings.update(mode='random', games=8, count=18, max_plies=120)

    def check_replay(self, result):
        games = {g['id']: g for g in result['setup_generation']['games']}
        placements = set()
        for entry in result['positions']:
            game = games[entry['game_id']]
            self.assertEqual(entry['split'], game['split'])
            board = chess.Board()
            for uci in game['moves_uci'][:entry['ply']]:
                move = chess.Move.from_uci(uci)
                self.assertIn(move, board.legal_moves)
                board.push(move)
            self.assertEqual(board.fen(), entry['fen'])
            self.assertTrue(board.is_valid())
            self.assertNotIn(board.board_fen(), placements)
            placements.add(board.board_fen())
            self.assertNotIn('piece_transforms', entry)
        self.assertNotIn('assets', result)
        self.assertNotIn('cameras', result)

    def test_replay_and_reproducibility(self):
        before = copy.deepcopy(self.settings)
        first = create_setups(self.settings)
        self.assertEqual(first, create_setups(self.settings))
        self.assertEqual(self.settings, before)
        self.assertEqual(len(first['positions']), self.settings['count'])
        self.check_replay(first)
        games = [g for g in first['setup_generation']['games'] if g['split'] is not None]
        counts = Counter(g['split'] for g in games)
        weights = self.settings['split_weights']
        for split, weight in weights.items():
            quota = len(games) * weight / sum(weights.values())
            self.assertLess(abs(counts[split] - quota), 1)
        self.settings['seed'] += 1
        self.assertNotEqual(first['positions'], create_setups(self.settings)['positions'])

    def test_invalid_bounds_and_insufficient_positions(self):
        for key, value in [('max_loss_cp', float('nan')),
                           ('sample_every_plies', 0), ('min_ply', -1), ('count', 10000)]:
            with self.subTest(key=key, value=value):
                settings = {**self.settings, key: value}
                with self.assertRaises(ValueError):
                    create_setups(settings)

    def test_stockfish_replay(self):
        path = ROOT / self.settings['engine_path']
        if not path.exists():
            self.skipTest('Stockfish is not installed')
        self.settings.update(mode='engine', engine_nodes=1000, games=2,
                             max_plies=24, count=4)
        with chess.engine.SimpleEngine.popen_uci(str(path)) as engine:
            result = create_setups(self.settings, engine=engine)
        self.assertIn('Stockfish', result['setup_generation']['engine']['name'])
        self.check_replay(result)


if __name__ == '__main__':
    unittest.main()
