"""Check that a terminal sweep forwards settings and produces experiment records."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SweepTest(unittest.TestCase):
    def test_small_sweep(self):
        (ROOT / 'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs') as directory:
            result = subprocess.run(
                [sys.executable, 'sweep.py', '--parameter', 'latent-dim', '--values', '8',
                 '--runs', directory, '--epochs', '1', '--limit', '16', '--batch-size', '8', '--threads', '2'],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            root = Path(directory)
            config = json.loads((root / 'sweep_latent-dim_8/config.json').read_text(encoding='utf-8'))
            self.assertEqual(config['latent_dim'], 8)
            self.assertEqual(config['limit'], 16)
            self.assertTrue((root / 'sweep_latent-dim_8/summary.md').is_file())
            self.assertTrue((root / 'experiments.csv').is_file())
            self.assertTrue((root / 'comparison_losses.png').is_file())
            self.assertTrue((root / 'comparison_samples.png').is_file())


if __name__ == '__main__':
    unittest.main()
