"""Integration check using cached pretrained weights; never downloads in this test."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from vae import VAE

ROOT = Path(__file__).resolve().parents[1]
CACHED = ROOT / '.cache/torch/checkpoints/inception_v3_google-0cc3c7bd.pth'


class FixedReferenceTest(unittest.TestCase):
    @unittest.skipUnless(CACHED.is_file(), 'Requires locally cached Inception weights')
    def test_reference_features_do_not_depend_on_model_size(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as directory:
            root = Path(directory)
            metrics, features = [], []
            for size in (64, 128):
                run = root / str(size)
                run.mkdir()
                model = VAE(8, size, depth=3, base_channels=8)
                config = {'latent_dim': 8, 'image_size': size, 'depth': 3, 'base_channels': 8,
                          'data': str(ROOT / 'chest_xray')}
                torch.save({'model': model.state_dict(), 'config': config, 'epoch': 1}, run / 'best.pt')
                result = subprocess.run(
                    [sys.executable, 'evaluate.py', '--checkpoint', str(run / 'best.pt'),
                     '--num-samples', '4', '--batch-size', '4', '--threads', '2', '--reference-size', '299'],
                    cwd=ROOT, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                metric = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
                metrics.append(metric)
                feature = torch.load(run / metric['output_directory'] / 'inception_probabilities.pt',
                                     weights_only=True)['real']
                features.append(feature)
                self.assertIn('fixedreal299-v2', metric['fid_protocol'])
                self.assertTrue(metric['subset_evaluation'])
                self.assertTrue((run / metric['output_directory'] / 'samples.png').exists())
            self.assertEqual(metrics[0]['real_manifest_sha256'], metrics[1]['real_manifest_sha256'])
            self.assertTrue(torch.equal(features[0], features[1]))


if __name__ == '__main__':
    unittest.main()
