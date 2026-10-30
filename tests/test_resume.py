"""Verify resumed training matches an uninterrupted CPU run, plus legacy compatibility."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]


class ResumeTest(unittest.TestCase):
    def test_resume(self):
        (ROOT / 'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs') as directory:
            root = Path(directory)

            def run(*args, success=True):
                result = subprocess.run([sys.executable, 'train.py', *map(str, args)],
                                        cwd=ROOT, capture_output=True, text=True)
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result

            common = ['--limit', 16, '--batch-size', 8, '--threads', 2]
            run(*common, '--epochs', 2, '--out', root / 'full')
            run(*common, '--epochs', 1, '--out', root / 'split')
            result = run('--resume', root / 'split/last.pt', '--epochs', 2, '--threads', 2)
            self.assertIn('Resuming after epoch 1', result.stdout)
            full = torch.load(root / 'full/last.pt', weights_only=True)
            resumed = torch.load(root / 'split/last.pt', weights_only=True)
            self.assertEqual(resumed['epoch'], 2)
            self.assertEqual(resumed['history'], full['history'])
            self.assertEqual(resumed['best_loss'], full['best_loss'])
            for name, value in full['model'].items():
                self.assertTrue(torch.equal(value, resumed['model'][name]), name)
            self.assertTrue(torch.equal(full['fixed_z'], resumed['fixed_z']))
            self.assertTrue(torch.equal(full['rng_state'], resumed['rng_state']))
            self.assertTrue((root / 'split/loss_curves.png').is_file())
            self.assertTrue((root / 'split/summary.md').is_file())
            self.assertTrue((root / 'experiments.csv').is_file())
            notes = root / 'split/notes.md'
            notes.write_text('Keep my experiment observations.', encoding='utf-8')
            from experiments import refresh_catalog
            refresh_catalog(root)
            self.assertEqual(notes.read_text(encoding='utf-8'), 'Keep my experiment observations.')
            run('--resume', root / 'split/last.pt', '--epochs', 3,
                '--latent-dim', 16, success=False)
            run('--resume', root / 'split/last.pt', '--epochs', 2, success=False)
            run(*common, '--epochs', 1, '--out', root / 'split', success=False)
            legacy = {key: resumed[key] for key in ('model', 'config', 'epoch', 'val_loss', 'optimizer')}
            legacy['config'] = {key: value for key, value in legacy['config'].items()
                                if key not in ('depth', 'base_channels', 'parameter_count')}
            torch.save(legacy, root / 'split/legacy.pt')
            result = run('--resume', root / 'split/legacy.pt', '--epochs', 3,
                         '--out', root / 'legacy', '--threads', 2)
            self.assertIn('Legacy checkpoint', result.stdout)
            continued = torch.load(root / 'legacy/last.pt', weights_only=True)
            self.assertEqual([row['epoch'] for row in continued['history']], [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
