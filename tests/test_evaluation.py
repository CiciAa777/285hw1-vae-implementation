"""Mathematical/format checks without downloading or running Inception."""
import unittest

import numpy as np
from PIL import Image
import torch

from evaluate import compute_scores, inception_input


class EvaluationTest(unittest.TestCase):
    def test_preprocessing_matches_pil_reference(self):
        source = torch.linspace(0, 1, 64 * 64).reshape(1, 1, 64, 64)
        result = inception_input(source)
        self.assertEqual(result.shape, (1, 3, 299, 299))
        reference = Image.fromarray((source[0, 0].numpy() * 255).astype(np.uint8)).resize(
            (299, 299), Image.Resampling.BILINEAR)
        self.assertTrue(np.array_equal(result[0, 0].numpy(), np.array(reference, dtype=np.float32) / 255))
        self.assertTrue(torch.equal(result[:, 0], result[:, 1]))
        self.assertTrue(torch.equal(result[:, 1], result[:, 2]))

    def test_known_metrics(self):
        # Identical point-mass distributions have FID=0 and IS=1.
        probabilities = torch.full((4, 1000), 0.001)
        fid, score = compute_scores(probabilities, probabilities)
        self.assertAlmostEqual(fid, 0, places=6)
        self.assertAlmostEqual(score, 1, places=6)


if __name__ == '__main__':
    unittest.main()
