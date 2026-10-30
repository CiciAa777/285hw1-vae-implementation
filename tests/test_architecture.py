"""Architecture shapes, gradients, and compatibility with the original checkpoint layout."""
import unittest
import torch
from torch import nn
from vae import VAE, vae_loss


class ArchitectureTest(unittest.TestCase):
    def test_depths_and_widths(self):
        torch.set_num_threads(2)
        for depth in (3, 4, 5):
            for width in (1, 8):
                with self.subTest(depth=depth, width=width):
                    model = VAE(8, 64, depth, width)
                    x = torch.rand(2, 1, 64, 64)
                    y, mu, logvar = model(x)
                    self.assertEqual(y.shape, x.shape)
                    loss, _, _ = vae_loss(y, x, mu, logvar)
                    loss.backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in model.parameters()))
                    self.assertEqual(sum(isinstance(layer, nn.Conv2d) for layer in model.encoder), depth)
                    self.assertEqual(sum(isinstance(layer, nn.ConvTranspose2d) for layer in model.decoder), depth)
                    self.assertIsInstance(model.decoder[-1], nn.Sigmoid)
                    self.assertEqual(sum(isinstance(layer, nn.Sigmoid) for layer in model.decoder), 1)

    def test_legacy_layout(self):
        # Exact layer names and shapes needed by existing 4-block checkpoints.
        model = VAE()
        self.assertEqual(model.state_dict()['encoder.6.weight'].shape, (128, 64, 4, 4))
        self.assertEqual(model.state_dict()['decoder.0.weight'].shape, (128, 64, 4, 4))
        self.assertEqual(model.state_dict()['decoder.6.weight'].shape, (16, 1, 4, 4))
        self.assertEqual(model.state_dict()['fc_mu.weight'].shape, (32, 2048))


if __name__ == '__main__':
    unittest.main()
