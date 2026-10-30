"""The three VAE operations: encode a distribution, sample, decode an image."""

import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim=32, image_size=64, depth=4, base_channels=16):
        super().__init__()
        if depth not in (2, 3, 4, 5) or base_channels < 1:
            raise ValueError('depth must be 2..5 and base_channels must be positive')
        if image_size < 32 or image_size % (2 ** depth) or image_size // (2 ** depth) < 2:
            raise ValueError('image_size must be >=32, divisible by 2**depth, with final spatial size >=2')
        self.latent_dim = latent_dim
        self.side = image_size // (2 ** depth)
        # Each stride-2 convolution halves height and width.
        channels = [1] + [base_channels * 2 ** i for i in range(depth)]
        self.feature_channels = channels[-1]
        layers = []
        for cin, cout in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(cin, cout, 4, 2, 1), nn.LeakyReLU(0.2)]
        self.encoder = nn.Sequential(*layers)
        features = self.feature_channels * self.side * self.side
        # These numbers PARAMETERIZE a diagonal Gaussian; they are not class scores.
        self.fc_mu = nn.Linear(features, latent_dim)
        self.fc_logvar = nn.Linear(features, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, features)
        layers = []
        for index, (cin, cout) in enumerate(zip(channels[:0:-1], channels[-2::-1])):
            layers += [nn.ConvTranspose2d(cin, cout, 4, 2, 1),
                       nn.Sigmoid() if index == depth - 1 else nn.ReLU()]
        self.decoder = nn.Sequential(*layers)

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        # logvar = log(sigma^2), so sigma = exp(logvar / 2).
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z):
        h = self.fc_decode(z).reshape(-1, self.feature_channels, self.side, self.side)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(reconstruction, x, mu, logvar, beta=1.0):
    # Sum over pixels/latent coordinates, then average over images, for BOTH terms.
    # Summed squared error corresponds to Gaussian NLL with fixed variance 1/2,
    # omitting an additive constant. beta=1 gives the standard negative ELBO.
    reconstruction_loss = F.mse_loss(reconstruction, x, reduction='none').flatten(1).sum(1).mean()
    kl_loss = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss
