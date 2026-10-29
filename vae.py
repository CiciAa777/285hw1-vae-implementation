"""The three VAE operations: encode a distribution, sample, decode an image."""

import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim=32, image_size=64):
        super().__init__()
        if image_size < 32 or image_size % 16:
            raise ValueError('image_size must be at least 32 and divisible by 16')
        self.latent_dim = latent_dim
        self.side = image_size // 16
        # Each stride-2 convolution halves height and width.
        channels = [1, 16, 32, 64, 128]
        layers = []
        for cin, cout in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(cin, cout, 4, 2, 1), nn.LeakyReLU(0.2)]
        self.encoder = nn.Sequential(*layers)
        features = 128 * self.side * self.side
        # These numbers PARAMETERIZE a diagonal Gaussian; they are not class scores.
        self.fc_mu = nn.Linear(features, latent_dim)
        self.fc_logvar = nn.Linear(features, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, features)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid(),
        )

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
        h = self.fc_decode(z).reshape(-1, 128, self.side, self.side)
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
