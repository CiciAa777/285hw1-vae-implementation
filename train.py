"""Run --inspect first, then a small smoke run, then full training. See README.md."""

import argparse
import csv
import json
import os
import random
from pathlib import Path

# Keep Matplotlib's writable font cache inside this project.
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.cache' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from vae import VAE, vae_loss


class Xrays(Dataset):
    def __init__(self, root, size):
        self.paths = sorted(p for p in Path(root).glob('*/*')
                            if p.suffix.lower() in {'.jpeg', '.jpg', '.png'} and not p.name.startswith('._'))
        self.size = size
        if not self.paths:
            raise ValueError(f'No images in {root}; expected split/NORMAL and split/PNEUMONIA')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            # Explicit square resize is a simple baseline; it changes aspect ratios.
            image = image.convert('L').resize((self.size, self.size), Image.Resampling.BILINEAR)
            pixels = np.array(image, dtype=np.float32) / 255.0
        # [H,W] -> [1,H,W]. Labels are unused in this unconditional VAE.
        return torch.from_numpy(pixels).unsqueeze(0)


def save_grid(images, path, columns=8):
    images = images.detach().cpu().clamp(0, 1)
    n, _, h, w = images.shape
    canvas = Image.new('L', (columns * w, ((n + columns - 1) // columns) * h))
    for i, image in enumerate(images):
        pixels = (image[0].numpy() * 255).round().astype(np.uint8)
        canvas.paste(Image.fromarray(pixels), ((i % columns) * w, (i // columns) * h))
    canvas.save(path)


def run_epoch(model, loader, device, beta, optimizer=None):
    model.train(optimizer is not None)
    totals = np.zeros(3)
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x in loader:
            x = x.to(device)
            reconstruction, mu, logvar = model(x)
            losses = vae_loss(reconstruction, x, mu, logvar, beta)
            if not all(torch.isfinite(loss).item() for loss in losses):
                raise RuntimeError('Non-finite loss; inspect inputs and lower the learning rate')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses[0].backward()
                optimizer.step()
            totals += np.array([loss.item() for loss in losses]) * len(x)
            count += len(x)
    return (totals / count).tolist()


def plot_history(rows, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, name in zip(axes, ('total', 'recon', 'kl')):
        for split in ('train', 'val'):
            ax.plot([r['epoch'] for r in rows], [r[f'{split}_{name}'] for r in rows], label=split)
        ax.set(title=name, xlabel='Epoch', ylabel='Loss per image')
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('chest_xray'))
    parser.add_argument('--out', type=Path, default=Path('runs/baseline'))
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='Random training subset for debugging; 0 means all')
    parser.add_argument('--inspect', action='store_true', help='Save real images and verify forward/backward only')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.latent_dim, args.threads) <= 0 or args.limit < 0:
        parser.error('epochs, batch size, latent dimension, threads must be positive; limit must be nonnegative')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Output directory is not empty; choose a new --out to preserve previous results')
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = Xrays(args.data / 'train', args.image_size)
    val = Xrays(args.data / 'val', args.image_size)
    if args.limit:
        indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed))[:args.limit]
        train = Subset(train, indices.tolist())
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = VAE(args.latent_dim, args.image_size).to(device)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device=str(device), torch_version=str(torch.__version__), train_images=len(train), val_images=len(val))
    (args.out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Device: {device}; train: {len(train)}; val: {len(val)}', flush=True)
    x = next(iter(train_loader)).to(device)
    reconstruction, mu, logvar = model(x)
    loss, recon, kl = vae_loss(reconstruction, x, mu, logvar, args.beta)
    assert reconstruction.shape == x.shape
    assert mu.shape == logvar.shape == (len(x), args.latent_dim)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(f'x: {tuple(x.shape)}; mu/logvar: {tuple(mu.shape)}; reconstruction: {tuple(reconstruction.shape)}', flush=True)
    print(f'Initial losses: total={loss.item():.3f}, recon={recon.item():.3f}, KL={kl.item():.3f}; backward passed', flush=True)
    save_grid(x[:32], args.out / 'real.png')
    if args.inspect:
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    fixed_z = torch.randn(32, args.latent_dim, device=device)
    fixed_x = next(iter(val_loader))[:8].to(device)
    history = []
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_losses = run_epoch(model, train_loader, device, args.beta, optimizer)
        # Keep validation sampling noise fixed across epochs for easier comparison.
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
            torch.manual_seed(args.seed + 1)
            val_losses = run_epoch(model, val_loader, device, args.beta)
        row = {'epoch': epoch}
        for split, values in [('train', train_losses), ('val', val_losses)]:
            row.update({f'{split}_{name}': value for name, value in zip(('total', 'recon', 'kl'), values)})
        history.append(row)
        with (args.out / 'history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        checkpoint = {'model': model.state_dict(), 'config': config, 'epoch': epoch,
                      'val_loss': val_losses[0], 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, args.out / 'last.pt')
        if val_losses[0] < best:
            best = val_losses[0]
            torch.save(checkpoint, args.out / 'best.pt')
        model.eval()
        with torch.no_grad():
            # NEW samples use only prior z, whereas reconstructions use real images.
            save_grid(model.decode(fixed_z), args.out / f'samples_{epoch:03d}.png')
            mean, _ = model.encode(fixed_x)
            save_grid(torch.cat([fixed_x, model.decode(mean)]), args.out / 'reconstructions.png')
        print(f'Epoch {epoch}/{args.epochs}: train={train_losses[0]:.3f}, val={val_losses[0]:.3f}, '
              f'recon={train_losses[1]:.3f}, KL={train_losses[2]:.3f}', flush=True)
    plot_history(history, args.out / 'loss_curves.png')


if __name__ == '__main__':
    main()
