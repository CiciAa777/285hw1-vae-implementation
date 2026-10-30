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
from experiments import refresh_catalog


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


def save_checkpoint(checkpoint, path):
    # An interruption during serialization must not destroy the previous checkpoint.
    temporary = path.with_suffix('.pt.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('chest_xray'))
    parser.add_argument('--out', type=Path, help='Fresh run directory; on resume defaults to checkpoint directory')
    parser.add_argument('--resume', type=Path, help='Continue from a checkpoint; --epochs is the total target')
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--depth', type=int, choices=[2, 3, 4, 5], default=4,
                        help='Number of stride-2 encoder blocks; decoder mirrors them')
    parser.add_argument('--base-channels', type=int, default=16,
                        help='First hidden channel count; doubles at each encoder block')
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
    checkpoint = None
    history = []
    if args.resume:
        if args.inspect:
            parser.error('--resume cannot be combined with --inspect')
        if not args.resume.is_file():
            parser.error(f'Checkpoint not found: {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        # Inherit the original experiment settings, rather than silently using defaults.
        inherited = ('data', 'image_size', 'latent_dim', 'batch_size', 'lr', 'beta', 'seed', 'limit',
                     'depth', 'base_channels')
        parser.set_defaults(**{key: checkpoint['config'].get(key, parser.get_default(key))
                               for key in (*inherited, 'workers', 'threads')})
        args = parser.parse_args()
        for key in inherited[1:]:
            if getattr(args, key) != checkpoint['config'].get(key, {'depth': 4, 'base_channels': 16}.get(key)):
                parser.error(f'Cannot change --{key.replace("_", "-")} when resuming this experiment')
        if args.epochs <= checkpoint['epoch']:
            parser.error(f'Checkpoint completed epoch {checkpoint["epoch"]}; --epochs must be larger')
        history = checkpoint.get('history', [])
        if not history:
            # Compatibility with checkpoints created before resume support.
            history_path = args.resume.parent / 'history.csv'
            if not history_path.is_file():
                parser.error('Legacy checkpoint requires its history.csv alongside it')
            with history_path.open(newline='', encoding='utf-8') as f:
                history = [{k: int(v) if k == 'epoch' else float(v) for k, v in row.items()}
                           for row in csv.DictReader(f)]
        history = [row for row in history if row['epoch'] <= checkpoint['epoch']]
        if not history or history[-1]['epoch'] != checkpoint['epoch']:
            parser.error('Loss history does not cover the checkpoint epoch')
    args.data = Path(args.data)
    args.out = args.out or (args.resume.parent if args.resume else Path('runs/baseline'))
    if min(args.epochs, args.batch_size, args.latent_dim, args.threads) <= 0 or args.limit < 0:
        parser.error('epochs, batch size, latent dimension, threads must be positive; limit must be nonnegative')
    if args.base_channels < 1 or args.image_size < 32 or args.image_size % (2 ** args.depth) or args.image_size // (2 ** args.depth) < 2:
        parser.error('Invalid architecture: positive channels, image size >=32 divisible by 2**depth, final spatial size >=2 required')
    if args.out.exists() and any(args.out.iterdir()):
        if not args.resume or args.out.resolve() != args.resume.parent.resolve():
            parser.error('Output directory is not empty; choose a new --out to preserve previous results')
        # Reject rollback into an existing run, which could mix results from different epochs.
        latest_path = args.out / 'last.pt'
        if latest_path.is_file():
            latest = torch.load(latest_path, map_location='cpu', weights_only=True)
            if latest['epoch'] > checkpoint['epoch']:
                parser.error('A newer last.pt exists; resume it or choose a new --out directory')
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
    model = VAE(args.latent_dim, args.image_size, args.depth, args.base_channels).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device=str(device), torch_version=str(torch.__version__), train_images=len(train), val_images=len(val),
                  parameter_count=sum(p.numel() for p in model.parameters()))
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
    best = float('inf')
    start_epoch = 1
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best = checkpoint.get('best_loss', min(row['val_total'] for row in history))
        if 'fixed_z' in checkpoint:
            fixed_z = checkpoint['fixed_z'].to(device)
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'])
            if device.type == 'cuda' and checkpoint.get('cuda_rng_state') is not None:
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        else:
            print('Legacy checkpoint: weights/optimizer/history restored; original RNG state unavailable.', flush=True)
        # A fork into a new directory retains the prior best model as well.
        previous_best = args.resume.parent / 'best.pt'
        if args.out.resolve() != args.resume.parent.resolve() and previous_best.is_file():
            old_best = torch.load(previous_best, map_location='cpu', weights_only=True)
            if old_best['epoch'] <= checkpoint['epoch']:
                save_checkpoint(old_best, args.out / 'best.pt')
        print(f'Resuming after epoch {checkpoint["epoch"]}; training through epoch {args.epochs}.', flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
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
        improved = val_losses[0] < best
        best = min(best, val_losses[0])
        model.eval()
        with torch.no_grad():
            # NEW samples use only prior z, whereas reconstructions use real images.
            save_grid(model.decode(fixed_z), args.out / f'samples_{epoch:03d}.png')
            mean, _ = model.encode(fixed_x)
            save_grid(torch.cat([fixed_x, model.decode(mean)]), args.out / 'reconstructions.png')
        checkpoint = {'model': model.state_dict(), 'config': config, 'epoch': epoch,
                      'val_loss': val_losses[0], 'optimizer': optimizer.state_dict(),
                      'best_loss': best, 'history': history, 'fixed_z': fixed_z.detach().cpu(),
                      'rng_state': torch.get_rng_state(),
                      'cuda_rng_state': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
        if improved:
            save_checkpoint(checkpoint, args.out / 'best.pt')
        save_checkpoint(checkpoint, args.out / 'last.pt')
        plot_history(history, args.out / 'loss_curves.png')
        refresh_catalog(args.out.parent)
        print(f'Epoch {epoch}/{args.epochs}: train={train_losses[0]:.3f}, val={val_losses[0]:.3f}, '
              f'recon={train_losses[1]:.3f}, KL={train_losses[2]:.3f}', flush=True)


if __name__ == '__main__':
    main()
