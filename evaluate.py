"""Evaluate a VAE using the assigned Ignite tutorial's default FID and IS protocol.

Reference: https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
This is explicitly the 1000-dimensional probability-output FID used by Ignite's
default extractor, NOT pytorch-fid's 2048-dimensional pool-feature protocol.
"""

import argparse
import hashlib
import importlib.metadata
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from experiments import atomic_text, file_hash, refresh_catalog
from train import Xrays, save_grid
from vae import VAE

PROTOCOL = 'ignite-default-probabilities-1000-pil299-v1'


def inception_input(images):
    """Match the tutorial's PIL uint8/bilinear interpolation, consistently for both sets."""
    resized = []
    for image in images.detach().cpu().clamp(0, 1):
        # ToPILImage's float-to-uint8 conversion truncates, rather than rounding.
        pixels = (image[0].numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(pixels).convert('RGB').resize((299, 299), Image.Resampling.BILINEAR)
        resized.append(torch.from_numpy(np.array(pil, dtype=np.float32) / 255).permute(2, 0, 1))
    return torch.stack(resized)


def compute_scores(fake_probabilities, real_probabilities):
    from ignite.metrics import FID, InceptionScore
    # Supplying num_features with no extractor uses Identity: Inception ran once upstream.
    fid = FID(num_features=1000)
    inception_score = InceptionScore(num_features=1000)
    fid.update((fake_probabilities, real_probabilities))
    inception_score.update(fake_probabilities)
    return float(fid.compute()), float(inception_score.compute())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/baseline/best.pt'))
    parser.add_argument('--data', type=Path, help='Defaults to checkpoint data path')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--num-samples', type=int, help='Equal real/generated count; default all split images')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--reference-size', type=int, default=299,
                        help='Fixed real-image resize, independent of model resolution (default 299)')
    parser.add_argument('--legacy-reference', action='store_true',
                        help='Reproduce the old model-resolution reference; not a cross-resolution comparison')
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}')
    if min(args.batch_size, args.threads, args.reference_size) <= 0:
        parser.error('batch-size, threads, and reference-size must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = VAE(config['latent_dim'], config['image_size'], config.get('depth', 4),
                config.get('base_channels', 16)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    data = args.data or Path(config['data'])
    reference_size = config['image_size'] if args.legacy_reference else args.reference_size
    protocol = PROTOCOL if args.legacy_reference else f'ignite-probabilities-1000-pil299-fixedreal{reference_size}-v2'
    real = Xrays(data / args.split, reference_size)
    count = args.num_samples if args.num_samples is not None else len(real)
    if not 2 <= count <= len(real):
        parser.error(f'num-samples must be between 2 and {len(real)}')
    indices = list(range(len(real))) if count == len(real) else torch.randperm(
        len(real), generator=torch.Generator().manual_seed(args.seed)).tolist()[:count]
    # Record actual image bytes, not just filenames, to identify the reference dataset.
    manifest = [{'path': str(real.paths[i].relative_to(data)), 'sha256': file_hash(real.paths[i])}
                for i in indices]
    checkpoint_hash = file_hash(args.checkpoint)
    run = args.checkpoint.parent
    tag = f'{args.checkpoint.stem}_{checkpoint_hash[:12]}_{args.split}_n{count}_seed{args.seed}_b{args.batch_size}'
    if not args.legacy_reference:
        tag += f'_fixedreal{reference_size}_v2'
    output = run / 'evaluations' / tag
    output.mkdir(parents=True, exist_ok=True)
    print(f'Evaluating {args.checkpoint}, epoch {checkpoint["epoch"]}: '
          f'{count} real + {count} generated images on {device}', flush=True)
    print(f'Protocol: {protocol}; first use downloads pretrained Inception weights (~104 MB).', flush=True)
    # Keep pretrained weights inside the project rather than the user's global cache.
    torch.hub.set_dir(str(Path(__file__).resolve().parent / '.cache' / 'torch'))
    from torchvision.models import inception_v3, Inception_V3_Weights
    weights = Inception_V3_Weights.IMAGENET1K_V1
    # Match Ignite's default extractor: pretrained Inception with its default
    # transform_input=True, followed by a 1000-class softmax. No extra normalization.
    extractor = nn.Sequential(inception_v3(weights=weights), nn.Softmax(dim=1)).to(device).eval()
    # Dedicated CPU generator keeps prior draws independent of model initialization.
    generator = torch.Generator().manual_seed(args.seed)
    latent = torch.randn(count, config['latent_dim'], generator=generator)
    loader = DataLoader(Subset(real, indices), batch_size=args.batch_size, shuffle=False)
    fake_features, real_features, generated = [], [], []
    start = time.perf_counter()
    offset = 0
    with torch.inference_mode():
        for real_batch in loader:
            size = len(real_batch)
            fake_batch = model.decode(latent[offset:offset + size].to(device)).cpu()
            generated.append(fake_batch)
            real_features.append(extractor(inception_input(real_batch).to(device)).cpu())
            fake_features.append(extractor(inception_input(fake_batch).to(device)).cpu())
            offset += size
            print(f'Extracted features: {offset}/{count} pairs', flush=True)
    fake_features, real_features = torch.cat(fake_features), torch.cat(real_features)
    print('Computing FID covariance matrix square root...', flush=True)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        fid, inception_score = compute_scores(fake_features, real_features)
    if not np.isfinite(fid) or not np.isfinite(inception_score):
        raise RuntimeError('Non-finite evaluation score; inspect feature extraction and numerical warnings')
    generated = torch.cat(generated)
    save_grid(generated[:32], output / 'samples.png')
    save_grid(torch.stack([real[i] for i in indices[:32]]), output / 'real.png')
    torch.save(generated, output / 'generated.pt')
    torch.save({'real': real_features, 'generated': fake_features}, output / 'inception_probabilities.pt')
    atomic_text(output / 'real_manifest.json', json.dumps(manifest, indent=2))
    messages = [str(w.message) for w in captured]
    messages += ['Small real reference set: sample-sensitive FID; ImageNet classes/features have limited medical meaning.',
                 'This FID uses 1000 softmax probabilities; do not compare with conventional pool2048 FID.',
                 'IS is one full-set estimate, not the mean of split estimates; no split standard deviation reported.']
    metrics = {
        'fid': fid, 'inception_score': inception_score, 'fid_protocol': protocol,
        'reference_size': reference_size, 'legacy_reference': args.legacy_reference,
        'checkpoint_name': args.checkpoint.name, 'checkpoint_sha256': checkpoint_hash,
        'checkpoint_epoch': checkpoint['epoch'], 'split': args.split, 'seed': args.seed,
        'real_count': count, 'generated_count': count, 'available_real_count': len(real),
        'subset_evaluation': count != len(real), 'batch_size': args.batch_size,
        'device': str(device), 'threads': args.threads,
        'output_directory': str(output.relative_to(run)),
        'feature_extractor': 'torchvision Inception v3; 1000 softmax probabilities; transform_input=True',
        'weights': 'Inception_V3_Weights.IMAGENET1K_V1', 'weights_url': weights.url,
        'preprocessing': f'Real: grayscale originals resized by PIL bilinear to {reference_size}; '
                         f'generated: decoder output at {config["image_size"]}; both [0,1]; '
                         'uint8 truncation; repeat grayscale as RGB; PIL bilinear resize to 299x299; '
                         '[0,1]; no external normalization; same pipeline for real/generated.',
        'real_manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        'packages': {name: importlib.metadata.version(name)
                     for name in ('torch', 'torchvision', 'pytorch-ignite', 'scipy', 'Pillow', 'numpy')},
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'evaluation_seconds': time.perf_counter() - start,
        'warnings': messages,
        'reference': 'https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/',
    }
    atomic_text(output / 'metrics.json', json.dumps(metrics, indent=2, allow_nan=False))
    # This is a pointer-like latest result; all evaluations remain in their own directory.
    atomic_text(run / 'metrics.json', json.dumps(metrics, indent=2, allow_nan=False))
    refresh_catalog(run.parent, plots=True)
    print(f'FID = {fid:.6f}; Inception Score = {inception_score:.6f}', flush=True)
    print(f'Saved evaluation and samples to {output}', flush=True)


if __name__ == '__main__':
    main()
