"""Rebuild report assets directly from saved histories and evaluated prior samples."""
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'legend.fontsize': 7})

# One training objective throughout this figure: 3-block, beta=0.5 model.
run = ROOT / 'runs/cnn128_depth_3'
with (run / 'history.csv').open() as stream:
    history = list(csv.DictReader(stream))
epochs = [int(row['epoch']) for row in history]
fig, axes = plt.subplots(1, 3, figsize=(6.0, 1.65), layout='constrained')
for ax, key, title in zip(axes, ('total', 'recon', 'kl'), ('Total objective', 'Reconstruction', 'KL divergence')):
    for split, color in [('train', '#2563a6'), ('val', '#bd5a20')]:
        ax.plot(epochs, [float(row[f'{split}_{key}']) for row in history],
                label=split, color=color, linewidth=1.2)
    ax.set(title=title, xlabel='Epoch')
    ax.grid(alpha=0.2)
axes[0].set_ylabel('Loss per image')
axes[-1].legend(frameon=False)
fig.savefig(OUT / 'training_curves.pdf', bbox_inches='tight')
fig.savefig(OUT / 'training_curves.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# Same first eight latent draws, evaluated best checkpoints, without selection.
cases = [('cnn128_depth_3', '3 blocks'), ('res128_beta_0.5', '4 blocks'), ('cnn128_depth_5', '5 blocks')]
canvas = Image.new('RGB', (1110, 3 * 132), 'white')
draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 22)
manifest = []
for row, (name, label) in enumerate(cases):
    run = ROOT / 'runs' / name
    metric = json.loads((run / 'metrics.json').read_text())
    assert metric['fid_protocol'] == 'ignite-probabilities-1000-pil299-fixedreal299-v2'
    checkpoint = run / metric['checkpoint_name']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == metric['checkpoint_sha256']
    samples = torch.load(run / metric['output_directory'] / 'generated.pt', weights_only=True)
    draw.text((3, row * 132 + 55), label, fill='black', font=font)
    for col, sample in enumerate(samples[:8]):
        pixels = (sample[0].clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        canvas.paste(Image.fromarray(pixels).convert('RGB'), (84 + col * 128, row * 132))
    manifest.append({'run': name, 'checkpoint_epoch': metric['checkpoint_epoch'],
                     'checkpoint_sha256': metric['checkpoint_sha256'],
                     'evaluation_seed': metric['seed'], 'sample_indices': list(range(8)),
                     'fid': metric['fid'], 'inception_score': metric['inception_score']})
canvas.save(OUT / 'depth_samples.png')
(OUT / 'sources.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('Prepared training curves and matched prior-sample grids.')
