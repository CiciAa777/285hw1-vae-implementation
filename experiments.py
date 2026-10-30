"""Create per-run notes, a comparison table, and comparison figures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def read_history(run):
    with (Path(run) / 'history.csv').open(newline='', encoding='utf-8') as stream:
        return [{key: int(value) if key == 'epoch' else float(value)
                 for key, value in row.items()} for row in csv.DictReader(stream)]


def summarize_run(run):
    run = Path(run)
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    history = read_history(run)
    if not history:
        return None
    final = history[-1]
    best = min(history, key=lambda row: row['val_total'])
    keys = ('image_size', 'latent_dim', 'lr', 'batch_size', 'beta', 'seed', 'limit')
    depth, base = config.get('depth', 4), config.get('base_channels', 16)
    channels = [1] + [base * 2 ** i for i in range(depth)]
    row = {'run': run.name, 'path': str(run.resolve()), **{key: config[key] for key in keys},
           'depth': depth, 'base_channels': base, 'parameter_count': config.get('parameter_count', ''),
           'completed_epochs': final['epoch'], 'target_epochs': config['epochs'],
           'best_epoch': best['epoch'], 'best_val_total': best['val_total'],
           'best_val_recon': best['val_recon'], 'best_val_kl': best['val_kl'],
           **{f'final_{key}': value for key, value in final.items() if key != 'epoch'},
           'fid': '', 'inception_score': '', 'evaluation_epoch': '', 'evaluation_split': '',
           'real_count': '', 'generated_count': '', 'evaluation_seed': '', 'fid_protocol': ''}
    row['subset_evaluation'] = ''
    metrics_path = run / 'metrics.json'
    metric_note = 'Not evaluated yet.'
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        checkpoint_path = run / metrics['checkpoint_name']
        if checkpoint_path.is_file() and file_hash(checkpoint_path) == metrics['checkpoint_sha256']:
            row.update({key: metrics[key] for key in ('fid', 'inception_score', 'real_count',
                                                     'generated_count', 'fid_protocol')})
            row.update(evaluation_epoch=metrics['checkpoint_epoch'], evaluation_split=metrics['split'],
                       evaluation_seed=metrics['seed'], subset_evaluation=metrics['subset_evaluation'])
            metric_note = (f"FID: {metrics['fid']:.6f}; IS: {metrics['inception_score']:.6f}. "
                           f"Checkpoint: {metrics['checkpoint_name']}, epoch {metrics['checkpoint_epoch']}; "
                           f"split: {metrics['split']}; real/generated: {metrics['real_count']}/"
                           f"{metrics['generated_count']}; seed: {metrics['seed']}. "
                           f"Protocol: {metrics['fid_protocol']}. See metrics.json for full provenance.")
        else:
            metric_note = 'Existing metrics belong to an older/missing checkpoint; re-evaluate before comparing.'
    settings = '\n'.join(f'- {key}: {config[key]}' for key in keys)
    samples = run / f"samples_{final['epoch']:03d}.png"
    note = (f'# Experiment: {run.name}\n\n## Settings\n\n{settings}\n'
            f"- optimizer: Adam\n- device: {config.get('device', 'unknown')}\n"
            f"- PyTorch: {config.get('torch_version', 'unknown')}\n"
            f"- train/validation images: {config.get('train_images')}/{config.get('val_images')}\n"
            f"- completed/target epochs: {final['epoch']}/{config['epochs']}\n\n"
            f'Architecture: {depth} encoder blocks, channels {channels}; kernel 4, stride 2, padding 1; '
            'two Gaussian heads; mirrored transposed-convolution decoder with sigmoid. '
            f'Parameters: {config.get("parameter_count", "not recorded in legacy run")}. '
            'Grayscale, direct square resize, pixels [0,1].\n\n'
            '## Recorded results\n\n'
            f"Best validation total: {best['val_total']:.6f} at epoch {best['epoch']}; "
            f"reconstruction: {best['val_recon']:.6f}; KL: {best['val_kl']:.6f}.\n\n"
            f"Final training total: {final['train_total']:.6f}; final validation total: {final['val_total']:.6f}.\n\n"
            f"Final validation pixel MSE: {final['val_recon'] / config['image_size']**2:.8f}. "
            'Losses sum pixels/latent coordinates per image and average across the batch. '
            'Total losses with different beta values or image sizes are not directly comparable.\n\n'
            f'## Evaluation\n\n{metric_note}\n\n'
            f'## Figures\n\n[Loss curves](loss_curves.png) · [Final prior samples]({samples.name}) · '
            '[Final reconstructions](reconstructions.png)\n\n'
            'The per-epoch training sample grids use fixed prior draws. Evaluation samples come from '
            'the explicitly selected checkpoint and a separate seed; see evaluations/.\n\n'
            '## Interpretation limits\n\nValidation has only 16 images in the original split. '
            'FID/IS use ImageNet features, not medical diagnostic assessment. Small evaluation sets '
            'give uncertain scores. Select experiments/checkpoints using validation, not repeated test tuning.\n\n'
            'Add your interpretation to notes.md; this generated summary is refreshed automatically.\n')
    atomic_text(run / 'summary.md', note)
    notes = run / 'notes.md'
    if not notes.exists():
        notes.write_text('# Observations\n\n- Hypothesis:\n- Image quality and diversity:\n'
                         '- What changed compared with the baseline:\n- Limitations:\n', encoding='utf-8')
    return row


def refresh_catalog(root, plots=False):
    root = Path(root)
    rows, runs = [], []
    for run in sorted(root.iterdir()):
        if run.is_dir() and (run / 'config.json').is_file() and (run / 'history.csv').is_file():
            row = summarize_run(run)
            if row:
                rows.append(row)
                runs.append(run)
    if rows:
        path = root / 'experiments.csv'
        temporary = path.with_suffix('.csv.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    if plots and rows:
        # Import plotting only when requested, so training-time notes do not touch RNG state.
        from train import plt
        from PIL import Image, ImageDraw
        # Keep debugging subsets in the table, but avoid presenting them as full-run comparisons.
        plot_runs = [run for run, row in zip(runs, rows) if row['limit'] == 0] or runs
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        for run in plot_runs:
            history = read_history(run)
            for ax, name in zip(axes, ('total', 'recon', 'kl')):
                ax.plot([r['epoch'] for r in history], [r[f'val_{name}'] for r in history], label=run.name)
                ax.set(title=f'Validation {name}', xlabel='Epoch', ylabel='Loss per image')
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root / 'comparison_losses.png', dpi=160)
        plt.close(fig)
        tiles = []
        for run in plot_runs:
            candidates = sorted((run / 'evaluations').glob('*/samples.png')) if (run / 'evaluations').exists() else []
            metric_file = run / 'metrics.json'
            selected_epoch = None
            if metric_file.exists():
                metric = json.loads(metric_file.read_text(encoding='utf-8'))
                selected = run / metric['output_directory'] / 'samples.png'
                selected_checkpoint = run / metric['checkpoint_name']
                valid = (selected_checkpoint.exists() and
                         file_hash(selected_checkpoint) == metric['checkpoint_sha256'])
                candidates = [selected] if selected.exists() and valid else []
                selected_epoch = metric['checkpoint_epoch']
            if candidates:
                path, label = candidates[-1], f'{run.name}: evaluated epoch {selected_epoch}'
            else:
                candidates = sorted(run.glob('samples_*.png'))
                if not candidates:
                    continue
                path, label = candidates[-1], f'{run.name}: {candidates[-1].stem} (final)'
            with Image.open(path) as source:
                preview = source.convert('RGB')
                preview.thumbnail((512, 256))
            tile = Image.new('RGB', (540, 290), 'white')
            ImageDraw.Draw(tile).text((8, 8), label, fill='black')
            tile.paste(preview, (8, 30))
            tiles.append(tile)
        if tiles:
            canvas = Image.new('RGB', (540, 290 * len(tiles)), 'white')
            for i, tile in enumerate(tiles):
                canvas.paste(tile, (0, i * 290))
            canvas.save(root / 'comparison_samples.png')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    args = parser.parse_args()
    rows = refresh_catalog(args.runs, plots=True)
    print(f'Updated {len(rows)} experiment summaries and {args.runs / "experiments.csv"}')
    if len({row['fid_protocol'] for row in rows if row['fid_protocol']}) > 1:
        print('Multiple FID protocols are present. Compare only rows using the same protocol and reference dataset.')
