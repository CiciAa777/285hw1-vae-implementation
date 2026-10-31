# ECE 285 HW1: Chest X-ray Generation

## Purpose

Implement and train a convolutional VAE in PyTorch to generate chest X-rays.
Experiments compare resolution, latent dimension, KL weight (beta), and CNN depth.

## Structure

| File/folder | Purpose |
|---|---|
| `vae.py` | Encoder, reparameterization, decoder, and reconstruction + beta * KL loss |
| `train.py` | Training, checkpoint resuming, loss curves, and image grids |
| `evaluate.py` | Generate new samples and compute Ignite FID/IS |
| `sweep.py` | Run multiple configurations sequentially |
| `experiments.py` | Refresh summaries, comparison CSV, and figures |
| `runs/` | Per-run settings, checkpoints, losses, images, and metrics |
| `report/` | LaTeX report draft and figures |
| `tests/` | Architecture, resume, sweep, and evaluation checks |

Dataset: [Kaggle Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## Run
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-eval.txt
```

Train one configuration, then resume to 40 total epochs:

```
python train.py --epochs 30 --out runs/example
python train.py --resume runs/example/last.pt --epochs 40
```

Defaults: 64x64 grayscale, latent dimension 32, four blocks, base channels 16,
beta 1, Adam, learning rate 0.001, batch size 32, seed 42.
Use a new output folder for each fresh experiment. Resume restores the last
completed epoch and the original settings.

Evaluate a saved checkpoint without retraining:

```
python evaluate.py --checkpoint runs/example/best.pt --reference-size 299
```

Evaluation uses 624 new prior samples and 624 test images, generation seed 123.
All models share real originals resized directly to 299x299. The
[assigned Ignite approach](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/)
uses 1,000 Inception softmax probabilities; this FID is not conventional pool2048 FID.

Train and evaluate a depth sweep:

```
python sweep.py --parameter depth --values 3 5 --image-size 128 --latent-dim 32 --beta 0.5 --epochs 40 --prefix depth_demo --evaluate
```

Other sweep parameters: `image-size`, `latent-dim`, `beta`, `lr`, `batch-size`, `base-channels`.
Append `--dry-run` to preview commands. Omit `--evaluate` to train only.

Each run saves `config.json`, `history.csv`, `loss_curves.png`, `best.pt`, `last.pt`,
`samples_*.png`, `reconstructions.png`, `summary.md`, and editable `notes.md`.
Evaluation adds `metrics.json` and images under `evaluations/`.
The experiment table updates automatically; refresh comparison figures with:

```powershell
python experiments.py --runs runs
```

## Results

Completed main runs, using the same fixed-reference v2 protocol and each run's
lowest-validation-loss checkpoint. FID: lower is better; IS: higher conventionally.

| Image size | Blocks | Latent dim | Beta | Training seed | FID | IS |
|---|---:|---:|---:|---:|---:|---:|
| 64 | 4 | 32 | 1 | 42 | 0.2351 | 1.4971 |
| 128 | 4 | 32 | 1 | 42 | 0.2462 | 1.8396 |
| 128 | 4 | 64 | 1 | 42 | 0.2459 | 1.8723 |
| 128 | 4 | 32 | 0.5 | 42 | 0.2440 | 1.9334 |
| 128 | 4 | 32 | 2 | 42 | 0.2518 | 1.7648 |
| 128 | 4 | 32 | 0.5 | 43 | 0.2334 | 1.9352 |
| 128 | 4 | 32 | 1 | 43 | 0.2766 | 1.9183 |
| 128 | 3 | 32 | 0.5 | 42 | **0.2150** | 1.9546 |
| 128 | 5 | 32 | 0.5 | 42 | 0.2785 | **1.9675** |

Beta 0.5 outperformed beta 1 on reconstruction and FID/IS in both tested seeds.
Three blocks achieved the best observed FID, but depth also changes parameter
count and spatial compression. Samples remain blurry; ImageNet-based scores
do not establish medical realism. Total losses across beta/resolution settings
are not directly comparable.

See [full results](runs/experiments.csv), [sample comparisons](runs/comparison_samples.png),
and [loss comparisons](runs/comparison_losses.png). These are exploratory results
with a 16-image validation set and limited training-seed replication.

