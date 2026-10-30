# ECE 285 HW1: learn a convolutional VAE

Start with a working 64 x 64 grayscale baseline. Read `vae.py` before changing the architecture. The implementation uses both NORMAL and PNEUMONIA images without labels: it learns the training mixture, not a classifier or a conditionally controlled generator.

What is our baseline?
Setting	Baseline
Input	Grayscale, 64×64, pixels [0,1]
Encoder	Four stride-2 convolution blocks
Encoder channels	1 → 16 → 32 → 64 → 128
Decoder	Four transposed-convolution layers
Latent dimension	32
Optimizer	Adam
Learning rate	0.001
Batch size	32
Beta	1
Training budget	30 epochs
Training seed	42
Checkpoint selection	Lowest validation total loss


The baseline completed 30 epochs; its selected checkpoint is epoch 23.


## 1. Environment and data

From this directory in PowerShell:

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The local environment reuses your installed PyTorch. Currently that installation is CPU-only. If moving to a GPU environment, install the appropriate PyTorch build there and copy this project and the dataset.

Expected data layout (use only the OUTER copy of this dataset):

```text
chest_xray/
  train/{NORMAL,PNEUMONIA}/   # 5,216 images
  val/{NORMAL,PNEUMONIA}/     # 16 images
  test/{NORMAL,PNEUMONIA}/    # 624 images, reserved for final evaluation
```

Preprocessing converts to grayscale, resizes directly to a square, and scales pixels to [0,1]. Direct resizing distorts aspect ratios; document this baseline choice. There is no augmentation. The original validation set is very small, so validation estimates and checkpoint selection can be noisy. Do not tune on the test set. Before larger experiments, consider a fixed patient-grouped validation split drawn from training images, recording its membership; an image-level random split can leak patient information.

## 2. Understand and check the model

```powershell
.\.venv\Scripts\python.exe train.py --inspect --out runs/inspect
```

Open `runs/inspect/real.png`. Can you see the lungs? Check the printed shapes:

```text
x                    [B, 1, 64, 64]
encoder features     [B, 128, 4, 4]
mu, logvar           [B, 32], [B, 32]
sampled z            [B, 32]
reconstruction       [B, 1, 64, 64]
```

Read `VAE.encode`, `VAE.reparameterize`, `VAE.decode`, then `vae_loss`.

- The encoder predicts parameters of q(z|x) = Normal(mu, diag(exp(logvar))). We choose the Gaussian family; the network learns its parameters.
- `std = exp(0.5 * logvar)` converts log variance into standard deviation.
- `z = mu + std * epsilon` keeps the random epsilon independent of the weights, allowing gradients through mu and std.
- The decoder predicts a pixel mean. Sigmoid matches the [0,1] input range.
- Reconstruction loss rewards matching the input. KL = 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar) regularizes the encoder toward the standard normal prior.

Both terms are summed per image, then averaged over the batch. Summed squared error is Gaussian negative log likelihood with fixed variance 1/2, up to an additive constant. With beta=1 this is the negative ELBO up to that constant; another beta changes the objective. Reconstruction values depend strongly on image resolution, so report the loss reduction and do not directly compare raw losses across resolutions.

Learning checks: Why do we output log variance rather than variance? Why multiply it by 0.5? What is KL when mu=0 and logvar=0? Which loss terms update the encoder, and which update the decoder?

## 3. Short debugging run

```powershell
.\.venv\Scripts\python.exe train.py --epochs 2 --limit 128 --out runs/smoke
```

This uses a seeded random subset of 128 training images, across both classes. It checks training, validation, checkpoints, plotting, and generation. Its images and metrics are debugging artifacts, not final homework results. Use a new output directory for every experiment; existing results are protected from overwriting.

## 4. Train the baseline

```powershell
.\.venv\Scripts\python.exe train.py --epochs 30 --out runs/baseline
```

Starting hyperparameters: Adam, learning rate 0.001, batch size 32, latent dimension 32, beta 1, seed 42, 64 x 64 images. Thirty epochs is a starting experiment, not a guarantee of convergence. Inspect results before choosing longer runs. CPU training may be slow.

Outputs:

- `config.json`: hyperparameters, device, sample counts, PyTorch version.
- `history.csv` and `loss_curves.png`: training/validation total, reconstruction, and KL losses per image. Validation uses a fixed posterior-noise seed for comparison.
- `samples_XXX.png`: new images decoded from the same 32 fixed prior draws across epochs.
- `reconstructions.png`: first row = 8 validation inputs; second row = decoder outputs using posterior means. This deterministic comparison differs from the stochastic reconstruction used in training.
- `best.pt`: checkpoint with lowest validation total loss; `last.pt`: final checkpoint. The displayed grids correspond to their respective epoch (reconstruction grid = final epoch), not necessarily the best checkpoint.

If reconstruction improves but KL stays approximately zero and generated samples all look similar, investigate posterior collapse. Compare images along with loss curves; low loss alone does not establish good generation. Change one hyperparameter at a time. A later 128 x 128 experiment is supported with `--image-size 128`, at higher memory and runtime cost.

### Interrupting and resuming

You can stop training with Ctrl+C. Resume from the last fully completed epoch:

```powershell
.\.venv\Scripts\python.exe train.py --resume runs/baseline/last.pt --epochs 30
```

`--epochs 30` means 30 total epochs, not 30 additional epochs. An interruption
during an epoch discards that epoch's partial progress. If no epoch finished,
there is no checkpoint yet: choose a new output directory or clear the incomplete
run directory before starting fresh.

Resume restores the model, Adam optimizer, original training settings, loss
history, and best validation loss. New checkpoints also preserve PyTorch random
states and the fixed latent vectors used for sample grids. With the same device,
environment, and execution settings this supports reproducible continuation;
cross-device and nondeterministic GPU operations may differ. Legacy checkpoints
still work using their neighboring `history.csv`, but lack the original random
state. Keep the dataset unchanged when resuming. `--data` may relocate that same
dataset; architecture and training hyperparameters cannot change during resume.

By default, resume writes into the checkpoint's directory. Add `--out` with a
new empty directory to continue into a separate folder. Prefer `last.pt` for
continuation; an older checkpoint cannot overwrite a run with a newer `last.pt`.
Checkpoint writes use a temporary file and replacement to protect the previous
checkpoint during serialization. Loss plots are updated after each epoch.

## 5. Generate from a selected checkpoint

```python
import torch
from vae import VAE
from train import save_grid

checkpoint = torch.load('runs/baseline/best.pt', map_location='cpu', weights_only=True)
cfg = checkpoint['config']
model = VAE(cfg['latent_dim'], cfg['image_size'], cfg.get('depth', 4), cfg.get('base_channels', 16))
model.load_state_dict(checkpoint['model'])
model.eval()
torch.manual_seed(123)
with torch.no_grad():
    z = torch.randn(32, model.latent_dim)
    save_grid(model.decode(z), 'runs/baseline/best_samples.png')
```

These are decoder means from new prior samples; no source image or encoder is involved. We display the decoder mean rather than adding independent Gaussian pixel noise.

## 6. Evaluation and report: after training

FID/IS evaluation is implemented in `evaluate.py`, adapted from the assignment's [Ignite FID and IS tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/). Its GAN training section is not needed for this VAE.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-eval.txt
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/baseline/best.pt
```

The first evaluation downloads pretrained Inception weights (~104 MB) into
`.cache/torch`. By default it uses all 624 test images and 624 new prior draws,
seed 123, and the checkpoint selected by validation loss. There is no retraining.
The final smaller batch is included. For a deliberate smaller debugging run use
`--num-samples 32`; this samples a seeded real subset and labels the subset in the
metrics. For another generation draw use `--seed 456`; all evaluation directories
are retained. `--split val` evaluates on validation instead, but its 16 images
make FID/IS especially unreliable. Do not repeatedly tune against test scores.

Installing Torchvision may resolve a newer compatible PyTorch inside `.venv`.
The baseline was trained with PyTorch 2.13.0; the evaluation environment installed
here uses PyTorch 2.15.0, Torchvision 0.29.2, and Ignite 0.5.5. Versions are recorded
in each run/evaluation. Keep the training environment fixed across controlled
experiments; rerun a baseline under that environment if strict comparability is needed.

**Protocol matters:** this implementation uses the actual Ignite default
extractor's **1000 softmax probabilities**, with `FID` and `InceptionScore` from
Ignite. The pretrained torchvision Inception v3 runs once per real/generated
batch; its outputs feed both metrics through identity extractors. This reproduces
the tutorial's default feature choice rather than conventional `pytorch-fid`
2048-dimensional pool features. Do not compare these FID values to pool2048
benchmarks. The original protocol was named `ignite-default-probabilities-1000-pil299-v1`.
New evaluations default to `ignite-probabilities-1000-pil299-fixedreal299-v2`:
the same real images are resized directly to a fixed 299 x 299 reference for
every model, independent of its training resolution. Generated outputs are
resized from their native resolution to 299 x 299. Use `--reference-size 299`
explicitly for cross-resolution comparisons. Given the same data and sample
selection, real reference tensors are identical across model resolutions.
This fixes evaluation preprocessing, not differences in training budgets,
model size, or the reconstruction/KL balance.

Real images use grayscale/fixed-reference square preprocessing. Both real and fake
images are converted from [0,1] to uint8 by truncation, repeated as RGB, resized to
299 x 299 with PIL bilinear interpolation, and converted back to [0,1]. There is
no external normalization; the model retains Ignite's default `transform_input=True`.
This is a documented tutorial-compatible protocol, not a claim that ImageNet
preprocessing is optimal for medical images. IS is computed once over the entire
generated set, without split averaging or a fabricated standard deviation.

Each evaluation saves `metrics.json`, `samples.png` (first 32, not cherry-picked),
`real.png`, all generated float images in `generated.pt` ([N,1,H,W]), extracted
Inception probabilities, and a real-image manifest with file hashes under
`runs/<case>/evaluations/<checkpoint-hash>_<split>_n<count>_seed<seed>_b<batch>/`.
Metadata records checkpoint hash/epoch, package versions, counts, seeds, weights,
preprocessing, numerical warnings, and protocol. `runs/<case>/metrics.json` holds
the latest evaluation; previous evaluations remain available in their directories.
If the checkpoint changes, its old metrics are excluded from the comparison table
until evaluation is rerun.

Fixed-reference results use a new `_fixedreal299_v2` evaluation directory suffix.
Old evaluation files are preserved. The run-level `metrics.json` and CSV show the
latest evaluation, so reevaluate all compared cases with the same protocol.
`--legacy-reference` reproduces the original model-resolution reference and its
old directory naming; it is not appropriate for cross-resolution comparisons.

```powershell
python evaluate.py --checkpoint runs/baseline/best.pt --reference-size 299
python evaluate.py --checkpoint runs/sweep_image-size_128/best.pt --reference-size 299
```

Use held-out real test images for FID, and NEW prior samples for generated images. Repeat grayscale into 3 channels and apply identical spatial/range preprocessing to real and generated images according to the selected feature extractor's contract. Record feature extractor, weights, normalization, resizing, sample counts, and seeds. Use all 624 real test images and state this limited reference size; FID covariance estimates are noisy with so few real images. IS uses generated images only. Standard Inception features/classes are trained on natural images, so discuss their limitations for X-rays. Do not compare scores from incompatible preprocessing or sample counts.

For the two-page NeurIPS-format report, reserve space for: method and objective; architecture and hyperparameters; loss curves; measured FID/IS and evaluation protocol; a prior-sample grid; and a short discussion of blur, diversity, and limitations. Write conclusions only after seeing actual results. Keep the code and report separate for submission; do not submit dataset or virtual environment files.

## 7. Experiments without editing Python

Change terminal options and give every fresh experiment a different folder:

```powershell
python train.py --latent-dim 16 --epochs 30 --out runs/latent16
python train.py --lr 0.0003 --epochs 30 --out runs/lr0003
```

Run a one-variable sweep sequentially (other train.py options are forwarded):

```powershell
python sweep.py --parameter latent-dim --values 16 64 --epochs 30 --dry-run
python sweep.py --parameter latent-dim --values 16 64 --epochs 30
```

Supported swept parameters: `latent-dim`, `lr`, `beta`, `batch-size`, `image-size`,
`depth`, `base-channels`. The existing baseline covers latent dimension
32. Use the same seed, epoch budget, data, and other settings for controlled
comparisons. `--resume-existing` continues interrupted sweep cases (target epoch
must exceed their saved epoch). Without it, existing run folders are protected.
No new full training sweeps are launched automatically by evaluation or reporting.
`--evaluate` opts into test evaluation after each sweep case; prefer comparing
validation/sample results first and evaluating the final planned cases together.

After each completed training epoch, `summary.md`, a separate editable `notes.md`,
and the shared `runs/experiments.csv` are updated. Summary notes capture settings,
architecture, best/final losses, and valid latest evaluation scores. Your edits
to `notes.md` are preserved; `summary.md` is generated and overwritten.
The `limit` column distinguishes debugging subsets from full-data experiments.
FID columns are blank until measured; scores are never inferred from losses.

### CNN depth and width experiments

`--depth` controls the number of stride-2 encoder convolution blocks (2 through
5); the decoder mirrors that number of upsampling blocks. `--base-channels`
controls the first hidden channel count, doubled at each successive encoder
block. Defaults (depth 4, base channels 16) exactly retain the old checkpoint
layout. Architecture settings are inherited and cannot change during resume.
Old checkpoints without these fields are interpreted as depth 4/base 16.

At 128 x 128, compare the existing four-block architecture with three and five:

```powershell
python sweep.py --parameter depth --values 3 5 --image-size 128 --latent-dim 32 --beta 0.5 --epochs 40 --seed 42 --prefix cnn128
```

Depth 3: channels 16,32,64 and final spatial size 16 x 16.
Depth 4: channels 16,32,64,128 and final spatial size 8 x 8 (existing model).
Depth 5: channels 16,32,64,128,256 and final spatial size 4 x 4.
These experiments change downsampling, receptive fields, and parameter counts,
so they test the complete architecture choice rather than depth in isolation.
Settings and parameter counts are recorded in new runs and in experiments.csv.
Input size must be divisible by 2**depth and leave at least a 2 x 2 spatial map.
To study width separately, sweep `base-channels` while keeping depth fixed.

Refresh existing runs and generate comparison figures at any time:

```powershell
python experiments.py --runs runs
```

Open `runs/experiments.csv`, `runs/comparison_losses.png`, and
`runs/comparison_samples.png`. Sample comparisons label whether they use an
evaluated checkpoint or the final training epoch; they may use different latent
dimensions and are not pixel-aligned comparisons. Curves show validation losses;
total losses for different beta/resolution settings are not directly comparable.
Debugging subsets stay in the CSV, but comparison figures omit them when full-data
runs are available.
For more reliable conclusions, repeat promising settings with additional seeds.

Implementation checks:

```powershell
python -m unittest discover -s tests -v
```
