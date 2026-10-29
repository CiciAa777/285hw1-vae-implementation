# ECE 285 HW1: learn a convolutional VAE

Start with a working 64 x 64 grayscale baseline. Read `vae.py` before changing the architecture. The implementation uses both NORMAL and PNEUMONIA images without labels: it learns the training mixture, not a classifier or a conditionally controlled generator.

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

## 5. Generate from a selected checkpoint

```python
import torch
from vae import VAE
from train import save_grid

checkpoint = torch.load('runs/baseline/best.pt', map_location='cpu', weights_only=True)
cfg = checkpoint['config']
model = VAE(cfg['latent_dim'], cfg['image_size'])
model.load_state_dict(checkpoint['model'])
model.eval()
torch.manual_seed(123)
with torch.no_grad():
    z = torch.randn(32, model.latent_dim)
    save_grid(model.decode(z), 'runs/baseline/best_samples.png')
```

These are decoder means from new prior samples; no source image or encoder is involved. We display the decoder mean rather than adding independent Gaussian pixel noise.

## 6. Evaluation and report: after training

FID/IS and the final report are the next stage; they are not implemented or measured yet. Follow the assignment's [Ignite FID and IS tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/). Its GAN training section is not needed for this VAE; reuse the evaluation approach with VAE prior samples.

Use held-out real test images for FID, and NEW prior samples for generated images. Repeat grayscale into 3 channels and apply identical spatial/range preprocessing to real and generated images according to the selected feature extractor's contract. Record feature extractor, weights, normalization, resizing, sample counts, and seeds. Use all 624 real test images and state this limited reference size; FID covariance estimates are noisy with so few real images. IS uses generated images only. Standard Inception features/classes are trained on natural images, so discuss their limitations for X-rays. Do not compare scores from incompatible preprocessing or sample counts.

For the two-page NeurIPS-format report, reserve space for: method and objective; architecture and hyperparameters; loss curves; measured FID/IS and evaluation protocol; a prior-sample grid; and a short discussion of blur, diversity, and limitations. Write conclusions only after seeing actual results. Keep the code and report separate for submission; do not submit dataset or virtual environment files.
