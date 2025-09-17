# Lab 6 — Variational Autoencoder on Fashion-MNIST

## What's inside

- `train_vae.py` — fully working VAE (encoder/decoder MLP, KL warm-up, sample generation, TensorBoard logging).
- `tests/` — unit test for forward shapes and an acceptance test verifying ≥20% training-loss reduction plus sample image export.

Method: stochastic latent variable model with reparameterisation (`μ`, `logσ²` heads), trained on a 10k subset of Fashion-MNIST using Adam. KL weight warms up over the first three epochs to avoid posterior collapse. After training, the script samples 64 images and saves them as a grid.

## How to run

```bash
poetry install
poetry run python -m lab6_generative.train_vae
```

TensorBoard logs live in `runs/lab6_generative`. Generated samples default to `artifacts/lab6_generative/samples.png`.

## Autograding

```bash
poetry run python -m pytest lab6_generative/tests -q
```

Acceptance checks:

- Training loss drops ≥ 20% from epoch 0 to the final epoch.
- `samples.png` is produced.

## Data & licensing

- **Dataset:** Fashion-MNIST — MIT License (via torchvision).
- Downloaded automatically to `.data/fashion_mnist` using torchvision’s dataset API.
- Training uses 10k images (subsampled for CPU speed); evaluation samples 2k test images.

## Moving pieces & extensions

- Adjust latent dimensionality or add β-VAE scaling to explore disentanglement vs. fidelity.
- Chart reconstruction vs. KL curves from `loss_history` to inspect optimisation dynamics.
- Extend the decoder to a convolutional architecture or plug VAE latents into anomaly detection tasks.

## Risk & bias note

Generative models can synthesise deceptive or biased content. Use course-produced VAEs responsibly, watermark generated imagery, and discuss ethical deployment scenarios with students.
