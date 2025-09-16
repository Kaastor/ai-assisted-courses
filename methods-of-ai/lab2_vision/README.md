# Lab 2 — Fashion-MNIST CNN & Calibration

## What's inside

- `train_cnn.py` — training script for a compact CNN with augmentations, StepLR, early stopping, and TensorBoard logging.
- `calibrate.py` — temperature scaling utilities for multiclass logits.
- `tests/` — shape test plus acceptance test asserting test accuracy ≥ 0.85 and worst-class recall ≥ 0.75.

Method: two-layer CNN, ReLU activations, dropout, AdamW optimiser, StepLR scheduler (gamma=0.5 every two epochs), augmentations via random affine transforms, and post-hoc temperature scaling.

## How to run

```bash
poetry install
poetry run python -m lab2_vision.train_cnn
```

Change hyperparameters by instantiating `VisionConfig`. Logs land in `runs/lab2_vision`.

## Autograding

```bash
poetry run python -m pytest lab2_vision/tests -q
```

Autograder expectations: test accuracy ≥ 0.85 and worst-class recall ≥ 0.75 on Fashion-MNIST (subset of 12k training images, CPU-friendly).

## Moving pieces

- Deterministic splits with seed 42 over a 12k-image training subset (85% train, 15% val).
- Per-class recall calculation to highlight minority-class performance; the minimum recall drives the acceptance test.
- Temperature scaling optimised with LBFGS on validation logits.
- Normalisation uses Fashion-MNIST statistics (mean 0.2861, std 0.3530).

## Data & licensing

- **Dataset:** Fashion-MNIST (MIT License). Downloaded automatically to `.data/` via `torchvision`.

## Risk & bias note

Dataset labels can reflect cultural biases about fashion categories. Report per-class errors, especially on underrepresented categories, before deploying the classifier in production or moderation settings.
