# Lab 4 — Transfer Learning for Vision

This lab follows the second vision lab and focuses on **transfer learning** and **efficient fine-tuning** on small image datasets. Students compare the accuracy and training cost of:

1. A lightweight convolutional network trained from scratch on a subset of CIFAR-10.
2. A ResNet feature extractor with either frozen backbones or partially unfrozen layers.

Key learning goals:

- Understand why pretrained weights speed up convergence and improve accuracy when data is scarce.
- Evaluate trade-offs between frozen vs. trainable backbones and different learning rates.
- Practise reproducible experimentation with fixed seeds, deterministic dataloaders, and TensorBoard traces.
- Record results and bottlenecks to motivate later deployment discussions.

The provided starter code exposes configuration knobs for subset size, optimizer, and fine-tuning strategy. Acceptance tests ensure that:

- Dataloaders produce the expected batch shapes on CPU.
- Baseline and transfer models can overfit a tiny synthetic dataset.
- Evaluation helpers return consistent accuracy/ loss values.

> For offline testing we stub the dataset; during the real lab students invoke the script with `--use-pretrained` to download torchvision weights.
