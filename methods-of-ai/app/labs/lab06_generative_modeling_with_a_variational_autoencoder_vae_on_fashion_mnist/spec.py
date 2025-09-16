"""Structured metadata for Lab 6."""

from __future__ import annotations

from pathlib import Path

from app.labs.base import AcceptanceTest, DatasetReference, LabSpecification

README_PATH = Path(__file__).with_name("README.md")

LAB_SPEC = LabSpecification(
    identifier="lab06_generative",
    title="Generative modeling with a Variational Autoencoder (VAE) on Fashion-MNIST",
    domain="Generative modeling",
    purpose=(
        "Teach variational autoencoders, ELBO decomposition, and basic anomaly detection using Fashion-MNIST."
    ),
    method_summary=(
        "Lightweight MLP-based VAE with KL warm-up, reconstruction monitoring, and latent sampling for CPU execution."
    ),
    dataset=DatasetReference(
        name="Fashion-MNIST",
        license="MIT License",
        url="https://github.com/zalandoresearch/fashion-mnist",
        notes="Shared dataset with Lab 2; reused here for generative training on a manageable subset.",
    ),
    acceptance_tests=(
        AcceptanceTest(
            description="Training loss decreases from epoch 0 by at least twenty percent",
            metric="ELBO",
            threshold="loss drops ≥20%",
            dataset_split="train",
        ),
        AcceptanceTest(
            description="Model produces samples.png after inference",
            metric="artifact",
            threshold="file generated",
            dataset_split="n/a",
        ),
        AcceptanceTest(
            description="VAE forward pass returns reconstruction with correct shape",
            metric="pytest",
            threshold="pass",
            dataset_split="synthetic",
        ),
    ),
    key_focus=(
        "Reparameterization trick and ELBO monitoring",
        "Balancing reconstruction and KL terms via beta schedules",
        "Latent sampling for generation and anomaly scoring",
        "Discussing ethical risks including deepfakes or misuse",
    ),
    failure_modes=(
        "Uniform gray reconstructions caused by decoder saturation (tune LR or epochs)",
        "KL collapse requiring warm-up or beta < 1 early in training",
    ),
    assignment_seed=(),
    starter_code=(
        "lab6_generative/train_vae.py",
        "lab6_generative/tests/test_shapes.py",
    ),
    stretch_goals=(),
    readings=(),
    comparison_table_markdown="""| Method       | When it shines                            | When it fails                | Data needs | Inference cost | Interpretability          | Typical metrics     |
| ------------ | ----------------------------------------- | ---------------------------- | ---------- | -------------- | ------------------------- | ------------------- |
| VAE          | Fast generative modeling; anomaly scoring | Sharpness & fidelity limited | 10k–100k   | Low            | Medium (latent traversal) | ELBO                |
| Denoising AE | Denoising/compression                     | No true generative sampling  | 10k–100k   | Very low       | Low                       | Recon loss          |
| Diffusion    | Photorealistic gen                        | Heavy compute                | 100k+      | High           | Low                       | FID (not used here) |
""",
    readme_path=README_PATH,
)
