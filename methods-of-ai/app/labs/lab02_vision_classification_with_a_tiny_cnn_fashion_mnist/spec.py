"""Structured metadata for Lab 2."""

from __future__ import annotations

from pathlib import Path

from app.labs.base import AcceptanceTest, DatasetReference, LabSpecification

README_PATH = Path(__file__).with_name("README.md")

LAB_SPEC = LabSpecification(
    identifier="lab02_vision",
    title="Vision classification with a tiny CNN (Fashion-MNIST)",
    domain="Image classification",
    purpose=(
        "Teach a compact CNN training loop with augmentation, regularization, and calibration "
        "techniques on Fashion-MNIST."
    ),
    method_summary=(
        "Two-layer convolutional network with affine data augmentation, AdamW optimization, and LR scheduling."
    ),
    dataset=DatasetReference(
        name="Fashion-MNIST",
        license="MIT License",
        url="https://github.com/zalandoresearch/fashion-mnist",
        notes="10-class clothing dataset built into torchvision; CPU-friendly size and download.",
    ),
    acceptance_tests=(
        AcceptanceTest(
            description="Ensure the tiny CNN can overfit a small batch for correctness",
            metric="pytest",
            threshold="pass",
            dataset_split="synthetic",
        ),
        AcceptanceTest(
            description="Top-1 accuracy on the evaluation split",
            metric="Accuracy",
            threshold="≥ 0.85",
            dataset_split="test",
        ),
        AcceptanceTest(
            description="Worst-class recall on evaluation data",
            metric="Recall",
            threshold="≥ 0.75",
            dataset_split="test",
        ),
    ),
    key_focus=(
        "Torchvision dataset usage and deterministic data transforms",
        "Data augmentation and regularization (dropout, weight decay)",
        "Learning rate scheduling via StepLR or OneCycle",
        "Confusion matrices, per-class accuracy, and reliability diagrams",
        "Temperature scaling for calibration",
    ),
    failure_modes=(
        "Underfitting signaled by low training accuracy (increase capacity or reduce augmentation)",
        "Overfitting with training accuracy far above validation (add regularization or early stop)",
        "Miscalibration evidenced by confident errors (apply validation temperature scaling)",
    ),
    assignment_seed=(
        "Compute confusion matrix and per-class F1; analyze three weakest classes",
        "Implement temperature scaling optimized on the validation set",
        "Summarize augmentation adjustments and their metric impact",
    ),
    starter_code=(
        "lab2_vision/train_cnn.py",
        "lab2_vision/calibrate.py",
        "lab2_vision/tests/test_overfit.py",
    ),
    stretch_goals=(
        "Experiment with stronger augmentations or CutOut/MixUp variants",
        "Benchmark transfer learning from pretrained CNN backbones",
        "Compare calibration approaches (Platt scaling vs. temperature)",
    ),
    readings=(
        "Introductory notes on CNN inductive biases",
        "Temperature scaling for neural network calibration (Guo et al.)",
        "Skim torchvision transform documentation",
    ),
    comparison_table_markdown="""| Method                     | When it shines                   | When it fails                       | Data needs    | Inference cost | Interpretability | Typical metrics |
| -------------------------- | -------------------------------- | ----------------------------------- | ------------- | -------------- | ---------------- | --------------- |
| Small CNN (from scratch)   | Simple tasks, limited compute    | Complex domains; small labeled data | 10k–100k imgs | Low–Med        | Low              | Top-1 acc       |
| Transfer learning (frozen) | Few labels; higher accuracy fast | Very small CPU? downloads heavy     | 1k–10k        | Low            | Low              | Top-1 acc       |
| Classic features + SVM     | Tight latency, tiny data         | Complex variations                  | 100–10k       | Very low       | Medium           | Top-1 acc       |
""",
    readme_path=README_PATH,
)
