"""Dataclasses describing the shared environment and tooling expectations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvironmentGuidelines:
    project_layout: tuple[str, ...]
    poetry: tuple[str, ...]
    modules_over_notebooks: tuple[str, ...]
    logging: tuple[str, ...]
    experiment_tracking: tuple[str, ...]
    version_control: tuple[str, ...]
    determinism: tuple[str, ...]
    compute_notes: tuple[str, ...]
    ethics_reliability: tuple[str, ...]
    reference_doc: Path


DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"

ENVIRONMENT_GUIDELINES = EnvironmentGuidelines(
    project_layout=(
        "pyproject.toml at repository root",
        "common/ module with shared helpers such as seed.py, metrics.py, viz.py",
        "Dedicated lab modules lab{1..6}_...",
        "scripts/ directory for CLI utilities like dataset downloads",
    ),
    poetry=(
        "Initialize with poetry; add torch, torchvision, pandas, scikit-learn, tensorboard, matplotlib",
        "Use `poetry lock` and `poetry install` to sync dependencies",
        "Run commands with `poetry run` to stay within the virtualenv",
    ),
    modules_over_notebooks=(
        "Prefer Python modules for import hygiene, testability, and reproducible runs",
        "Expose CLI flags for runtime configuration (batch size, epochs) to aid grading",
    ),
    logging=(
        "Use torch.utils.tensorboard.SummaryWriter for scalar logging",
        "Persist runs under runs/labX_* and avoid committing large logs",
    ),
    experiment_tracking=(
        "Maintain run cards (YAML/Markdown) capturing dataset version, hyperparameters, metrics, and checkpoints",
        "Adopt descriptive run naming such as lab2_cnn_bs128_lr1e-3_wd1e-4_seed42",
    ),
    version_control=(
        "Create feature branches per lab (e.g., feat/lab3-thresholding)",
        "PR template should cover what changed, why, testing evidence, and risks",
        "Tag hand-ins (lab1-v1.0) and export dependencies via `poetry export`",
    ),
    determinism=(
        "Call set_seed(42) for Python, NumPy, and PyTorch",
        "Enable torch.use_deterministic_algorithms(True)",
        "Configure DataLoader with num_workers=0, shuffle, drop_last for reproducibility",
        "Log package versions and OS details",
    ),
    compute_notes=(
        "Train on CPU-friendly subsets with small architectures",
        "Use batch sizes 64–256, AdamW lr around 1e-3, weight decay 1e-4, 5–6 epochs",
        "Prefer EmbeddingBag for text to avoid padding",
        "For time-series use window sizes ~24–48 and hidden dimension ~32",
    ),
    ethics_reliability=(
        "Document risk & bias notes per lab (e.g., Adult demographics, recommender filter bubbles, generative misuse)",
        "Include calibration plots and slice-based metrics",
        "Respect dataset licenses and privacy requirements",
    ),
    reference_doc=DOCS_ROOT / "environment_and_tooling.md",
)
