"""Capstone reflection checklist representation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReflectionSection:
    title: str
    checkpoints: tuple[str, ...]


@dataclass(frozen=True)
class ReflectionChecklist:
    sections: tuple[ReflectionSection, ...]
    reference_doc: Path


DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"

REFLECTION_CHECKLIST = ReflectionChecklist(
    sections=(
        ReflectionSection(
            title="Problem types",
            checkpoints=(
                "Match each domain (vision, text, tabular, time-series, recommendation, generative) to a PyTorch method built in labs",
                "Assess whether deep learning is appropriate for a new scenario and articulate reasoning versus classical methods",
            ),
        ),
        ReflectionSection(
            title="Pros/cons of DL",
            checkpoints=(
                "State deep learning trade-offs around data/compute, latency, maintenance debt, and interpretability",
                "Identify when a simpler baseline is the proper choice",
            ),
        ),
        ReflectionSection(
            title="Method choice & trade-offs",
            checkpoints=(
                "Justify selection between methods using constraints like data size, latency, interpretability, and maintenance",
                "Set sensible defaults (batch size, learning rate schedules) and communicate tuning rationale",
            ),
        ),
        ReflectionSection(
            title="Implement / test / debug",
            checkpoints=(
                "Implement clean training loops separating data, model, optimizer, scheduler, and metrics",
                "Write unit tests including shape checks and overfit-one-batch guards and interpret failures",
                "Perform error analysis via confusion matrices, per-class metrics, calibration, and slice checks",
            ),
        ),
        ReflectionSection(
            title="Data work",
            checkpoints=(
                "Build modality-appropriate input pipelines that prevent leakage (temporal/user splits, train-only transforms)",
                "Select metrics suited to the task (AUC/F1, MAE/MAPE, P@K/R@K, ELBO)",
                "Track experiments reproducibly with seeds, logs, and run cards",
            ),
        ),
        ReflectionSection(
            title="Ethics & reliability",
            checkpoints=(
                "Identify lab-specific biases or harms and outline mitigations",
                "Respect dataset licenses and privacy constraints",
            ),
        ),
    ),
    reference_doc=DOCS_ROOT / "capstone_reflection.md",
)
