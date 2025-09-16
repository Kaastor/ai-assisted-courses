"""Acceptance test summary derived from the outline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AcceptanceSummaryItem:
    lab_identifier: str
    description: str


@dataclass(frozen=True)
class AcceptanceSummary:
    items: tuple[AcceptanceSummaryItem, ...]
    reference_doc: Path


DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"

ACCEPTANCE_SUMMARY = AcceptanceSummary(
    items=(
        AcceptanceSummaryItem(
            lab_identifier="lab01_tabular",
            description="ROC-AUC ≥ 0.88, ECE ≤ 0.08, unit tests pass",
        ),
        AcceptanceSummaryItem(
            lab_identifier="lab02_vision",
            description="Accuracy ≥ 0.85, worst-class recall ≥ 0.75, overfit-one-batch test passes",
        ),
        AcceptanceSummaryItem(
            lab_identifier="lab03_text",
            description="Macro-F1 ≥ 0.90, vocabulary bounds respected, shape tests pass",
        ),
        AcceptanceSummaryItem(
            lab_identifier="lab04_timeseries",
            description="MAE improves ≥ 5% over naive baseline, window test passes",
        ),
        AcceptanceSummaryItem(
            lab_identifier="lab05_recsys",
            description="Precision@10 ≥ 0.07, Recall@10 ≥ 0.10, forward shape test passes",
        ),
        AcceptanceSummaryItem(
            lab_identifier="lab06_generative",
            description="Loss drops by ≥ 20% from epoch 0, samples.png created, shape test passes",
        ),
    ),
    reference_doc=DOCS_ROOT / "acceptance_summary.md",
)
