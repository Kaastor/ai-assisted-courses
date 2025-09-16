"""Common data structures for lab specifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DatasetReference:
    """Reference to a dataset used in a lab."""

    name: str
    license: str
    url: str
    notes: str


@dataclass(frozen=True)
class AcceptanceTest:
    """Acceptance test requirement for a lab."""

    description: str
    metric: str
    threshold: str
    dataset_split: str


@dataclass(frozen=True)
class LabSpecification:
    """Structured metadata for a lab assignment."""

    identifier: str
    title: str
    domain: str
    purpose: str
    method_summary: str
    dataset: DatasetReference
    acceptance_tests: tuple[AcceptanceTest, ...]
    key_focus: tuple[str, ...]
    failure_modes: tuple[str, ...]
    assignment_seed: tuple[str, ...]
    starter_code: tuple[str, ...]
    stretch_goals: tuple[str, ...]
    readings: tuple[str, ...]
    comparison_table_markdown: str
    readme_path: Path

    def ensure_readme_exists(self) -> None:
        """Validate that the backing README contains the full lab instructions."""

        if not self.readme_path.exists():  # pragma: no cover - defensive guard
            raise FileNotFoundError(self.readme_path)


def ensure_all_specs(specs: Iterable[LabSpecification]) -> tuple[LabSpecification, ...]:
    """Materialize lab specifications and ensure their READMEs exist."""

    realized = tuple(specs)
    for spec in realized:
        spec.ensure_readme_exists()
    return realized
