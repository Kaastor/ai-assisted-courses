"""Dataclass representation of the course overview."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GradingComponent:
    """Represents a graded component of the course."""

    name: str
    weight_percentage: int
    notes: str


@dataclass(frozen=True)
class CourseOverview:
    """Top-level overview aligned with OUTLINE.md."""

    title: str
    audience: str
    format: str
    learning_objectives: tuple[str, ...]
    grading: tuple[GradingComponent, ...]
    tools_and_compute: tuple[str, ...]
    workflow_policies: tuple[str, ...]
    course_overview_doc: Path
    labs_overview_doc: Path


DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"

COURSE_OVERVIEW = CourseOverview(
    title="Methods of Deep Learning",
    audience=(
        "Computer science undergraduates comfortable with Python and general programming; "
        "no deep learning prerequisites."
    ),
    format=(
        "Six hands-on labs (2.5–3 hours each) combining front-loaded instruction with guided build-modify-test exercises."
    ),
    learning_objectives=(
        "Recognize main deep learning problem types including vision, text, tabular, time-series, generative, and recommendation",
        "Explain advantages and limitations of deep learning versus classical ML (data/compute, latency, interpretability, maintenance)",
        "Choose appropriate methods with trade-off reasoning",
        "Implement, test, debug, and validate PyTorch models",
        "Work with data pipelines, preprocessing, metrics, and leakage pitfalls across modalities",
        "Practice reproducibility, reliability, and ethical considerations such as bias and privacy",
    ),
    grading=(
        GradingComponent(
            name="Labs 1–6",
            weight_percentage=60,
            notes=(
                "Each lab contributes 10% via acceptance tests (unit tests plus metric thresholds) and a brief error-analysis writeup."
            ),
        ),
        GradingComponent(
            name="Capstone reflection checklist",
            weight_percentage=10,
            notes="Ensures student work aligns with course goals and evidence requirements.",
        ),
        GradingComponent(
            name="Method choice memos (two short submissions)",
            weight_percentage=10,
            notes="Focus on trade-off reasoning based on practical scenarios.",
        ),
        GradingComponent(
            name="Open-book practical quiz",
            weight_percentage=20,
            notes="Covers debugging workflows and metric interpretation.",
        ),
    ),
    tools_and_compute=(
        "PyTorch framework running CPU-only with optional torchvision/torchtext/torchaudio",
        "Each lab designed to finish training/eval on CPU in <= 15 minutes via small models or subsets",
        "TensorBoard and CSV logging with optional Weights & Biases offline mode",
        "Reproducibility via seeded runs and deterministic flags",
    ),
    workflow_policies=(
        "Poetry for dependency management; deliverables as Python modules (no notebooks)",
        "Git workflow using branches and pull requests with descriptive commit summaries",
        "Collaboration allowed for discussion/debugging but submitted code must be original; cite AI helpers",
        "Two weekly office hours, including a debugging-focused bug clinic",
        "Academic integrity policy: plagiarism or undisclosed copying results in lab failure",
    ),
    course_overview_doc=DOCS_ROOT / "course_overview.md",
    labs_overview_doc=DOCS_ROOT / "labs_overview.md",
)
