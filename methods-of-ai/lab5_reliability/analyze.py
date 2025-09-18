"""High-level API for reliability assessment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import compute_calibration_curve, compute_classification_metrics, brier_score
from .slices import SliceMetrics, evaluate_slices


@dataclass
class ReliabilityConfig:
    """Configuration options for generating a reliability report."""

    threshold: float = 0.5
    temperature: float = 1.0
    n_bins: int = 10
    min_slice_size: int = 10
    device: str = "cpu"


@dataclass
class ReliabilityReport:
    """Structured summary of reliability checks."""

    overall: Dict[str, float]
    calibration: Dict[str, np.ndarray]
    brier: float
    slices: Dict[str, list[SliceMetrics]]


@torch.inference_mode()
def collect_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Collect sigmoid probabilities and labels from a binary classifier."""

    model.eval()
    probs = []
    labels = []
    extras: Dict[str, list[np.ndarray]] = {}
    for batch in loader:
        if isinstance(batch, Mapping):
            inputs = batch["inputs"].to(device)
            label = batch["labels"].to(device)
            slice_features = {k: v for k, v in batch.items() if k not in {"inputs", "labels"}}
        else:
            inputs, label = batch[:2]
            inputs = inputs.to(device)
            label = label.to(device)
            slice_features = {}
        logits = model(inputs)
        probs.append(torch.sigmoid(logits / temperature).cpu().numpy())
        labels.append(label.cpu().numpy())
        for key, value in slice_features.items():
            extras.setdefault(key, []).append(np.asarray(value))
    probabilities = np.concatenate(probs).astype(float)
    label_array = np.concatenate(labels).astype(int)
    slices = {key: np.concatenate(values) for key, values in extras.items()}
    return {"probabilities": probabilities, "labels": label_array, "slices": slices}


def analyze_predictions(
    probabilities: Iterable[float],
    labels: Iterable[int],
    slice_features: Optional[Mapping[str, Iterable[str]]] = None,
    config: ReliabilityConfig = ReliabilityConfig(),
) -> ReliabilityReport:
    """Produce a reliability report from raw probabilistic predictions."""

    probs = np.asarray(list(probabilities), dtype=float)
    labs = np.asarray(list(labels), dtype=int)
    overall = compute_classification_metrics(probs, labs, threshold=config.threshold)
    calibration = compute_calibration_curve(probs, labs, n_bins=config.n_bins)
    report_slices: Dict[str, list[SliceMetrics]] = {}
    if slice_features:
        mapped = {name: list(values) for name, values in slice_features.items()}
        metrics = evaluate_slices(
            probs,
            labs,
            mapped,
            threshold=config.threshold,
            min_samples=config.min_slice_size,
        )
        for item in metrics:
            report_slices.setdefault(item.slice_name, []).append(item)
    return ReliabilityReport(
        overall=overall,
        calibration=calibration,
        brier=brier_score(probs, labs),
        slices=report_slices,
    )
