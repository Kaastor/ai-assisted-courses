"""Slice-based metric computation for reliability analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .metrics import compute_classification_metrics


@dataclass
class SliceMetrics:
    """Container for per-slice evaluation results."""

    slice_name: str
    value: str
    sample_count: int
    metrics: Dict[str, float]


FeatureMap = Mapping[str, Sequence[str]]


def evaluate_slices(
    probabilities: Iterable[float],
    labels: Iterable[int],
    slice_features: FeatureMap,
    threshold: float = 0.5,
    min_samples: int = 10,
) -> List[SliceMetrics]:
    """Compute metrics for each feature/value slice above ``min_samples``."""

    probs = np.asarray(list(probabilities), dtype=float)
    labs = np.asarray(list(labels), dtype=int)
    if probs.shape != labs.shape:
        raise ValueError("probabilities and labels must have the same length")

    results: List[SliceMetrics] = []
    for feature_name, values in slice_features.items():
        values_array = np.asarray(list(values))
        if values_array.shape[0] != probs.shape[0]:
            raise ValueError(f"Feature '{feature_name}' has mismatched length")
        for unique_value in np.unique(values_array):
            mask = values_array == unique_value
            if mask.sum() < min_samples:
                continue
            metrics = compute_classification_metrics(probs[mask], labs[mask], threshold=threshold)
            results.append(
                SliceMetrics(
                    slice_name=feature_name,
                    value=str(unique_value),
                    sample_count=int(mask.sum()),
                    metrics=metrics,
                )
            )
    return results
