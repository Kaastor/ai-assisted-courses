"""Metric helpers for the reliability lab."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn import metrics

from common.metrics import expected_calibration_error


def compute_classification_metrics(
    probabilities: Iterable[float],
    labels: Iterable[int],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Return accuracy/precision/recall/F1/ROC-AUC for binary predictions."""

    probs = np.asarray(list(probabilities), dtype=float)
    labs = np.asarray(list(labels), dtype=int)
    preds = (probs >= threshold).astype(int)
    result = {
        "accuracy": float(metrics.accuracy_score(labs, preds)),
        "precision": float(metrics.precision_score(labs, preds, zero_division=0)),
        "recall": float(metrics.recall_score(labs, preds, zero_division=0)),
        "f1": float(metrics.f1_score(labs, preds, zero_division=0)),
    }
    if len(np.unique(labs)) > 1:
        result["roc_auc"] = float(metrics.roc_auc_score(labs, probs))
    else:
        result["roc_auc"] = float("nan")
    result["ece"] = expected_calibration_error(probs, labs)
    return result


def compute_calibration_curve(
    probabilities: Iterable[float],
    labels: Iterable[int],
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """Return per-bin confidence, accuracy, and counts for plotting."""

    probs = np.asarray(list(probabilities), dtype=float)
    labs = np.asarray(list(labels), dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1

    confidences = np.zeros(n_bins, dtype=float)
    accuracies = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for idx in range(n_bins):
        mask = bin_indices == idx
        if not np.any(mask):
            continue
        counts[idx] = int(mask.sum())
        confidences[idx] = float(probs[mask].mean())
        accuracies[idx] = float(labs[mask].mean())

    return {"confidence": confidences, "accuracy": accuracies, "count": counts}


def brier_score(probabilities: Iterable[float], labels: Iterable[int]) -> float:
    """Compute the Brier score (squared error) for calibrated probabilities."""

    probs = np.asarray(list(probabilities), dtype=float)
    labs = np.asarray(list(labels), dtype=float)
    return float(np.mean((probs - labs) ** 2))
