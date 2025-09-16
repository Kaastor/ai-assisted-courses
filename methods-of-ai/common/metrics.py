"""Reusable metric helpers for the labs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn import metrics


@dataclass
class BinaryMetrics:
    """Container for binary classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def _to_numpy(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values)
    return arr if arr.ndim else arr.reshape(-1)


def binary_classification_metrics(
    probabilities: Iterable[float],
    labels: Iterable[int],
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Compute accuracy, precision, recall, F1, and ROC-AUC."""

    probs = _to_numpy(probabilities)
    y_true = _to_numpy(labels).astype(int)
    y_pred = (probs >= threshold).astype(int)
    return BinaryMetrics(
        accuracy=float(metrics.accuracy_score(y_true, y_pred)),
        precision=float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        recall=float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        f1=float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(metrics.roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan"),
    )


def macro_f1(logits_or_probs: Iterable[Iterable[float]], labels: Iterable[int]) -> float:
    """Compute macro F1 for multi-class tasks; accepts logits or probabilities."""

    scores = np.asarray(logits_or_probs)
    y_true = _to_numpy(labels).astype(int)
    if scores.ndim == 1:
        scores = np.expand_dims(scores, 1)
    y_pred = scores.argmax(axis=1)
    return float(metrics.f1_score(y_true, y_pred, average="macro"))


def per_class_recall(logits_or_probs: Iterable[Iterable[float]], labels: Iterable[int]) -> Dict[int, float]:
    """Return recall by class index."""

    scores = np.asarray(logits_or_probs)
    y_true = _to_numpy(labels).astype(int)
    if scores.ndim == 1:
        scores = np.expand_dims(scores, 1)
    y_pred = scores.argmax(axis=1)
    recalls = metrics.recall_score(y_true, y_pred, average=None, labels=np.unique(y_true))
    return {int(cls): float(rec) for cls, rec in zip(sorted(np.unique(y_true)), recalls)}


def confusion_matrix(logits_or_probs: Iterable[Iterable[float]], labels: Iterable[int]) -> np.ndarray:
    """Return the confusion matrix for multi-class predictions."""

    scores = np.asarray(logits_or_probs)
    y_true = _to_numpy(labels).astype(int)
    if scores.ndim == 1:
        scores = np.expand_dims(scores, 1)
    y_pred = scores.argmax(axis=1)
    return metrics.confusion_matrix(y_true, y_pred)


def expected_calibration_error(
    probabilities: Iterable[float], labels: Iterable[int], n_bins: int = 15
) -> float:
    """Compute Expected Calibration Error for binary classifiers."""

    probs = _to_numpy(probabilities).astype(float)
    y_true = _to_numpy(labels).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(probs, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = binids == b
        if not np.any(mask):
            continue
        avg_conf = probs[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += abs(avg_conf - avg_acc) * mask.mean()
    return float(ece)


def precision_recall_at_k(scores: np.ndarray, positives: Iterable[int], k: int) -> Tuple[float, float]:
    """Compute Precision@k and Recall@k given item scores."""

    ranking = np.argsort(scores)[::-1][:k]
    pos_set = set(int(i) for i in positives)
    hits = len(pos_set.intersection(ranking))
    precision = hits / max(1, k)
    recall = hits / max(1, len(pos_set))
    return precision, recall
