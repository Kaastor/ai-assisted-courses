import numpy as np
import pytest

from lab5_reliability.metrics import compute_calibration_curve, compute_classification_metrics, brier_score


def test_compute_classification_metrics_handles_perfect_predictions() -> None:
    probs = np.array([0.1, 0.9, 0.2, 0.8])
    labels = np.array([0, 1, 0, 1])
    metrics = compute_classification_metrics(probs, labels, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["ece"] == pytest.approx(0.15)


def test_compute_calibration_curve_counts_match_samples() -> None:
    probs = np.linspace(0.05, 0.95, 10)
    labels = np.round(probs)
    curve = compute_calibration_curve(probs, labels, n_bins=5)
    assert curve["count"].sum() == probs.shape[0]
    assert curve["confidence"].shape == (5,)


def test_brier_score_non_negative() -> None:
    probs = [0.2, 0.7, 0.1]
    labels = [0, 1, 0]
    score = brier_score(probs, labels)
    assert score >= 0.0
