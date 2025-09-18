import numpy as np

from lab5_reliability.analyze import ReliabilityConfig, analyze_predictions
from lab5_reliability.slices import evaluate_slices


def test_evaluate_slices_filters_small_groups() -> None:
    probs = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    labels = np.array([0, 1, 0, 1, 1])
    slices = {"group": ["A", "A", "B", "B", "B"]}
    metrics = evaluate_slices(probs, labels, slices, min_samples=2)
    assert len(metrics) == 2
    assert {m.value for m in metrics} == {"A", "B"}


def test_analyze_predictions_returns_structured_report() -> None:
    probs = np.array([0.1, 0.9, 0.2, 0.8])
    labels = np.array([0, 1, 0, 1])
    slices = {"bucket": ["low", "high", "low", "high"]}
    report = analyze_predictions(probs, labels, slices, ReliabilityConfig(n_bins=4, min_slice_size=2))
    assert set(report.overall.keys()) >= {"accuracy", "precision", "recall", "f1"}
    assert report.calibration["confidence"].shape == (4,)
    assert "bucket" in report.slices
    assert all(item.sample_count >= 2 for item in report.slices["bucket"])
