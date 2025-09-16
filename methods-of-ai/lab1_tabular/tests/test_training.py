from pathlib import Path

from lab1_tabular.train_tabular import TrainingConfig, train


def test_training_meets_acceptance(tmp_path):
    config = TrainingConfig(log_dir=tmp_path / "runs")
    result = train(config)
    assert result.test_metrics["roc_auc"] >= 0.88
    assert result.val_metrics["ece"] <= 0.08
