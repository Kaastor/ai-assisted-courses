from lab4_timeseries.train_gru import TrainingConfig, train


def test_gru_beats_naive_baseline(tmp_path):
    config = TrainingConfig(
        log_dir=tmp_path / "runs",
        artifacts_dir=tmp_path / "artifacts",
    )
    result = train(config)
    assert result.test_metrics["mae"] < result.naive_mae
    assert result.improvement_pct >= 5.0
    assert "mape" in result.test_metrics
