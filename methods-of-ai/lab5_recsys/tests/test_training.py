from lab5_recsys.train_mf import TrainingConfig, train


def test_mf_hits_precision_and_recall(tmp_path):
    config = TrainingConfig(
        log_dir=tmp_path / "runs",
        artifacts_dir=tmp_path / "artifacts",
    )
    result = train(config)
    precision_key = f"precision@{config.k_eval}"
    recall_key = f"recall@{config.k_eval}"
    assert result.test_metrics[precision_key] >= 0.07
    assert result.test_metrics[recall_key] >= 0.10
