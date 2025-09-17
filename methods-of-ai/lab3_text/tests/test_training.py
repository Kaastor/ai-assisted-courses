from lab3_text.train_spam import TrainingConfig, train


def test_training_hits_macro_f1_and_vocab_bounds(tmp_path):
    config = TrainingConfig(
        log_dir=tmp_path / "runs",
        artifacts_dir=tmp_path / "artifacts",
    )
    result = train(config)
    assert 0.0 < result.threshold < 1.0
    assert result.test_metrics["f1"] >= 0.90
    assert 3000 <= result.vocab_size <= 30000
    assert result.test_metrics["roc_auc"] >= 0.95
