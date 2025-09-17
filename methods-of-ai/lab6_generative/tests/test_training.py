from lab6_generative.train_vae import TrainingConfig, train


def test_vae_produces_samples_and_reduces_loss(tmp_path):
    config = TrainingConfig(
        log_dir=tmp_path / "runs",
        artifacts_dir=tmp_path / "artifacts",
    )
    result = train(config)
    assert result.loss_drop_pct >= 20.0
    assert result.samples_path is not None
    assert result.samples_path.exists()
