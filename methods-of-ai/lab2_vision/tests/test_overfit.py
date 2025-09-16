import torch

from lab2_vision.train_cnn import SmallCNN, VisionConfig, train


def test_forward_shape():
    model = SmallCNN()
    out = model(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)


def test_training_hits_accuracy(tmp_path):
    config = VisionConfig(log_dir=tmp_path / "runs")
    result = train(config)
    assert result.test_metrics["accuracy"] >= 0.85
    assert result.test_metrics["worst_class_recall"] >= 0.75
