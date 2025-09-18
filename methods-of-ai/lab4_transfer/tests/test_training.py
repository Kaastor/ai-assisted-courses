import torch
from torch.utils.data import Dataset

from lab4_transfer.train_transfer import TransferConfig, TransferTrainingResult, build_dataloaders, evaluate, train


class TinyVisionDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.targets.size(0)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


def _dummy_builder(config: TransferConfig):
    num_classes = 3
    per_split = 24
    rng = torch.Generator().manual_seed(config.seed)

    def make_split() -> TinyVisionDataset:
        images = []
        labels = []
        for cls in range(num_classes):
            base = torch.full((3, 32, 32), float(cls) / num_classes)
            noise = torch.randn((per_split // num_classes, 3, 32, 32), generator=rng) * 0.01
            split_images = base.unsqueeze(0) + noise
            images.append(split_images)
            labels.append(torch.full((per_split // num_classes,), cls, dtype=torch.long))
        stacked = torch.cat(images, dim=0)
        stacked = stacked.clamp(0.0, 1.0).to(torch.float32)
        target = torch.cat(labels, dim=0)
        return TinyVisionDataset(stacked, target)

    train_ds = make_split()
    val_ds = make_split()
    test_ds = make_split()
    class_names = tuple(str(i) for i in range(num_classes))
    return train_ds, val_ds, test_ds, class_names


def test_build_dataloaders_shapes() -> None:
    config = TransferConfig(batch_size=8, subset_size=24)
    loaders, class_names = build_dataloaders(config, dataset_builder=_dummy_builder)
    batch = next(iter(loaders["train"]))
    images, labels = batch
    assert images.shape == (8, 3, 32, 32)
    assert labels.shape == (8,)
    assert class_names == ("0", "1", "2")


def test_evaluate_reports_accuracy_between_zero_and_one() -> None:
    config = TransferConfig(batch_size=12)
    loaders, _ = build_dataloaders(config, dataset_builder=_dummy_builder)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(32 * 32 * 3, 3))
    metrics = evaluate(model, ((x, y) for x, y in loaders["val"]), torch.device("cpu"))
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_train_returns_result_with_metrics() -> None:
    config = TransferConfig(batch_size=12, max_epochs=2, learning_rate=5e-3, use_pretrained=False)
    result = train(config, dataset_builder=_dummy_builder)
    assert isinstance(result, TransferTrainingResult)
    assert set(result.scratch_metrics.keys()) == {"loss", "accuracy"}
    assert set(result.transfer_metrics.keys()) == {"loss", "accuracy"}
    assert result.best_epoch >= 0
