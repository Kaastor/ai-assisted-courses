"""Training entrypoint for the transfer learning lab."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from common.seed import set_seed
from common.tensorboard import create_summary_writer

try:  # torchvision is optional at test time
    from torchvision import datasets, models, transforms
    _TORCHVISION_AVAILABLE = True
except Exception:  # pragma: no cover - torchvision missing only in unusual envs
    datasets = models = transforms = None  # type: ignore
    _TORCHVISION_AVAILABLE = False


class SimpleCNN(nn.Module):
    """Small CNN baseline trained from scratch."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass
class TransferConfig:
    """Configuration for comparing scratch vs. transfer learning."""

    batch_size: int = 128
    max_epochs: int = 4
    learning_rate: float = 1e-3
    head_learning_rate: Optional[float] = None
    weight_decay: float = 1e-4
    subset_size: int = 5000
    freeze_backbone: bool = True
    use_pretrained: bool = False
    seed: int = 42
    log_dir: Optional[Path] = Path("runs/lab4_transfer")
    device: str = "cpu"


@dataclass
class TransferTrainingResult:
    """Metrics collected for both training strategies."""

    best_epoch: int
    scratch_metrics: Dict[str, float]
    transfer_metrics: Dict[str, float]
    class_names: Tuple[str, ...]


DatasetBuilder = Callable[[TransferConfig], Tuple[Dataset, Dataset, Dataset, Tuple[str, ...]]]


def _default_dataset_builder(config: TransferConfig) -> Tuple[Dataset, Dataset, Dataset, Tuple[str, ...]]:
    """Download CIFAR-10 and return train/val/test subsets."""

    if not _TORCHVISION_AVAILABLE:  # pragma: no cover - used only without torchvision installed
        raise RuntimeError("torchvision is required for the default dataset builder")

    augment = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ]
    )
    eval_tfm = transforms.ToTensor()
    train_ds_full = datasets.CIFAR10(".data", train=True, download=True, transform=augment)
    eval_ds_full = datasets.CIFAR10(".data", train=True, download=True, transform=eval_tfm)
    test_ds = datasets.CIFAR10(".data", train=False, download=True, transform=eval_tfm)
    generator = torch.Generator().manual_seed(config.seed)
    subset_size = min(config.subset_size, len(train_ds_full))
    indices = torch.randperm(len(train_ds_full), generator=generator)[:subset_size]
    train_size = int(len(indices) * 0.8)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_ds = Subset(train_ds_full, train_indices.tolist())
    val_ds = Subset(eval_ds_full, val_indices.tolist())
    class_names = tuple(train_ds_full.classes)
    return train_ds, val_ds, test_ds, class_names


def build_dataloaders(
    config: TransferConfig,
    dataset_builder: Optional[DatasetBuilder] = None,
) -> Tuple[Dict[str, DataLoader], Tuple[str, ...]]:
    """Construct train/val/test dataloaders using the provided builder."""

    builder = dataset_builder or _default_dataset_builder
    train_ds, val_ds, test_ds, class_names = builder(config)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=len(train_ds) > config.batch_size,
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    return {"train": train_loader, "val": val_loader, "test": test_loader}, class_names


def _build_transfer_model(num_classes: int, config: TransferConfig) -> nn.Module:
    """Create a ResNet18 backbone with a new classifier head."""

    if not _TORCHVISION_AVAILABLE:  # pragma: no cover - handled by tests via custom builder
        raise RuntimeError("torchvision is required for the default transfer model")

    weights = None
    if config.use_pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT  # type: ignore[attr-defined]
        except AttributeError:  # older torchvision
            weights = None
    backbone = models.resnet18(weights=weights)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)
    if config.freeze_backbone:
        for name, param in backbone.named_parameters():
            if name.startswith("fc."):
                continue
            param.requires_grad = False
    return backbone


def _train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Train a model for a single epoch and return average loss."""

    model.train()
    total_loss = 0.0
    total_items = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * yb.size(0)
        total_items += yb.size(0)
    return total_loss / max(1, total_items)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: Iterable, device: torch.device) -> Dict[str, float]:
    """Compute loss and accuracy for a model on a dataset."""

    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_items = 0
    correct = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * yb.size(0)
        total_items += yb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
    accuracy = correct / max(1, total_items)
    return {"loss": total_loss / max(1, total_items), "accuracy": accuracy}


def train(
    config: TransferConfig = TransferConfig(),
    dataset_builder: Optional[DatasetBuilder] = None,
) -> TransferTrainingResult:
    """Train both baseline and transfer-learning models and compare metrics."""

    set_seed(config.seed)
    device = torch.device(config.device)
    loaders, class_names = build_dataloaders(config, dataset_builder=dataset_builder)

    num_classes = len(class_names)
    scratch_model = SimpleCNN(num_classes).to(device)
    if config.head_learning_rate is None:
        head_lr = config.learning_rate
    else:
        head_lr = config.head_learning_rate

    if _TORCHVISION_AVAILABLE:
        transfer_model = _build_transfer_model(num_classes, config).to(device)
    else:
        transfer_model = SimpleCNN(num_classes).to(device)

    scratch_optimizer = torch.optim.Adam(scratch_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    transfer_parameters = [p for p in transfer_model.parameters() if p.requires_grad]
    transfer_optimizer = torch.optim.Adam(transfer_parameters, lr=head_lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    writer = create_summary_writer(config.log_dir)

    best_metric = -math.inf
    best_epoch = -1
    best_state = None

    for epoch in range(config.max_epochs):
        scratch_loss = _train_one_epoch(scratch_model, loaders["train"], device, scratch_optimizer, criterion)
        transfer_loss = _train_one_epoch(transfer_model, loaders["train"], device, transfer_optimizer, criterion)
        scratch_val = evaluate(scratch_model, loaders["val"], device)
        transfer_val = evaluate(transfer_model, loaders["val"], device)
        if writer is not None:
            writer.add_scalars("loss/train", {"scratch": scratch_loss, "transfer": transfer_loss}, epoch)
            writer.add_scalars("accuracy/val", {"scratch": scratch_val["accuracy"], "transfer": transfer_val["accuracy"]}, epoch)
        if transfer_val["accuracy"] > best_metric:
            best_metric = transfer_val["accuracy"]
            best_epoch = epoch
            best_state = {"scratch": scratch_model.state_dict(), "transfer": transfer_model.state_dict()}

    if writer is not None:
        writer.close()

    if best_state is not None:
        scratch_model.load_state_dict(best_state["scratch"])
        transfer_model.load_state_dict(best_state["transfer"])

    scratch_metrics = evaluate(scratch_model, loaders["test"], device)
    transfer_metrics = evaluate(transfer_model, loaders["test"], device)
    return TransferTrainingResult(
        best_epoch=best_epoch,
        scratch_metrics=scratch_metrics,
        transfer_metrics=transfer_metrics,
        class_names=class_names,
    )


def main() -> None:  # pragma: no cover - CLI helper
    result = train()
    print("Best epoch", result.best_epoch)
    print("Scratch metrics", result.scratch_metrics)
    print("Transfer metrics", result.transfer_metrics)


if __name__ == "__main__":  # pragma: no cover
    main()
