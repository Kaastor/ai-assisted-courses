"""Training loop for the Fashion-MNIST CNN lab."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from common.metrics import macro_f1, per_class_recall
from common.seed import set_seed
from lab2_vision.calibrate import fit_temperature


class SmallCNN(nn.Module):
    """Compact CNN suitable for CPU training."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass
class VisionConfig:
    batch_size: int = 128
    max_epochs: int = 6
    patience: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    subset_size: int = 12000
    seed: int = 42
    log_dir: Optional[Path] = Path("runs/lab2_vision")
    device: str = "cpu"


@dataclass
class VisionTrainingResult:
    best_epoch: int
    temperature: float
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]


def _transforms() -> Dict[str, transforms.Compose]:
    normalize = transforms.Normalize((0.2861,), (0.3530,))
    train_tfm = transforms.Compose(
        [
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tfm = transforms.Compose([transforms.ToTensor(), normalize])
    return {"train": train_tfm, "eval": eval_tfm}


def _make_dataloaders(config: VisionConfig) -> Dict[str, DataLoader]:
    tfms = _transforms()
    train_aug = datasets.FashionMNIST(".data", train=True, download=True, transform=tfms["train"])
    train_eval = datasets.FashionMNIST(".data", train=True, download=True, transform=tfms["eval"])
    test_ds = datasets.FashionMNIST(".data", train=False, download=True, transform=tfms["eval"])
    indices = torch.arange(config.subset_size)
    generator = torch.Generator().manual_seed(config.seed)
    perm = indices[torch.randperm(len(indices), generator=generator)]
    train_len = int(len(perm) * 0.85)
    train_indices = perm[:train_len].tolist()
    val_indices = perm[train_len:].tolist()
    train_ds = Subset(train_aug, train_indices)
    val_ds = Subset(train_eval, val_indices)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    return {"train": train_loader, "val": val_loader, "test": test_loader}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    logits_all = []
    labels_all = []
    total_loss = 0.0
    total_items = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * yb.size(0)
        total_items += yb.size(0)
        logits_all.append(logits.cpu())
        labels_all.append(yb.cpu())
    logits_cat = torch.cat(logits_all)
    labels_cat = torch.cat(labels_all)
    scaled_logits = logits_cat / temperature
    preds = scaled_logits.argmax(dim=1)
    accuracy = (preds == labels_cat).float().mean().item()
    macro = macro_f1(scaled_logits.numpy(), labels_cat.numpy())
    per_cls = per_class_recall(scaled_logits.numpy(), labels_cat.numpy())
    worst_recall = min(per_cls.values())
    return {
        "loss": total_loss / max(1, total_items),
        "accuracy": accuracy,
        "macro_f1": macro,
        "worst_class_recall": worst_recall,
    }


def train(config: VisionConfig = VisionConfig()) -> VisionTrainingResult:
    set_seed(config.seed)
    device = torch.device(config.device)
    loaders = _make_dataloaders(config)
    model = SmallCNN().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    writer: Optional[SummaryWriter] = None
    if config.log_dir is not None:
        config.log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(config.log_dir))

    best_metric = -math.inf
    best_state = None
    best_epoch = -1
    patience = config.patience

    for epoch in range(config.max_epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, yb in loaders["train"]:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            seen += yb.size(0)
        scheduler.step()

        train_loss = running_loss / max(1, seen)
        val_metrics = evaluate(model, loaders["val"], device)
        metric = val_metrics["accuracy"]
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
            writer.add_scalar("val/worst_class_recall", val_metrics["worst_class_recall"], epoch)
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = config.patience
        else:
            patience -= 1
            if patience <= 0:
                break

    if writer is not None:
        writer.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    temperature = fit_temperature(model, loaders["val"], device)
    val_metrics = evaluate(model, loaders["val"], device, temperature=temperature)
    test_metrics = evaluate(model, loaders["test"], device, temperature=temperature)
    return VisionTrainingResult(
        best_epoch=best_epoch,
        temperature=temperature,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )


def main() -> None:  # pragma: no cover
    result = train()
    print("Best epoch", result.best_epoch)
    print("Temperature", result.temperature)
    print("Validation", result.val_metrics)
    print("Test", result.test_metrics)


if __name__ == "__main__":  # pragma: no cover
    main()
