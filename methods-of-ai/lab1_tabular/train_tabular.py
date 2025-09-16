"""Training entrypoint for Lab 1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from common.seed import set_seed
from lab1_tabular.data import prepare_datasets
from lab1_tabular.eval import evaluate
from lab1_tabular.model import TabularNet


@dataclass
class TrainingConfig:
    """Configuration for the tabular training loop."""

    batch_size: int = 256
    max_epochs: int = 8
    patience: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_step: int = 2
    scheduler_gamma: float = 0.5
    seed: int = 42
    log_dir: Optional[Path] = Path("runs/lab1_tabular")
    device: str = "cpu"


@dataclass
class TrainingResult:
    best_epoch: int
    temperature: float
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]


def _collect_logits(model: nn.Module, loader, device: torch.device) -> Dict[str, torch.Tensor]:
    logits = []
    labels = []
    with torch.no_grad():
        model.eval()
        for cats, nums, ys in loader:
            logits.append(model(cats.to(device), nums.to(device)).detach())
            labels.append(ys.to(device))
    return {
        "logits": torch.cat(logits),
        "labels": torch.cat(labels),
    }


def calibrate_temperature(model: nn.Module, loader, device: torch.device) -> float:
    """Temperature scaling using LBFGS on validation logits."""

    cached = _collect_logits(model, loader, device)
    logits = cached["logits"]
    labels = cached["labels"]
    temperature = torch.nn.Parameter(torch.ones(1, device=device))
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        temp = temperature.clamp(1e-3, 10.0)
        loss = loss_fn(logits / temp, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().cpu().item())


def train(config: TrainingConfig = TrainingConfig()) -> TrainingResult:
    """Run the full training loop and return metrics."""

    set_seed(config.seed)
    device = torch.device(config.device)
    train_loader, val_loader, test_loader, metadata = prepare_datasets(
        batch_size=config.batch_size, seed=config.seed
    )
    model = TabularNet(metadata.categorical_cardinalities, metadata.numeric_features)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )
    loss_fn = nn.BCEWithLogitsLoss()

    writer: Optional[SummaryWriter] = None
    if config.log_dir is not None:
        config.log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(config.log_dir))

    best_metric = -math.inf
    best_epoch = -1
    best_state = None
    remaining_patience = config.patience

    for epoch in range(config.max_epochs):
        model.train()
        total_loss = 0.0
        total_items = 0
        for cats, nums, ys in train_loader:
            cats = cats.to(device)
            nums = nums.to(device)
            ys = ys.to(device)
            optimizer.zero_grad()
            logits = model(cats, nums)
            loss = loss_fn(logits, ys)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * ys.size(0)
            total_items += ys.size(0)
        scheduler.step()

        train_loss = total_loss / max(1, total_items)
        val_metrics = evaluate(model, val_loader, device)
        metric = val_metrics["roc_auc"]
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/roc_auc", metric, epoch)
            writer.add_scalar("val/f1", val_metrics["f1"], epoch)
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            remaining_patience = config.patience
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                break

    if writer is not None:
        writer.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    temperature = calibrate_temperature(model, val_loader, device)
    val_metrics = evaluate(model, val_loader, device, temperature=temperature)
    test_metrics = evaluate(model, test_loader, device, temperature=temperature)
    return TrainingResult(
        best_epoch=best_epoch,
        temperature=temperature,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )


def main() -> None:  # pragma: no cover - CLI wrapper
    result = train()
    print("Best epoch", result.best_epoch)
    print("Temperature", result.temperature)
    print("Validation metrics", result.val_metrics)
    print("Test metrics", result.test_metrics)


if __name__ == "__main__":  # pragma: no cover
    main()
