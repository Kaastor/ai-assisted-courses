"""Temperature scaling for multiclass classifiers."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


@torch.inference_mode()
def collect_logits(model: nn.Module, loader: Iterable, device: torch.device) -> dict:
    logits = []
    labels = []
    model.eval()
    for xb, yb in loader:
        logits.append(model(xb.to(device)).detach())
        labels.append(yb.to(device))
    return {"logits": torch.cat(logits), "labels": torch.cat(labels)}


def fit_temperature(model: nn.Module, loader: Iterable, device: torch.device) -> float:
    """Fit a scalar temperature using validation data."""

    cached = collect_logits(model, loader, device)
    logits = cached["logits"].clone().detach()
    labels = cached["labels"].clone().detach()
    temperature = torch.nn.Parameter(torch.ones(1, device=device))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        temp = temperature.clamp(0.5, 5.0)
        loss = criterion(logits / temp, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().cpu().item())
