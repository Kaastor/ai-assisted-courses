"""Evaluation helpers for Lab 1."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from common.metrics import expected_calibration_error


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: torch.device,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """Evaluate a model and compute key metrics."""

    model.eval()
    logits: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    loss_fn = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        cats, nums, ys = batch
        cats = cats.to(device)
        nums = nums.to(device)
        ys = ys.to(device)
        batch_logits = model(cats, nums)
        total_loss += loss_fn(batch_logits, ys).item() * ys.size(0)
        total_samples += ys.size(0)
        logits.append(batch_logits.cpu())
        targets.append(ys.cpu())
    logits_cat = torch.cat(logits)
    targets_cat = torch.cat(targets)
    probs = torch.sigmoid(logits_cat / temperature).numpy()
    labels = targets_cat.numpy()
    metrics = {
        "loss": total_loss / max(1, total_samples),
        "accuracy": float(accuracy_score(labels, (probs >= 0.5).astype(int))),
        "f1": float(f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "ece": expected_calibration_error(probs, labels),
    }
    return metrics
