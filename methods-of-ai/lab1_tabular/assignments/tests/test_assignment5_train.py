from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lab1_tabular.assignments import student as S
from lab1_tabular.assignments.variant import build_variant, make_numeric_dataset


def _make_loader_from_df(df, batch_size=64):
    feature_cols = [c for c in df.columns if c != "y"]
    X = torch.tensor(df[feature_cols].to_numpy(dtype=np.float32))
    y = torch.tensor(df["y"].to_numpy(dtype=np.float32))
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def test_train_one_epoch_reduces_loss():
    # Per-student variant
    cfg = build_variant(os.environ.get("STUDENT_ID") or os.environ.get("GITHUB_ACTOR"))
    df = make_numeric_dataset(cfg)

    loader = _make_loader_from_df(df, batch_size=64)
    model = S.SimpleMLP(input_dim=cfg.n_features, hidden_dims=(64, 32), dropout=0.1)
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    # Measure initial loss (no training)
    model.eval()
    with torch.no_grad():
        losses = []
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X).squeeze(-1)
            losses.append(loss_fn(logits, y).item())
        initial_loss = float(np.mean(losses))

    # Train for a few epochs and expect a substantial reduction
    for _ in range(6):
        avg_loss = S.train_one_epoch(model, loader, device, optimizer, loss_fn)

    assert avg_loss < initial_loss * 0.7

