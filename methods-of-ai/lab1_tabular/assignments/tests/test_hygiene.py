from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lab1_tabular.assignments import student as S


def test_split_dataframe_does_not_mutate_input():
    df = pd.DataFrame({"x": list(range(20)), "y": [0] * 10 + [1] * 10})
    original = df.copy(deep=True)
    _ = S.split_dataframe(df, seed=123)
    pd.testing.assert_frame_equal(df, original)


def test_preprocessing_functions_do_not_mutate_input():
    df = pd.DataFrame({
        "x0": [1.0, 2.0, 3.0, 4.0],
        "x1": [10.0, 10.0, 10.0, 10.0],
        "cat": ["a", "b", "a", "c"],
        "y": [0, 1, 0, 1],
    })
    df_numeric = df[["x0", "x1"]].copy(deep=True)
    df_categorical = df[["cat"]].copy(deep=True)

    # Numeric stats and standardization should not modify inputs
    means, stds = S.prepare_numeric_stats(df, ["x0", "x1"])
    _ = S.standardize_numeric(df, ["x0", "x1"], means, stds)
    pd.testing.assert_frame_equal(df[["x0", "x1"]], df_numeric)

    # Categorical encoding should not modify inputs
    mapping = S.build_categorical_mapping(df, ["cat"])
    _ = S.encode_categoricals(df, mapping)
    pd.testing.assert_frame_equal(df[["cat"]], df_categorical)


class _SentinelModel(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(input_dim, 1)
        self.forward_calls = 0

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Must be in train mode during train_one_epoch
        assert self.training, "Model should be in training mode during train_one_epoch"
        self.forward_calls += 1
        return self.lin(X)


def test_train_epoch_updates_params_and_uses_train_mode():
    # Small synthetic dataset
    N, D = 64, 4
    X = torch.randn(N, D)
    y = (torch.randn(N) > 0).float()
    loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)

    model = _SentinelModel(D)
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss()

    before = [p.detach().clone() for p in model.parameters()]
    avg_loss = S.train_one_epoch(model, loader, device, opt, loss_fn)
    assert np.isfinite(avg_loss)
    assert model.forward_calls > 0

    after = list(model.parameters())
    changed = any(not torch.allclose(b, a.detach(), atol=0, rtol=0) for b, a in zip(before, after))
    assert changed, "Parameters should update during training (optimizer.step should be called)"


def _contains_sigmoid_calls(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr.lower() == "sigmoid":
                return True
            if isinstance(f, ast.Name) and f.id.lower() == "sigmoid":
                return True
        if isinstance(n, ast.Attribute) and n.attr == "Sigmoid":
            return True
        if isinstance(n, ast.Name) and n.id == "Sigmoid":
            return True
    return False


def test_no_sigmoid_in_model_or_train_loop():
    # Guardrail: logits should be used with BCEWithLogitsLoss
    src = Path("lab1_tabular/assignments/student.py").read_text()
    tree = ast.parse(src)

    found_bad = []
    for node in tree.body:
        # class SimpleMLP
        if isinstance(node, ast.ClassDef) and node.name == "SimpleMLP":
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if sub.name in {"__init__", "forward"}:
                        if _contains_sigmoid_calls(sub):
                            found_bad.append(f"SimpleMLP.{sub.name}")
        # def train_one_epoch
        if isinstance(node, ast.FunctionDef) and node.name == "train_one_epoch":
            if _contains_sigmoid_calls(node):
                found_bad.append("train_one_epoch")

    assert not found_bad, (
        "Avoid sigmoid in model forward/init and train_one_epoch; use logits with "
        "BCEWithLogitsLoss. Found in: " + ", ".join(found_bad)
    )

