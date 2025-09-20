"""Student task stubs for Lab 1 assignments.

Only edit this file to complete Assignments 1–5. Read the TODO blocks for each
task. Keep implementations simple, readable, and robust, following PEP8 and the
course style in AGENTS.md. You may use numpy/pandas/torch but no extra deps.

Quick run:
- Local: `RUN_ASSIGNMENT_TESTS=1 poetry run pytest -q lab1_tabular/assignments/tests`
- CI sets `STUDENT_ID` for per‑student variants; locally you can set it too.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn


def split_dataframe(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assignment 1: 70/15/15 data split.

    Implement a deterministic split into (train, val, test) using the provided
    seed for shuffling. Reset indices on each output dataframe.

    Guidelines:
    - Use `df.sample(frac=1.0, random_state=seed)` to shuffle without mutation.
    - Compute counts: `n_train = int(0.7 * n)`, `n_val = int(0.15 * n)`;
      the remainder goes to test.
    - Slice the shuffled frame into contiguous blocks and `reset_index(drop=True)`.
    - Do not modify the input `df` in‑place.
    """

    # TODO(Student): implement the split as described above.
    raise NotImplementedError


def prepare_numeric_stats(train_df: pd.DataFrame, numeric_cols: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Assignment 2a: numeric stats.

    Compute column‑wise means and standard deviations on the TRAIN split only.
    Replace any zero standard deviation with 1.0 to avoid divide‑by‑zero later.

    Tips:
    - Use `to_numpy(dtype=np.float32)` for numeric stability and dtype.
    - Shapes must be `(len(numeric_cols),)`; cast to `np.float32`.
    - Do not clip or center here beyond mean/std computation.
    """

    # TODO(Student): implement computation of `means` and `stds` per column.
    raise NotImplementedError


def standardize_numeric(
    df: pd.DataFrame, numeric_cols: list[str], means: np.ndarray, stds: np.ndarray
) -> np.ndarray:
    """Assignment 2b: numeric standardization.

    Return an `np.ndarray` of shape `(N, D)` where `N = len(df)` and `D = len(numeric_cols)`
    with dtype float32, computed as `(values - means) / stds` using the provided
    train‑derived `means` and `stds`.

    Tips:
    - Extract with `df[numeric_cols].to_numpy(dtype=np.float32)`.
    - Broadcasting will handle `(N, D) - (D,)` → `(N, D)`.
    - Do not recompute stats here; use the given ones.
    """

    # TODO(Student): implement standardization using provided means/stds.
    raise NotImplementedError


def build_categorical_mapping(train_df: pd.DataFrame, categorical_cols: list[str]) -> dict[str, dict[str, int]]:
    """Assignment 3a: categorical mapping with unknown=0.

    For each column, map unique training values to indices starting at 1, and
    reserve 0 for unknown/unseen values at inference.

    Tips:
    - Ensure deterministic ordering (e.g., `sorted(train_df[col].unique())`).
    - Build `{column: {value: index}}` with indices `1..K`.
    - Do not include the 0 mapping explicitly; tests only check that 0 is unused.
    """

    # TODO(Student): implement the mapping dictionary.
    raise NotImplementedError


def encode_categoricals(df: pd.DataFrame, mapping: dict[str, dict[str, int]]) -> np.ndarray:
    """Assignment 3b: categorical encoding with unknown handling.

    Return an array of shape `(N, C)` where `C = len(mapping)` in the order of
    `mapping` keys. Use `0` when the value is not present in the mapping for the
    column.

    Tips:
    - Iterate columns in a stable order (e.g., for `col in mapping:`).
    - Use `dict.get(value, 0)` to fall back to unknown.
    - Return dtype `np.int64`.
    """

    # TODO(Student): implement encoding to an int64 ndarray.
    raise NotImplementedError


class SimpleMLP(nn.Module):
    """Assignment 4: small MLP that outputs logits.

    Build an MLP: for each hidden dim H: `Linear(prev, H) → ReLU → Dropout`,
    then a final `Linear(last_hidden, 1)` producing a single logit per row.

    Implementation notes:
    - Use `nn.Sequential` to assemble layers.
    - Apply dropout after ReLU; accept `dropout=0.0`.
    - Forward returns shape `(N,)` (via `.squeeze(-1)`) or `(N, 1)`; tests accept either.
    """

    def __init__(self, input_dim: int, hidden_dims: Iterable[int] = (64, 32), dropout: float = 0.2) -> None:
        super().__init__()
        # TODO(Student): build layers per spec into `self.net`.
        raise NotImplementedError

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # TODO(Student): implement a forward pass returning logits.
        raise NotImplementedError


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """Assignment 5: train for one epoch and return avg loss.

    Loop over batches of `(X, y)`:
    - Move tensors to `device`.
    - `optimizer.zero_grad()`, forward to logits, compute loss, `backward()`, `step()`.
    - Accumulate `loss.item() * batch_size`; divide by total samples at the end.
    - Put the model in train mode at the start with `model.train()`.
    """

    # TODO(Student): implement standard training epoch and average loss.
    raise NotImplementedError
