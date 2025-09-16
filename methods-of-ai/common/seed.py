"""Deterministic seeding utilities shared across labs."""

from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch and enable deterministic algorithms."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:  # pragma: no cover - DataLoader callback
    """Seed DataLoader workers for determinism."""

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


worker_init_fn: Callable[[int], None] = seed_worker
"""Alias to plug directly into ``DataLoader(worker_init_fn=...)``."""
