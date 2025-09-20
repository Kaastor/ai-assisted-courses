"""Per-student variant utilities for Lab 1 assignments.

This module generates small, synthetic datasets that are deterministic per
student. The `STUDENT_ID` environment variable (or `GITHUB_ACTOR` in CI)
controls the seed so each student receives a slightly different dataset
while keeping difficulty consistent across variants.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def get_student_id() -> str:
    """Return a student identifier from environment or a default value.

    Order of precedence:
    - `STUDENT_ID` (local override)
    - `GITHUB_ACTOR` (GitHub Classroom/Actions)
    - "local" (fallback)
    """

    return os.environ.get("STUDENT_ID") or os.environ.get("GITHUB_ACTOR") or "local"


def seed_from_student_id(student_id: str) -> int:
    """Derive a stable 32-bit seed from an arbitrary student id string."""

    h = hashlib.sha256(student_id.encode("utf-8")).digest()
    # Use first 4 bytes as an unsigned 32-bit int
    return int.from_bytes(h[:4], byteorder="big", signed=False)


@dataclass
class VariantConfig:
    seed: int
    n_samples: int
    n_features: int
    class_sep: float


def build_variant(student_id: Optional[str] = None) -> VariantConfig:
    """Build a numeric classification variant config for a student.

    The ranges are chosen to keep problem difficulty comparable across variants
    while ensuring uniqueness.
    """

    sid = student_id or get_student_id()
    seed = seed_from_student_id(sid)
    rng = np.random.default_rng(seed)
    # Keep sample size and dimensionality modest for CI speed
    n_samples = int(800 + (rng.integers(0, 6) * 40))  # 800..1000
    n_features = int(4 + (rng.integers(0, 3)))  # 4..6
    class_sep = float(1.6 + (rng.random() * 0.8))  # 1.6..2.4
    return VariantConfig(seed=seed, n_samples=n_samples, n_features=n_features, class_sep=class_sep)


def make_numeric_dataset(cfg: VariantConfig) -> pd.DataFrame:
    """Create a linearly-separable-ish binary classification dataset as a DataFrame.

    Columns: `x0..x{n_features-1}`, `y` (0/1)
    """

    rng = np.random.default_rng(cfg.seed)
    X = rng.normal(size=(cfg.n_samples, cfg.n_features)).astype(np.float32)
    # Random weight vector scaled by `class_sep` for clearer decision boundary
    w = rng.normal(size=(cfg.n_features,)).astype(np.float32)
    w /= (np.linalg.norm(w) + 1e-8)
    w *= cfg.class_sep
    # Additive noise
    noise = rng.normal(scale=0.5, size=(cfg.n_samples,)).astype(np.float32)
    logits = X @ w + noise
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs >= 0.5).astype(np.int64)
    data = {f"x{i}": X[:, i] for i in range(cfg.n_features)}
    data["y"] = y
    return pd.DataFrame(data)

