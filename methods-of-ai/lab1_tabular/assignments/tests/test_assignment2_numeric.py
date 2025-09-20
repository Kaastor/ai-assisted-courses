from __future__ import annotations

import numpy as np
import pandas as pd

from lab1_tabular.assignments import student as S


def test_numeric_standardization_mean_zero_std_one():
    train = pd.DataFrame({
        "x0": [1.0, 2.0, 3.0, 4.0],
        "x1": [10.0, 10.0, 10.0, 10.0],  # zero variance
        "y": [0, 1, 0, 1],
    })
    numeric_cols = ["x0", "x1"]
    means, stds = S.prepare_numeric_stats(train, numeric_cols)
    assert means.shape == (2,)
    assert stds.shape == (2,)
    assert means.dtype == np.float32
    assert stds.dtype == np.float32
    assert stds[1] == 1.0  # zero-variance handled

    full = pd.concat([train, train], ignore_index=True)
    Z = S.standardize_numeric(full, numeric_cols, means, stds)
    assert Z.shape == (len(full), 2)
    # Check train slice stats approximately
    Z_train = Z[: len(train)]
    assert np.allclose(Z_train.mean(axis=0), np.array([0.0, 0.0]), atol=1e-6)
    assert np.allclose(Z_train.std(axis=0), np.array([1.0, 1.0]), atol=1e-6)

