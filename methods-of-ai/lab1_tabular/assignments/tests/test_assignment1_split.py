from __future__ import annotations

import pandas as pd

from lab1_tabular.assignments import student as S


def test_split_dataframe_shapes_and_reproducibility():
    # Build a simple synthetic dataframe
    n = 100
    df = pd.DataFrame({
        "x": range(n),
        "y": [0] * (n // 2) + [1] * (n - n // 2),
    })

    t1, v1, te1 = S.split_dataframe(df, seed=123)
    t2, v2, te2 = S.split_dataframe(df, seed=123)

    assert len(t1) == int(0.7 * n)
    assert len(v1) == int(0.15 * n)
    assert len(te1) == n - len(t1) - len(v1)
    # Reproducible with same seed
    pd.testing.assert_frame_equal(t1, t2)
    pd.testing.assert_frame_equal(v1, v2)
    pd.testing.assert_frame_equal(te1, te2)

