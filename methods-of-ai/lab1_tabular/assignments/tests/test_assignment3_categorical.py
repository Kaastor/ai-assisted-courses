from __future__ import annotations

import numpy as np
import pandas as pd

from lab1_tabular.assignments import student as S


def test_categorical_mapping_and_encoding_with_unknown():
    train = pd.DataFrame({
        "color": ["red", "green", "blue", "green"],
        "shape": ["square", "circle", "triangle", "circle"],
    })
    mapping = S.build_categorical_mapping(train, ["color", "shape"])
    assert set(mapping.keys()) == {"color", "shape"}
    # Index 0 must be reserved for unknown
    assert 0 not in mapping["color"].values()
    assert 0 not in mapping["shape"].values()

    # Encode a frame with an unknown category
    frame = pd.DataFrame({
        "color": ["red", "cyan"],  # 'cyan' is unknown
        "shape": ["triangle", "hexagon"],  # 'hexagon' is unknown
    })
    enc = S.encode_categoricals(frame, mapping)
    assert enc.shape == (2, 2)
    assert enc.dtype == np.int64
    # First row known values must be non-zero, second row has zero for unknowns
    assert enc[0, 0] != 0 and enc[0, 1] != 0
    assert enc[1, 0] == 0 and enc[1, 1] == 0

