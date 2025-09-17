import numpy as np

from lab4_timeseries.train_gru import WindowDataset


def test_window_dataset_shapes():
    series = np.arange(40, dtype=np.float32)
    hours = np.tile(np.arange(24), 40 // 24 + 1)[: series.size]
    ds = WindowDataset(series, hours, window=5)
    x, y, h = ds[0]
    assert x.shape == (5, 1)
    assert y.shape == ()
    assert 0 <= int(h) < 24
