from __future__ import annotations

import torch
from torch import nn

from lab1_tabular.assignments import student as S


def test_simple_mlp_structure_and_forward():
    model = S.SimpleMLP(input_dim=10, hidden_dims=(16, 8), dropout=0.1)
    assert isinstance(model, nn.Module)
    X = torch.randn(4, 10)
    out = model(X)
    assert out.shape[0] == 4
    # Count dropout layers equals number of hidden layers when dropout > 0
    dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    assert len(dropouts) == 2
    # Contains ReLU activations
    relus = [m for m in model.modules() if isinstance(m, nn.ReLU)]
    assert len(relus) == 2

