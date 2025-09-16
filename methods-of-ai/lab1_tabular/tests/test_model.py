import torch

from lab1_tabular.model import TabularNet


def test_tabular_net_output_shape():
    model = TabularNet([5, 10], numeric_dim=3)
    cats = torch.zeros(4, 2, dtype=torch.long)
    nums = torch.zeros(4, 3)
    logits = model(cats, nums)
    assert logits.shape == (4,)
