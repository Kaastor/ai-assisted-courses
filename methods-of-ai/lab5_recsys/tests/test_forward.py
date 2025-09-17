import torch

from lab5_recsys.train_mf import MatrixFactorization


def test_matrix_factorization_scores_per_pair():
    model = MatrixFactorization(num_users=5, num_items=7, embedding_dim=8)
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([1, 3, 5])
    scores = model(users, items)
    assert scores.shape == (3,)
