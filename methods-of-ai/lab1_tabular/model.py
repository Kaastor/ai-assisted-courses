"""Model definition for the tabular classification lab."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


def default_embedding_dim(cardinality: int) -> int:
    """Heuristic for embedding sizes."""

    return min(50, max(4, int(round(cardinality ** 0.5 * 2))))


class TabularNet(nn.Module):
    """MLP with categorical embeddings and numeric inputs."""

    def __init__(
        self,
        categorical_cardinalities: Sequence[int],
        numeric_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, default_embedding_dim(cardinality))
            for cardinality in categorical_cardinalities
        )
        total_dim = sum(emb.embedding_dim for emb in self.embeddings) + numeric_dim
        layers = []
        dims = [total_dim, *hidden_dims, 1]
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        embs = []
        for idx, emb in enumerate(self.embeddings):
            embs.append(emb(categorical[:, idx]))
        features = torch.cat(embs + [numeric], dim=1)
        logits = self.mlp(features).squeeze(1)
        return logits
