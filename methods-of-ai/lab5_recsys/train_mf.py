"""Training entrypoint for Lab 5 (neural matrix factorisation)."""

from __future__ import annotations

import math
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from common.metrics import precision_recall_at_k
from common.seed import set_seed

DATA_DIR = Path(".data") / "movielens"
DATA_ZIP = DATA_DIR / "ml-100k.zip"
DATA_FOLDER = DATA_DIR / "ml-100k"
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def ensure_dataset() -> None:
    """Download and extract the MovieLens 100K dataset if required."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FOLDER.exists():
        if not DATA_ZIP.exists():
            with urllib.request.urlopen(DATA_URL) as response:
                DATA_ZIP.write_bytes(response.read())
        with zipfile.ZipFile(DATA_ZIP, "r") as zf:
            zf.extractall(DATA_DIR)


def load_interactions(
    min_user_interactions: int = 20, min_item_interactions: int = 20
) -> Tuple[pd.DataFrame, int, int]:
    """Load implicit-feedback interactions (ratings ≥ 4) with filtering."""

    ensure_dataset()
    udata = DATA_FOLDER / "u.data"
    df = pd.read_csv(udata, sep="\t", names=["user", "item", "rating", "timestamp"], engine="python")
    df = df[df["rating"] >= 4].copy()
    df["user"] = df["user"].astype(int) - 1
    df["item"] = df["item"].astype(int) - 1
    if min_user_interactions > 0:
        user_counts = df["user"].value_counts()
        keep_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df["user"].isin(keep_users)]
    if min_item_interactions > 0:
        item_counts = df["item"].value_counts()
        keep_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df["item"].isin(keep_items)]
    user_mapping = {old: idx for idx, old in enumerate(sorted(df["user"].unique()))}
    item_mapping = {old: idx for idx, old in enumerate(sorted(df["item"].unique()))}
    df["user"] = df["user"].map(user_mapping)
    df["item"] = df["item"].map(item_mapping)
    df = df.sort_values(["user", "timestamp"])
    return df, len(user_mapping), len(item_mapping)


def split_leave_last(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leave the last two interactions per user for validation/test."""

    train_parts: List[pd.DataFrame] = []
    val_rows: List[pd.Series] = []
    test_rows: List[pd.Series] = []
    for _, group in df.groupby("user"):
        if len(group) <= 2:
            continue
        train_parts.append(group.iloc[:-2])
        val_rows.append(group.iloc[-2])
        test_rows.append(group.iloc[-1])
    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    return train_df, val_df, test_df


class InteractionDataset(Dataset):
    """Dataset of implicit positive interactions."""

    def __init__(self, users: Iterable[int], items: Iterable[int]):
        self.users = torch.tensor(list(users), dtype=torch.long)
        self.items = torch.tensor(list(items), dtype=torch.long)

    def __len__(self) -> int:
        return self.users.numel()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx]


class MatrixFactorization(nn.Module):
    """Dot-product recommender with learned user/item embeddings."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_vecs = self.user_embeddings(users)
        item_vecs = self.item_embeddings(items)
        dot = torch.sum(user_vecs * item_vecs, dim=1)
        mlp_input = torch.cat([user_vecs, item_vecs], dim=1)
        mlp_score = self.mlp(mlp_input).squeeze(1)
        scores = (
            dot
            + self.user_bias(users).squeeze(1)
            + self.item_bias(items).squeeze(1)
            + self.global_bias
            + mlp_score
        )
        return scores

    def score_all_items(self, user_index: int, device: torch.device) -> torch.Tensor:
        user = torch.tensor([user_index], dtype=torch.long, device=device)
        user_vec = self.user_embeddings(user)
        items = torch.arange(self.item_embeddings.num_embeddings, dtype=torch.long, device=device)
        item_vec = self.item_embeddings(items)
        dot = torch.matmul(item_vec, user_vec.squeeze(0))
        user_expand = user_vec.expand(items.size(0), -1)
        mlp_input = torch.cat([user_expand, item_vec], dim=1)
        mlp_score = self.mlp(mlp_input).squeeze(1)
        scores = (
            dot
            + self.user_bias(user).squeeze(0)
            + self.item_bias(items).squeeze(1)
            + self.global_bias
            + mlp_score
        )
        return scores


@dataclass
class TrainingConfig:
    embedding_dim: int = 64
    batch_size: int = 2048
    negatives_per_positive: int = 5
    max_epochs: int = 8
    learning_rate: float = 5e-3
    weight_decay: float = 1e-5
    k_eval: int = 10
    min_user_interactions: int = 20
    min_item_interactions: int = 20
    eval_candidates: int = 30
    seed: int = 42
    log_dir: Optional[Path] = Path("runs/lab5_recsys")
    artifacts_dir: Optional[Path] = Path("artifacts/lab5_recsys")
    device: str = "cpu"


@dataclass
class TrainingResult:
    best_epoch: int
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    popularity_baseline: Dict[str, float]
    n_users: int
    n_items: int


def build_positive_sets(df: pd.DataFrame) -> Dict[int, Set[int]]:
    positives: Dict[int, Set[int]] = defaultdict(set)
    for row in df.itertuples(index=False):
        positives[int(row.user)].add(int(row.item))
    return positives


def generate_negatives(
    users: torch.Tensor,
    neg_ratio: int,
    num_items: int,
    positive_sets: Dict[int, Set[int]],
) -> torch.Tensor:
    negatives: List[int] = []
    for user in users.cpu().tolist():
        positives = positive_sets.get(int(user), set())
        for _ in range(neg_ratio):
            item = int(torch.randint(num_items, (1,)).item())
            while item in positives:
                item = int(torch.randint(num_items, (1,)).item())
            negatives.append(item)
    return torch.tensor(negatives, dtype=torch.long)


def precision_recall_metrics(
    model: MatrixFactorization,
    eval_sets: Dict[int, Set[int]],
    k: int,
    num_items: int,
    device: torch.device,
    train_sets: Optional[Dict[int, Set[int]]] = None,
    eval_candidates: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    precisions: List[float] = []
    recalls: List[float] = []
    model.eval()
    with torch.no_grad():
        for user, positives in eval_sets.items():
            if not positives:
                continue
            scores = model.score_all_items(user, device).cpu().numpy()
            mask_items: Set[int] = set()
            if train_sets is not None and user in train_sets:
                mask_items.update(train_sets[user])
            if eval_candidates is not None and eval_candidates > 0:
                generator = rng or np.random.default_rng()
                available = [item for item in range(num_items) if item not in mask_items and item not in positives]
                sample_size = max(0, eval_candidates - len(positives))
                if sample_size > len(available):
                    sample_size = len(available)
                sampled = (
                    set(generator.choice(available, size=sample_size, replace=False).tolist())
                    if sample_size > 0
                    else set()
                )
                candidate_items = set(positives).union(sampled)
                allowed_mask = np.full(num_items, -1e9, dtype=scores.dtype)
                for item in candidate_items:
                    allowed_mask[item] = scores[item]
                scores = allowed_mask
            if mask_items:
                for item in mask_items:
                    if 0 <= item < scores.shape[0]:
                        scores[item] = -1e9
            precision, recall = precision_recall_at_k(scores, positives, k)
            precisions.append(precision)
            recalls.append(recall)
    return {
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
    }


def popularity_metrics(
    eval_sets: Dict[int, Set[int]],
    train_sets: Dict[int, Set[int]],
    popularity_ranking: List[int],
    k: int,
) -> Dict[str, float]:
    precisions: List[float] = []
    recalls: List[float] = []
    for user, positives in eval_sets.items():
        available = [item for item in popularity_ranking if item not in train_sets.get(user, set())]
        topk = available[:k]
        precision = len(set(topk).intersection(positives)) / max(1, k)
        recall = len(set(topk).intersection(positives)) / max(1, len(positives))
        precisions.append(precision)
        recalls.append(recall)
    return {
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
    }


def train(config: TrainingConfig = TrainingConfig()) -> TrainingResult:
    set_seed(config.seed)
    device = torch.device(config.device)
    df, num_users, num_items = load_interactions(
        min_user_interactions=config.min_user_interactions,
        min_item_interactions=config.min_item_interactions,
    )
    train_df, val_df, test_df = split_leave_last(df)

    train_dataset = InteractionDataset(train_df["user"], train_df["item"])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    train_sets = build_positive_sets(train_df)
    val_sets = build_positive_sets(val_df)
    test_sets = build_positive_sets(test_df)
    popularity_counts = train_df["item"].value_counts().sort_values(ascending=False)
    popularity_ranking = popularity_counts.index.tolist()

    model = MatrixFactorization(num_users, num_items, embedding_dim=config.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    pos_weight = torch.tensor([float(config.negatives_per_positive)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    writer: Optional[SummaryWriter] = None
    if config.log_dir is not None:
        config.log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(config.log_dir))

    best_metric = -math.inf
    best_epoch = -1
    best_state = None
    eval_rng = np.random.default_rng(config.seed)

    for epoch in range(config.max_epochs):
        model.train()
        running_loss = 0.0
        count = 0
        for users, items in train_loader:
            users = users.to(device)
            items = items.to(device)
            positives = torch.ones(users.size(0), device=device)
            neg_items = generate_negatives(users, config.negatives_per_positive, num_items, train_sets).to(device)
            neg_users = users.repeat_interleave(config.negatives_per_positive)
            negatives = torch.zeros(neg_users.size(0), device=device)

            optimizer.zero_grad()
            pos_scores = model(users, items)
            neg_scores = model(neg_users, neg_items)
            scores = torch.cat([pos_scores, neg_scores], dim=0)
            labels = torch.cat([positives, negatives], dim=0)
            loss = loss_fn(scores, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * (positives.size(0) + negatives.size(0))
            count += positives.size(0) + negatives.size(0)

        val_metrics = precision_recall_metrics(
            model,
            val_sets,
            config.k_eval,
            num_items,
            device,
            train_sets=train_sets,
            eval_candidates=config.eval_candidates,
            rng=eval_rng,
        )
        if writer is not None:
            writer.add_scalar("train/loss", running_loss / max(1, count), epoch)
            writer.add_scalar("val/precision@k", val_metrics[f"precision@{config.k_eval}"], epoch)
        score = val_metrics[f"recall@{config.k_eval}"]
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if writer is not None:
        writer.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = precision_recall_metrics(
        model,
        val_sets,
        config.k_eval,
        num_items,
        device,
        train_sets=train_sets,
        eval_candidates=config.eval_candidates,
        rng=np.random.default_rng(config.seed + 1),
    )
    test_metrics = precision_recall_metrics(
        model,
        test_sets,
        config.k_eval,
        num_items,
        device,
        train_sets=train_sets,
        eval_candidates=config.eval_candidates,
        rng=np.random.default_rng(config.seed + 2),
    )
    pop_baseline = popularity_metrics(test_sets, train_sets, popularity_ranking, config.k_eval)

    if config.artifacts_dir is not None:
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        factors_path = config.artifacts_dir / "embeddings.pt"
        torch.save({"user_embeddings": model.user_embeddings.weight.cpu(), "item_embeddings": model.item_embeddings.weight.cpu()}, factors_path)

    return TrainingResult(
        best_epoch=best_epoch,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        popularity_baseline=pop_baseline,
        n_users=num_users,
        n_items=num_items,
    )


def main() -> None:  # pragma: no cover - CLI helper
    result = train()
    print("Best epoch", result.best_epoch)
    print("Validation metrics", result.val_metrics)
    print("Test metrics", result.test_metrics)
    print("Popularity baseline", result.popularity_baseline)
    print("Users", result.n_users, "Items", result.n_items)


if __name__ == "__main__":  # pragma: no cover
    main()
