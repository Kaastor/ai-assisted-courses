"""Training entrypoint for Lab 3 (SMS spam detection)."""

from __future__ import annotations

import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import urllib.request

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from common.metrics import binary_classification_metrics, expected_calibration_error
from common.seed import set_seed
from common.viz import plot_pr_curve
from lab3_text.vocab import Vocab, build_char_vocab, build_word_vocab

DATA_DIR = Path(".data") / "sms_spam"
DATA_ZIP = DATA_DIR / "smsspamcollection.zip"
DATA_FILE = DATA_DIR / "SMSSpamCollection"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

LABEL_MAP = {"ham": 0, "spam": 1}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def ensure_dataset() -> None:
    """Download and extract the SMS Spam Collection dataset if missing."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        if not DATA_ZIP.exists():
            with urllib.request.urlopen(DATA_URL) as response:
                DATA_ZIP.write_bytes(response.read())
        with zipfile.ZipFile(DATA_ZIP, "r") as zf:
            zf.extractall(DATA_DIR)


def load_dataframe() -> pd.DataFrame:
    """Load the dataset into a DataFrame with ``label`` and ``text`` columns."""

    ensure_dataset()
    df = pd.read_csv(DATA_FILE, sep="\t", header=None, names=["label", "text"], encoding="latin-1")
    df["label"] = df["label"].map(LABEL_MAP)
    return df


def tokenize(text: str) -> List[str]:
    """Lowercase tokenization with alphanumeric word pieces."""

    tokens = TOKEN_PATTERN.findall(text.lower())
    return tokens or ["<empty>"]


def build_threshold(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Select the validation threshold that maximises F1."""

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = np.zeros_like(precision)
    for i in range(len(precision)):
        denom = precision[i] + recall[i]
        if denom > 0:
            f1_scores[i] = 2 * precision[i] * recall[i] / denom
    best_index = int(np.argmax(f1_scores))
    # precision_recall_curve omits threshold for the last point; guard accordingly
    threshold = float(thresholds[best_index]) if best_index < len(thresholds) else 0.5
    pr_curve = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1_scores.tolist(),
    }
    return threshold, pr_curve


class SmsDataset(Dataset):
    """Dataset returning tokenized SMS messages."""

    def __init__(self, texts: Iterable[str], labels: Iterable[int], word_vocab: Vocab, char_vocab: Vocab):
        self.texts = list(texts)
        self.labels = list(labels)
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        word_tokens = tokenize(text)
        words = torch.tensor(self.word_vocab.encode(word_tokens), dtype=torch.long)
        chars = torch.tensor(self.char_vocab.encode(list(text.lower())), dtype=torch.long)
        label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return words, chars, label


def collate_batch(batch: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    word_tokens: List[torch.Tensor] = []
    char_tokens: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    word_offsets = [0]
    char_offsets = [0]
    for words, chars, label in batch:
        word_tokens.append(words)
        char_tokens.append(chars)
        labels.append(label)
        word_offsets.append(words.numel())
        char_offsets.append(chars.numel())
    word_offsets = torch.tensor(np.cumsum(word_offsets[:-1]), dtype=torch.long)
    char_offsets = torch.tensor(np.cumsum(char_offsets[:-1]), dtype=torch.long)
    word_tensor = torch.cat(word_tokens)
    char_tensor = torch.cat(char_tokens)
    label_tensor = torch.stack(labels)
    return word_tensor, word_offsets, char_tensor, char_offsets, label_tensor


class HybridBagModel(nn.Module):
    """EmbeddingBag-based classifier with character fallback."""

    def __init__(self, word_vocab: int, char_vocab: int, word_dim: int = 64, char_dim: int = 32, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.word_emb = nn.EmbeddingBag(word_vocab, word_dim, mode="mean")
        self.char_emb = nn.EmbeddingBag(char_vocab, char_dim, mode="mean")
        nn.init.normal_(self.word_emb.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.char_emb.weight, mean=0.0, std=0.1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(word_dim + char_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, word_tokens: torch.Tensor, word_offsets: torch.Tensor, char_tokens: torch.Tensor, char_offsets: torch.Tensor) -> torch.Tensor:
        word_vec = self.word_emb(word_tokens, word_offsets)
        char_vec = self.char_emb(char_tokens, char_offsets)
        features = torch.cat([word_vec, char_vec], dim=1)
        out = self.fc(self.dropout(features))
        return out.squeeze(1)


@dataclass
class TrainingConfig:
    batch_size: int = 128
    max_epochs: int = 6
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    min_freq: int = 2
    max_vocab: int = 30000
    char_vocab: int = 256
    seed: int = 42
    log_dir: Optional[Path] = Path("runs/lab3_text")
    artifacts_dir: Optional[Path] = Path("artifacts/lab3_text")
    device: str = "cpu"


@dataclass
class TrainingResult:
    best_epoch: int
    threshold: float
    vocab_size: int
    char_vocab_size: int
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    pr_curve_path: Optional[Path]


def prepare_loaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], float]:
    df = load_dataframe()
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=config.seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=config.seed)
    word_vocab = build_word_vocab(train_df["text"], tokenize, min_freq=config.min_freq, max_size=config.max_vocab)
    char_vocab = build_char_vocab(train_df["text"], max_size=config.char_vocab)
    train_dataset = SmsDataset(train_df["text"], train_df["label"], word_vocab, char_vocab)
    val_dataset = SmsDataset(val_df["text"], val_df["label"], word_vocab, char_vocab)
    test_dataset = SmsDataset(test_df["text"], test_df["label"], word_vocab, char_vocab)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
        generator=generator,
    )
    eval_loader = lambda ds: DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_batch, num_workers=0)
    val_loader = eval_loader(val_dataset)
    test_loader = eval_loader(test_dataset)
    pos_weight = float((train_df["label"] == 0).sum() / max(1, (train_df["label"] == 1).sum()))
    metadata = {
        "word_vocab": len(word_vocab),
        "char_vocab": len(char_vocab),
    }
    return train_loader, val_loader, test_loader, metadata, pos_weight


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probs: List[float] = []
    labels: List[int] = []
    model.eval()
    with torch.no_grad():
        for word_tokens, word_offsets, char_tokens, char_offsets, yb in loader:
            logits = model(
                word_tokens.to(device),
                word_offsets.to(device),
                char_tokens.to(device),
                char_offsets.to(device),
            )
            probs.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(yb.cpu().tolist())
    metrics = binary_classification_metrics(probs, labels, threshold=threshold)
    metrics_dict = {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "roc_auc": metrics.roc_auc,
        "ece": expected_calibration_error(probs, labels),
    }
    return metrics_dict


def train(config: TrainingConfig = TrainingConfig()) -> TrainingResult:
    set_seed(config.seed)
    device = torch.device(config.device)
    train_loader, val_loader, test_loader, metadata, pos_weight = prepare_loaders(config)
    model = HybridBagModel(
        word_vocab=metadata["word_vocab"],
        char_vocab=metadata["char_vocab"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    writer: Optional[SummaryWriter] = None
    if config.log_dir is not None:
        config.log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(config.log_dir))

    best_metric = -math.inf
    best_epoch = -1
    best_state = None

    for epoch in range(config.max_epochs):
        model.train()
        running_loss = 0.0
        count = 0
        for batch in train_loader:
            word_tokens, word_offsets, char_tokens, char_offsets, yb = batch
            word_tokens = word_tokens.to(device)
            word_offsets = word_offsets.to(device)
            char_tokens = char_tokens.to(device)
            char_offsets = char_offsets.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(word_tokens, word_offsets, char_tokens, char_offsets)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            count += yb.size(0)
        val_metrics = evaluate(model, val_loader, device)
        if writer is not None:
            writer.add_scalar("train/loss", running_loss / max(1, count), epoch)
            writer.add_scalar("val/macro_f1", val_metrics["f1"], epoch)
            writer.add_scalar("val/roc_auc", val_metrics["roc_auc"], epoch)
        if val_metrics["f1"] > best_metric:
            best_metric = val_metrics["f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if writer is not None:
        writer.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Determine threshold via PR curve on validation set
    probs: List[float] = []
    labels: List[int] = []
    model.eval()
    with torch.no_grad():
        for word_tokens, word_offsets, char_tokens, char_offsets, yb in val_loader:
            logits = model(
                word_tokens.to(device),
                word_offsets.to(device),
                char_tokens.to(device),
                char_offsets.to(device),
            )
            probs.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(yb.cpu().tolist())
    threshold, pr_curve = build_threshold(np.asarray(labels), np.asarray(probs))

    pr_curve_path: Optional[Path] = None
    if config.artifacts_dir is not None:
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        pr_curve_path = config.artifacts_dir / "precision_recall.png"
        plot_pr_curve(pr_curve["recall"], pr_curve["precision"], pr_curve_path)

    val_metrics = evaluate(model, val_loader, device, threshold=threshold)
    test_metrics = evaluate(model, test_loader, device, threshold=threshold)
    return TrainingResult(
        best_epoch=best_epoch,
        threshold=threshold,
        vocab_size=metadata["word_vocab"],
        char_vocab_size=metadata["char_vocab"],
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        pr_curve_path=pr_curve_path,
    )


def main() -> None:  # pragma: no cover - CLI helper
    result = train()
    print("Best epoch", result.best_epoch)
    print("Threshold", result.threshold)
    print("Validation metrics", result.val_metrics)
    print("Test metrics", result.test_metrics)
    if result.pr_curve_path:
        print("Precision-recall curve saved to", result.pr_curve_path)


if __name__ == "__main__":  # pragma: no cover
    main()
