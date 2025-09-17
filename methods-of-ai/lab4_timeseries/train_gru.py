"""Training entrypoint for Lab 4 (electricity load forecasting)."""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from common.seed import set_seed

DATA_DIR = Path(".data") / "electricity_load"
DATA_ZIP = DATA_DIR / "LD2011_2014.txt.zip"
DATA_FILE = DATA_DIR / "LD2011_2014.txt"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"


def ensure_dataset() -> None:
    """Download and extract the Electricity Load Diagrams dataset if needed."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        if not DATA_ZIP.exists():
            with urllib.request.urlopen(DATA_URL) as response:
                DATA_ZIP.write_bytes(response.read())
        with zipfile.ZipFile(DATA_ZIP, "r") as zf:
            zf.extractall(DATA_DIR)


def load_series(column: str = "MT_016") -> pd.Series:
    """Load an individual client's series and resample to hourly frequency."""

    ensure_dataset()
    df = pd.read_csv(
        DATA_FILE,
        sep=";",
        index_col=0,
        parse_dates=[0],
        low_memory=False,
        decimal=",",
    )
    series = df[column].resample("1h").mean().dropna()
    return series


class WindowDataset(Dataset):
    """Sliding window dataset for sequence-to-one forecasting."""

    def __init__(self, values: np.ndarray, hours: np.ndarray, window: int):
        self.inputs: torch.Tensor
        self.targets: torch.Tensor
        self.hours: torch.Tensor
        xs: List[np.ndarray] = []
        ys: List[float] = []
        hs: List[int] = []
        for idx in range(window, len(values)):
            xs.append(values[idx - window : idx])
            ys.append(values[idx])
            hs.append(int(hours[idx]))
        array_x = np.stack(xs, axis=0)
        array_y = np.stack(ys, axis=0)
        self.inputs = torch.tensor(array_x, dtype=torch.float32).unsqueeze(-1)
        self.targets = torch.tensor(array_y, dtype=torch.float32)
        self.hours = torch.tensor(hs, dtype=torch.int64)

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx], self.hours[idx]


class GRUForecaster(nn.Module):
    """GRU-based forecaster mapping windows to next-step predictions."""

    def __init__(self, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.GRU(1, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(inputs)
        representation = hidden[-1]
        out = self.fc(representation)
        return out.squeeze(1)


@dataclass
class TrainingConfig:
    window_size: int = 24
    batch_size: int = 128
    max_epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_size: int = 32
    num_layers: int = 1
    seed: int = 42
    log_dir: Optional[Path] = Path("runs/lab4_timeseries")
    artifacts_dir: Optional[Path] = Path("artifacts/lab4_timeseries")
    device: str = "cpu"
    series_column: str = "MT_016"


@dataclass
class TrainingResult:
    best_epoch: int
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    naive_mae: float
    improvement_pct: float
    hourly_mae: Dict[int, float]
    forecast_path: Optional[Path]


def split_series(series: pd.Series, window: int) -> Tuple[WindowDataset, WindowDataset, WindowDataset, float, float]:
    values = series.to_numpy(dtype=np.float32)
    hours = series.index.hour.to_numpy()
    n = len(series)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train_vals = values[:n_train]
    val_vals = values[n_train : n_train + n_val]
    test_vals = values[n_train + n_val :]
    train_hours = hours[:n_train]
    val_hours = hours[n_train : n_train + n_val]
    test_hours = hours[n_train + n_val :]
    mean = float(train_vals.mean())
    std = float(train_vals.std())
    if std == 0.0:
        std = 1.0

    def norm(x: np.ndarray) -> np.ndarray:
        return (x - mean) / std

    train_ds = WindowDataset(norm(train_vals), train_hours, window)
    val_ds = WindowDataset(norm(np.concatenate([train_vals[-window:], val_vals])), np.concatenate([train_hours[-window:], val_hours]), window)
    test_ds = WindowDataset(
        norm(np.concatenate([val_vals[-window:], test_vals])),
        np.concatenate([val_hours[-window:], test_hours]),
        window,
    )
    return train_ds, val_ds, test_ds, mean, std


def prepare_loaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader, DataLoader, float, float]:
    series = load_series(config.series_column)
    train_ds, val_ds, test_ds, mean, std = split_series(series, config.window_size)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True)
    eval_loader = lambda ds: DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    val_loader = eval_loader(val_ds)
    test_loader = eval_loader(test_ds)
    return train_loader, val_loader, test_loader, mean, std


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    hours: np.ndarray,
) -> Tuple[Dict[str, float], Dict[int, float]]:
    errors = np.abs(predictions - targets)
    mae = float(errors.mean())
    nonzero_targets = np.where(np.abs(targets) < 1e-6, 1.0, np.abs(targets))
    mape = float(np.mean(np.abs(predictions - targets) / nonzero_targets) * 100.0)
    hourly_mae: Dict[int, float] = {}
    for hour in range(24):
        mask = hours == hour
        if np.any(mask):
            hourly_mae[hour] = float(np.mean(errors[mask]))
    metrics = {
        "mae": mae,
        "mape": mape,
    }
    metrics.update({f"mae_hour_{hour:02d}": value for hour, value in hourly_mae.items()})
    return metrics, hourly_mae


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    std: float,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds: List[np.ndarray] = []
    targs: List[np.ndarray] = []
    hours_list: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb, hb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds.append(model(xb).cpu().numpy())
            targs.append(yb.cpu().numpy())
            hours_list.append(hb.numpy())
    pred_arr = np.concatenate(preds)
    target_arr = np.concatenate(targs)
    hours_arr = np.concatenate(hours_list)
    pred_denorm = pred_arr * std + mean
    target_denorm = target_arr * std + mean
    metrics, hourly_mae = compute_metrics(pred_denorm, target_denorm, hours_arr)
    metrics["hourly_mae"] = hourly_mae
    return metrics, pred_denorm, target_denorm, hours_arr


def naive_baseline(loader: DataLoader, mean: float, std: float) -> float:
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb, _ in loader:
            naive = xb[:, -1, 0].numpy()
            preds.append(naive)
            targets.append(yb.numpy())
    pred_arr = np.concatenate(preds) * std + mean
    target_arr = np.concatenate(targets) * std + mean
    return float(np.mean(np.abs(pred_arr - target_arr)))


def train(config: TrainingConfig = TrainingConfig()) -> TrainingResult:
    set_seed(config.seed)
    device = torch.device(config.device)
    train_loader, val_loader, test_loader, mean, std = prepare_loaders(config)
    model = GRUForecaster(hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=0.1 if config.num_layers > 1 else 0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    writer: Optional[SummaryWriter] = None
    if config.log_dir is not None:
        config.log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(config.log_dir))

    best_state = None
    best_metric = math.inf
    best_epoch = -1

    for epoch in range(config.max_epochs):
        model.train()
        running_loss = 0.0
        count = 0
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            count += yb.size(0)
        val_metrics, _, _, _ = evaluate(model, val_loader, device, mean, std)
        if writer is not None:
            writer.add_scalar("train/loss", running_loss / max(1, count), epoch)
            writer.add_scalar("val/mae", val_metrics["mae"], epoch)
        if val_metrics["mae"] < best_metric:
            best_metric = val_metrics["mae"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if writer is not None:
        writer.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics, _, _, _ = evaluate(model, val_loader, device, mean, std)
    test_metrics, predictions, targets, hours = evaluate(model, test_loader, device, mean, std)
    naive = naive_baseline(test_loader, mean, std)
    improvement = (naive - test_metrics["mae"]) / max(naive, 1e-6) * 100.0

    forecast_path: Optional[Path] = None
    if config.artifacts_dir is not None:
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        forecast_path = config.artifacts_dir / "forecast_vs_actual.csv"
        pd.DataFrame({
            "prediction": predictions,
            "target": targets,
            "hour": hours,
        }).to_csv(forecast_path, index=False)

    hourly_mae = {int(k): float(v) for k, v in test_metrics.get("hourly_mae", {}).items()}
    return TrainingResult(
        best_epoch=best_epoch,
        val_metrics={k: v for k, v in val_metrics.items() if not k.startswith("mae_hour_") and k != "hourly_mae"},
        test_metrics={k: v for k, v in test_metrics.items() if not k.startswith("mae_hour_") and k != "hourly_mae"},
        naive_mae=naive,
        improvement_pct=float(improvement),
        hourly_mae=hourly_mae,
        forecast_path=forecast_path,
    )


def main() -> None:  # pragma: no cover - CLI helper
    result = train()
    print("Best epoch", result.best_epoch)
    print("Validation metrics", result.val_metrics)
    print("Test metrics", result.test_metrics)
    print("Naive baseline MAE", result.naive_mae)
    print("Improvement %", result.improvement_pct)
    if result.forecast_path:
        print("Saved forecast samples to", result.forecast_path)


if __name__ == "__main__":  # pragma: no cover
    main()
