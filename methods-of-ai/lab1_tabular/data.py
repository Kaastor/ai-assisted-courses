"""Data utilities for the Adult income classification lab."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common.seed import set_seed


DATA_DIR = Path(".data") / "adult"
TRAIN_FILE = DATA_DIR / "adult.data"
TEST_FILE = DATA_DIR / "adult.test"

ADULT_URLS = {
    TRAIN_FILE: "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    TEST_FILE: "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
}

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

TARGET_COLUMN = "income"


@dataclass
class AdultMetadata:
    """Metadata for model construction and evaluation."""

    categorical_cardinalities: List[int]
    numeric_features: int
    label_names: Sequence[str]
    categorical_mapping: Dict[str, Dict[str, int]]
    numeric_means: np.ndarray
    numeric_stds: np.ndarray


def ensure_downloads() -> None:
    """Download the Adult dataset to ``.data`` if missing."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in ADULT_URLS.items():
        if path.exists():
            continue
        with urllib.request.urlopen(url) as response:
            content = response.read()
        path.write_bytes(content)


def _read_csv(path: Path) -> pd.DataFrame:
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    df = pd.read_csv(
        path,
        names=columns,
        na_values=" ?",
        skipinitialspace=True,
        engine="python",
        quoting=csv.QUOTE_NONE,
    )
    if path == TEST_FILE:
        df = df.iloc[1:]  # first row is header info
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.replace(".", "", regex=False)
    return df


def load_adult_dataframe() -> pd.DataFrame:
    """Return the cleaned Adult dataset as a DataFrame."""

    ensure_downloads()
    df_train = _read_csv(TRAIN_FILE)
    df_test = _read_csv(TEST_FILE)
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df.dropna().reset_index(drop=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"<=50K": 0, ">50K": 1})
    return df


def train_val_test_split(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/val/test according to 70/15/15 proportions."""

    set_seed(seed)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].reset_index(drop=True)
    return train_df, val_df, test_df


def _build_categorical_mapping(train_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    mapping: Dict[str, Dict[str, int]] = {}
    for col in CATEGORICAL_COLUMNS:
        values = sorted(train_df[col].unique())
        mapping[col] = {v: i + 1 for i, v in enumerate(values)}  # 0 reserved for unknown
    return mapping


def _encode_categoricals(
    df: pd.DataFrame, mapping: Dict[str, Dict[str, int]]
) -> np.ndarray:
    encoded = []
    for col in CATEGORICAL_COLUMNS:
        map_col = mapping[col]
        encoded.append(df[col].map(lambda v: map_col.get(v, 0)).astype(np.int64).to_numpy())
    return np.stack(encoded, axis=1)


def _standardize_numeric(df: pd.DataFrame, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    values = df[NUMERIC_COLUMNS].to_numpy(dtype=np.float32)
    return (values - means) / stds


def _prepare_numeric_stats(train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    values = train_df[NUMERIC_COLUMNS].to_numpy(dtype=np.float32)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1.0
    return means, stds


class AdultDataset(Dataset):
    """PyTorch dataset providing categorical, numeric, and target tensors."""

    def __init__(self, categorical: np.ndarray, numeric: np.ndarray, targets: np.ndarray):
        self.categorical = torch.from_numpy(categorical.astype(np.int64))
        self.numeric = torch.from_numpy(numeric.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:  # pragma: no cover - simple
        return self.targets.shape[0]

    def __getitem__(self, idx: int):
        return self.categorical[idx], self.numeric[idx], self.targets[idx]


def prepare_datasets(
    batch_size: int = 256,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, AdultMetadata]:
    """Build DataLoaders and metadata for training."""

    df = load_adult_dataframe()
    train_df, val_df, test_df = train_val_test_split(df, seed=seed)
    categorical_mapping = _build_categorical_mapping(train_df)
    cat_cardinalities = [len(mapping) + 1 for mapping in categorical_mapping.values()]
    numeric_means, numeric_stds = _prepare_numeric_stats(train_df)

    def encode_split(split_df: pd.DataFrame) -> AdultDataset:
        categorical = _encode_categoricals(split_df, categorical_mapping)
        numeric = _standardize_numeric(split_df, numeric_means, numeric_stds)
        targets = split_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
        return AdultDataset(categorical, numeric, targets)

    train_ds = encode_split(train_df)
    val_ds = encode_split(val_df)
    test_ds = encode_split(test_df)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "drop_last": True,
        "shuffle": True,
    }
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    metadata = AdultMetadata(
        categorical_cardinalities=cat_cardinalities,
        numeric_features=len(NUMERIC_COLUMNS),
        label_names=["<=50K", ">50K"],
        categorical_mapping=categorical_mapping,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
    )
    return train_loader, val_loader, test_loader, metadata
