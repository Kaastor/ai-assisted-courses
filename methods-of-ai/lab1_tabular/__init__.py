"""Lab 1: Tabular classification with an MLP."""

from .model import TabularNet
from .train_tabular import TrainingConfig, TrainingResult, train

__all__ = ["TabularNet", "TrainingConfig", "TrainingResult", "train"]
