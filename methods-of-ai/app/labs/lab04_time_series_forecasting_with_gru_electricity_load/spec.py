"""Structured metadata for Lab 4."""

from __future__ import annotations

from pathlib import Path

from app.labs.base import AcceptanceTest, DatasetReference, LabSpecification

README_PATH = Path(__file__).with_name("README.md")

LAB_SPEC = LabSpecification(
    identifier="lab04_timeseries",
    title="Time-series forecasting with GRU (Electricity Load)",
    domain="Time-series forecasting",
    purpose=(
        "Introduce temporal splits, sliding windows, and baseline comparisons for short-horizon energy forecasts."
    ),
    method_summary=(
        "Single-layer GRU forecaster with windowed inputs, trained against MAE/MAPE targets and compared to naive baselines."
    ),
    dataset=DatasetReference(
        name="Electricity Load Diagrams 2011-2014",
        license="CC BY 4.0",
        url="https://archive.ics.uci.edu/ml/datasets/electricityloaddiagrams20112014?utm_source=chatgpt.com",
        notes="Hourly electricity consumption data; subset to one client for CPU-friendly workloads.",
    ),
    acceptance_tests=(
        AcceptanceTest(
            description="Model beats naive last-value baseline on MAE",
            metric="MAE improvement",
            threshold="≥5%",
            dataset_split="test",
        ),
        AcceptanceTest(
            description="Window dataset emits expected tensor shapes",
            metric="pytest",
            threshold="pass",
            dataset_split="synthetic",
        ),
    ),
    key_focus=(
        "Chronological train/validation/test splits with blocked evaluation",
        "Fitting scalers on training data only to avoid future leakage",
        "Comparing GRU forecasts against naive baselines",
        "Inspecting MAE/MAPE trade-offs and hour-of-day slices",
        "Maintaining determinism through seeds and DataLoader settings",
    ),
    failure_modes=(
        "Model underperforms naive baseline (adjust window size or learning rate)",
        "Metric instability across runs due to missing deterministic settings",
    ),
    assignment_seed=(),
    starter_code=(
        "lab4_timeseries/train_gru.py",
        "lab4_timeseries/tests/test_window.py",
    ),
    stretch_goals=(),
    readings=(),
    comparison_table_markdown="""| Method        | When it shines                     | When it fails          | Data needs    | Inference cost | Interpretability | Typical metrics |
| ------------- | ---------------------------------- | ---------------------- | ------------- | -------------- | ---------------- | --------------- |
| GRU next-step | Short-horizon sequences            | Very long dependencies | 1k–100k steps | Very low       | Low              | MAE, MAPE       |
| Temporal CNN  | Parallelism, fixed receptive field | Irregular sampling     | 10k–1M steps  | Low            | Low              | MAE, sMAPE      |
| ARIMA/ETS     | Clear seasonality, linear          | Nonlinear/volatile     | 100+ steps    | Very low       | Medium           | MAE, MASE       |
""",
    readme_path=README_PATH,
)
