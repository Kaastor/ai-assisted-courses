# Lab 4 — Electricity Load Forecasting with a GRU

## What's inside

- `train_gru.py` — GRU forecaster with sliding-window dataset construction, deterministic splits, valuation against a naive last-value baseline, hourly slice analysis, and CSV artifact export.
- `tests/` — shape/unit test for the dataset and a training acceptance test asserting an MAE improvement ≥ 5% over the naive baseline.

Method: single-layer GRU (defaulting to client `MT_016`) consuming 24-hour windows of normalised demand, trained with AdamW on MSE loss. Validation is chronological, metrics include MAE, MAPE, and per-hour MAE slices. A CSV artifact captures predictions vs. actuals for downstream plotting.

## How to run

Install deps then train:

```bash
poetry install
poetry run python -m lab4_timeseries.train_gru
```

Logs land in `runs/lab4_timeseries`; `forecast_vs_actual.csv` is written to `artifacts/lab4_timeseries`.

## Autograding

```bash
poetry run python -m pytest lab4_timeseries/tests -q
```

The acceptance test checks:

- Test MAE beats the naive last-value baseline by ≥ 5%.
- Reported metrics include MAE and MAPE on the held-out test set.

## Data & licensing

- **Dataset:** Electricity Load Diagrams 2011–2014 — Creative Commons CC BY 4.0.
- Automatically downloaded from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/electricityloaddiagrams20112014) into `.data/electricity_load/`.
- Train/val/test split is chronological (70/15/15); scaling uses train-only statistics to avoid leakage.

## Moving pieces & extensions

- Sliding-window dataset ensures temporal consistency; adjust `window_size` or `series_column` (e.g., `MT_002`) to explore different clients.
- Hour-of-day MAE slices highlight systematic bias; extend to weekday/weekend breakdowns.
- Experiment with deeper GRUs via `num_layers` or add dropout for regularisation.
- Swap the GRU with a dilated Temporal CNN or add exogenous regressors (weather, holidays) for comparison.

## Risk & bias note

Electric load forecasts feed into infrastructure planning; underestimates risk outages while overestimates waste resources. Always monitor error slices around peak hours, validate against business-critical thresholds, and combine with operational safeguards (alerts, manual overrides).
