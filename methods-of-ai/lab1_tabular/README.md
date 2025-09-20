# Lab 1 — Tabular Classification with an MLP

## What's inside

- `data.py` — downloads UCI Adult, builds train/val/test splits, encodes categoricals, and standardises numeric features without leakage.
- `model.py` — `TabularNet` with per-column embeddings concatenated with numeric features and a small MLP head.
- `train_tabular.py` — full training loop with AdamW, StepLR, early stopping, TensorBoard logging, and temperature scaling for calibration.
- `eval.py` — evaluation helpers producing accuracy, F1, ROC-AUC, and calibration error.
- `tests/` — unit test for model shapes plus an acceptance test that ensures ROC-AUC ≥ 0.88 and validation ECE ≤ 0.08.

Method: multilayer perceptron with categorical embeddings, binary cross-entropy loss, AdamW optimiser, StepLR scheduler (gamma=0.5 every two epochs), early stopping on validation ROC-AUC, and post-hoc temperature scaling.

## How to run

Install deps with Poetry then train:

```bash
poetry install
poetry run python -m lab1_tabular.train_tabular
```

TensorBoard logs land in `runs/lab1_tabular` by default. Override hyperparameters via `TrainingConfig` if you create your own CLI wrapper.

## Autograding

Acceptance tests live in `tests/`. Run all checks:

```bash
poetry run python -m pytest lab1_tabular/tests -q
```

The autograder ensures ROC-AUC ≥ 0.88 on the held-out test set and validation ECE ≤ 0.08 after temperature scaling.

### Student Assignments (GitHub Classroom)

Five practical, autograded assignments live under `lab1_tabular/assignments/` with per‑student variants:

- Edit only `lab1_tabular/assignments/student.py` to complete tasks.
- Local run: `RUN_ASSIGNMENT_TESTS=1 poetry run pytest -q lab1_tabular/assignments/tests`
- In GitHub Classroom, the workflow in `.github/workflows/classroom.yml` runs these with `STUDENT_ID` set to the GitHub username; variant logic is in `lab1_tabular/assignments/variant.py`.


## Data & licensing

- **Dataset:** UCI Adult (Census Income) — Creative Commons CC BY 4.0.
- Downloaded automatically to `.data/adult/`. No files are tracked in Git.
- Script removes rows with missing values (`?`) and builds 70/15/15 splits with seed 42 to avoid leakage.

## Moving pieces & extensions

- `Category` embeddings sized via √cardinality heuristic.
- Numeric features standardised with train-only mean/std.
- Early stopping resets patience whenever validation ROC-AUC improves.
- Temperature scaling solved with LBFGS on validation logits.
- `common/metrics.py` contains reusable metrics including Expected Calibration Error.

Stretch ideas mirror the outline: compare against one-hot encodings, tune per-group thresholds, analyse calibration per demographic slice (`sex`, `race`).

## Risk & bias note

Adult income predictions risk amplifying socioeconomic and demographic biases. Always inspect per-group confusion matrices and calibration before deployment, document fairness mitigations, and avoid using these predictions for high-stakes decisions without legal review.
