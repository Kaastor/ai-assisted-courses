# Lab 1 — Practical Assignments (Autograded)

These five hands-on assignments reinforce the core skills of tabular ML used in Lab 1. They’re designed for GitHub Classroom autograding and support per‑student variants so each learner gets a slightly different dataset to work with.

- Files to edit: `lab1_tabular/assignments/student.py`
- Do not edit: `lab1_tabular/assignments/variant.py`, tests under `lab1_tabular/assignments/tests/`
- Run locally: `RUN_ASSIGNMENT_TESTS=1 poetry run pytest -q lab1_tabular/assignments/tests`

Per‑student variants are generated deterministically from your GitHub username (`GITHUB_ACTOR` in CI) or from `STUDENT_ID` when running locally.

## Assignment 1 — Data Split
Implement `split_dataframe(df, seed)`:
- Input: a `pandas.DataFrame` with a binary target column `y`.
- Output: three dataframes `(train, val, test)` with 70/15/15 proportions using the provided `seed` for shuffling.
- Constraints: no leakage, preserve reproducibility.

## Assignment 2 — Numeric Standardization
Implement numeric preprocessing:
- `prepare_numeric_stats(train_df, numeric_cols)` → `(means, stds)` with `stds == 1.0` for any zero‑variance columns.
- `standardize_numeric(df, numeric_cols, means, stds)` → `np.ndarray` standardized per column.
- Tests check train means ≈ 0 and stds ≈ 1.

## Assignment 3 — Categorical Encoding
Implement categorical encoding with unknown handling:
- `build_categorical_mapping(train_df, categorical_cols)` returns `{col: {value: index}}` using `0` for unknown.
- `encode_categoricals(df, mapping)` returns a `np.ndarray` of encoded columns using `0` for unseen categories.

## Assignment 4 — Simple MLP
Implement `SimpleMLP(input_dim, hidden_dims, dropout)` in PyTorch:
- Architecture: Linear → ReLU → Dropout repeated for each hidden layer, then a final `Linear(..., 1)` producing logits.
- Shape: forward accepts `X: (N, input_dim)` and returns `(N,)` or `(N, 1)` (both accepted by tests).

## Assignment 5 — Train One Epoch
Implement `train_one_epoch(model, loader, device, optimizer, loss_fn)`:
- Loop over batches, compute logits, loss (`BCEWithLogitsLoss`), backprop, and step.
- Return the average loss over all samples.
- Tests train on a per‑student synthetic dataset and expect a clear loss reduction across epochs.

## Per‑Student Variants
- Source: `lab1_tabular/assignments/variant.py`
- Local run: `STUDENT_ID=<your-id> RUN_ASSIGNMENT_TESTS=1 poetry run pytest -q lab1_tabular/assignments/tests`
- In CI (GitHub Classroom), the workflow sets `STUDENT_ID` to your GitHub username automatically.

## Tips
- Keep functions pure; avoid reading global state in `student.py`.
- Use `numpy`/`pandas` for data transforms and PyTorch for models/training.
- Prefer `pathlib.Path` and type hints as in the main lab.

