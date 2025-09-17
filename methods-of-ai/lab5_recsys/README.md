# Lab 5 — Neural Matrix Factorisation on MovieLens 100K

## What's inside

- `train_mf.py` — implicit-feedback recommender with leave-last-N splits, negative sampling, validation-driven checkpointing, and popularity baselines.
- `tests/` — shape/unit test for the MF scorer and an acceptance test verifying Precision@10 ≥ 0.07 and Recall@10 ≥ 0.10 on the held-out set.

Method: user/item embeddings with bias terms plus a small MLP head, trained via logistic loss on implicit positives (ratings ≥ 4) with randomly sampled negatives. Evaluation uses sampled candidate sets (30 negatives + held-out positives) to compute Precision@K/Recall@K and compares against a popularity baseline while persisting learned embeddings for downstream analysis.

## How to run

```bash
poetry install
poetry run python -m lab5_recsys.train_mf
```

Artifacts default to `artifacts/lab5_recsys/embeddings.pt`; TensorBoard logs go to `runs/lab5_recsys`.

## Autograding

```bash
poetry run python -m pytest lab5_recsys/tests -q
```

Acceptance criteria:

- Precision@10 ≥ 0.07.
- Recall@10 ≥ 0.10.

## Data & licensing

- **Dataset:** MovieLens 100K — research use only; cite GroupLens and do not redistribute commercially.
- Downloaded to `.data/movielens/ml-100k`, filtered to users/items with ≥20 positives to keep IDs dense. Students must acknowledge the [MovieLens license](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt) in submissions.
- Leave-last-two split per user prevents timeline leakage; only train data builds the popularity baseline, negative sampler, and evaluation candidate pool.

## Moving pieces & extensions

- Tune embedding dimensionality, negative ratios, candidate sample size, or try BPR loss for ranking-aware training.
- Swap the neural MF head for different architectures or add side features (genres, timestamps).
- Analyse cold-start performance by holding out new users/items, or add feature-based warm-start strategies.

## Risk & bias note

Recommender systems influence consumption and can reinforce filter bubbles. Inspect results for fairness across user cohorts, log why recommendations are made, and ensure opt-outs plus content safeguards before deploying beyond coursework contexts.
