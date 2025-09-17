# Lab 3 — SMS Spam Detection with Hybrid Embedding Bags

## What's inside

- `train_spam.py` — full training loop using an EmbeddingBag model with a character-level fallback, validation-driven thresholding, TensorBoard hooks, and precision–recall analysis.
- `vocab.py` — lightweight vocabulary builders for word and character tokens.
- `tests/` — unit and acceptance tests ensuring the model emits logits per sample and reaches the target macro-F1 with a bounded vocabulary size.

Method: word-level `nn.EmbeddingBag` paired with a character-level `nn.EmbeddingBag` fallback, concatenated features passed through a two-layer MLP and optimised with AdamW plus positive-class weighting. Validation threshold is chosen via the precision–recall curve to maximise macro-F1, and the resulting curve is stored for inspection.

## How to run

Install dependencies and start training:

```bash
poetry install
poetry run python -m lab3_text.train_spam
```

Artifacts go to `runs/lab3_text` and `artifacts/lab3_text` by default (TensorBoard logs and `precision_recall.png`). Override paths with `TrainingConfig` if you wrap the trainer.

## Autograding

Run the lab-specific tests:

```bash
poetry run python -m pytest lab3_text/tests -q
```

The acceptance test asserts:

- Macro-F1 ≥ 0.90 on the held-out test set (threshold tuned on validation).
- Vocabulary size stays between 3k and 30k tokens.

## Data & licensing

- **Dataset:** SMS Spam Collection — Creative Commons CC BY 4.0.
- Downloaded on demand from the [UCI repository](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection) into `.data/sms_spam/`.
- Stratified 70/15/15 splits with seed 42 prevent leakage; vocabularies are built on the training partition only.

## Moving pieces & extensions

- Character-level fallback ensures OOV-heavy messages still generate informative embeddings.
- Precision–recall curve plotting plus validation-driven threshold selection highlights imbalance trade-offs.
- Positive-class weighting mitigates the ham/spam skew; adjust via `TrainingConfig`.
- Extend by adding TextCNN/BiLSTM alternatives or DistilBERT feature extraction for comparison, mirroring the outline suggestions.

## Risk & bias note

SMS spam detection can misclassify legitimate (ham) messages — false positives harm user trust. Keep humans in the loop for high-stakes channels, monitor per-sender slices (short vs. long messages), and review regulations for automated filtering in your jurisdiction.
