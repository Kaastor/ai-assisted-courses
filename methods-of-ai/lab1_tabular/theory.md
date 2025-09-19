# Lab 1 Theory & Rationale — Tabular Classification with an MLP (Adult Income)

**Audience:** Python-savvy undergraduates who are new to Deep Learning.  
**Goal of this doc:** Give you the *mental model* and the *why* behind each design decision so you can (1) prepare for the lab and (2) look things up while you work.

---

## 1) Big Picture

We predict whether a person’s annual income is **> \$50K** from census-style, **tabular** inputs (age, education, occupation, etc.). We use a small **neural network (MLP)** that:
- learns **embeddings** (dense vectors) for categorical columns,
- concatenates them with standardized numeric features,
- and feeds everything into a few fully connected layers to produce a **logit** (a raw score which we turn into a probability with a sigmoid).

Why this is useful:
- Tabular data is everywhere (finance, operations, marketing, health).  
- Learning **embeddings** lets the model discover similarity between categories (e.g., two occupations that behave similarly in the data).  
- The setup is a gentle introduction to deep learning tooling: data pipelines, model definition, training loops, metrics, logging, and **calibration** (turning scores into trustworthy probabilities).

A simple mental picture of one forward pass (one batch):
```
[cat col 1 ids] --
[cat col 2 ids] ----> [embedding 1]
... [embedding 2] ----> concat --> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> logit
[cat col 8 ids] --/ ... (hidden sizes like 128, 64)
[numeric features] ----------------------------------------------------------/
```


---

## 2) Data & Preprocessing — What happens and **why**

**Dataset.** We use the UCI Adult (“Census Income”) dataset. The code downloads, loads, cleans, and merges train/test splits from UCI automatically. Rows with missing values are dropped; labels are mapped `<=50K → 0`, `>50K → 1`.

**Why?**
- Automatic download makes the lab reproducible.
- Dropping rows with missing values keeps the tutorial focused on representation learning and calibration; handling missingness is important but orthogonal.
- Binary labels make it a clear case for **binary cross-entropy** and calibration.

**Columns.** We model:
- **Categorical:** `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`.  
- **Numeric:** `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`.

**Why these choices?**  
They are standard for this benchmark and cover a mix of demographic, education, and work-related features that are predictive in the dataset.

**Train/Val/Test split (70% / 15% / 15%) with a fixed seed.**

**Why?**
- **Validation** is for model selection (e.g., early stopping), so we don’t “peek” at the test set.  
- A fixed seed makes experiments repeatable, so your results won’t fluctuate randomly.

**Encoding categoricals:**  
For each categorical column, we map each unique string **from the training split** to an integer ID. ID **0 is reserved for “unknown”** so that previously unseen categories in validation/test (or future data) won’t crash the model.

**Why embeddings (instead of one-hot)?**
- One-hot vectors are huge and sparse; embeddings are compact and **learn similarity** between levels.
- In deep learning, embeddings are the standard way to represent discrete tokens.

**Embedding sizes (heuristic):**  
For a column with `cardinality` distinct values, we pick `min(50, max(4, round(2 * sqrt(cardinality))))`.

**Why this heuristic?**  
It balances capacity and parameter count: small columns don’t get oversized embeddings; large columns don’t blow up memory.

**Standardizing numeric features:**  
For each numeric column, compute **(value − mean) / std** using *training* data statistics. If std is 0, we set it to 1 to avoid division by zero.

**Why standardize?**  
- Helps optimization: features on similar scales keep gradients well-behaved.  
- Using **train-only** stats avoids **data leakage** (accidentally letting test/val information influence training).

**DataLoaders:**  
- **Training** loader shuffles and uses `drop_last=True` (discard the final small batch).  
- **Validation/Test** loaders do not shuffle and keep all samples.

**Why these loader settings?**  
- Shuffling breaks any order correlations and improves SGD.  
- Dropping the last partial batch keeps batch sizes consistent (useful for some layers and for stable training statistics).  
- No shuffling at eval time ensures metrics are stable and reproducible.

---

## 3) Model — What we build and **why**

**Architecture:** `TabularNet`  
1. A list of `nn.Embedding` layers—**one per categorical column** (each with its own dimensionality based on cardinality).  
2. Concatenate **all embedding vectors** with the standardized numeric feature vector.  
3. An MLP head with hidden sizes like `(128, 64)`, each hidden layer: `Linear → ReLU → Dropout`.  
4. A final `Linear` to **one logit** per example.

**Why this design?**
- Per-column embeddings keep the model interpretable at the feature level and let each categorical feature learn its own geometry.  
- Concatenation is a simple, effective way to fuse heterogeneous features.  
- ReLU is a solid default nonlinearity; Dropout regularizes (reduces overfitting).  
- A single logit is natural for binary classification and works well with `BCEWithLogitsLoss`.

**Shapes to build intuition (batch size = B):**
- Categorical tensor: `[B, 8]` integers (one column per feature).  
- Numeric tensor: `[B, 6]` floats.  
- After embeddings + concat: `[B, sum(embed_dims) + 6]`.  
- Model output: `[B]` logits (one scalar per row).

---

## 4) Training — How learning happens and **why these choices**

**Loss:** `BCEWithLogitsLoss` (binary cross-entropy + sigmoid baked in, stable).

**Why this loss?**  
- It’s the textbook choice for binary classification with raw logits.  
- Numerically stable vs. doing `sigmoid` yourself and then using `BCELoss`.

**Optimizer:** `AdamW` with a small `weight_decay`.  
**Why AdamW?**  
- Adam converges quickly and is robust to feature scaling; **AdamW** corrects how L2 regularization should interact with Adam (decoupled weight decay).

**Learning-rate schedule:** `StepLR` (halve LR every few epochs).  
**Why schedule at all?**  
- Larger LR at the start helps exploration; smaller LR later helps fine-tuning near a minimum.

**Early stopping on validation ROC–AUC.**  
**Why ROC–AUC for stopping?**  
- It’s **threshold-free** (doesn’t depend on picking 0.5) and is robust when classes are imbalanced.  
- Using validation performance to stop training reduces overfitting.

**TensorBoard logging.**  
**Why log?**  
- Seeing loss/metrics curves helps you debug and develop intuition (e.g., overfitting shows up as training loss down while validation AUC stalls).

**Temperature calibration after training (on the validation set) using LBFGS.**  
We fit a single scalar **temperature** `T` so that `sigmoid(logit / T)` is better calibrated. Typically `T > 1` *softens* probabilities (reduces overconfidence).

**Why calibrate?**  
- Many neural nets output overconfident probabilities.  
- If you use scores to make decisions (e.g., “call if P>0.7”), you want **probabilistic correctness**—that 0.7 really means ~70% positive in aggregate.

**Why LBFGS and a clamp on T?**  
- This is a tiny 1-parameter optimization; second-order methods like LBFGS converge quickly.  
- Clamping T to a reasonable range avoids degenerate solutions.

---

## 5) Evaluation — What we report and **why it matters**

We compute:
- **Loss** (BCE with logits) — sanity check for optimization.
- **Accuracy** and **F1** at a 0.5 threshold — easy-to-interpret point metrics.
- **ROC–AUC** — threshold-free ranking quality; good for early stopping and model selection.
- **ECE (Expected Calibration Error)** — how close predicted probabilities are to empirical frequencies (lower is better).

**Why both ranking and calibration metrics?**  
- A model can rank well (high AUC) but be poorly calibrated (overconfident or underconfident).  
- In practice, you often care about **both**: ranking for prioritization, calibration for decision thresholds and risk.

---

## 6) What to watch for while training (debugging intuition)

- **Training loss ↓, Validation AUC ↑**: you’re learning.  
- **Training loss ↓, Validation AUC ↔/↓**: overfitting; try more dropout, fewer/lower hidden layers, or early stopping sooner.  
- **Validation ECE improves after temperature scaling**: probabilities got more trustworthy (typical).  
- **Plateaus early**: raise `max_epochs`, adjust LR or scheduler step, or enlarge the model modestly.

---

## 7) Practical ethics & fairness

This dataset contains demographic attributes (`sex`, `race`). Any predictive model risks **amplifying biases** present in the data. As you work:
- Inspect metrics and **calibration per group** (e.g., by `sex`, `race`).  
- Consider separate thresholds or other mitigations if you deploy downstream.  
- Do **not** use such models for high-stakes decisions without rigorous review.

---

## 8) Study checklist (prepare before the lab)

- PyTorch basics: `Tensor`, `nn.Module`, `forward`, `optimizer.step()`, `DataLoader`.
- What an **embedding** is and why we use it for categoricals.
- The difference between **logits** and **probabilities**; why `BCEWithLogitsLoss` is preferred.
- What **ROC–AUC**, **F1**, and **ECE** measure.
- Why calibration (temperature scaling) can change probabilities without changing AUC.  
- How to read TensorBoard curves.

---

## 9) Common “why” questions (quick answers)

**Why an MLP for tabular data instead of gradient-boosted trees?**  
We’re practicing deep learning tools (embeddings, calibration, end-to-end training). Trees often win on tabular data, but here the focus is DL fundamentals and good ML hygiene.

**Why reserve ID 0 as “unknown” for categoricals?**  
It safely handles **unseen categories** at validation/test time (and in production) instead of crashing or leaking information.

**Why standardize using train-only stats?**  
To avoid **data leakage**: the model should not see anything about validation/test distributions during training.

**Why early-stop on ROC–AUC, not accuracy/F1?**  
Accuracy/F1 depend on an arbitrary threshold; ROC–AUC doesn’t, so it’s a more stable early-stopping signal.

**Why a learning-rate schedule?**  
It helps you start with bigger steps (explore) and finish with smaller steps (refine).

**Why calibrate after training instead of during training?**  
Post-hoc calibration is simple, stable, and doesn’t interfere with learning a good **ranker**. It adjusts only the **confidence**, not the **ranking**.

---

## 10) Minimal “shapes-first” walk-through

1. **Batch from DataLoader:**  
   `cats: [B, 8]` (ints), `nums: [B, 6]` (floats), `ys: [B]` (0/1).

2. **Forward pass:**  
   Per-column embedding lookups → concatenate all embeddings + `nums` → MLP → logits `[B]`.

3. **Train step:**  
   Compute `BCEWithLogitsLoss(logits, ys)` → `backward()` → `optimizer.step()`.

4. **Validation:**  
   Convert logits to probabilities via `sigmoid(logits / T)` (with `T=1` during training; learned later) → compute Loss, Accuracy, F1, ROC–AUC, ECE.

5. **Calibration (after training):**  
   Fit `T` on validation logits only → re-evaluate with calibrated probabilities → report validation and test metrics.

---

## 11) How to run (quick reference)

- Install with Poetry, then:

```bash
poetry install
poetry run python -m lab1_tabular.train_tabular
```

TensorBoard logs appear under runs/lab1_tabular. The acceptance tests check that the final model reaches strong ROC–AUC and good calibration.

## 12) Extensions (if you want to explore)

One-hot baseline: Compare performance/params vs. embeddings.

Hyperparameters: Try different hidden sizes, dropout rates, and learning schedules.

Threshold tuning: Optimize the decision threshold for F1 or cost-sensitive metrics.

Group-wise analysis: Plot calibration/error per demographic group; consider group-specific thresholds.

Alternative calibration: Try Platt scaling or isotonic regression and compare to temperature scaling.

Tree baselines: Train a gradient-boosted tree (e.g., XGBoost) for perspective.

## 13) Glossary

Logit: A real-valued score; sigmoid(logit) gives a probability in (0,1).

Embedding: Learned dense vector representing a discrete token/category.

ROC–AUC: Probability that a random positive ranks higher than a random negative; threshold-free.

ECE (Expected Calibration Error): Average gap between predicted probability and actual frequency across confidence bins.

Temperature scaling: Post-hoc calibration using a single scalar T dividing logits.

## 14) Map to source files (for curious readers)

Data download, cleaning, splits, encoding, standardization → `lab1_tabular/data.py`

Model with per-column embeddings + MLP → `lab1_tabular/model.py`

Training loop, optimizer/scheduler, early stopping, TensorBoard, temperature scaling → `lab1_tabular/train_tabular.py`

Evaluation metrics (loss, accuracy, F1, ROC–AUC, ECE) → lab1_tabular/eval.py

Package exports for easy imports → `lab1_tabular/__init__.py`

Overview, how-to-run, acceptance tests, notes on bias → `lab1_tabular/README.md`