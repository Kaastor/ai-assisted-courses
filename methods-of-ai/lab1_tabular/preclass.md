# One-Hour Class Plan — Intro to Tabular DL Lab (Undergrad, No DL Prereq)

---

## 0) Pre‑class prep (before you walk in)

**Materials to share**
- *Lab Start — One‑Page Checklist (TA & Students)*
- *Single‑Batch Walk‑through (Shapes‑First, End‑to‑End)*
- Repository link and “How to run” commands.

**Environment sanity checks**
- Install deps (Poetry or pip) and run a smoke test:
  ```bash
  poetry install
  poetry run python -m lab1_tabular.train_tabular
  ```
- Ensure `common.seed`, `common.tensorboard`, and `common.metrics.expected_calibration_error` are available (or vendor small stubs).
- Confirm dataset access (Adult will download to `.data/adult/`). Keep a zipped copy as fallback.
- Verify TensorBoard opens: `tensorboard --logdir runs/lab1_tabular`.

**Optional demo shortcut**
```python
from pathlib import Path
from lab1_tabular import train, TrainingConfig
result = train(TrainingConfig(max_epochs=2, device="cpu", log_dir=Path("runs/lecture_demo")))
print(result.val_metrics, result.test_metrics)
```

---

## 1) 60‑minute agenda (time‑boxed)

### 0–5 min — Framing & learning goals
**Say:** Today we’ll train a small neural net for **tabular** data that outputs a **probability**. You’ll learn: tensors vs **logits** vs probabilities, why **embeddings** for categoricals, how to read **TensorBoard**, and why we **calibrate** probabilities (temperature scaling).

**Board bullets:**
- Task: Adult/Census Income (binary)
- Output: one **logit** → `sigmoid` → probability
- Metrics: **ROC–AUC** (ranking) + **ECE** (calibration)

---

### 5–12 min — Data & hygiene
**Say:** Pipeline downloads UCI Adult, drops rows with `?`, maps labels (`<=50K`→0, `>50K`→1), splits **70/15/15** with a fixed seed.

**Emphasize:**
- 8 categorical + 6 numeric columns.
- **Train‑only** encoders/scalers; **0 = unknown** for unseen categories.
- Anti‑**leakage** hygiene.

**Ask:** Why not compute stats on all data? What happens without an “unknown” ID?

---

### 12–22 min — Model (embeddings + MLP, shapes‑first)
**Say:** One **embedding** table per categorical column; concatenate all embeddings with standardized numerics → MLP → **one logit per row**.

**Show:** `TabularNet` and expected shapes  
- cats `[B, 8]`, nums `[B, 6]`, concat `[B, sum(embed_dims)+6]`, logits `[B]`.

**Ask:** Why embeddings instead of one‑hot? (Compact, learn similarity.)

---

### 22–30 min — Training loop & early stopping
**Say:** `BCEWithLogitsLoss`, **AdamW**, **StepLR**, **early stopping** on validation **ROC–AUC**, logs to **TensorBoard**.

**Live demo #1 (2–3 min):**
- Run a quick 2‑epoch training and open TensorBoard (`runs/lecture_demo`).
- Call out `train/loss`, `val/roc_auc`, `val/f1` curves.

**Ask:** Why AUC for stopping vs accuracy/F1? (Threshold‑free.)

---

### 30–38 min — Evaluation & calibration
**Say:** We compute loss, accuracy, F1, **ROC–AUC**, **ECE**. After training we fit a single **temperature** on validation logits → re‑evaluate. AUC won’t change; **ECE should drop** (probabilities more trustworthy).

**Show:** `evaluate(...)` metrics and a glance at `calibrate_temperature(...)` (LBFGS; clamped T).

**Talk track:** Calibration turns a good **ranker** into reliable **probabilities**.

---

### 38–45 min — Single‑batch walk‑through (hands‑on intuition)
**Live demo #2:** From the handout, run one batch end‑to‑end: load → forward → `BCEWithLogitsLoss` → `backward()` → `optimizer.step()`; print shapes. Reinforce the shapes‑first mental model.

**Ask:** What if numeric standardization used validation stats? (Leakage.)

---

### 45–52 min — Logistics & success criteria
**Say:** How to run with Poetry; where logs go; acceptance tests.  
**Targets:** **test ROC–AUC ≥ 0.88** and **validation ECE ≤ 0.08** *after* temperature scaling. Explain why these two cover ranking + calibration.

**Tip:** `from lab1_tabular import train, TrainingConfig` for quick experiments.

---

### 52–58 min — Responsible use & dataset caveats
**Say:** Dataset has demographic attributes. Models can **amplify bias**. If you explore variants, inspect metrics/calibration **per group** (`sex`, `race`) and read the fairness notes. Brainstorm mitigations (per‑group thresholds, auditing).

---

### 58–60 min — Exit ticket & handoff
- Exit ticket (2 minutes):
  1) Define **logit** vs **probability**  
  2) One reason to use **embeddings**  
  3) What **ECE** tells you that AUC does not
- Point to the two handouts and “How to run.” Share the support channel.

---

## 2) First lab sprint (students right after intro)

**Goal:** complete a run + TensorBoard + pass acceptance thresholds.

**Steps:**
1) Run training; open TensorBoard; screenshot curves.  
2) Record **AUC** and **ECE** before/after temperature scaling; one‑sentence takeaway.  
3) Do the single‑batch shape check and paste outputs.  
4) (Stretch) Change `hidden_dims`/`dropout`, run 2–4 epochs, note effect on val AUC/ECE.

---

## 3) Talk‑track snippets (drop‑in lines)

- **What is a logit?** “Real‑valued score; `sigmoid` → probability. We optimize **BCEWithLogitsLoss** on logits for stability.”
- **Why embeddings?** “Compact, learn similarity; one table per categorical column; concat with numerics.”
- **Why ROC–AUC for stopping?** “Threshold‑free, robust when classes are imbalanced.”
- **Why temperature scaling?** “Fixes **calibration** without changing ranking; ECE improves after fitting on validation.”

---

## 4) Quick file map (for your live tour)

- Data pipeline & anti‑leakage → `lab1_tabular/data.py`  
- Embedding + MLP model → `lab1_tabular/model.py`  
- Training loop, early stopping, calibration → `lab1_tabular/train_tabular.py`  
- Metrics (loss, accuracy, F1, ROC–AUC, ECE) → `lab1_tabular/eval.py`  
- How‑to‑run & thresholds → `README.md`  
- Theory (mental model & shapes‑first) → `theory.md`  
- Reading list (embeddings, calibration, fairness) → `reading.md`  
- Easy imports (`train`, `TrainingConfig`) → `__init__.py`
