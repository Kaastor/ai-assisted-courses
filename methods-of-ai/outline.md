# 6‑Meeting Beginner Course: **Methods of AI (Computer Vision)**

## 1) Course snapshot

This course is for undergrads and early‑career developers with basic Python who want to become **applied** AI engineers in computer vision. In 6 two‑hour meetings you’ll build a tiny, reproducible **image classifier** end‑to‑end: from framing and metrics, through data handling and transfer learning, to validation, error analysis, explainability, and a lightweight **Gradio** demo. You’ll practice the habits that matter on the job—**decision‑making, verification, and communication**—using small public datasets, pinned environments, Git/GitHub workflows, fixed seeds, and measurable checkpoints every session.

---

## 2) Skill map (where each habit is practiced)

| Core Engineer Skill ↓ / Meeting →      |   M1  |   M2  |   M3  |   M4  |   M5  |   M6  |
| -------------------------------------- | :---: | :---: | :---: | :---: | :---: | :---: |
| Problem framing & metric choice        | **✓** |   ✓   |       |   ✓   |       |   ✓   |
| Data handling & EDA                    | **✓** | **✓** |   ✓   |   ✓   |   ✓   |       |
| Modeling (baseline → improved)         |       | **✓** | **✓** |   ✓   |       |       |
| Validation & thresholding              |       |   ✓   |   ✓   | **✓** |   ✓   |       |
| Error analysis & slicing               |       |       |       |   ✓   | **✓** |   ✓   |
| Explainability & ethics                |       |       |       |       |   ✓   | **✓** |
| Simple deployment (CLI/Gradio)         |       |       |       |       |       | **✓** |
| Communication (repo, model card, demo) |   ✓   |   ✓   |   ✓   |   ✓   |   ✓   | **✓** |

---

## 3) Six‑meeting plan (concise)

### Meeting 1 — **From Problem to Metric + Reproducible Setup**

**Objectives**

* Frame a CV task, define **target**, **success metric**, and a basic **cost matrix**.
* Create a deterministic, versioned environment; practice Git from day 1.
* Make a **majority/random baseline** to anchor expectations.

**Core topics**: pipeline overview; accuracy vs F1 vs PR‑AUC; class imbalance; fixed seeds; directory structure; GitHub flow; dataset licenses.

**Hands‑on lab (20–40 min)**
Dataset: **Hymenoptera (Ants vs Bees)** from torchvision tutorial (≈400 train/val images).
Steps:

1. Clone course template; create `venv`; install pinned `requirements.txt`.
2. Implement `seed_everything()` and **stratified** train/val/test split saved to disk (CSV of file paths + labels).
3. Compute class balance; implement **majority** and **random** baselines; log metrics.

**Homework (≤60 min)**

* Write a 6–8 sentence **problem brief**: use case, metric, cost matrix, and acceptance threshold.
* Commit a `data_card.md` (source, license, size, known biases).

**Success criteria / checkpoint**

* Reproducible split artifacts; baseline metrics table in `reports/metrics_baseline.csv`; problem brief merged via PR.

---

### Meeting 2 — **Data to First Model (Classical Features)**

**Objectives**

* Implement minimal preprocessing & **deterministic** augmentations.
* Extract **HOG** features and train a **Linear SVM** or **Logistic Regression** baseline.
* Read a **confusion matrix** and pick a threshold using your cost matrix.

**Core topics**: resizing/normalization; HOG/color histograms; train/val/test discipline; confusion matrix; precision‑recall & thresholding.

**Hands‑on lab**
Dataset: Hymenoptera.
Steps:

1. Build `torchvision.transforms` pipeline (resize→center crop; deterministic).
2. Extract HOG (via scikit‑image) → train **Linear SVM** (`scikit‑learn`) with class weights.
3. Evaluate accuracy, macro‑F1, **PR‑AUC**; pick operational threshold; save plots.

**Homework (≤60 min)**

* Plot top‑12 **false positives/negatives** with captions; propose 1 concrete data/feature fix.

**Checkpoint**

* `reports/confusion_matrix.png`, `reports/pr_curve.png`, and `errors/` thumbnails; baseline beats majority.

---

### Meeting 3 — **Stronger Model via Transfer Learning**

**Objectives**

* Explain transfer learning; **freeze** vs **fine‑tune**.
* Train a **ResNet‑18 / MobileNetV3** head on CPU in minutes (frozen backbone).
* Track learning curves; avoid overfitting with early stopping.

**Core topics**: pre‑trained encoders; cross‑entropy; learning rate; early stopping; saving **best** checkpoints.

**Hands‑on lab**
Steps:

1. Use torchvision pretrained `resnet18` (ImageNet). Freeze all but final layer; train for 3–5 epochs.
2. Optional: unfreeze last block for +1 epoch.
3. Compare to HOG‑SVM; log both to `reports/compare_models.csv`.

**Homework (≤60 min)**

* Safely tune **one** hyperparameter (LR or augmentation intensity) and justify; update comparison table.

**Checkpoint**

* Transfer model improves macro‑F1 by ≥5 points on **validation** without test peeking; best checkpoint saved.

---

### Meeting 4 — **Validate Right (No Leakage)**

**Objectives**

* Design validation: **stratified hold‑out** vs **(light) K‑fold** on train only.
* Compute **bootstrap confidence intervals**; set threshold from PR curve.
* Catch/avoid **leakage** in transforms and file handling.

**Core topics**: data splits; repeated seeds; CI via bootstrap; calibration basics; reproducible eval script.

**Hands‑on lab**
Steps:

1. Add `kfold.py` (e.g., 3× repeated 3‑fold on train).
2. Aggregate metrics with 95% CI; freeze a threshold from validation PR.
3. Run **one** final test set evaluation; write `evaluation_report.json`.

**Homework (≤60 min)**

* 1‑page **Validation Note**: split choice, CI, threshold, leakage checks.

**Checkpoint**

* CIs reported; single sealed test result; `tests/` autograder passes “no‑leakage” checks.

---

### Meeting 5 — **Error Analysis, Slicing & Fairness**

**Objectives**

* Slice performance by **brightness/blur/viewpoint** proxies; read gaps.
* Propose and validate a targeted fix (data augmentation, class weights, threshold).
* Document trade‑offs and risks.

**Core topics**: slice metrics; subgroup support sizes; ablations; fairness thinking for CV (domain fairness, not demographics).

**Hands‑on lab**
Steps:

1. Compute slices (e.g., luminance quartiles, motion‑blur score).
2. Make a small **ablation table** (before/after your fix).
3. Write a short **risk statement** (where the model fails).

**Homework (≤60 min)**

* Finalize the fix; update ablation & narrative.

**Checkpoint**

* Slice report with at least one gap reduced **without** regressing global F1 >2 points.

---

### Meeting 6 — **Explainability + Lightweight Deploy + Demo**

**Objectives**

* Use **Grad‑CAM** to visualize predictions and spot spurious cues.
* Ship a local **Gradio** app and a CLI; write a **Model Card**.
* Deliver a 3‑minute demo and answer questions.

**Core topics**: Grad‑CAM basics; model cards (intended use, data, metrics, limits); packaging & reproducible runs.

**Hands‑on lab**
Steps:

1. Generate Grad‑CAM overlays for 8 correctly and 8 incorrectly classified images.
2. Build `app.py` (Gradio) + `predict.py` (CLI).
3. Finish `MODEL_CARD.md` and a 3‑slide deck.

**Homework (≤60 min)**

* Polish mini‑project; merge PR; tag a release.

**Checkpoint**

* App runs locally on CPU; explanations rendered; model card complete; demo delivered.

---

### Compact schedule table

| Meeting | Objectives (short)                           | Topics                             | Lab                                                           | Homework                        | Checkpoint                         |
| ------- | -------------------------------------------- | ---------------------------------- | ------------------------------------------------------------- | ------------------------------- | ---------------------------------- |
| **M1**  | Frame task, metric, cost; reproducible setup | Pipeline, metrics, seeds, Git      | Set up repo/venv; stratified split; majority/random baselines | Problem brief + data card       | Split artifacts + baseline metrics |
| **M2**  | Preprocess; classical baseline; threshold    | HOG, SVM/LogReg, PR curve          | HOG→SVM; confusion matrix; choose threshold                   | Visualize top errors + fix idea | Plots + better‑than‑majority       |
| **M3**  | Transfer learning; early stopping            | ResNet18/MobileNetV3, freezing, LR | Train frozen head; compare to baseline                        | Tune one hyperparam             | F1 ↑ ≥5 pts; best checkpoint       |
| **M4**  | Solid validation; CI; anti‑leakage           | K‑fold, bootstrap CI, calibration  | Repeated CV on train; freeze threshold; single test eval      | 1‑page Validation Note          | CIs + sealed test result           |
| **M5**  | Slice & fix; fairness trade‑offs             | Slice metrics, ablations, risk     | Brightness/blur slices; targeted fix                          | Update ablation + narrative     | Gap reduced; no big regression     |
| **M6**  | Explain, deploy, present                     | Grad‑CAM, model cards, Gradio      | Grad‑CAM overlays; CLI+Gradio app                             | Polish & release                | App + model card + demo            |

---

## 4) Mini‑project brief (half page)

**Scope:** Build a reproducible **binary or 3‑class** image classifier with a tiny local pipeline and demo. Recommended datasets (pick one):

* **Hymenoptera (ants vs bees)** — tiny, ideal for CPU.
* **CIFAR‑10 (3‑class subset)** — e.g., airplane/automobile/ship, downsample to ≤10k train images.
* **Oxford‑IIIT Pet (binary subset)** — cat vs dog; downsample to ≤3k images.

**Acceptance criteria (the “ritual”):**

1. **Define metric & threshold** using a short cost matrix.
2. **Split** deterministically (stratified); save split manifests.
3. **Baseline**: HOG+Linear SVM (or majority) with confusion matrix & PR curve.
4. **Improve**: transfer learning (ResNet‑18/MobileNetV3) with frozen backbone; show +Δ on validation.
5. **Validate**: repeated CV on train; 95% CI; one final test run; no leakage.
6. **Error analysis**: at least **two slices** (e.g., brightness quartiles), an ablation of one fix.
7. **Explainability**: Grad‑CAM overlays for ≥16 images (balanced across classes and errors).
8. **Simple app**: Gradio UI + CLI; CPU‑only; seeds fixed; instructions in `README.md`.
9. **Communication**: `MODEL_CARD.md` (intended use, data, metrics with CI, limits, ethics), and a **3‑minute demo**.

---

## 5) Assessment & rubric (brief)

* **Labs (M1–M5)** – 30 pts (6 pts each): completeness (3), correctness/repro (2), clarity of artifacts (1).
* **Homework (M1–M6)** – 10 pts (≈1.6 each): on‑time, focused (≤60 min), clean commit history.
* **Mini‑project (code & report)** – 35 pts: pipeline & reproducibility (10), modeling & validation quality (10), error analysis & fix (8), explainability & ethics (7).
* **Demo (M6)** – 20 pts: concise story (5), live app + CLI (7), defend choices w/ evidence (5), timing & Q\&A (3).
* **Ethics & limitations reflection** – 5 pts: concrete risks, failure modes, and boundaries of safe use.

**Grading notes:** CI must accompany every reported metric; any leakage → −5 pts on project; no random seeds → −3 pts.

---

## 6) Tech stack & setup

**Versions (pinned):**

* Python **3.11**; PyTorch **2.3** (CPU build), Torchvision **0.18**; scikit‑learn **1.4**; numpy **1.26**; pandas **2.2**; matplotlib **3.8**; Pillow **10**; scikit‑image **0.22**; gradio **4.x**; pytorch‑grad‑cam **1.5**; pytest **8**; black **24**; ruff **0.5**; pre‑commit **3.7**.

**Quickstart links:** PyTorch & Torchvision (pytorch.org), scikit‑learn (scikit-learn.org), Gradio (gradio.app), pytorch‑grad‑cam (github.com/jacobgil/pytorch-grad-cam).

**Repo scaffold (provided):**

```
.
├─ data/                 # empty; .gitkeep
├─ notebooks/
│  ├─ 00_setup.ipynb
│  ├─ 01_eda_splits.ipynb
│  ├─ 02_hog_svm_baseline.ipynb
│  ├─ 03_transfer_learning.ipynb
│  ├─ 04_validate_ci.ipynb
│  ├─ 05_error_slices_explain.ipynb
│  └─ 06_deploy_gradio.ipynb
├─ src/
│  ├─ data.py            # loaders, deterministic transforms
│  ├─ features.py        # HOG/color hist
│  ├─ models.py          # head builder for ResNet18
│  ├─ train.py           # train loop (seeded), early stop
│  ├─ evaluate.py        # metrics, PR curve, CIs
│  ├─ kfold.py           # repeated CV on train only
│  ├─ explain.py         # Grad-CAM utilities
│  ├─ predict.py         # CLI
│  └─ app.py             # Gradio UI
├─ tests/                # autograder: split determinism, no leakage, metric shapes
├─ reports/              # auto-generated plots/tables
├─ MODEL_CARD.md         # template
├─ README.md             # run instructions
├─ requirements.txt      # pinned
└─ pyproject.toml        # black/ruff config
```

**Repro recipe:**
`python -m venv .venv && source .venv/bin/activate` (Win: `.\.venv\Scripts\activate`) → `pip install -r requirements.txt` → `python src/train.py --seed 2024 --epochs 5 --freeze true` → `python src/evaluate.py --use_test_once` → `python src/app.py`.

**GitHub Classroom:** starter tests check: (1) deterministic splits, (2) baseline metrics file exists, (3) model head has correct output size, (4) CV never touches test, (5) threshold stored, (6) slice report present, (7) app launches.

---

## 7) Risks & mitigations

* **Data leakage** (augments applied after seeing labels; test images in train):
  *Mitigation*: saved manifest CSVs; test set read‑only; autograder asserts no test file appears elsewhere; transforms defined by split (train vs eval) not by label.

* **Overfitting** (tiny data):
  *Mitigation*: frozen backbone, early stopping on val, small LR, limited epochs; report CI not just point estimate.

* **Metric confusion** (accuracy vs PR‑AUC):
  *Mitigation*: cost matrix exercise in M1–M2; choose threshold from PR; macro‑averaging by default.

* **Environment drift**:
  *Mitigation*: pinned versions; `requirements.txt`; seed function enforces `torch.use_deterministic_algorithms(True)`; all random sources seeded.

* **Compute limits** (CPU only):
  *Mitigation*: tiny dataset; frozen features; capped epochs; downsampled CIFAR subset.

* **Repo hygiene** (messy results):
  *Mitigation*: fixed folder structure; pre‑commit hooks (black/ruff); single entry scripts for train/eval.

---

## 8) Accessibility & ethics notes

* **Dataset considerations**: avoid sensitive attributes; document sources/licensing in `data_card.md`; show class balance and potential spurious cues (e.g., background/lighting).
* **Bias checks (group metrics)**: use **domain slices** (brightness/blur/crop) as proxies; report per‑slice precision/recall with support; if a slice underperforms, propose targeted mitigation.
* **Disclosure of limitations**: model card must state: intended use, out‑of‑scope cases, expected failure modes (e.g., heavy motion blur), and operational **confidence threshold**; include CI and test‑only‑once policy.
* **Accessibility**: plots use color‑blind friendly defaults; always include text labels; Gradio app supports keyboard navigation and alt‑text on example images.

---

## 9) Stretch paths (optional)

1. **Cross‑validation & ensembling**: 5× CV heads; average logits at inference; compare with bootstrap significance.
2. **Explainability+**: Grad‑CAM++, **Integrated Gradients** for the head, or counterfactuals via simple occlusion sensitivity.
3. **A/B experiment**: MobileNetV3 vs ResNet18 with identical data & seed; test for statistically significant difference; discuss throughput/latency trade‑offs.

---

### Summary of what students will really learn

They’ll practice *decision‑making + verification + communication*: define the right **metric**, design sound **validation**, spot and reduce **errors** on important **slices**, **explain** and **document** results, and ship a tiny but trustworthy **demo**—all reproducibly, locally, and under version control.
