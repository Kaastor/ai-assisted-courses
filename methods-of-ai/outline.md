# 6-Meeting Beginner Course: **Methods of AI (Computer Vision)** — *Graduate Edition*

## 1) Course snapshot

For grad students who can program but are new to AI/ML. In 6×2-hour meetings you’ll build a small, fully reproducible **image classifier** end-to-end: framing and metrics → data handling → classical baseline → **transfer learning** → validation with **CIs & expected-utility thresholding** → error slicing & fairness → **Grad-CAM** explainability → a lightweight **Gradio** demo (plus a tiny CPU **load test**). You’ll practice the core applied-engineer habits: **decision-making, verification (tests & calibration), and communication**, with small public datasets and pinned environments.

---

## 2) Skill map (where each habit is practiced)

| Core Engineer Skill ↓ / Meeting →          |   M1  |   M2  |   M3  |   M4  |   M5  |   M6  |
| ------------------------------------------ | :---: | :---: | :---: | :---: | :---: | :---: |
| Problem framing & metric choice            | **✓** |   ✓   |       |   ✓   |       |   ✓   |
| Data handling & EDA                        | **✓** | **✓** |   ✓   |   ✓   |   ✓   |       |
| Modeling (baseline → improved)             |       | **✓** | **✓** |   ✓   |       |       |
| Validation, CIs & thresholding             |       |   ✓   |   ✓   | **✓** |   ✓   |   ✓   |
| Error analysis & slicing                   |       |       |       |   ✓   | **✓** |   ✓   |
| Explainability & ethics                    |       |       |       |       |   ✓   | **✓** |
| Simple deployment (CLI/Gradio + load test) |       |       |       |       |       | **✓** |
| Communication (notes, model card, demo)    |   ✓   |   ✓   |   ✓   | **✓** |   ✓   | **✓** |

---

## 3) Six-meeting plan (concise)

### Meeting 1 — **From Problem to Metric + Reproducible Setup**

**Objectives**

* Frame a CV task; define target, **success metric**, and a simple **cost matrix**.
* Build a deterministic, versioned environment; introduce GitHub Classroom workflow.
* Implement **majority/random baselines** and a “spec → generate → **verify**” habit.

**Core topics**: pipeline overview; accuracy vs F1 vs **PR-AUC**; class imbalance; fixed seeds; directory structure; dataset licensing.

**Hands-on lab (20–40 min)**
Dataset: **Hymenoptera (Ants vs Bees)** (tiny, public).
Steps:

1. Clone course template; create `venv`; install pinned deps.
2. Write a docstring/spec for `seed_everything()`; implement; **unit test** for determinism.
3. Deterministic **stratified** train/val/test split saved as CSV manifests.
4. Compute class balance; implement majority & random baselines; log metrics.

**Homework (≤60 min)**

* **Problem brief** (6–8 sentences): use case, primary metric, cost matrix, acceptance threshold.
* **Data card** (source, license, size, class balance, obvious biases).

**Success criteria / checkpoint**

* Split manifests on disk; passing unit test for seeding; baseline metrics file in `reports/`.

---

### Meeting 2 — **Data to First Model (Classical Features + Calibration)**

**Objectives**

* Build a deterministic preprocessing pipeline.
* Extract **HOG** (or HOG+color) features; train **Linear SVM** (or Logistic Regression).
* Read a **confusion matrix**; plot **PR curve**; compute **Brier score** & **calibration curve**.

**Core topics**: resizing/normalization; classical features; threshold from PR & costs; calibration 101.

**Hands-on lab**
Steps:

1. `torchvision.transforms` (train vs eval paths).
2. HOG feature extraction → Linear SVM with class weights.
3. Metrics: accuracy, macro-F1, PR-AUC, **Brier score**; calibration plot; choose threshold.
4. **Unit tests**: (a) HOG output shape, (b) transform determinism.

**Homework (≤60 min)**

* Visualize top-12 false positives/negatives with captions; propose 1 concrete fix (data/feature/threshold).

**Checkpoint**

* Plots: confusion, PR, **calibration**; baseline beats majority; tests passing.

---

### Meeting 3 — **Stronger Model via Transfer Learning (with Ablation)**

**Objectives**

* Understand **freeze vs fine-tune**; train a **ResNet-18/MobileNetV3** head on CPU.
* Track learning curves; early stopping; compare against classical baseline.
* Run a tiny **ablation**: frozen backbone vs “last block unfrozen”.

**Core topics**: pre-trained encoders; LR & weight decay; capacity control; early stopping.

**Hands-on lab**
Steps:

1. Load pretrained `resnet18`; freeze all but final layer; train 3–5 epochs CPU-only.
2. Optional +1 epoch with last block unfrozen.
3. Compare models in `reports/compare_models.csv`.
4. Save **best checkpoint**; scriptable train/eval.

**Homework (≤60 min)**

* Safely tune **one** hyperparameter (LR or augmentation intensity).
* Compute a small **paired bootstrap CI** (on validation predictions) for frozen vs unfrozen F1.

**Checkpoint**

* Transfer model improves macro-F1 ≥5 pts (val); ablation table with CI; best checkpoint saved.

---

### Meeting 4 — **Validate Right (CIs, Expected Utility & Sealed Test)**

**Objectives**

* Design validation: **stratified hold-out** vs light **K-fold** (on train only).
* Compute **bootstrap 95% CIs**; set threshold by **expected utility** using cost matrix.
* Run one final **sealed test** evaluation; write a concise **Validation Note**.

**Core topics**: repeated CV; bootstrap; calibration touch-up; threshold as decision, not score.

**Hands-on lab**
Steps:

1. 3× repeated 3-fold CV on train; aggregate metrics + 95% CI.
2. Pick threshold from PR curve that **minimizes expected cost** (per your matrix).
3. Single sealed-test run; save `evaluation_report.json` (+ CI & threshold).

**Homework (≤60 min)**

* **Validation Note** (≤1 page): split design, CI method, chosen threshold & expected cost, leakage checks.

**Checkpoint**

* CI reported; threshold justified by expected utility; sealed test result stored.

---

### Meeting 5 — **Error Analysis, Slicing, Fairness & Robustness**

**Objectives**

* Slice performance by **brightness/blur/viewpoint** proxies; identify gaps.
* Perform a targeted **fix** (e.g., augmentation, class weights, threshold tweak) and re-measure.
* Add a simple **distribution-shift probe** (e.g., held-out lighting condition).
* Write a short **risk memo** (limitations & safe use).

**Core topics**: slice metrics, supports; ablations; fairness as performance equity across domain slices; robustness.

**Hands-on lab**
Steps:

1. Compute slice metrics (e.g., luminance quartiles, blur score).
2. Implement one fix; produce an **ablation table** (before/after, global + slices).
3. Create a mini **shift set** (e.g., deliberately darkened images) and report metrics.

**Homework (≤60 min)**

* Finalize fix; update ablation & narrative; draft **risk memo** (≤300 words).

**Checkpoint**

* At least one problematic slice improved without regressing global F1 by >2 pts; shift probe reported.

---

### Meeting 6 — **Explainability + Lightweight Deploy + Demo (with Load Test)**

**Objectives**

* Use **Grad-CAM** to visualize decisions; spot spurious cues.
* Ship a CPU-only **Gradio** app and a CLI; run a tiny **load test** (throughput/latency).
* Complete a **Model Card**; deliver a crisp 3-minute demo.

**Core topics**: Grad-CAM basics; model cards (intended use, data, metrics with CIs, limits); packaging; perf sanity checks.

**Hands-on lab**
Steps:

1. Generate Grad-CAM overlays for 8 correct + 8 incorrect predictions; discuss failure patterns.
2. Build `app.py` (Gradio) + `predict.py` (CLI).
3. Add a **10-line load test** (images/sec & median latency on CPU).
4. Finish `MODEL_CARD.md` and 3-slide demo deck.

**Homework (≤60 min)**

* Polish mini-project; tag a release; rehearse demo.

**Checkpoint**

* App runs locally; explanations rendered; **calibration & throughput** briefly shown in demo; model card complete.

---

### Compact schedule table

| Meeting | Objectives (short)                                   | Topics                                   | Lab                                                  | Homework                            | Checkpoint                      |
| ------- | ---------------------------------------------------- | ---------------------------------------- | ---------------------------------------------------- | ----------------------------------- | ------------------------------- |
| **M1**  | Frame task; metric & costs; repro setup; spec→verify | Pipeline, metrics, seeds, Git            | Split manifests; baseline; unit test for seeding     | Problem brief + data card           | Repro splits + baseline metrics |
| **M2**  | Classical baseline + **calibration**                 | HOG/SVM, PR curve, Brier                 | HOG→SVM; PR & calibration; tests for determinism     | Error gallery + fix idea            | Confusion/PR/calibration plots  |
| **M3**  | Transfer learning + **ablation**                     | ResNet18/MobileNetV3, freeze vs unfreeze | Train frozen head; optional last-block; compare      | Tune 1 hyperparam; **bootstrap CI** | F1 ↑ ≥5 pts; ablation w/ CI     |
| **M4**  | Validation with **CIs & utility**; sealed test       | K-fold, bootstrap, thresholding          | Repeated CV; choose threshold by expected cost; test | **Validation Note**                 | CI + justified threshold + test |
| **M5**  | Slicing, fix, **shift probe**                        | Slice metrics, ablations, fairness       | Slice report; targeted fix; shift set                | Update ablation; **risk memo**      | Slice gap ↓; no big regression  |
| **M6**  | Explain, **deploy, load test**, present              | Grad-CAM, model cards, Gradio            | Grad-CAM; CLI+Gradio; latency/throughput             | Polish & release                    | App + model card + demo         |

---

## 4) Mini-project brief (half page)

**Scope:** Build a reproducible **binary or 3-class** image classifier with a tiny local pipeline and demo. Choose one dataset:

* **Hymenoptera (ants vs bees)** (torchvision tutorial) — ideal for CPU; \~400 train/val images.
* **CIFAR-10 (3-class subset)** — e.g., airplane/automobile/ship; cap to ≤10k train images.
* **Oxford-IIIT Pet (binary subset)** — cat vs dog; cap to ≤3k images.

**Acceptance criteria (the ritual):**

1. **Define metric & cost matrix** → declare an **acceptance threshold**.
2. **Deterministic split** (stratified) → manifests stored.
3. **Baseline**: HOG+Linear SVM (or majority) → confusion + PR + **calibration** (Brier, reliability plot).
4. **Improve**: transfer learning (ResNet-18/MobileNetV3, frozen backbone) → +Δ on validation.
5. **Validate**: light K-fold or repeated CV on train; **95% CIs** (bootstrap); choose threshold by **expected utility**; one sealed-test run.
6. **Error analysis**: ≥2 slices (e.g., brightness & blur) + one targeted **fix** with an ablation table.
7. **Explainability**: **Grad-CAM** overlays for ≥16 images (balanced across right/wrong).
8. **Deploy**: **Gradio app** + **CLI**; **CPU load test** (images/s, median latency) and brief calibration check.
9. **Communicate**: `MODEL_CARD.md` (intended use, data, metrics with CI, threshold & costs, limitations, known risks) and a **3-minute demo**.

---

## 5) Assessment & rubric (brief)

* **Labs (M1–M5)** – 30 pts (6 each): completeness (2), correctness & reproducibility (2), tests/plots & artifacts quality (2).
* **Homework (M1–M6)** – 10 pts: clear, on-time, ≤60 min scope, integrates required plots/tables.
* **Mini-project (code & report)** – 35 pts: pipeline & reproducibility (8), modeling & improvement (8), validation quality (**CIs & expected-utility thresholding**) (9), error analysis & robustness (6), explainability & ethics (4).
* **Demo (M6)** – 20 pts: story & clarity (5), live app + CLI (7), **calibration & threshold defense** with numbers (5), timing/Q\&A (3).
* **Ethics & limitations reflection** – 5 pts: concrete risks, failure modes, and safe-use boundaries.

> Deductions: any sealed-test reuse during development (−10), missing seeds or nondeterministic training (−5), absent calibration or CI where required (−3).

---

## 6) Tech stack & setup

**Versions (pinned)**
Python **3.11**; PyTorch **2.3** (CPU), Torchvision **0.18**; scikit-learn **1.4**; numpy **1.26**; pandas **2.2**; matplotlib **3.8**; Pillow **10**; scikit-image **0.22**; gradio **4.x**; pytorch-grad-cam **1.5**; **pytest 8**, **pytest-cov**, **pytest-xdist**; black **24**; ruff **0.5**; pre-commit **3.7**; typer **0.12**; rich **13**.

**Quickstart links:** PyTorch/Torchvision, scikit-learn, Gradio, pytorch-grad-cam.

**Template notebook & repo scaffold**

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
│  ├─ evaluate.py        # metrics, PR curve, CIs, calibration
│  ├─ kfold.py           # repeated CV on train only
│  ├─ explain.py         # Grad-CAM utilities
│  ├─ predict.py         # CLI (typer)
│  └─ app.py             # Gradio UI
├─ tests/                # determinism, leakage, metrics, calibration, coverage
├─ reports/              # auto-generated plots/tables
├─ MODEL_CARD.md         # template
├─ README.md             # run instructions
├─ requirements.txt      # pinned
└─ pyproject.toml        # black/ruff config
```

**Repro recipe**
`python -m venv .venv && source .venv/bin/activate` → `pip install -r requirements.txt` →
`python src/train.py --seed 2024 --epochs 5 --freeze true` →
`python src/evaluate.py --use_test_once --compute_ci --expected_cost "FN:5,FP:1"` →
`python src/app.py` (then run `python -m src.predict path/to/img.jpg`).

**GitHub Classroom autograder checks**
(1) deterministic splits; (2) baseline metrics file exists; (3) model head output size; (4) CV never touches test; (5) threshold persisted; (6) slice report present; (7) **calibration artifacts** present; (8) **bootstrap CI** computed; (9) app launches; (10) **load test script** runs.

---

## 7) Risks & mitigations

* **Data leakage** (test in train; normalization fit on full data).
  *Mitigation*: split manifests created **before** any stats; autograder checks no test path appears elsewhere; eval script reads test once.

* **Overfitting on tiny data**.
  *Mitigation*: frozen backbone; early stopping; small LR; report **CIs** not just point estimates.

* **Metric confusion** (accuracy vs PR-AUC vs utility).
  *Mitigation*: cost matrix exercise; **expected-utility thresholding**; macro-averaging by default.

* **Environment drift**.
  *Mitigation*: pinned versions; seed function enforces deterministic algorithms; tests assert determinism.

* **Compute limits (CPU only)**.
  *Mitigation*: tiny datasets; capped epochs; frozen features; optional downsampling.

---

## 8) Accessibility & ethics notes

* **Datasets**: use small, permissively licensed sets; document in `data_card.md`; show class balance and potential spurious cues (backgrounds/lighting).
* **Bias checks (group metrics)**: domain-based slices (brightness/blur/viewpoint); report per-slice precision/recall with support; note trade-offs when applying fixes.
* **Limitations disclosure**: model card states intended use, out-of-scope cases (e.g., heavy motion blur or non-natural images), confidence threshold, **CI**, and test-only-once policy.
* **Accessibility**: plots labeled; color-blind-safe palettes; Gradio app includes alt-text and keyboard navigation notes.

---

## 9) Stretch paths (optional)

1. **Statistical comparison**: paired bootstrap for MobileNetV3 vs ResNet18; report p-value/CI of ΔF1.
2. **Robustness sweep**: corruptions (brightness/blur/noise) with a small “robust augment” ablation; report robustness curves.
3. **Interpretability+**: Grad-CAM++ vs Occlusion Sensitivity; discuss disagreements and implications for trust.

---

### Bottom line

Assignments now emphasize **verification** (unit tests, **calibration**, **CIs**, **expected-utility thresholds**), **robustness** (slice + shift probe), and **operationalization** (CPU **load test** + Gradio app). Students may use AI assistants as they like—the assignments are designed so the only way to succeed is to **make correct, reproducible, well-justified work**.
