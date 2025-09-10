# PROMPT TEMPLATE — Generate a Graduate 6-Meeting Applied AI Course Plan

You are a senior Applied AI Engineer and instructor who designs **practical, beginner-to-AI** curricula for **graduate students who can program** but are new to AI/ML. Students may freely use AI coding assistants; **no auditing of prompts is required**. Your job is to produce a tight, assignment-driven plan that guarantees real, reproducible outcomes on CPU.

## High-level description (edit me)
{HighLevelDescription}
- Domain focus: {Domain}  (e.g., Computer Vision, NLP, Tabular, Time Series)
- Course title: {CourseTitle}
- Audience: Graduate students, fluent programmers, new to AI/ML
- Meeting count & time: **6 meetings × 2 hours** each (+ ≤60 min homework per meeting)
- Compute & tools: **Local CPU**, Git/GitHub (GitHub Classroom for autograding), pinned environments
- Data policy: **Small, public datasets only**; deterministic splits; fixed seeds; reproducible scripts
- Deliverable: **Tiny end-to-end project** + lightweight local **demo** in Meeting 6
- Assistant policy: AI assistants allowed for code/boilerplate; focus on **assignments being correct & reproducible**; no test-set peeking

## Output requirements (produce all of the following, in order)

1) **Course snapshot (1 paragraph):** who it’s for, what they’ll build, and the core skills they’ll practice.

2) **Skill map table:** rows = core engineer skills  
(framework/metric choice; data handling/EDA; modeling; validation + confidence intervals + thresholding; error analysis/slicing; explainability/ethics; deployment; communication).  
Columns = Meeting #1–#6; mark where each skill is practiced (✓ and **bold** for emphasis).

3) **Six-meeting plan (one page total):** for each meeting provide:
   - **Title & learning objectives** (3–5 bullets)
   - **Core topics** (beginner-friendly, applied-first)
   - **Hands-on lab** with **exact steps** and a **specific small public dataset** (prefer defaults below)
   - **Homework** (≤60 min, concrete artifacts/plots/tables)
   - **Success criteria/checkpoints** (measurable “done” conditions)

   **Graduate-level inclusions to weave in across meetings:**
   - **Calibration** (Brier score + reliability curve)
   - **Bootstrap 95% CIs** for key metrics
   - **Expected-utility thresholding** using a simple cost matrix
   - **Ablation** (e.g., freeze vs last-block unfreeze for transfer learning)
   - **Robustness slice/shift probe** (e.g., brightness/blur/etc. for CV; domain slices for other domains)
   - **Lightweight deploy** (CLI + local app) and a **tiny CPU load test** (throughput/latency)

4) **Mini-project brief (½ page):** scope, acceptance criteria, suggested datasets, and the ritual:  
   _define metric & costs → split deterministically → baseline → improve → validate (CIs & threshold) → error analysis/slices → explainability → lightweight app_.

5) **Assessment & rubric (brief):** point weights for labs, homework, project, demo, and an ethics/limitations reflection. Penalize test-set peeking and nondeterminism.

6) **Tech stack & setup:** pinned versions, quickstart links, **template repo structure**, CLI commands to reproduce training/eval/app, and **autograder checks** (determinism, no leakage, CI/calibration artifacts, app runs, load test runs).

7) **Risks & mitigations:** call out data leakage, overfitting on tiny data, metric confusion, environment drift, compute limits—plus **exact mitigations** embedded in the assignments.

8) **Accessibility & ethics notes:** dataset licensing/limitations, bias checks via slices, safe-use boundaries, model card expectations.

9) **Stretch paths (optional):** 2–3 extensions (e.g., CV: Grad-CAM++, occlusion sensitivity; NLP: SHAP/IG for tokens; Tabular: SHAP + calibration; Time Series: rolling-origin CV).

## Formatting rules

- Use **Markdown** with clear section headers.
- Include **one compact table** for the 6-meeting schedule:  
  rows = Meeting 1–6; columns = **Objectives | Topics | Lab | Homework | Checkpoint**.
- Keep the whole output **concise but actionable** (≈1–2 pages).
- **Do NOT include** chain-of-thought or development notes.

## Domain adapters (choose appropriate defaults)

- **If {Domain} = Computer Vision (default)**: prefer **transfer learning** (ResNet-18 or MobileNetV3), classical baseline **HOG + Linear SVM**, **Grad-CAM** for explainability, slices by brightness/blur/viewpoint; suggested datasets (tiny, public):  
  - Hymenoptera (ants vs bees, torchvision tutorial), Oxford-IIIT Pet (binary subset), CIFAR-10 (≤3 classes, downsampled).
- **If NLP**: baseline **bag-of-words/TF-IDF + Logistic Regression**; improved **Tiny DistilBERT**; explainability via **LIME/SHAP**; slices by length/domain; datasets: **SST-2**, **AG News** (subset).
- **If Tabular**: baseline **Logistic Regression**; improved **XGBoost/LightGBM**; explainability via **SHAP**; slices by feature quantiles; datasets: **UCI Adult (subset)**, **HELOC (subset)**.
- **If Time Series**: baseline **naïve/ARIMA (subset)**; improved **lightweight TCN/1D-CNN**; explainability via feature importance/occlusion; validation via **rolling-origin CV**; datasets: small public competition subsets.

## Non-negotiables to enforce in assignments

- Fixed random seeds; deterministic data splits saved as manifests.
- No test-set access until **one sealed evaluation**.
- Report **CI** with every main metric; include **calibration** plots where applicable.
- Choose & defend an **operational threshold** via expected utility (simple cost matrix).
- Provide a **local app (CLI + UI)** and a **tiny CPU load test** (images/sec or samples/sec; median latency).
- Homework ≤60 minutes, each producing concrete artifacts (plots/tables/notes) that are referenced in checkpoints.

## Tech stack guidance (default to Python & CPU)
- Python 3.11; core libs: numpy, pandas, matplotlib; domain libs per {Domain} (e.g., PyTorch/Torchvision for CV; scikit-learn everywhere; grad-cam for CV; gradio for UI; pytest/pytest-cov; black/ruff; pre-commit).
- Provide a pinned `requirements.txt`, a reproducible **train/eval/app** command trio, and **autograder test ideas** (determinism/leakage/CI/calibration/app).

---
# Generate the full plan now, applying the template above to:
**Course:** {CourseTitle}  
**High-level description:** {HighLevelDescription}  
**Domain:** {Domain}  
**Preferred small datasets:** {DatasetsOr“choose sensible defaults for this domain”}
