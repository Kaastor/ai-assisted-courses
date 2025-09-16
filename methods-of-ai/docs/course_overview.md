# Course Overview

## 1) One‑page course overview / syllabus

**Course title:** *Methods of Deep Learning*
**Audience:** CS undergraduates comfortable with Python & general programming; *no prior deep learning required.*
**Format:** 6× hands‑on labs (2.5–3h each). Each lab starts with a 60‑minute instructor intro + live coding, followed by structured build‑modify‑test exercises.

### Learning objectives (explicitly tied to your goals)

By the end, students can:

* Recognize main DL **problem types** (vision, text, tabular, time‑series, generative, recommendation).
* Explain **pros/cons of DL** vs. classical ML (data/compute needs, latency, interpretability, maintenance).
* **Choose appropriate methods** with trade‑off reasoning.
* **Implement, test, debug, and validate** PyTorch models.
* **Work with data**: loading, preprocessing/augmentation, proper splits, metrics, leakage pitfalls across modalities.
* Practice **reproducibility** (seeds, determinism), **reliability** (calibration, slice checks), **ethics** (bias, privacy, misuse).

### Grading breakdown

* Labs 1–6 (10% each): 60% total

  * Each lab includes: acceptance tests (unit tests + metric thresholds), short error‑analysis writeup.
* Capstone reflection checklist (course‑goal alignment): 10%
* Two short “method choice” memos (trade‑off reasoning based on a scenario): 10%
* Open‑book practical quiz (debugging & metrics): 20%

### Tools & compute

* **Framework:** PyTorch (CPU‑only; no GPUs required). Optional: torchvision/torchtext/torchaudio where useful.
* **Runtime:** Each lab set up to complete training/eval **on CPU in ≤15 minutes** using small models / subsets.
* **Tracking:** TensorBoard logs + CSV logs; optional Weights & Biases (offline mode ok).
* **Reproducibility:** Seeds set, sources of nondeterminism noted, deterministic flags where possible.

### Workflow & policies

* **Environment:** `poetry` for dependency management; project as Python **modules** (no notebooks in submissions).
* **Version control:** Git required; feature branches + pull requests; commit messages must summarize what changed & why.
* **Collaboration:** Discuss ideas/debugging freely; **code you submit must be your own**. Cite any AI assistant help (a one‑line note in README like “Used AI for: refactor trainer” is sufficient).
* **Office hours:** 2×/week; one “bug clinic” focused on debugging workflows.
* **Academic integrity:** Plagiarism or undisclosed code copying fails the lab.

---

[1]: https://archive.ics.uci.edu/dataset/2/adult?utm_source=chatgpt.com "Adult - UCI Machine Learning Repository"
[2]: https://github.com/zalandoresearch/fashion-mnist "GitHub - zalandoresearch/fashion-mnist: A MNIST-like fashion product database. Benchmark"
[3]: https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection?utm_source=chatgpt.com "SMS Spam Collection"
[4]: https://archive.ics.uci.edu/ml/datasets/electricityloaddiagrams20112014?utm_source=chatgpt.com "ElectricityLoadDiagrams20112014"
[5]: https://files.grouplens.org/datasets/movielens/ml-100k-README.txt "files.grouplens.org"
[6]: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html "files.grouplens.org"

