# Environment & Tooling

## 3) Environment & tooling (poetry, modules, logging, tracking, version control)

**Project layout (reused across labs)**

```
dl-methods/
├─ pyproject.toml
├─ common/
│  ├─ seed.py            # set_seed(), deterministic flags
│  ├─ metrics.py         # accuracy, F1, confusion, ECE
│  └─ viz.py             # optional plotting helpers
├─ lab{1..6}_.../        # each lab as a module
└─ scripts/              # small CLIs (e.g., download datasets)
```

**Poetry**

* `poetry init` → add deps: `torch`, `torchvision` (for labs 2/6), `pandas`, `scikit-learn`, `tensorboard`, `matplotlib` (for local plots).
* Lock & sync: `poetry lock && poetry install`.
* Run: `poetry run python lab2_vision/train_cnn.py`.

**Modules over notebooks**

* Benefits: import hygiene, **unit tests**, reproducible runs, smaller diffs.
* Provide **CLI flags** for batch size/epochs so TAs can test quickly.

**Logging**

* TensorBoard: `from torch.utils.tensorboard import SummaryWriter` → log scalars: loss/metrics each epoch.
* Save `runs/labX_*` and commit only **small** sample logs, not full runs.

**Experiment tracking tips**

* Keep a **run card** (YAML/Markdown) with: dataset version, seed, hyperparams, metrics, best checkpoint path.
* Name runs with a schema: `lab2_cnn_bs128_lr1e-3_wd1e-4_seed42`.

**Version control workflow**

* Branch per lab: `feat/lab3-thresholding`.
* PR template: **What changed**, **Why**, **How to test**, **Risks**.
* Use **tags** `lab1-v1.0` for submissions; include `requirements.txt` export for reproducibility (`poetry export`).

**Determinism checklist**

* `set_seed(42)` for Python, NumPy, PyTorch.
* `torch.use_deterministic_algorithms(True)` (note: if some ops aren’t deterministic, PyTorch will warn).
* `DataLoader(num_workers=0, shuffle=True, drop_last=True)`.
* Log package versions + OS info.

**Compute notes (CPU‑only)**

* Use **subsets** (e.g., 10–12k images) and **small models**.
* Batch sizes 64–256; AdamW lr 1e‑3; weight decay 1e‑4; 5–6 epochs.
* Prefer **EmbeddingBag** for text to avoid padding overhead.
* For time‑series, use small window (24–48) and hidden size (32).

**Ethics & reliability**

* Each lab: add a “**Risk & Bias Note**” section in README (e.g., Adult dataset demographic bias; recommender filter bubbles; generative misuse).
* Include **calibration** plots, **slice‑based** metrics (by group/length/hour).
* Keep data privacy in mind (no re‑sharing restricted datasets; follow dataset licenses).

---

[1]: https://archive.ics.uci.edu/dataset/2/adult?utm_source=chatgpt.com "Adult - UCI Machine Learning Repository"
[2]: https://github.com/zalandoresearch/fashion-mnist "GitHub - zalandoresearch/fashion-mnist: A MNIST-like fashion product database. Benchmark"
[3]: https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection?utm_source=chatgpt.com "SMS Spam Collection"
[4]: https://archive.ics.uci.edu/ml/datasets/electricityloaddiagrams20112014?utm_source=chatgpt.com "ElectricityLoadDiagrams20112014"
[5]: https://files.grouplens.org/datasets/movielens/ml-100k-README.txt "files.grouplens.org"
[6]: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html "files.grouplens.org"

