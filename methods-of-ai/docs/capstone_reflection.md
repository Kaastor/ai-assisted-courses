# Capstone Reflection Checklist

## 4) Capstone reflection checklist (tie labs → course goals)

Use this checklist to prepare your final submission. Check each item only if you have evidence (plots, metrics, diffs, or code pointers).

**Problem types**

* [ ] I can match each domain (vision, text, tabular, time‑series, recommendation, generative) to an appropriate PyTorch method we built.
* [ ] For a new scenario, I can identify if DL fits and **why** (vs classical).

**Pros/cons of DL**

* [ ] I can state DL trade‑offs: **data & compute needs**, **latency**, **maintenance debt**, **interpretability**.
* [ ] I know when a **simpler baseline** is the right choice.

**Method choice & trade‑offs**

* [ ] I can justify **method A vs B** using constraints (data size, latency, interpretability, maintenance).
* [ ] I can set **sensible defaults** (batch size, LR/decay, epochs, schedulers) and explain tuning steps.

**Implement / test / debug**

* [ ] I can implement a training loop with clean separation: data, model, optimizer, scheduler, metrics.
* [ ] I can write **unit tests** (shape checks, **overfit‑one‑batch**) and interpret failures.
* [ ] I can perform **error analysis**: confusion matrices, per‑class metrics, calibration, and **slice‑based** checks.

**Data work**

* [ ] I can build correct **input pipelines** per modality and prevent **leakage** (temporal/user splits, train‑only transforms).
* [ ] I can choose **metrics** appropriate to the task (AUC/F1, MAE/MAPE, P\@K/R\@K, ELBO).
* [ ] I can track experiments reproducibly (seeds, logs, run cards).

**Ethics & reliability**

* [ ] I can identify bias/harms per lab and propose mitigations (e.g., thresholding by group, calibrated scores).
* [ ] I respect **dataset licenses** & privacy constraints.

---

[1]: https://archive.ics.uci.edu/dataset/2/adult?utm_source=chatgpt.com "Adult - UCI Machine Learning Repository"
[2]: https://github.com/zalandoresearch/fashion-mnist "GitHub - zalandoresearch/fashion-mnist: A MNIST-like fashion product database. Benchmark"
[3]: https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection?utm_source=chatgpt.com "SMS Spam Collection"
[4]: https://archive.ics.uci.edu/ml/datasets/electricityloaddiagrams20112014?utm_source=chatgpt.com "ElectricityLoadDiagrams20112014"
[5]: https://files.grouplens.org/datasets/movielens/ml-100k-README.txt "files.grouplens.org"
[6]: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html "files.grouplens.org"

