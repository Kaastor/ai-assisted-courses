# Lab 1 — Tabular classification with an MLP (UCI Adult)

**Purpose (1‑liner):** First end‑to‑end PyTorch project: data splits without leakage, embeddings for categoricals, clean training loop, and reliable metrics on a classic tabular problem.

* **Problem type & use cases:** Binary income classification; similar flows appear in churn prediction, credit risk, and fraud pre‑screens.

* **Method taught:** **MLP with per‑column embeddings** for categorical features + numeric standardization.
  **Alternatives:**

  * Gradient‑boosted trees (XGBoost/LightGBM): prefer with smaller datasets and strong tabular structure; often **faster & more accurate** than MLPs on tabular.
  * Linear/logistic baseline: strong starting point, highly interpretable; use when latency/interpretability dominates.

* **Dataset:** **UCI Adult (Census Income)** — **CC BY 4.0**; includes demographic features; well‑known leakage pitfalls. Download from UCI ML Repository. ([UCI Machine Learning Repository][1])

#### 60‑minute instructor intro (time‑boxed)

* **Learning objectives (5 min)**

  * Build a minimal PyTorch pipeline for tabular data.
  * Avoid leakage (fit scalers/encoders only on train).
  * Implement clean training/eval loops with seeds & logs.
* **Conceptual walkthrough (10 min) — key ideas**

  1. Structured vs. unstructured data; why DL isn’t a silver bullet on tabular.
  2. Train/val/test **by row** (random) vs **by time/user** (later labs).
  3. Categorical embeddings vs. one‑hot.
  4. Losses for imbalance; thresholding vs. calibration.
  5. Reproducibility: `PYTHONHASHSEED`, `torch.use_deterministic_algorithms(True)`.
  6. Metrics: accuracy, ROC‑AUC, F1; majority baseline.
  7. Regularization: weight decay, dropout, early stopping.
  8. Error analysis: confusion matrix by **sensitive attributes** (ethics).
  9. Logging with TensorBoard scalars.
  10. “Overfit a tiny batch” sanity test.
* **Live code outline (25 min)**

  * Modules: `data.py` (CSV → tensors; encoders), `model.py` (MLP+emb), `train.py` (loop+TB), `eval.py` (metrics, confusion).
  * Training loop: zero‑grad → forward → loss → backward → step; Gradient clipping; StepLR.
  * Eval: ROC‑AUC, PR‑AUC; confusion matrix; per‑slice metrics.
* **Diagrams/mental model (5 min)**

  * **Blocks:** \[Categorical Embeddings] + \[Standardized Numerics] → \[Concatenate] → \[FC→ReLU→Dropout]\* → \[Logit].
  * **Leakage alert:** Any transform learned from data (encoders, scalers) must see **train only**.

#### Full working example (CPU‑friendly)

```python
# lab1_tabular/train_tabular.py
import os, random, math, argparse
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    os.environ["PYTHONHASHSEED"]=str(s)
    torch.manual_seed(s)
    torch.use_deterministic_algorithms(True)

CAT_COLS = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
NUM_COLS = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]

class AdultDS(Dataset):
    def __init__(self, df, cat_maps, num_stats):
        self.y = (df["income"]==">50K").astype(np.int64).values
        self.cats = [torch.tensor(df[c].map(cat_maps[c]).fillna(0).astype(int).values) for c in CAT_COLS]
        mu, sigma = num_stats
        Xn = (df[NUM_COLS].values - mu)/np.clip(sigma,1e-6,None)
        self.nums = torch.tensor(Xn, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        return [c[i] for c in self.cats], self.nums[i], torch.tensor(self.y[i])

class TabularNet(nn.Module):
    def __init__(self, cat_sizes, num_dim, hidden=[128,64], p=0.2):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(n+1, min(16, (n+1)//2)) for n in cat_sizes])
        emb_dim = sum(e.embedding_dim for e in self.embeds)
        layers=[]; in_dim=emb_dim+num_dim
        for h in hidden:
            layers += [nn.Linear(in_dim,h), nn.ReLU(), nn.Dropout(p)]
            in_dim=h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim,1)
    def forward(self, cats, nums):
        x = torch.cat([e(c) for e,c in zip(self.embeds, cats)], dim=1)
        x = torch.cat([x, nums], dim=1)
        x = self.mlp(x)
        return self.out(x).squeeze(1)

def build_splits(df, seed=42):
    df = df.sample(frac=1, random_state=seed)
    n = len(df); n_tr=int(0.7*n); n_va=int(0.15*n)
    return df[:n_tr], df[n_tr:n_tr+n_va], df[n_tr+n_va:]

def fit_preprocess(train_df):
    cat_maps = {}
    for c in CAT_COLS:
        cats = sorted(train_df[c].dropna().astype(str).unique())
        cat_maps[c] = {k:i+1 for i,k in enumerate(cats)}  # 0 for unk
    mu = train_df[NUM_COLS].mean().values
    sigma = train_df[NUM_COLS].std(ddof=0).values
    return cat_maps, (mu, sigma)

def collate(batch):
    cats = list(zip(*[b[0] for b in batch]))
    cats = [torch.stack(c,0) for c in cats]
    nums = torch.stack([b[1] for b in batch],0)
    y = torch.stack([b[2] for b in batch],0).float()
    return cats, nums, y

def train_epoch(model, loader, opt, crit):
    model.train(); total=0.
    for cats,nums,y in loader:
        opt.zero_grad()
        logits = model(cats, nums)
        loss = crit(logits, y)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); total += loss.item()*y.size(0)
    return total/len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); ys, ps = [], []
    for cats,nums,y in loader:
        p = torch.sigmoid(model(cats,nums))
        ys.append(y.numpy()); ps.append(p.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    yhat = (p>=0.5).astype(int)
    return dict(acc=accuracy_score(y,yhat),
                f1=f1_score(y,yhat),
                auc=roc_auc_score(y,p))

if __name__=="__main__":
    set_seed(42)
    parser=argparse.ArgumentParser(); parser.add_argument("--data", default="adult.csv")
    args=parser.parse_args()
    # Simple CSV loader (expect pre-downloaded adult.csv with headers as UCI names)
    df = pd.read_csv(args.data).dropna()
    tr, va, te = build_splits(df)
    cat_maps, num_stats = fit_preprocess(tr)
    cat_sizes=[len(cat_maps[c]) for c in CAT_COLS]
    ds_tr = AdultDS(tr, cat_maps, num_stats)
    ds_va = AdultDS(va, cat_maps, num_stats)
    ds_te = AdultDS(te, cat_maps, num_stats)
    bs=256
    tr_ld = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=0, collate_fn=collate, drop_last=True)
    va_ld = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate)
    te_ld = DataLoader(ds_te, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate)

    model = TabularNet(cat_sizes, len(NUM_COLS))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    for epoch in range(6):
        tr_loss = train_epoch(model, tr_ld, opt, crit)
        metrics = evaluate(model, va_ld)
        print(f"epoch {epoch} loss {tr_loss:.4f} | val {metrics}")
    print("test:", evaluate(model, te_ld))
```

* **Input pipeline & preprocessing**

  * **Download:** UCI “Adult” CSV; remove missing rows; train/val/test = 70/15/15 with seed=42.
  * **Categoricals:** per‑column integer mapping (0 reserved for unknown).
  * **Numerics:** standardize using **train mean/std** only.
  * **Batching:** `DataLoader(..., num_workers=0, drop_last=True)` for determinism on CPU.

* **Metrics & validation**

  * Report **Accuracy, F1, ROC‑AUC**.
  * **Baselines**: majority class accuracy (≈0.75) and logistic regression (students compute).
  * **Targets (CPU, ≤6 epochs):** Acc ≥0.84, F1 ≥0.70, ROC‑AUC ≥0.88 (achievable with the small MLP).

* **Methodology focus:** Proper **splits & leakage prevention**, **reproducibility** (seeds, deterministic algorithms), **logging**, initial **regularization** (dropout, weight decay), and **slice‑based error analysis** (e.g., by `sex`, `race`).

* **Why this over that? (trade‑offs)**

  * MLP handles mixed numeric/categorical with embeddings; **scales** to many features; can learn interactions.
  * vs. **Gradient‑boosted trees:** often stronger on small/medium tabular; far easier to tune; more interpretable with SHAP; **lower latency**. Choose MLP when you also need joint training with other modalities or large categorical vocabularies.
  * vs. **Logistic regression:** interpretable and fast; but limited nonlinearity, weaker with high‑order interactions.

* **Common failure modes & debugging**

  * **Symptom:** Validation > training loss. **Likely:** Dropout too high or underfitting; **Fix:** reduce dropout, increase hidden units/epochs.
  * **Symptom:** Great val, bad test. **Likely:** leakage in preprocess; **Fix:** recompute encoders/scalers on train only.
  * **Symptom:** AUC good, F1 poor. **Likely:** thresholding suboptimal; **Fix:** tune threshold on val for chosen metric; consider calibration.

* **Assignment seed (students implement)**

  * Add **early stopping** + **LR scheduler** (StepLR gamma=0.5 every 2 epochs).
  * Implement **temperature scaling** for calibration on the validation set; report ECE.
  * **Acceptance tests:** (a) unit tests pass; (b) ROC‑AUC ≥0.88 on test; (c) **ECE ≤0.08** on val.

* **Starter code structure**

  ```
  lab1_tabular/
  ├─ data.py          # load & preprocess (fit encoders on train only)
  ├─ model.py         # TabularNet
  ├─ train_tabular.py # main training loop
  ├─ eval.py          # metrics, confusion matrix, slice eval
  └─ tests/
     └─ test_sanity.py
  ```

* **Unit‑test examples**

  ```python
  # lab1_tabular/tests/test_sanity.py
  import torch
  from lab1_tabular.model import TabularNet
  def test_output_shape():
    m = TabularNet(cat_sizes=[10,5], num_dim=3)
    cats = [torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)]
    nums = torch.zeros(4,3)
    out = m(cats, nums)
    assert out.shape == (4,), "Logit per sample"
  def test_overfit_one_batch():
    m = TabularNet([5], 2); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    xcats = [torch.randint(0,5,(32,))]; xnum = torch.randn(32,2)
    y = (xnum.sum(dim=1)>0).float()  # easy target
    for _ in range(200):
      opt.zero_grad(); loss = torch.nn.BCEWithLogitsLoss()(m(xcats,xnum), y); loss.backward(); opt.step()
    assert loss.item() < 0.1
  ```

* **Stretch goals**

  * Replace embeddings with **one‑hots** and compare speed/accuracy.
  * Add **L2 feature normalization** vs. standardization; compare.
  * Implement **per‑group** thresholding to equalize opportunity (ethics lens).

* **Short reading/viewing**

  * *Practical tips for ML reproducibility* (short blog / class note).
  * *Confusion matrix & ROC/PR curves (sklearn docs)*.
  * *Responsible AI: bias, harm, and mitigation checklists* (course handout).

**Comparison table — Lab 1**

| Method                 | When it shines                                  | When it fails                                  | Data needs  | Inference cost | Interpretability           | Typical metrics  |
| ---------------------- | ----------------------------------------------- | ---------------------------------------------- | ----------- | -------------- | -------------------------- | ---------------- |
| MLP + cat embeddings   | Mixed dtypes, many categoricals, joint training | Small data, heavy feature engineering required | 10k–1M rows | Medium         | Medium (feature ablations) | Acc, F1, ROC‑AUC |
| Gradient‑boosted trees | Tabular with limited data, quick iteration      | Very high‑cardinality categoricals             | 1k–1M rows  | Low            | High (SHAP)                | Acc, F1, ROC‑AUC |
| Logistic regression    | Need maximum simplicity/interpretability        | Nonlinear boundaries                           | 100+ rows   | Very low       | Very high                  | Acc, ROC‑AUC     |

---

[1]: https://archive.ics.uci.edu/dataset/2/adult?utm_source=chatgpt.com "Adult - UCI Machine Learning Repository"
[2]: https://github.com/zalandoresearch/fashion-mnist "GitHub - zalandoresearch/fashion-mnist: A MNIST-like fashion product database. Benchmark"
[3]: https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection?utm_source=chatgpt.com "SMS Spam Collection"
[4]: https://archive.ics.uci.edu/ml/datasets/electricityloaddiagrams20112014?utm_source=chatgpt.com "ElectricityLoadDiagrams20112014"
[5]: https://files.grouplens.org/datasets/movielens/ml-100k-README.txt "files.grouplens.org"
[6]: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html "files.grouplens.org"

