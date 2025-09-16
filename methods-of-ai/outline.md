# Methods of Deep Learning — complete course design (PyTorch, CPU-friendly)

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

## 2) Six Labs — different domains/methods, ramping methodology

> **Order:** Builds from fundamentals (tabular) → vision → text → time‑series → recommendation → generative.
> **All labs** include: a working **starter code** baseline (students modify it), **unit tests**, CPU‑friendly hyperparameters, and **acceptance tests**.

---

### **Lab 1 — Tabular classification with an MLP (UCI Adult)**

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

### **Lab 2 — Vision classification with a tiny CNN (Fashion‑MNIST)**

**Purpose:** Train a small CNN with data augmentation, weight decay, and LR scheduling; practice confusion matrices & calibration.

* **Problem type & use cases:** Image recognition; e.g., defect detection, digit/product classification.

* **Method taught:** **Small CNN** (2×Conv → 2×FC) with **augmentation** and **cross‑entropy**.
  **Alternatives:**

  * **Transfer learning** (frozen ImageNet backbones): faster convergence, better accuracy with few labels.
  * **Linear probes** on pretrained features: lowest compute, strong baselines; choose when you can’t fine‑tune.

* **Dataset:** **Fashion‑MNIST (MIT License)** — built into torchvision; modern MNIST‑style images. ([GitHub][2])

#### 60‑minute instructor intro

* **Objectives (5 min):** Set up torchvision datasets; implement CNN; log/visualize learning curves; avoid over/underfit.
* **Key ideas (10 min):**
  Conv filters as **pattern detectors**, padding/stride; **augmentation** ≈ synthetic data; **weight decay** vs **dropout**; **OneCycle/StepLR**; **calibration** (temperature scaling).
* **Live code (35 min):** Dataset + transforms; model; training loop; StepLR; confusion matrix; reliability diagram (ECE).
* **Diagram (5 min):** Input 28×28 → \[Conv→ReLU→Pool]×2 → Flatten → FC→ReLU→FC→Softmax.

#### Full working example

```python
# lab2_vision/train_cnn.py
import os, random, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def set_seed(s=42):
    random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"]=str(s)
    torch.manual_seed(s); torch.use_deterministic_algorithms(True)

class SmallCNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*7*7, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n))
    def forward(self,x): return self.net(x)

def accuracy(logits, y): return (logits.argmax(1)==y).float().mean().item()

def main():
    set_seed()
    tfm_train = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.05,0.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.2861,), (0.3530,))
    ])
    tfm_test = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.2861,), (0.3530,))])
    tr = datasets.FashionMNIST(root=".data", train=True, download=True, transform=tfm_train)
    te = datasets.FashionMNIST(root=".data", train=False, download=True, transform=tfm_test)
    n_sub = 12000  # CPU-friendly subset
    tr_subset = torch.utils.data.Subset(tr, list(range(n_sub)))
    tr_ld = DataLoader(tr_subset, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
    te_ld = DataLoader(te, batch_size=256, shuffle=False, num_workers=0)

    model = SmallCNN(); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
    crit = nn.CrossEntropyLoss()
    for ep in range(6):
        model.train(); tot=0
        for xb,yb in tr_ld:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step(); tot+=loss.item()*yb.size(0)
        sch.step()
        # quick eval
        model.eval(); accs=[]
        with torch.no_grad():
            for xb,yb in te_ld: accs.append(accuracy(model(xb), yb))
        print(f"epoch {ep}: loss {tot/len(tr_subset):.4f} | test acc {np.mean(accs):.3f}")
if __name__=="__main__": main()
```

* **Metrics & target:** Top‑1 accuracy; **baseline** (logistic on raw pixels) \~0.78; **target** for tiny CNN (6 epochs) **≥0.85**.

* **Methodology focus:** Augmentations; **regularization** (weight decay, dropout); **learning‑rate scheduling**; confusion matrix & per‑class accuracy; simple **temperature scaling** for calibration.

* **Why this over that?**

  * CNNs exploit spatial locality; more **parameter‑efficient** than MLP on images.
  * vs. transfer learning: higher start‑up cost to download backbones; but transfer gives higher accuracy with fewer labels.
  * vs. classical (HOG+SVM): faster inference, lower data needs; but less flexible than learned features.

* **Failure modes & fixes**

  * Underfitting: low train acc → increase channels/epochs or reduce augmentation.
  * Overfitting: train≫val acc → add dropout/augment, early stop.
  * Miscalibration: confident wrong predictions → apply temperature scaling on val.

* **Assignment seed**

  * Add confusion matrix & **per‑class F1**; identify **3 worst classes**; propose augmentation tweaks.
  * Implement temperature scaling (optimize T on val).
  * **Acceptance tests:** test acc ≥0.85; worst‑class recall ≥0.75.

* **Starter structure**

  ```
  lab2_vision/
  ├─ train_cnn.py
  ├─ calibrate.py
  └─ tests/test_overfit.py
  ```

* **Unit‑test snippet**

  ```python
  # lab2_vision/tests/test_overfit.py
  import torch
  from lab2_vision.train_cnn import SmallCNN
  def test_forward_shape():
    m=SmallCNN(); x=torch.randn(4,1,28,28); y=m(x)
    assert y.shape==(4,10)
  ```

**Comparison table — Lab 2**

| Method                     | When it shines                   | When it fails                       | Data needs    | Inference cost | Interpretability | Typical metrics |
| -------------------------- | -------------------------------- | ----------------------------------- | ------------- | -------------- | ---------------- | --------------- |
| Small CNN (from scratch)   | Simple tasks, limited compute    | Complex domains; small labeled data | 10k–100k imgs | Low–Med        | Low              | Top‑1 acc       |
| Transfer learning (frozen) | Few labels; higher accuracy fast | Very small CPU? downloads heavy     | 1k–10k        | Low            | Low              | Top‑1 acc       |
| Classic features + SVM     | Tight latency, tiny data         | Complex variations                  | 100–10k       | Very low       | Medium           | Top‑1 acc       |

---

### **Lab 3 — Text classification with EmbeddingBag (SMS Spam)**

**Purpose:** Text preprocessing, tokenization, building a bag‑of‑embeddings model, stratified splits, and F1 for imbalanced classes.

* **Problem type & use cases:** Spam/phishing detection, toxicity filters, routing.

* **Method taught:** **EmbeddingBag + Linear** (bag‑of‑words embeddings).
  **Alternatives:**

  * TextCNN / bi‑LSTM: stronger on local order; small compute.
  * DistilBERT feature extractor + linear head: best accuracy for short texts; slower/on‑disk weights.

* **Dataset:** **SMS Spam Collection** (5,574 SMS, **CC BY 4.0**). UCI Machine Learning Repository. ([UCI Machine Learning Repository][3])

#### 60‑minute intro

* **Objectives:** Clean text → tokens → vocab; handle imbalance; choose metrics; slice analysis (e.g., by message length).
* **Key ideas:** Tokenization; OOV; padding vs **EmbeddingBag** (no padding); F1 vs accuracy; threshold tuning; reproducibility.
* **Live code:** Build vocab (min frequency=2); EmbeddingBag model; train BCEWithLogits; macro F1; stratified split; calibration check.
* **Diagram:** Text → \[tokenize] → \[lookup indices] → \[EmbeddingBag (mean)] → \[Linear] → \[sigmoid].

#### Full working example

```python
# lab3_text/train_spam.py
import os, re, random, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

def set_seed(s=42):
    random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"]=str(s)
    torch.manual_seed(s); torch.use_deterministic_algorithms(True)

def tokenize(s): return re.findall(r"\b\w+\b", s.lower())

class Vocab:
    def __init__(self, texts, min_freq=2):
        from collections import Counter
        c=Counter()
        for t in texts: c.update(tokenize(t))
        self.itos=["<unk>"]+[w for w,f in c.items() if f>=min_freq]
        self.stoi={w:i for i,w in enumerate(self.itos)}
    def encode(self, text): return [self.stoi.get(t,0) for t in tokenize(text)]

class SmsDS(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = [torch.tensor(vocab.encode(t), dtype=torch.long) for t in texts]
        self.offsets = torch.tensor([0]+[len(x) for x in self.texts[:-1]]).cumsum(0)
        self.all_tokens = torch.cat([x for x in self.texts]) if self.texts else torch.tensor([],dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.labels)
    def __getitem__(self,i):
        # EmbeddingBag likes (all_tokens, offsets)
        start = self.offsets[i].item()
        end = self.offsets[i+1].item() if i+1<len(self.offsets) else len(self.all_tokens)
        return self.all_tokens[start:end], self.offsets[i]-self.offsets[i], self.labels[i]

class BagModel(nn.Module):
    def __init__(self, vocab_size, dim=64):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, dim, mode="mean")
        self.fc = nn.Linear(dim, 1)
    def forward(self, tokens, offsets):
        x = self.emb(tokens, offsets)
        return self.fc(x).squeeze(1)

if __name__=="__main__":
    set_seed()
    # Expect CSV with columns: label (spam|ham), text
    import pandas as pd
    df = pd.read_csv("SMSSpamCollection.csv")  # students: include a small loader from UCI zip
    y = (df["label"].map({"ham":0,"spam":1}).values)
    X_train, X_temp, y_train, y_temp = train_test_split(df["text"], y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    vocab = Vocab(X_train.tolist(), min_freq=2)
    def mkds(X,y): return SmsDS(X.tolist(), y.tolist(), vocab)
    tr, va, te = mkds(X_train, y_train), mkds(X_val, y_val), mkds(X_test, y_test)
    def collate(batch):
        toks = torch.cat([b[0] for b in batch])
        offs = torch.tensor([0]+[len(b[0]) for b in batch[:-1]]).cumsum(0)
        ys = torch.stack([b[2] for b in batch])
        return toks, offs, ys
    tr_ld = DataLoader(tr, batch_size=128, shuffle=True, collate_fn=collate, num_workers=0, drop_last=True)
    va_ld = DataLoader(va, batch_size=256, shuffle=False, collate_fn=collate)
    te_ld = DataLoader(te, batch_size=256, shuffle=False, collate_fn=collate)
    model = BagModel(len(vocab.itos), 64)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(1., (y_train==0).sum()/(y_train==1).sum())]))
    for ep in range(5):
        model.train()
        for tok,off,yb in tr_ld:
            opt.zero_grad(); loss=crit(model(tok,off), yb); loss.backward(); opt.step()
        # quick val F1
        model.eval(); ys=[]; ps=[]
        with torch.no_grad():
            for tok,off,yb in va_ld:
                p = torch.sigmoid(model(tok,off)); ys+=yb.tolist(); ps+=p.tolist()
        thr=0.5; yhat=[1 if p>=thr else 0 for p in ps]
        print(f"ep{ep} val f1={f1_score(ys,yhat):.3f}")
    # test
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for tok,off,yb in te_ld:
            p = torch.sigmoid(model(tok,off)); ys+=yb.tolist(); ps+=p.tolist()
    thr=0.5; yhat=[1 if p>=thr else 0 for p in ps]
    print("test acc", accuracy_score(ys,yhat), "f1", f1_score(ys,yhat))
```

* **Metrics & target:** **Macro F1** (class imbalance). **Baseline** = majority class ≈ 0.86 accuracy but poor F1; **Target:** Macro‑F1 ≥ **0.90** on test with 5 epochs.

* **Methodology focus:** Imbalance handling (pos\_weight/threshold), **stratified splits**, OOV handling, simple **slice‑based eval** (long vs short messages).

* **Why this over that?**

  * EmbeddingBag is **fast on CPU** and robust; no padding.
  * vs. TextCNN/BiLSTM: captures **word order** and n‑grams → higher accuracy; slightly slower.
  * vs. DistilBERT features: best accuracy; cost: larger downloads & slower inference; choose when precision critical.

* **Common failure modes**

  * Degenerate F1 (near 0): likely tokenization bug or label mapping inverted.
  * Val >> Test: leakage via building vocab on full data; fix to train‑only vocab.

* **Assignment seed**

  * Add **precision‑recall curve** & choose threshold to maximize F1 on val.
  * Implement **character‑level** fallback for OOV‑heavy cases; compare.
  * **Acceptance tests:** Macro‑F1 ≥0.90; vocab size between 3k–30k.

* **Starter structure**

  ```
  lab3_text/
  ├─ train_spam.py
  ├─ vocab.py
  └─ tests/test_shapes.py
  ```

* **Unit test**

  ```python
  def test_logit_scalar_per_sample():
      from lab3_text.train_spam import BagModel
      import torch
      m=BagModel(100,32)
      toks=torch.randint(0,100,(50,)); offs=torch.tensor([0,10,20,35])
      out=m(toks,offs); assert out.shape==(4,)
  ```

**Comparison table — Lab 3**

| Method                | When it shines                        | When it fails                   | Data needs   | Inference cost | Interpretability         | Typical metrics |
| --------------------- | ------------------------------------- | ------------------------------- | ------------ | -------------- | ------------------------ | --------------- |
| EmbeddingBag + Linear | Short texts, CPU‑only, quick training | Long context; nuanced semantics | 5k–100k msgs | Very low       | Medium (n‑gram saliency) | Macro F1        |
| TextCNN / BiLSTM      | Order matters, still light            | Very long docs                  | 10k–100k     | Low–Med        | Medium                   | Macro F1        |
| DistilBERT features   | Best accuracy                         | Download/inference cost         | 1k–50k       | Med–High       | Low                      | Macro F1        |

---

### **Lab 4 — Time‑series forecasting with GRU (Electricity Load)**

**Purpose:** Proper **temporal splits**, sliding‑window datasets, naive baselines, and MAE/MAPE for forecasting.

* **Problem & use cases:** Energy demand forecasting, sensor telemetry, capacity planning.

* **Method taught:** **GRU sequence‑to‑one** next‑step forecaster with sliding windows.
  **Alternatives:**

  * **Temporal CNN (1D dilated)**: parallelizable; strong for local patterns.
  * **Classical naive/ARIMA**: strong baselines when seasonality dominates.

* **Dataset:** **Electricity Load Diagrams 2011–2014** (Portugal) — **CC BY 4.0** at UCI repository. We subselect **1 client** and **hourly resample** for CPU speed. ([UCI Machine Learning Repository][4])

#### 60‑minute intro

* **Objectives:** Make **temporal** train/val/test split; implement sliding windows; compare to **naive last‑value** baseline; avoid leakage from scalers.
* **Key ideas:** Stationarity; scale only on **train**; blocked validation; MAPE pitfalls near zero; seasonality hints; coverage vs accuracy.
* **Live code:** Window maker; GRU(32) → Linear; MAE/MAPE; compare to naive; plot predictions (students).
* **Diagram:** Time axis windows: `[t−W…t−1] → y_t`; shift windows by stride.

#### Full working example

```python
# lab4_timeseries/train_gru.py
import os, random, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def set_seed(s=42):
    random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"]=str(s)
    torch.manual_seed(s); torch.use_deterministic_algorithms(True)

class WindowDS(Dataset):
    def __init__(self, series, W=24):
        self.x=[]; self.y=[]
        for i in range(W, len(series)):
            self.x.append(series[i-W:i]); self.y.append(series[i])
        self.x=torch.tensor(self.x, dtype=torch.float32).unsqueeze(-1) # [N,W,1]
        self.y=torch.tensor(self.y, dtype=torch.float32)               # [N]
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.x[i], self.y[i]

class GRUForecaster(nn.Module):
    def __init__(self, hid=32):
        super().__init__()
        self.rnn = nn.GRU(1, hid, batch_first=True)
        self.fc = nn.Linear(hid, 1)
    def forward(self, x):
        _, h = self.rnn(x)      # [1,B,H]
        return self.fc(h[-1]).squeeze(1)

def mape(y,p): return (torch.abs((y-p).clamp_min(1e-6)/y.clamp_min(1.0))).mean().item()
def mae(y,p):  return torch.mean(torch.abs(y-p)).item()

if __name__=="__main__":
    set_seed()
    # Expect CSV "LD2011_2014.txt" pre-downloaded; we'll read one column
    df = pd.read_csv("LD2011_2014.txt", sep=";", index_col=0, parse_dates=[0])
    s = df.iloc[:,0].resample("1H").mean().dropna()  # pick client 0; hourly
    # temporal split: first 70% train, next 15% val, last 15% test
    n=len(s); tr=int(0.7*n); va=int(0.15*n)
    s_train, s_val, s_test = s.iloc[:tr], s.iloc[tr:tr+va], s.iloc[tr+va:]
    mu, sigma = s_train.mean(), s_train.std()
    def norm(x): return (x-mu)/max(sigma,1e-6)
    tr_ds = WindowDS(norm(s_train.values).astype(np.float32), W=24)
    va_ds = WindowDS(norm(s_val.values).astype(np.float32), W=24)
    te_ds = WindowDS(norm(s_test.values).astype(np.float32), W=24)
    tr_ld = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=256, shuffle=False, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=256, shuffle=False, num_workers=0)

    model = GRUForecaster(32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()
    for ep in range(6):
        model.train(); tot=0
        for xb,yb in tr_ld:
            opt.zero_grad(); loss=crit(model(xb), yb); loss.backward(); opt.step(); tot+=loss.item()*yb.size(0)
        # quick val
        model.eval(); pred=[]; targ=[]
        with torch.no_grad():
            for xb,yb in va_ld: pred+=model(xb).tolist(); targ+=yb.tolist()
        yp = torch.tensor(pred); yt = torch.tensor(targ)
        print(f"ep{ep}: val MAE={mae(yt,yp):.3f}")
    # test vs naive
    with torch.no_grad():
        pred=[]; targ=[]
        for xb,yb in te_ld: pred+=model(xb).tolist(); targ+=yb.tolist()
    yp=torch.tensor(pred); yt=torch.tensor(targ)
    mae_model=mae(yt,yp)
    # naive: last value
    naive=[]
    for xb,_ in te_ld:
        naive += xb[:,-1,0].tolist()
    mae_naive=mae(yt, torch.tensor(naive))
    print({"test_mae":mae_model, "naive_mae":mae_naive, "improve_pct":(mae_naive-mae_model)/mae_naive*100})
```

* **Metrics & targets:** **MAE** and **MAPE**; **Acceptance:** model **beats naive** MAE by **≥5%** on test.

* **Methodology focus:** **Temporal split**, avoiding future leakage, standardization on **train only**, comparing to naive; error analysis by **hour‑of‑day** slice.

* **Why this over that?**

  * GRU is simple & effective for short‑horizon forecasting.
  * vs. Temporal CNN: faster training & receptive‑field control; choose when parallelism matters.
  * vs. ARIMA: great for linear seasonality but brittle with regime changes; DL handles nonlinearities.

* **Failure modes**

  * Model worse than naive → window too small / learning rate too high; try W=48 or reduce LR.
  * Instability across runs → check seeds & `num_workers=0`.

* **Starter structure & tests**

  ```
  lab4_timeseries/
  ├─ train_gru.py
  └─ tests/test_window.py
  ```

  ```python
  def test_window_shapes():
      import torch, numpy as np
      from lab4_timeseries.train_gru import WindowDS
      ds=WindowDS(np.arange(30,dtype=np.float32), W=5)
      x,y=ds[0]; assert x.shape==(5,1) and y.ndim==0
  ```

**Comparison table — Lab 4**

| Method        | When it shines                     | When it fails          | Data needs    | Inference cost | Interpretability | Typical metrics |
| ------------- | ---------------------------------- | ---------------------- | ------------- | -------------- | ---------------- | --------------- |
| GRU next‑step | Short‑horizon sequences            | Very long dependencies | 1k–100k steps | Very low       | Low              | MAE, MAPE       |
| Temporal CNN  | Parallelism, fixed receptive field | Irregular sampling     | 10k–1M steps  | Low            | Low              | MAE, sMAPE      |
| ARIMA/ETS     | Clear seasonality, linear          | Nonlinear/volatile     | 100+ steps    | Very low       | Medium           | MAE, MASE       |

---

### **Lab 5 — Recommendations with neural matrix factorization (MovieLens)**

**Purpose:** Build an implicit‑feedback recommender (user/item **embeddings** + dot product), evaluate with **Precision\@K/Recall\@K**, and learn timeline‑aware splits.

* **Problem & use cases:** Personalized ranking: products, content, courses, news.

* **Method taught:** **Neural MF (dot‑product)** with negative sampling; train on implicit positives.
  **Alternatives:**

  * Item‑item **nearest neighbors**: strong cold‑start baseline; fast to deploy.
  * **BPR loss** with pairwise ranking: optimizes ranking metrics more directly.

* **Dataset:** **MovieLens 100K** (stable benchmark). **Usage license** in README allows **research use**, requires **acknowledgement**, disallows **commercial use**; redistribution requires permission. Students must download from GroupLens. ([GroupLens][5])

> *(If availability is a concern, the “ml‑latest‑small” dataset has similar terms; also research‑oriented.)* ([GroupLens][6])

#### 60‑minute intro

* **Objectives:** Proper **user‑level** splits (leave‑last‑N per user); implement negative sampling; compute P\@K/R\@K.
* **Key ideas:** Implicit vs explicit; popularity bias; cold‑start; train/test leakage (don’t sample negatives from future).
* **Live code:** Parse `u.data`; build train/val/test by user; embeddings (64‑dim); BCEWithLogits with negatives; top‑K eval.
* **Diagram:** Users & Items mapped to points; **score = ⟨u, v⟩**; rank items by score.

#### Full working example (condensed)

```python
# lab5_recsys/train_mf.py
import os, random, numpy as np, pandas as pd, torch
from torch import nn
from collections import defaultdict

def set_seed(s=42):
    random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"]=str(s)
    torch.manual_seed(s); torch.use_deterministic_algorithms(True)

class MF(nn.Module):
    def __init__(self, n_users, n_items, d=64):
        super().__init__()
        self.U = nn.Embedding(n_users, d); self.I = nn.Embedding(n_items, d)
        nn.init.normal_(self.U.weight, std=0.01); nn.init.normal_(self.I.weight, std=0.01)
    def forward(self, u, i): return (self.U(u)*self.I(i)).sum(1)

def split_leave_last(df, n_last=1):
    df = df.sort_values(["user","time"])
    tr=[]; va=[]; te=[]
    for u,g in df.groupby("user"):
        if len(g)<=2: continue
        te.append(g.iloc[-1]); va.append(g.iloc[-2]); tr.append(g.iloc[:-2])
    return pd.concat(tr), pd.DataFrame(va), pd.DataFrame(te)

def sample_batch(pos, n_items, neg_ratio=3, bs=2048):
    # pos: DataFrame with (user,item)
    for i in range(0, len(pos), bs):
        b = pos.iloc[i:i+bs]
        u = torch.tensor(b["user"].values, dtype=torch.long)
        pi = torch.tensor(b["item"].values, dtype=torch.long)
        ni = torch.randint(0, n_items, (len(b)*neg_ratio,))
        nu = u.repeat_interleave(neg_ratio)
        yield u, pi, nu, ni

def precision_recall_at_k(model, inter_by_user, k=10, n_items=1682):
    model.eval(); P=[]; R=[]
    all_items = torch.arange(n_items)
    with torch.no_grad():
        for u, pos_items in inter_by_user.items():
            u_t = torch.tensor([u]*n_items)
            scores = model(u_t, all_items)
            topk = torch.topk(scores, k).indices.cpu().numpy().tolist()
            hits = len(set(topk) & pos_items)
            P.append(hits/k); R.append(hits/max(1,len(pos_items)))
    return float(np.mean(P)), float(np.mean(R))

if __name__=="__main__":
    set_seed()
    # Expect ml-100k u.data (tab-separated: user item rating timestamp)
    df = pd.read_csv("u.data", sep="\t", names=["user","item","rating","time"])
    # implicit positives: rating >= 4
    df = df[df["rating"]>=4][["user","item","time"]]
    n_users = df["user"].max()+1; n_items=df["item"].max()+1
    tr, va, te = split_leave_last(df)
    model = MF(n_users, n_items, d=64)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss()
    for ep in range(5):
        model.train()
        for u,pi,nu,ni in sample_batch(tr, n_items):
            opt.zero_grad()
            y_pos = torch.ones(len(pi)); y_neg = torch.zeros(len(ni))
            loss = crit(model(u,pi), y_pos) + crit(model(nu,ni), y_neg)
            loss.backward(); opt.step()
        # quick val P@10/R@10
        inter_by_user = defaultdict(set)
        for _,r in va.iterrows(): inter_by_user[int(r.user)].add(int(r.item))
        P,R = precision_recall_at_k(model, inter_by_user, k=10, n_items=n_items)
        print(f"ep{ep} val P@10={P:.3f} R@10={R:.3f}")
    # test
    inter_by_user = defaultdict(set)
    for _,r in te.iterrows(): inter_by_user[int(r.user)].add(int(r.item))
    P,R = precision_recall_at_k(model, inter_by_user, k=10, n_items=n_items)
    print("test", {"P@10":P, "R@10":R})
```

* **Metrics & targets:** **Precision\@10** and **Recall\@10** across users. **Acceptance:** Recall\@10 ≥ **0.10** and Precision\@10 ≥ **0.07** (achievable with 5 epochs on CPU).

* **Methodology focus:** **User‑timeline splits**, **negative sampling**, popularity baseline, cold‑start discussion; **responsible AI** (filter bubbles, fairness).

* **Why this over that?**

  * Neural MF is simple, fast, and strong; embeddings reusable.
  * vs. Item‑item KNN: transparent and trivial to ship; weaker personalization.
  * vs. BPR: better ranking optimization but slightly more complex sampling.

* **Common failure modes**

  * All zeros at eval → indexing mismatch (off‑by‑one); confirm max IDs.
  * Model predicts popular items only → increase neg\_ratio or add weight decay.

* **Starter structure & tests**

  ```
  lab5_recsys/
  ├─ train_mf.py
  └─ tests/test_forward.py
  ```

  ```python
  def test_score_shape():
      from lab5_recsys.train_mf import MF
      import torch; m=MF(10,20,8)
      u=torch.tensor([0,1,2]); i=torch.tensor([1,3,5])
      s=m(u,i); assert s.shape==(3,)
  ```

**Comparison table — Lab 5**

| Method                  | When it shines                  | When it fails             | Data needs          | Inference cost | Interpretability | Typical metrics |
| ----------------------- | ------------------------------- | ------------------------- | ------------------- | -------------- | ---------------- | --------------- |
| Neural MF (dot‑product) | Warm users/items; implicit data | Extreme cold‑start        | 10k–1M interactions | Very low       | Low              | P\@K, R\@K      |
| Item‑item KNN           | Simplicity & transparency       | Personalization depth     | 1k–1M               | Low            | Medium           | P\@K            |
| BPR (pairwise)          | Ranking quality                 | Implementation complexity | 50k–10M             | Low–Med        | Low              | MAP, NDCG       |

---

### **Lab 6 — Generative modeling with a Variational Autoencoder (VAE) on Fashion‑MNIST**

**Purpose:** Learn representation learning & generation; practice **ELBO**, **KL annealing**, and **anomaly detection** via reconstruction error.

* **Problem & use cases:** Image generation, denoising, anomaly detection (manufacturing).

* **Method taught:** **VAE (MLP)** on 28×28 images; CPU‑friendly.
  **Alternatives:**

  * **Denoising autoencoder:** simpler, but no generative sampling.
  * **Diffusion models:** state‑of‑the‑art generation, but heavy compute.

* **Dataset:** **Fashion‑MNIST (MIT License)** via torchvision. ([GitHub][2])

#### 60‑minute intro

* **Objectives:** Implement VAE with reparameterization; monitor **recon vs KL**; sample images; simple anomaly scores.
* **Key ideas:** ELBO = Recon + KL; **β‑VAE** (trade‑off disentanglement vs fidelity); KL warm‑up; stability tricks.
* **Live code:** Encoder/decoder MLP; train few epochs; save sample grid; anomaly score on one class as “out‑of‑class”.
* **Diagram:** Input x → **Encoder** (μ, logσ²) → **reparam** z → **Decoder** → x̂; backprop via reparameterization trick.

#### Full working example

```python
# lab6_generative/train_vae.py
import os, random, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils

def set_seed(s=42):
    random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"]=str(s)
    torch.manual_seed(s); torch.use_deterministic_algorithms(True)

class VAE(nn.Module):
    def __init__(self, z=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(28*28,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU())
        self.mu = nn.Linear(128,z); self.logvar = nn.Linear(128,z)
        self.dec = nn.Sequential(nn.Linear(z,128), nn.ReLU(), nn.Linear(128,256), nn.ReLU(), nn.Linear(256,28*28), nn.Sigmoid())
    def forward(self, x, beta=1.0):
        h = self.enc(x); mu, logv = self.mu(h), self.logvar(h)
        std = torch.exp(0.5*logv)
        eps = torch.randn_like(std)
        z = mu + eps*std
        xrec = self.dec(z).view(-1,1,28,28)
        # losses
        recon = nn.functional.binary_cross_entropy(xrec, x, reduction='sum')/x.size(0)
        kl = -0.5*torch.sum(1 + logv - mu.pow(2) - logv.exp())/x.size(0)
        return xrec, recon + beta*kl, recon.item(), kl.item()

if __name__=="__main__":
    set_seed()
    tfm = transforms.ToTensor()
    tr = datasets.FashionMNIST(".data", train=True, download=True, transform=tfm)
    te = datasets.FashionMNIST(".data", train=False, download=True, transform=tfm)
    # CPU-friendly subset
    tr = Subset(tr, list(range(10000)))
    tr_ld = DataLoader(tr, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
    model = VAE(z=16); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(5):
        model.train(); tot=0; rsum=0; ksum=0
        beta = min(1.0, ep/3)  # KL warm-up
        for xb,yb in tr_ld:
            opt.zero_grad(); _, loss, recon, kl = model(xb, beta=beta); loss.backward(); opt.step()
            tot+=loss.item(); rsum+=recon; ksum+=kl
        print(f"ep{ep}: loss {tot/len(tr_ld):.2f} recon {rsum/len(tr_ld):.2f} kl {ksum/len(tr_ld):.2f}")
    # sample grid
    with torch.no_grad():
        z = torch.randn(64,16)
        imgs = model.dec(z).view(-1,1,28,28)
        utils.save_image(imgs, "samples.png", nrow=8)
        print("saved samples.png")
```

* **Metrics & targets:** Track **ELBO**, Recon, KL; visually inspect samples. **Acceptance:** Train loss drops ≥20% from epoch 0; produce **samples.png**.

* **Methodology focus:** Representation learning; β‑VAE trade‑offs; basic **anomaly detection** via reconstruction error; discuss **misuse risks** (deepfakes).

* **Why this over that?**

  * VAE gives a **probabilistic latent space** with sampling & interpolation; lightweight on CPU.
  * vs. Denoising autoencoder: simpler, but no explicit generative model;
  * vs. Diffusion: higher fidelity but heavy compute and tuning.

* **Common failure modes**

  * All gray samples → decoder saturation; reduce LR, increase epochs slightly.
  * KL collapse → use **KL warm‑up** or **β<1** early.

* **Starter structure & test**

  ```
  lab6_generative/
  ├─ train_vae.py
  └─ tests/test_shapes.py
  ```

  ```python
  def test_vae_shapes():
      from lab6_generative.train_vae import VAE
      import torch
      m=VAE(z=8); x=torch.rand(4,1,28,28)
      xr, loss, r, k = m(x, beta=1.0)
      assert xr.shape==(4,1,28,28)
  ```

**Comparison table — Lab 6**

| Method       | When it shines                            | When it fails                | Data needs | Inference cost | Interpretability          | Typical metrics     |
| ------------ | ----------------------------------------- | ---------------------------- | ---------- | -------------- | ------------------------- | ------------------- |
| VAE          | Fast generative modeling; anomaly scoring | Sharpness & fidelity limited | 10k–100k   | Low            | Medium (latent traversal) | ELBO                |
| Denoising AE | Denoising/compression                     | No true generative sampling  | 10k–100k   | Very low       | Low                       | Recon loss          |
| Diffusion    | Photorealistic gen                        | Heavy compute                | 100k+      | High           | Low                       | FID (not used here) |

---

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

## Dataset links & licenses (for quick reference)

* **Fashion‑MNIST** — MIT License (GitHub repository). ([GitHub][2])
* **UCI Adult (Census Income)** — Creative Commons **CC BY 4.0** (UCI page). ([UCI Machine Learning Repository][1])
* **SMS Spam Collection** — Creative Commons **CC BY 4.0** (UCI page). ([UCI Machine Learning Repository][3])
* **Electricity Load Diagrams 2011–2014** — Creative Commons **CC BY 4.0** (UCI page). ([UCI Machine Learning Repository][4])
* **MovieLens 100K** — research use license; acknowledgement required; no commercial use; redistribution requires permission (README). ([GroupLens][5])

---

## Why students still learn even with AI assistants

* **Acceptance tests** require *understanding* (e.g., beat naive baselines; calibration ECE thresholds; slice‑based metrics).
* **Structured modifications** (threshold tuning, early stopping, KL warm‑up, negative sampling) force contact with *core ideas*, not just code generation.
* **Error analysis writeups** require students to explain *why* a change helped/hurt, with logs and plots.
* AI can help draft code, but **only careful debugging & evaluation** passes the tests.

---

### Appendix: common nondeterminism sources & fixes (quick list)

* Data order & random sampling → **fix seeds**, `num_workers=0`, `shuffle=True` with seed, disable Python hash randomization.
* Dropout & initialization → seed and log initial weights’ stats.
* Different library versions → log `pip freeze` / `poetry export`.
* Floating‑point non‑associativity → expect tiny metric jitter; compare with tolerances in tests.

---

### One‑page acceptance‑test summary (all labs)

* **Lab 1 (Adult):** ROC‑AUC ≥ 0.88, ECE ≤ 0.08, unit tests pass.
* **Lab 2 (Vision):** acc ≥ 0.85, worst‑class recall ≥ 0.75, overfit‑one‑batch test passes.
* **Lab 3 (Text):** Macro‑F1 ≥ 0.90, vocab size bounds, shape tests pass.
* **Lab 4 (TS):** MAE improves ≥5% over naive on test; window test passes.
* **Lab 5 (RecSys):** P\@10 ≥ 0.07, R\@10 ≥ 0.10, forward shape test passes.
* **Lab 6 (VAE):** Loss ↓ by ≥20% from epoch 0, samples.png created, shape test passes.

---

> *All code is CPU‑friendly and uses small models, subsets, or few epochs. Set seeds, respect dataset licenses, and keep logs for reproducibility.*

[1]: https://archive.ics.uci.edu/dataset/2/adult?utm_source=chatgpt.com "Adult - UCI Machine Learning Repository"
[2]: https://github.com/zalandoresearch/fashion-mnist "GitHub - zalandoresearch/fashion-mnist: A MNIST-like fashion product database. Benchmark"
[3]: https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection?utm_source=chatgpt.com "SMS Spam Collection"
[4]: https://archive.ics.uci.edu/ml/datasets/electricityloaddiagrams20112014?utm_source=chatgpt.com "ElectricityLoadDiagrams20112014"
[5]: https://files.grouplens.org/datasets/movielens/ml-100k-README.txt "files.grouplens.org"
[6]: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html "files.grouplens.org"
