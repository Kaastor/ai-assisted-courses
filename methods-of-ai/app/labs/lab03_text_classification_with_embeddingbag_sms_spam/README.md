# Lab 3 — Text classification with EmbeddingBag (SMS Spam)

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

[1]: https://archive.ics.uci.edu/dataset/2/adult?utm_source=chatgpt.com "Adult - UCI Machine Learning Repository"
[2]: https://github.com/zalandoresearch/fashion-mnist "GitHub - zalandoresearch/fashion-mnist: A MNIST-like fashion product database. Benchmark"
[3]: https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection?utm_source=chatgpt.com "SMS Spam Collection"
[4]: https://archive.ics.uci.edu/ml/datasets/electricityloaddiagrams20112014?utm_source=chatgpt.com "ElectricityLoadDiagrams20112014"
[5]: https://files.grouplens.org/datasets/movielens/ml-100k-README.txt "files.grouplens.org"
[6]: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html "files.grouplens.org"

