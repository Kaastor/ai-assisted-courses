# Lab 5 — Recommendations with neural matrix factorization (MovieLens)

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

[1]: https://archive.ics.uci.edu/dataset/2/adult?utm_source=chatgpt.com "Adult - UCI Machine Learning Repository"
[2]: https://github.com/zalandoresearch/fashion-mnist "GitHub - zalandoresearch/fashion-mnist: A MNIST-like fashion product database. Benchmark"
[3]: https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection?utm_source=chatgpt.com "SMS Spam Collection"
[4]: https://archive.ics.uci.edu/ml/datasets/electricityloaddiagrams20112014?utm_source=chatgpt.com "ElectricityLoadDiagrams20112014"
[5]: https://files.grouplens.org/datasets/movielens/ml-100k-README.txt "files.grouplens.org"
[6]: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html "files.grouplens.org"

