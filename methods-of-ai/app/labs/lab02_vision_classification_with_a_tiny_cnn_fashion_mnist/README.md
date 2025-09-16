# Lab 2 — Vision classification with a tiny CNN (Fashion‑MNIST)

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

[1]: https://archive.ics.uci.edu/dataset/2/adult?utm_source=chatgpt.com "Adult - UCI Machine Learning Repository"
[2]: https://github.com/zalandoresearch/fashion-mnist "GitHub - zalandoresearch/fashion-mnist: A MNIST-like fashion product database. Benchmark"
[3]: https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection?utm_source=chatgpt.com "SMS Spam Collection"
[4]: https://archive.ics.uci.edu/ml/datasets/electricityloaddiagrams20112014?utm_source=chatgpt.com "ElectricityLoadDiagrams20112014"
[5]: https://files.grouplens.org/datasets/movielens/ml-100k-README.txt "files.grouplens.org"
[6]: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html "files.grouplens.org"

