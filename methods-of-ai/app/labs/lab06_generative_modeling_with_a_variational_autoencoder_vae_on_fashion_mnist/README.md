# Lab 6 — Generative modeling with a Variational Autoencoder (VAE) on Fashion‑MNIST

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

[1]: https://archive.ics.uci.edu/dataset/2/adult?utm_source=chatgpt.com "Adult - UCI Machine Learning Repository"
[2]: https://github.com/zalandoresearch/fashion-mnist "GitHub - zalandoresearch/fashion-mnist: A MNIST-like fashion product database. Benchmark"
[3]: https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection?utm_source=chatgpt.com "SMS Spam Collection"
[4]: https://archive.ics.uci.edu/ml/datasets/electricityloaddiagrams20112014?utm_source=chatgpt.com "ElectricityLoadDiagrams20112014"
[5]: https://files.grouplens.org/datasets/movielens/ml-100k-README.txt "files.grouplens.org"
[6]: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html "files.grouplens.org"

