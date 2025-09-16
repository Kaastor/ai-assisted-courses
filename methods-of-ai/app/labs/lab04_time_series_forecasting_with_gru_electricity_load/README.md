# Lab 4 — Time‑series forecasting with GRU (Electricity Load)

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

[1]: https://archive.ics.uci.edu/dataset/2/adult?utm_source=chatgpt.com "Adult - UCI Machine Learning Repository"
[2]: https://github.com/zalandoresearch/fashion-mnist "GitHub - zalandoresearch/fashion-mnist: A MNIST-like fashion product database. Benchmark"
[3]: https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection?utm_source=chatgpt.com "SMS Spam Collection"
[4]: https://archive.ics.uci.edu/ml/datasets/electricityloaddiagrams20112014?utm_source=chatgpt.com "ElectricityLoadDiagrams20112014"
[5]: https://files.grouplens.org/datasets/movielens/ml-100k-README.txt "files.grouplens.org"
[6]: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html "files.grouplens.org"

