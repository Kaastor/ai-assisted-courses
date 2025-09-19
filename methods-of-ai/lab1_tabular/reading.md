
# Supplementary Reading Handout — Tabular Classification, Calibration & Good ML Hygiene

**Use this as a quick, annotated map of what to read and why.**  
Legend: **★ Must‑read** (short & directly relevant to this lab).

---

## 0) PyTorch quick references (★ short, hands‑on)
- **★ Learn the Basics — PyTorch Tutorials.**  
  https://docs.pytorch.org/tutorials/beginner/basics/intro.html  
  *Why:* maps our lab’s workflow (data → model → train → save) to idiomatic PyTorch.

- **★ `nn.Embedding` documentation.**  
  https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html  
  *Why:* we use one embedding table *per categorical column*.

- **★ `BCEWithLogitsLoss` documentation.**  
  https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html  
  *Why:* explains why we don’t apply `sigmoid` inside the model.

- **★ `AdamW` optimizer** and **`StepLR` scheduler**.  
  AdamW: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html  
  StepLR: https://docs.pytorch.org/docs/2.8/generated/torch.optim.lr_scheduler.StepLR.html  
  *Why:* matches the lab’s optimizer/schedule choices.

- **Learning PyTorch with Examples (Justin Johnson).**  
  https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html  
  *Why:* quick tensor‑to‑Module crash course.

---

## 1) Tabular deep learning & embeddings
- **★ Entity Embeddings of Categorical Variables (Guo & Berkhahn, 2016).**  
  https://arxiv.org/abs/1604.06737  
  *Why:* core idea behind replacing one‑hots with dense per‑category vectors.

- **★ Why do tree‑based models still outperform deep learning on tabular data? (Grinsztajn et al., NeurIPS 2022).**  
  arXiv: https://arxiv.org/abs/2207.08815  • NeurIPS PDF: https://papers.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf  
  *Why:* honest baseline context for where MLPs help vs. gradient‑boosted trees.

- **Deep Neural Networks and Tabular Data: A Survey (Borisov et al., 2021).**  
  https://arxiv.org/abs/2110.01889  
  *Why:* broad overview of architectures and regularization for tabular data.

- **Revisiting Deep Learning Models for Tabular Data — FT‑Transformer (Gorishniy et al., 2021).**  
  https://arxiv.org/pdf/2106.11959  
  *Why:* state‑of‑the‑art transformer variant for tabular inputs.

---

## 2) Optimization & regularization (used in the lab)
- **★ Decoupled Weight Decay Regularization = AdamW (Loshchilov & Hutter, 2017).**  
  https://arxiv.org/abs/1711.05101  
  *Why:* why “Adam + L2” ≠ “AdamW”, and why decoupled weight decay generalizes better.

- **★ Dropout: A Simple Way to Prevent Neural Networks from Overfitting (Srivastava et al., JMLR 2014).**  
  JMLR: https://jmlr.org/papers/v15/srivastava14a.html  • PDF: https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf  
  *Why:* the regularizer we use in the MLP head.

- **Early Stopping — But When? (Prechelt, ‘Neural Networks: Tricks of the Trade’).**  
  PDF: https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf  • Springer: https://link.springer.com/chapter/10.1007/978-3-642-35289-8_5  
  *Why:* theory & practice for validation‑based early stopping.

---

## 3) Evaluation & decision‑making (AUC, PR, thresholds)
- **★ An Introduction to ROC Analysis (Fawcett, 2006).**  
  https://people.inf.elte.hu/kiss/13dwhdm/roc.pdf  
  *Why:* intuition for ROC/AUC and choosing operating points.

- **The Relationship Between Precision‑Recall and ROC Curves (Davis & Goadrich, ICML 2006).**  
  https://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf  
  *Why:* when PR is more informative than ROC (class imbalance).

---

## 4) Probability calibration (centerpiece of this lab)
- **★ On Calibration of Modern Neural Networks (Guo et al., ICML 2017).**  
  PMLR PDF: https://proceedings.mlr.press/v70/guo17a/guo17a.pdf  
  *Why:* shows modern nets are over‑confident; **temperature scaling** is a simple, strong fix.

- **★ Predicting Good Probabilities with Supervised Learning (Niculescu‑Mizil & Caruana, ICML 2005).**  
  PDF: https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf  
  *Why:* classic study of probability quality across models.

- **Platt scaling (Platt, 1999).**  
  PDF: https://home.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf  
  *Why:* logistic calibration for margins/scores.

- **Isotonic regression for calibration (Zadrozny & Elkan, KDD 2002).**  
  PDF: https://www.cs.cornell.edu/courses/cs678/2007sp/ZadroznyElkan.pdf  
  *Why:* non‑parametric alternative to sigmoid scaling.

- **Beta Calibration (Kull, Silva Filho, Flach, AISTATS 2017).**  
  PDF: https://proceedings.mlr.press/v54/kull17a/kull17a.pdf  
  *Why:* simple, often stronger variant than logistic calibration.

- **Reliability diagrams & ECE:**  
  BBQ (Naeini et al., AAAI 2015): https://ojs.aaai.org/index.php/AAAI/article/view/9602  • scikit‑learn guide: https://scikit-learn.org/stable/modules/calibration.html  • API (`CalibratedClassifierCV`): https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html  
  *Why:* concepts & practical tools to *see* and fix miscalibration.

---

## 5) Data hygiene: avoiding **leakage**
- **★ Leakage in Data Mining: Formulation, Detection, and Avoidance (Kaufman & Rosset et al., TKDD 2012).**  
  Open PDF mirror: https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf  • ACM: https://dl.acm.org/doi/10.1145/2382577.2382579  
  *Why:* why we compute scalers/encoders on *train only* and keep a held‑out validation set.

- **On Leakage in Machine Learning Pipelines (Sasse et al., 2023).**  
  arXiv PDF: https://arxiv.org/pdf/2311.04179  
  *Why:* modern catalogue of leakage modes across real ML workflows.

---

## 6) Fairness & responsible ML (relevant to Adult/Census features)
- **★ *Fairness and Machine Learning: Limitations and Opportunities* (Barocas, Hardt, Narayanan) — free book.**  
  https://fairmlbook.org/  • PDF: https://fairmlbook.org/pdf/fairmlbook.pdf  
  *Why:* rigorous foundation for working with demographic attributes.

- **Model Cards for Model Reporting (Mitchell et al., 2019).**  
  arXiv PDF: https://arxiv.org/pdf/1810.03993  
  *Why:* how to document scope, per‑group performance, and caveats.

- **Datasheets for Datasets (Gebru et al., 2018).**  
  arXiv PDF: https://arxiv.org/pdf/1803.09010  
  *Why:* standardizing dataset documentation and limitations.

- **Fairlearn — User Guide & Metrics.**  
  Guide: https://fairlearn.org/main/user_guide/index.html  • Metrics: https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html  
  *Why:* practical tooling for disaggregated metrics and parity analysis.

- **Retiring Adult: New Datasets for Fair ML (Ding, Hardt, Miller, Schmidt, NeurIPS 2021).**  
  PDF: https://proceedings.neurips.cc/paper/2021/file/32e54441e6382a7fbacbbbaf3c450059-Paper.pdf  
  *Why:* context on why the classic “Adult” dataset has issues and modern alternatives.

---

## 7) Dataset background (for your EDA notes)
- **UCI Machine Learning Repository — Adult (Census Income).**  
  https://archive.ics.uci.edu/dataset/2/adult  
  *Why:* official dataset description, features, provenance.

---

## Suggested learning path (time‑boxed)
1. **Start (1–2h):** PyTorch basics; `Embedding`, `BCEWithLogitsLoss`, `AdamW`, `StepLR`.  
2. **Then (2–3h):** Entity embeddings + tabular DL vs. trees; ROC vs. PR.  
3. **Core (2h):** Calibration (Guo 2017), scikit‑learn calibration guide, reliability diagrams.  
4. **Hygiene (1–2h):** AdamW, dropout, early stopping; leakage primer.  
5. **Responsible use (1–2h):** Fairness book chapters; Model Cards; Datasheets.

---

*Tip:* As you read, relate each item back to the lab: embeddings ↔ categorical columns, `BCEWithLogitsLoss` ↔ logits, AdamW/StepLR ↔ training loop, ROC‑AUC/F1/ECE ↔ evaluation, and temperature scaling ↔ calibrated probabilities.