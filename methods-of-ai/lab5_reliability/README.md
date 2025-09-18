# Lab 5 — Reliability, Calibration, and Error Analysis

This lab teaches students how to **trust but verify** their deep learning models. Rather than training a new network, they load checkpoints produced in earlier labs, run them against curated evaluation splits, and build a reliability report.

Core activities:

- Collect logits/probabilities on validation data with deterministic seeds.
- Compute classification and calibration metrics (accuracy, precision/recall/F1, ROC-AUC, expected calibration error).
- Slice results by demographic or feature buckets to surface blind spots.
- Apply targeted perturbations to inputs and quantify the impact.
- Summarise findings and remediation ideas in a short method-choice memo.

The starter code in `analyze.py` exposes a composable pipeline: gather predictions → compute overall metrics → compute slice metrics → produce a JSON/Markdown report. Acceptance tests cover metric correctness, calibration curve binning, and slice filtering logic. Students extend the tooling to match the dataset used in their project.
