"""Structured metadata for Lab 1."""

from __future__ import annotations

from pathlib import Path

from app.labs.base import AcceptanceTest, DatasetReference, LabSpecification

README_PATH = Path(__file__).with_name("README.md")

LAB_SPEC = LabSpecification(
    identifier="lab01_tabular",
    title="Tabular classification with an MLP (UCI Adult)",
    domain="Tabular classification",
    purpose=(
        "First end-to-end PyTorch project covering leakage-free splits, "
        "categorical embeddings, and reliable evaluation on the Adult dataset."
    ),
    method_summary=(
        "Multilayer perceptron with per-column categorical embeddings and "
        "standardized numerical features, trained with AdamW and deterministic seeds."
    ),
    dataset=DatasetReference(
        name="UCI Adult (Census Income)",
        license="CC BY 4.0",
        url="https://archive.ics.uci.edu/dataset/2/adult?utm_source=chatgpt.com",
        notes=(
            "Classic demographic dataset with known leakage pitfalls; enables slice-based "
            "fairness analysis across sensitive attributes."
        ),
    ),
    acceptance_tests=(
        AcceptanceTest(
            description="Starter unit tests covering model shapes and overfit-one-batch",
            metric="pytest",
            threshold="pass",
            dataset_split="synthetic",
        ),
        AcceptanceTest(
            description="ROC-AUC performance on held-out test data",
            metric="ROC-AUC",
            threshold="≥ 0.88",
            dataset_split="test",
        ),
        AcceptanceTest(
            description="Calibration quality on validation data",
            metric="ECE",
            threshold="≤ 0.08",
            dataset_split="validation",
        ),
    ),
    key_focus=(
        "Preventing train/validation/test leakage via fit-on-train encoders and scalers",
        "Reproducibility with seeds, deterministic algorithms, and logging",
        "Baseline comparison with accuracy/F1/ROC-AUC metrics and confusion analysis",
        "Regularization with dropout, weight decay, and early stopping heuristics",
        "Slice-based error analysis on sensitive demographic attributes",
    ),
    failure_modes=(
        "Validation loss higher than training due to underfitting or aggressive dropout",
        "Strong validation but weak test metrics indicating preprocessing leakage",
        "Good ROC-AUC yet poor F1 suggesting threshold tuning or calibration needs",
    ),
    assignment_seed=(
        "Implement early stopping coupled with a StepLR scheduler (gamma=0.5 every 2 epochs)",
        "Add temperature scaling using validation data and report ECE",
    ),
    starter_code=(
        "lab1_tabular/data.py",
        "lab1_tabular/model.py",
        "lab1_tabular/train_tabular.py",
        "lab1_tabular/eval.py",
        "lab1_tabular/tests/test_sanity.py",
    ),
    stretch_goals=(
        "Swap embeddings for one-hot encodings to compare speed and accuracy",
        "Experiment with L2 feature normalization versus standardization",
        "Implement per-group thresholding to equalize opportunity across demographics",
    ),
    readings=(
        "Practical tips for ML reproducibility (course note)",
        "Confusion matrix & ROC/PR curves (scikit-learn documentation)",
        "Responsible AI bias and harm mitigation checklist",
    ),
    comparison_table_markdown="""| Method                 | When it shines                                  | When it fails                                  | Data needs  | Inference cost | Interpretability           | Typical metrics  |
| ---------------------- | ----------------------------------------------- | ---------------------------------------------- | ----------- | -------------- | -------------------------- | ---------------- |
| MLP + cat embeddings   | Mixed dtypes, many categoricals, joint training | Small data, heavy feature engineering required | 10k–1M rows | Medium         | Medium (feature ablations) | Acc, F1, ROC-AUC |
| Gradient-boosted trees | Tabular with limited data, quick iteration      | Very high-cardinality categoricals             | 1k–1M rows  | Low            | High (SHAP)                | Acc, F1, ROC-AUC |
| Logistic regression    | Need maximum simplicity/interpretability        | Nonlinear boundaries                           | 100+ rows   | Very low       | Very high                  | Acc, ROC-AUC     |
""",
    readme_path=README_PATH,
)
