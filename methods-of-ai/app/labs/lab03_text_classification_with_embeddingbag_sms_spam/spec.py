"""Structured metadata for Lab 3."""

from __future__ import annotations

from pathlib import Path

from app.labs.base import AcceptanceTest, DatasetReference, LabSpecification

README_PATH = Path(__file__).with_name("README.md")

LAB_SPEC = LabSpecification(
    identifier="lab03_text",
    title="Text classification with EmbeddingBag (SMS Spam)",
    domain="Text classification",
    purpose=(
        "Guide students through tokenization, vocab building, imbalance handling, and macro-F1 evaluation "
        "for SMS spam detection."
    ),
    method_summary=(
        "EmbeddingBag plus linear classifier trained with BCEWithLogitsLoss, using stratified splits and threshold tuning."
    ),
    dataset=DatasetReference(
        name="SMS Spam Collection",
        license="CC BY 4.0",
        url="https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection?utm_source=chatgpt.com",
        notes="5,574 labeled SMS messages suitable for lightweight text classification exercises.",
    ),
    acceptance_tests=(
        AcceptanceTest(
            description="Macro-averaged F1 score on held-out test data",
            metric="Macro F1",
            threshold="≥ 0.90",
            dataset_split="test",
        ),
        AcceptanceTest(
            description="Vocabulary size stays within configured bounds",
            metric="Vocabulary size",
            threshold="3k–30k",
            dataset_split="train",
        ),
        AcceptanceTest(
            description="Model forward shape/unit checks",
            metric="pytest",
            threshold="pass",
            dataset_split="synthetic",
        ),
    ),
    key_focus=(
        "Tokenization and vocab construction with reproducible seeds",
        "Handling class imbalance via weighting and threshold tuning",
        "Stratified train/validation/test splits for text data",
        "Out-of-vocabulary mitigation using embedding fallbacks",
        "Slice-based evaluation by message length",
    ),
    failure_modes=(
        "Near-zero F1 due to tokenization or label mapping issues",
        "Validation outperforming test metrics because of vocabulary leakage",
    ),
    assignment_seed=(
        "Plot precision-recall curve and select a validation-optimized threshold",
        "Add character-level fallback representations for heavy OOV scenarios",
    ),
    starter_code=(
        "lab3_text/train_spam.py",
        "lab3_text/vocab.py",
        "lab3_text/tests/test_shapes.py",
    ),
    stretch_goals=(),
    readings=(),
    comparison_table_markdown="""| Method                | When it shines                        | When it fails                   | Data needs   | Inference cost | Interpretability         | Typical metrics |
| --------------------- | ------------------------------------- | ------------------------------- | ------------ | -------------- | ------------------------ | --------------- |
| EmbeddingBag + Linear | Short texts, CPU-only, quick training | Long context; nuanced semantics | 5k–100k msgs | Very low       | Medium (n-gram saliency) | Macro F1        |
| TextCNN / BiLSTM      | Order matters, still light            | Very long docs                  | 10k–100k     | Low–Med        | Medium                   | Macro F1        |
| DistilBERT features   | Best accuracy                         | Download/inference cost         | 1k–50k       | Med–High       | Low                      | Macro F1        |
""",
    readme_path=README_PATH,
)
