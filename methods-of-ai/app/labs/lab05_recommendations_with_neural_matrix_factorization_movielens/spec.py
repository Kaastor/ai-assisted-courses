"""Structured metadata for Lab 5."""

from __future__ import annotations

from pathlib import Path

from app.labs.base import AcceptanceTest, DatasetReference, LabSpecification

README_PATH = Path(__file__).with_name("README.md")

LAB_SPEC = LabSpecification(
    identifier="lab05_recsys",
    title="Recommendations with neural matrix factorization (MovieLens)",
    domain="Recommendation systems",
    purpose=(
        "Construct an implicit-feedback recommender with user/item embeddings, negative sampling, and top-K evaluation."
    ),
    method_summary=(
        "Neural matrix factorization with dot products, BCE loss, and leave-last-N evaluation per user."
    ),
    dataset=DatasetReference(
        name="MovieLens 100K",
        license="GroupLens research use license",
        url="https://files.grouplens.org/datasets/movielens/ml-100k-README.txt",
        notes=(
            "Requires acknowledgement, is non-commercial, and must be downloaded directly by students; "
            "ml-latest-small offers similar terms."
        ),
    ),
    acceptance_tests=(
        AcceptanceTest(
            description="Precision@10 on held-out interactions",
            metric="Precision@10",
            threshold="≥ 0.07",
            dataset_split="test",
        ),
        AcceptanceTest(
            description="Recall@10 across users",
            metric="Recall@10",
            threshold="≥ 0.10",
            dataset_split="test",
        ),
        AcceptanceTest(
            description="Matrix factorization scoring shape checks",
            metric="pytest",
            threshold="pass",
            dataset_split="synthetic",
        ),
    ),
    key_focus=(
        "User-level chronological splits (leave-last-N)",
        "Negative sampling for implicit feedback",
        "Top-K ranking metrics including Precision@K and Recall@K",
        "Popularity baseline comparisons and cold-start considerations",
        "Ethical discussion of filter bubbles and fairness",
    ),
    failure_modes=(
        "Zero hits at evaluation caused by ID indexing mismatches",
        "Overly popular recommendations due to insufficient negative sampling or regularization",
    ),
    assignment_seed=(),
    starter_code=(
        "lab5_recsys/train_mf.py",
        "lab5_recsys/tests/test_forward.py",
    ),
    stretch_goals=(),
    readings=(),
    comparison_table_markdown="""| Method                  | When it shines                  | When it fails             | Data needs          | Inference cost | Interpretability | Typical metrics |
| ----------------------- | ------------------------------- | ------------------------- | ------------------- | -------------- | ---------------- | --------------- |
| Neural MF (dot-product) | Warm users/items; implicit data | Extreme cold-start        | 10k–1M interactions | Very low       | Low              | P@K, R@K        |
| Item-item KNN           | Simplicity & transparency       | Personalization depth     | 1k–1M               | Low            | Medium           | P@K             |
| BPR (pairwise)          | Ranking quality                 | Implementation complexity | 50k–10M             | Low–Med        | Low              | MAP, NDCG       |
""",
    readme_path=README_PATH,
)
