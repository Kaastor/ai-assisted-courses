"""Regression tests ensuring outline data integrity."""

from __future__ import annotations

import pytest

from app.assessments import ACCEPTANCE_SUMMARY, REFLECTION_CHECKLIST
from app.course import COURSE_OVERVIEW
from app.labs import iter_lab_specs
from app.tooling import ENVIRONMENT_GUIDELINES
from app.resources import DATASET_REFERENCES, DATASETS_DOC


def test_lab_specifications_have_complete_metadata() -> None:
    specs = list(iter_lab_specs())
    assert len(specs) == 6, "Expected six lab specifications"
    for spec in specs:
        readme_text = spec.readme_path.read_text(encoding="utf-8")
        normalized_readme = readme_text.replace("‑", "-")
        normalized_title = spec.title.replace("‑", "-")
        assert normalized_title in normalized_readme
        assert spec.comparison_table_markdown.strip().splitlines()[0] in readme_text
        condensed_readme = readme_text.replace(" ", "").replace("*", "")
        for acceptance in spec.acceptance_tests:
            normalized_threshold = acceptance.threshold.replace(" ", "")
            if acceptance.metric in {"pytest", "artifact"}:
                continue
            assert normalized_threshold in condensed_readme, (
                f"Threshold {acceptance.threshold} missing from README for {spec.identifier}"
            )


def test_course_overview_documents_exist() -> None:
    assert COURSE_OVERVIEW.course_overview_doc.exists()
    assert COURSE_OVERVIEW.labs_overview_doc.exists()


def test_environment_guidelines_reference_exists() -> None:
    assert ENVIRONMENT_GUIDELINES.reference_doc.exists()


def test_reflection_and_acceptance_summary_docs_exist() -> None:
    assert REFLECTION_CHECKLIST.reference_doc.exists()
    assert ACCEPTANCE_SUMMARY.reference_doc.exists()


@pytest.mark.parametrize("dataset_key", sorted(DATASET_REFERENCES))
def test_dataset_reference_docs(dataset_key: str) -> None:
    assert DATASET_REFERENCES[dataset_key].url
    assert DATASETS_DOC.exists()
