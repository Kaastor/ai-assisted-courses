"""Split OUTLINE.md into lab-specific READMEs and supporting docs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTLINE_PATH = ROOT / "OUTLINE.md"
DOCS_ROOT = ROOT / "docs"
LABS_ROOT = ROOT / "app" / "labs"


@dataclass
class LabSection:
    number: int
    title: str
    body: str

    @property
    def slug(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self.title.lower()).strip("_")
        return f"lab{self.number:02d}_{slug}"



def extract_sections() -> tuple[str, str, list[LabSection], str, str, str, str, str, str, str]:
    outline = OUTLINE_PATH.read_text(encoding="utf-8")
    footnote_match = re.search(r"\n\[1\]:", outline)
    if not footnote_match:
        raise ValueError("Footnotes not found in OUTLINE.md")
    footnotes = outline[footnote_match.start() :].strip()
    body = outline[: footnote_match.start()].rstrip()

    def slice_between(start: str, end: str | None) -> str:
        start_idx = body.index(start)
        if end is None:
            return body[start_idx:].rstrip()
        end_idx = body.index(end, start_idx)
        return body[start_idx:end_idx].rstrip()

    section1 = slice_between("## 1)", "## 2)")
    section2 = slice_between("## 2)", "## 3)")
    section3 = slice_between("## 3)", "## 4)")
    section4 = slice_between("## 4)", "## Dataset")
    datasets = slice_between("## Dataset", "## Why")
    ai_learning = slice_between("## Why", "### Appendix")
    appendix = slice_between("### Appendix", "### One")
    acceptance_summary = slice_between("### One", None)

    lab_pattern = re.compile(
        r"### \*\*Lab (?P<num>\d+) — (?P<title>.+?)\*\*\n(?P<body>.*?)(?=\n### \*\*Lab |\n-+\n\n## 3\)|\n## 3\)|\Z)",
        re.DOTALL,
    )
    labs: list[LabSection] = []
    for match in lab_pattern.finditer(section2):
        number = int(match.group("num"))
        title = match.group("title").strip()
        body_text = match.group("body").strip()
        labs.append(LabSection(number, title, body_text))
    labs.sort(key=lambda s: s.number)

    if not labs:
        raise ValueError("No lab sections were parsed from OUTLINE.md")
    labs_overview_end = section2.index("### **Lab 1")
    labs_overview = section2[:labs_overview_end].rstrip()

    return (
        section1,
        labs_overview,
        labs,
        section3,
        section4,
        datasets,
        ai_learning,
        appendix,
        acceptance_summary,
        footnotes,
    )



def write_text(path: Path, content: str, footnotes: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = content.rstrip()
    if footnotes:
        text = f"{text}\n\n{footnotes.strip()}\n"
    path.write_text(text + "\n", encoding="utf-8")



def main() -> None:
    (
        overview,
        labs_overview,
        labs,
        environment,
        capstone,
        datasets,
        ai_learning,
        appendix,
        acceptance_summary,
        footnotes,
    ) = extract_sections()

    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    write_text(DOCS_ROOT / "course_overview.md", "# Course Overview\n\n" + overview, footnotes)
    write_text(DOCS_ROOT / "labs_overview.md", "# Labs Overview\n\n" + labs_overview, footnotes)
    write_text(DOCS_ROOT / "environment_and_tooling.md", "# Environment & Tooling\n\n" + environment, footnotes)
    write_text(DOCS_ROOT / "capstone_reflection.md", "# Capstone Reflection Checklist\n\n" + capstone, footnotes)
    write_text(DOCS_ROOT / "dataset_links_and_licenses.md", "# Dataset Links & Licenses\n\n" + datasets, footnotes)
    write_text(DOCS_ROOT / "learning_with_ai_support.md", "# Learning with AI Support\n\n" + ai_learning, footnotes)
    write_text(DOCS_ROOT / "appendix.md", "# Appendix\n\n" + appendix, footnotes)
    write_text(
        DOCS_ROOT / "acceptance_summary.md",
        "# Acceptance Test Summary\n\n" + acceptance_summary,
        footnotes,
    )

    for lab in labs:
        lab_dir = LABS_ROOT / lab.slug
        lab_dir.mkdir(parents=True, exist_ok=True)
        heading = f"# Lab {lab.number} — {lab.title}\n\n"
        write_text(lab_dir / "README.md", heading + lab.body, footnotes)
        (lab_dir / "__init__.py").write_text("\"\"\"Lab package.\"\"\"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
