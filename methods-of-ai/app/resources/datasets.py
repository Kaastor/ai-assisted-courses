"""Dataset reference information keyed by lab usage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    license: str
    url: str
    notes: str


DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"

DATASET_REFERENCES: dict[str, DatasetInfo] = {
    "fashion_mnist": DatasetInfo(
        name="Fashion-MNIST",
        license="MIT License",
        url="https://github.com/zalandoresearch/fashion-mnist",
        notes="Used in Labs 2 and 6; built into torchvision and supports CPU-only workflows.",
    ),
    "uci_adult": DatasetInfo(
        name="UCI Adult (Census Income)",
        license="CC BY 4.0",
        url="https://archive.ics.uci.edu/dataset/2/adult?utm_source=chatgpt.com",
        notes="Tabular dataset with demographic attributes; highlight leakage and fairness considerations.",
    ),
    "sms_spam": DatasetInfo(
        name="SMS Spam Collection",
        license="CC BY 4.0",
        url="https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection?utm_source=chatgpt.com",
        notes="5,574 SMS messages for binary spam classification; available as a UCI zip archive.",
    ),
    "electricity_load": DatasetInfo(
        name="Electricity Load Diagrams 2011–2014",
        license="CC BY 4.0",
        url="https://archive.ics.uci.edu/ml/datasets/electricityloaddiagrams20112014?utm_source=chatgpt.com",
        notes="Hourly electricity consumption; subset to one client and resample hourly for CPU-friendly training.",
    ),
    "movielens_100k": DatasetInfo(
        name="MovieLens 100K",
        license="GroupLens research-only",
        url="https://files.grouplens.org/datasets/movielens/ml-100k-README.txt",
        notes="Implicit feedback lab (Lab 5); requires acknowledgement and prohibits commercial use.",
    ),
    "movielens_latest_small": DatasetInfo(
        name="MovieLens latest small",
        license="GroupLens research-only",
        url="https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html",
        notes="Backup dataset with similar license conditions when ml-100k access is constrained.",
    ),
}

DATASETS_DOC = DOCS_ROOT / "dataset_links_and_licenses.md"
