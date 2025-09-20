from __future__ import annotations

import json
from dataclasses import asdict

from .variant import build_variant


def main() -> None:  # pragma: no cover - convenience CLI
    cfg = build_variant()
    print(json.dumps(asdict(cfg), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()

