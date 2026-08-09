from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_progressive_spectral_specs import progressive_specs


SELECTED = {"progressive_late", "progressive_conservative"}


def validation_specs() -> list[dict]:
    return [spec for spec in progressive_specs() if spec["name"] in SELECTED]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"stage": "progressive_validation", "specs": validation_specs()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
