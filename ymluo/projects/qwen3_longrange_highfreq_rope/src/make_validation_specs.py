from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_highfreq_specs import discovery_specs


SELECTED = (
    "native_rope",
    "global_f00_07_delete",
    "late_f00_07_delete",
    "deep_f08_15_delete",
)


def validation_specs() -> list[dict]:
    by_name = {spec["name"]: spec for spec in discovery_specs()}
    return [by_name[name] for name in SELECTED]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"stage": "validation", "specs": validation_specs()}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
