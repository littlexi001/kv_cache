from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = PROJECT_ROOT / "configs" / "p0_four_machine_plan.json"


def load_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("machines")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("the P0 plan must contain exactly four tasks")
    identifiers = [int(row["machine_id"]) for row in rows]
    if sorted(identifiers) != list(range(4)):
        raise ValueError(f"machine IDs must be 0..3, found {identifiers}")
    strategies = [str(row["strategy"]) for row in rows]
    if len(set(strategies)) != 4:
        raise ValueError("the four P0 strategies must be unique")
    return payload


def assignment(payload: dict[str, Any], machine_id: int) -> dict[str, Any]:
    for row in payload["machines"]:
        if int(row["machine_id"]) == machine_id:
            return row
    raise ValueError(f"unknown P0 task ID {machine_id}; expected 0 through 3")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--machine-id", type=int, required=True)
    parser.add_argument("--field", choices=["strategy", "role", "json"], default="strategy")
    args = parser.parse_args()
    row = assignment(load_plan(args.plan), args.machine_id)
    print(json.dumps(row, ensure_ascii=False, sort_keys=True) if args.field == "json" else row[args.field])


if __name__ == "__main__":
    main()
