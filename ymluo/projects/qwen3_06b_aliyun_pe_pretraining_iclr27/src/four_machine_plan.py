from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = PROJECT_ROOT / "configs" / "four_machine_plan.json"


def load_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    machines = payload.get("machines")
    if not isinstance(machines, list) or len(machines) != 4:
        raise ValueError("the four-machine plan must contain exactly four assignments")
    identifiers = [int(row["machine_id"]) for row in machines]
    strategies = [str(row["strategy"]) for row in machines]
    if sorted(identifiers) != [0, 1, 2, 3]:
        raise ValueError(f"machine IDs must be exactly 0,1,2,3; found {identifiers}")
    if len(strategies) != len(set(strategies)):
        raise ValueError("each machine must receive a different strategy")
    for strategy in strategies:
        config = PROJECT_ROOT / "configs" / "strategies" / f"{strategy}.json"
        if not config.is_file():
            raise FileNotFoundError(f"missing strategy config: {config}")
    if "native_rope" not in strategies:
        raise ValueError("a matched-token native_rope control is mandatory")
    return payload


def assignment(payload: dict[str, Any], machine_id: int) -> dict[str, Any]:
    for row in payload["machines"]:
        if int(row["machine_id"]) == machine_id:
            return row
    raise ValueError(f"unknown machine ID {machine_id}; expected 0, 1, 2, or 3")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--machine-id", type=int, required=True)
    parser.add_argument("--field", choices=["strategy", "role", "json"], default="strategy")
    args = parser.parse_args()
    row = assignment(load_plan(args.plan), args.machine_id)
    if args.field == "json":
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    else:
        print(row[args.field])


if __name__ == "__main__":
    main()
