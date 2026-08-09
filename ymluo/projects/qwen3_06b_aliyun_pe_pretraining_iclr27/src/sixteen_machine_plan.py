from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = PROJECT_ROOT / "configs" / "sixteen_machine_plan.json"


def load_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("machines")
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("the pretraining plan must contain exactly 16 tasks")
    identifiers = [int(row["machine_id"]) for row in rows]
    if sorted(identifiers) != list(range(16)):
        raise ValueError(f"machine IDs must be 0..15, found {identifiers}")
    strategies = [str(row["strategy"]) for row in rows]
    if len(set(strategies)) != 16:
        raise ValueError("all sixteen strategies must be unique")
    if strategies[0] != "native_rope":
        raise ValueError("task 0 must be the native_rope control")
    for row in rows:
        strategy = PROJECT_ROOT / "configs" / "strategies" / f"{row['strategy']}.json"
        method_doc = PROJECT_ROOT / str(row["method_doc"])
        if not strategy.is_file():
            raise FileNotFoundError(f"missing strategy config: {strategy}")
        if not method_doc.is_file():
            raise FileNotFoundError(f"missing method document: {method_doc}")
    return payload


def assignment(payload: dict[str, Any], machine_id: int) -> dict[str, Any]:
    for row in payload["machines"]:
        if int(row["machine_id"]) == machine_id:
            return row
    raise ValueError(f"unknown task ID {machine_id}; expected 0 through 15")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--machine-id", type=int, required=True)
    parser.add_argument(
        "--field", choices=["strategy", "family", "method_doc", "json"], default="strategy"
    )
    args = parser.parse_args()
    row = assignment(load_plan(args.plan), args.machine_id)
    print(json.dumps(row, ensure_ascii=False, sort_keys=True) if args.field == "json" else row[args.field])


if __name__ == "__main__":
    main()
