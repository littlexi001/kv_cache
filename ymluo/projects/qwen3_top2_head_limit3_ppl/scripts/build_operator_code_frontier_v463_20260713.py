#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/riskkv_operator_contract_schema_v461_20260713.json"
VARIANTS = {
    "c512": {"budget_tokens": 512, "page_tokens": 64, "sink_tokens": 32, "recent_tokens": 256},
    "c1024": {"budget_tokens": 1024, "page_tokens": 64, "sink_tokens": 32, "recent_tokens": 512},
    "c1536": {"budget_tokens": 1536, "page_tokens": 128, "sink_tokens": 64, "recent_tokens": 768},
}


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    for name, overrides in VARIANTS.items():
        variant = json.loads(json.dumps(payload))
        variant["__comment"] = (
            "v463 code locality frontier: generic contract router with a recent-heavy code action; "
            "all non-code actions are unchanged from v461."
        )
        variant["__operator_router"]["actions"]["code"].update(overrides)
        path = ROOT / f"configs/riskkv_operator_contract_code_v463_{name}_20260713.json"
        path.write_text(json.dumps(variant, indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
