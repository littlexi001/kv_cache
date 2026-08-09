#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/riskkv_operator_contract_schema_v461_20260713.json"
VARIANTS = {
    "base": {},
    "closure20": {
        "graph_bridge": True,
        "graph_bridge_budget_fraction": 0.20,
        "graph_bridge_seed_pages": 8,
        "graph_bridge_max_terms": 24,
        "graph_bridge_min_score": 0.0,
    },
    "closure35": {
        "graph_bridge": True,
        "graph_bridge_budget_fraction": 0.35,
        "graph_bridge_seed_pages": 12,
        "graph_bridge_max_terms": 32,
        "graph_bridge_min_score": 0.0,
    },
}


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    for name, overrides in VARIANTS.items():
        variant = json.loads(json.dumps(payload))
        variant["__comment"] = (
            "v464 paired retrieve diagnostic: fixed 1024-token action with bounded entity-evidence closure. "
            "The base arm has no closure and all arms use the same contract parser and QA sample split."
        )
        variant["__operator_router"]["actions"]["retrieve"].update(overrides)
        path = ROOT / f"configs/riskkv_operator_contract_retrieve_v464_{name}_20260713.json"
        path.write_text(json.dumps(variant, indent=2) + "\n", encoding="utf-8")
        print(path)

    v465 = json.loads(json.dumps(payload))
    v465["__comment"] = (
        "v465 runnable Pareto configuration: 35% bounded entity-evidence closure for retrieve "
        "and a 512-token recent-heavy code action."
    )
    v465["__operator_router"]["actions"]["retrieve"].update(VARIANTS["closure35"])
    v465["__operator_router"]["actions"]["code"].update(
        {"budget_tokens": 512, "page_tokens": 64, "sink_tokens": 32, "recent_tokens": 256}
    )
    v465_path = ROOT / "configs/riskkv_operator_contract_v465_closure35_code512_20260713.json"
    v465_path.write_text(json.dumps(v465, indent=2) + "\n", encoding="utf-8")
    print(v465_path)

    v466 = json.loads(json.dumps(v465))
    v466["__comment"] = (
        "v466 sub-10% KV frontier probe: v465 evidence closure with retrieve=896 and code=256. "
        "The two operator changes are evaluated independently before composing the final action set."
    )
    v466["__operator_router"]["actions"]["retrieve"]["budget_tokens"] = 896
    v466["__operator_router"]["actions"]["code"].update(
        {"budget_tokens": 256, "page_tokens": 32, "sink_tokens": 32, "recent_tokens": 160}
    )
    v466_path = ROOT / "configs/riskkv_operator_contract_v466_retrieve896_code256_20260713.json"
    v466_path.write_text(json.dumps(v466, indent=2) + "\n", encoding="utf-8")
    print(v466_path)


if __name__ == "__main__":
    main()
