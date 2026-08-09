from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "papers"
    / "countcap_iclr2027"
    / "scripts"
    / "make_qksieve_quality_generalization_figure.py"
)
SPEC = importlib.util.spec_from_file_location("qksieve_paper_figure", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def frozen_contract() -> dict[str, object]:
    return {
        "method": "qksieve_robust",
        "budget": "min(N,1280,max(256,ceil(0.06*N)))",
        "full_attention_fallback": False,
        "length_switch": False,
        "value_sketch": {
            "rank": 16,
            "bits": 4,
            "block_tokens": 256,
            "tail_alpha": 0.5,
        },
    }


def valid_payloads() -> tuple[dict[str, object], dict[str, object]]:
    contract = frozen_contract()
    method = str(contract["method"])
    per_length = {
        length: {
            "full_kv": {"score": 1.0},
            method: {"score": 0.99},
        }
        for length in MODULE.RULER_LENGTHS
    }
    ruler: dict[str, object] = {
        "schema": "qksieve_robust_ruler_summary_v1",
        "strict_pairs": 650,
        "rows": 1300,
        "tasks": [f"task{i}" for i in range(13)],
        "per_length": per_length,
        "fallback_count": 0,
        "bootstrap": {"quality_retention_95ci": [0.98, 1.0]},
        "frozen_contract": contract,
    }
    multimodel: dict[str, object] = {
        "schema": "qksieve_robust_multimodel_summary_v1",
        "frozen_contract": deepcopy(contract),
        "models": {
            name: {
                "strict_pairs": 160,
                "tasks": 16,
                "quality_retention": 0.99,
                "quality_retention_95ci": [0.98, 1.0],
            }
            for name in MODULE.MODEL_ORDER
        },
    }
    return ruler, multimodel


def test_complete_frozen_evidence_is_accepted() -> None:
    ruler, multimodel = valid_payloads()
    assert MODULE.validate(ruler, multimodel) == "qksieve_robust"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("strict_pairs",), 649),
        (("fallback_count",), 1),
        (("frozen_contract", "length_switch"), True),
        (("frozen_contract", "value_sketch", "tail_alpha"), 0.6),
    ],
)
def test_partial_or_drifted_ruler_evidence_is_rejected(
    path: tuple[str, ...], value: object
) -> None:
    ruler, multimodel = valid_payloads()
    target: dict[str, object] = ruler
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        MODULE.validate(ruler, multimodel)
