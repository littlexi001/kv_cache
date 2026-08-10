from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from verify_qksieve_shrinkage_grid_equivalence_20260810 import (  # noqa: E402
    verify,
)


FIELDS = [
    "label",
    "layer",
    "heldout_step",
    "kv_head",
    "query_head",
    "method",
    "selected_fraction_target",
    "top2_recall",
    "selected_attention_mass",
    "oracle_top2_attention_mass",
    "top2_attention_mass_recall",
    "score_pearson",
    "score_rmse",
]


def write_artifact(root: Path, delta: float, *, duplicate: bool = False) -> None:
    root.mkdir(parents=True)
    row = {
        "label": "trace",
        "layer": 0,
        "heldout_step": 0,
        "kv_head": 0,
        "query_head": 0,
        "method": "qk_balanced",
        "selected_fraction_target": 0.02,
        "top2_recall": 0.9 + delta,
        "selected_attention_mass": 0.95 + delta,
        "oracle_top2_attention_mass": 0.96 + delta,
        "top2_attention_mass_recall": 0.99 + delta,
        "score_pearson": 0.9 + delta,
        "score_rmse": 0.1 + delta,
    }
    with (root / "per_head.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)
        if duplicate:
            writer.writerow(row)
    with (root / "allocations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["layer", "kv_head", "method", "allocation"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "layer": 0,
                "kv_head": 0,
                "method": "qk_balanced",
                "allocation": "4-4-2-1-1-1-1-0",
            }
        )


def test_equivalence_verifier_accepts_roundoff(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    write_artifact(reference, 0.0)
    write_artifact(candidate, 1e-7)
    report = verify(reference, candidate)
    assert report["passed"]
    assert report["conditions"] == 1
    assert report["allocations_identical"]


def test_equivalence_verifier_rejects_metric_drift(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    write_artifact(reference, 0.0)
    write_artifact(candidate, 0.01)
    with pytest.raises(AssertionError, match="exceeds equivalence tolerance"):
        verify(reference, candidate)


def test_equivalence_verifier_rejects_duplicate_conditions(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    write_artifact(reference, 0.0, duplicate=True)
    write_artifact(candidate, 0.0)
    with pytest.raises(AssertionError, match="duplicate qk_balanced condition"):
        verify(reference, candidate)
