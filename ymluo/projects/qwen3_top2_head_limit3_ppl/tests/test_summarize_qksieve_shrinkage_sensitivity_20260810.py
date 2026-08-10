from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from summarize_qksieve_shrinkage_sensitivity_20260810 import (  # noqa: E402
    lambda_tag,
    summarize,
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
    "top2_attention_mass_recall",
    "score_pearson",
    "score_rmse",
]


def write_run(root: Path, label: str, shrinkage: float, *, drop_last: bool = False) -> None:
    output = root / "analysis" / label / lambda_tag(shrinkage)
    output.mkdir(parents=True)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "query_shrinkage": shrinkage,
                    "calibration_source": "prefill_tail",
                }
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for layer in (0, 1):
        quality = 0.80 + 0.01 * int(shrinkage == 0.75)
        rows.append(
            {
                "label": label,
                "layer": layer,
                "heldout_step": 0,
                "kv_head": 0,
                "query_head": 0,
                "method": "qk_balanced",
                "selected_fraction_target": 0.01,
                "top2_recall": quality,
                "selected_attention_mass": quality + 0.10,
                "top2_attention_mass_recall": quality + 0.05,
                "score_pearson": quality,
                "score_rmse": 0.10 - 0.01 * int(shrinkage == 0.75),
            }
        )
    if drop_last:
        rows.pop()
    with (output / "per_head.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_summarizer_requires_strict_pairing_and_accepts_stable_production(
    tmp_path: Path,
) -> None:
    for label in ("qwen_sports", "llama_medicine"):
        for shrinkage in (0.0, 0.75):
            write_run(tmp_path, label, shrinkage)

    report = summarize(
        tmp_path,
        labels=("qwen_sports", "llama_medicine"),
        shrinkages=(0.0, 0.75),
        fractions=(0.01,),
        bootstrap_samples=100,
    )

    assert report["complete"]
    assert report["strict_paired_conditions"] == 4
    assert report["acceptance"]["passed"]
    assert len(report["source_sha256"]) == 8


def test_summarizer_rejects_an_unpaired_lambda(tmp_path: Path) -> None:
    write_run(tmp_path, "qwen_sports", 0.0)
    write_run(tmp_path, "qwen_sports", 0.75, drop_last=True)

    with pytest.raises(AssertionError, match="unpaired shrinkage"):
        summarize(
            tmp_path,
            labels=("qwen_sports",),
            shrinkages=(0.0, 0.75),
            fractions=(0.01,),
            bootstrap_samples=100,
        )
