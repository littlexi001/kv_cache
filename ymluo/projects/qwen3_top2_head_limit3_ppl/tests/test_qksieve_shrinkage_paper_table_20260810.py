from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "papers"
    / "countcap_iclr2027"
    / "scripts"
    / "make_qksieve_shrinkage_tables.py"
)
SPEC = importlib.util.spec_from_file_location("qksieve_shrinkage_tables", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def summary(passed: bool = True) -> dict:
    aggregate = []
    for fraction in (0.01, 0.02, 0.04):
        for shrinkage in (0.0, 0.25, 0.5, 0.75, 0.9):
            aggregate.append(
                {
                    "selected_fraction": fraction,
                    "shrinkage": shrinkage,
                    "top2_recall": 0.8 + 0.01 * shrinkage,
                    "selected_attention_mass": 0.9 + 0.01 * shrinkage,
                    "top2_attention_mass_recall": 0.95,
                    "score_pearson": 0.97,
                    "score_rmse": 0.1,
                }
            )
    checks = [
        {
            "selected_fraction": fraction,
            "production_recall_regret": 0.002,
            "production_mass_regret": 0.001,
            "production_rmse_ratio_to_best": 1.02,
        }
        for fraction in (0.01, 0.02, 0.04)
    ]
    return {
        "aggregate": aggregate,
        "acceptance": {
            "passed": passed,
            "checks": checks,
            "failures": [] if passed else ["mass regret exceeds 1 point at 1.0%"],
        },
    }


def test_shrinkage_table_renders_full_grid_and_registered_regret() -> None:
    text = MODULE.render(summary(), chinese=False, provenance="abc123")
    assert "Generated from paired shrinkage evidence: abc123" in text
    assert text.count("\\textbf{") == 21
    assert "1.00\\% & 80.75\\% & 0.20\\%" in text
    assert "The preregistered stability check passed" in text
    assert "selector-level numerical sensitivity" in text


def test_shrinkage_table_exposes_failures_in_both_languages() -> None:
    english = MODULE.render(summary(False), chinese=False, provenance="abc123")
    chinese = MODULE.render(summary(False), chinese=True, provenance="abc123")
    assert "failed; failures:" in english
    assert "mass regret exceeds 1 point at 1.0\\%" in english
    assert "预注册稳定性检查未通过" in chinese
    assert "失败项" in chinese
