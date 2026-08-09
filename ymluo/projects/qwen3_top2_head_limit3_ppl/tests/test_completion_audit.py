from __future__ import annotations

from audit_20260716_experiment_completion import overall_validator, rows_validator


def test_overall_validator_requires_exact_method_counts() -> None:
    validator = overall_validator({"full_kv": 2, "sparse": 2})
    detail = validator(
        {
            "overall": [
                {"method": "full_kv", "samples": 2},
                {"method": "sparse", "samples": 2},
            ]
        }
    )
    assert "full_kv" in detail


def test_rows_validator_accepts_dict_or_list() -> None:
    validator = rows_validator(2)
    assert validator({"rows": [{}, {}]}) == "rows=2"
    assert validator([{}, {}]) == "rows=2"
