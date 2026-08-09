import csv
from pathlib import Path

import pytest

from analyze_countcap_token_nll_tail_20260726 import (
    paired_deltas,
    summarize,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_paired_delta_summary(tmp_path: Path) -> None:
    path = tmp_path / "length64000_x" / "token_results.csv"
    write_rows(
        path,
        [
            {
                "topic": "x",
                "window": 0,
                "target_index": 0,
                "token_id": 10,
                "method": "full_attention",
                "nll": 1.0,
            },
            {
                "topic": "x",
                "window": 0,
                "target_index": 0,
                "token_id": 10,
                "method": "direct_countcap",
                "nll": 1.5,
            },
            {
                "topic": "x",
                "window": 0,
                "target_index": 1,
                "token_id": 11,
                "method": "full_attention",
                "nll": 2.0,
            },
            {
                "topic": "x",
                "window": 0,
                "target_index": 1,
                "token_id": 11,
                "method": "direct_countcap",
                "nll": 1.5,
            },
        ],
    )
    rows = paired_deltas([path])
    result = summarize(rows)
    assert result["tokens"] == 2
    assert result["mean_delta_nll"] == pytest.approx(0.0)
    assert result["positive_delta_fraction"] == pytest.approx(0.5)
    assert {row["history_tokens"] for row in rows} == {64000}


def test_unpaired_token_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "length64000_x" / "token_results.csv"
    write_rows(
        path,
        [
            {
                "topic": "x",
                "window": 0,
                "target_index": 0,
                "token_id": 10,
                "method": "full_attention",
                "nll": 1.0,
            }
        ],
    )
    with pytest.raises(ValueError, match="unpaired token"):
        paired_deltas([path])


def test_same_token_key_at_two_lengths_does_not_collide(
    tmp_path: Path,
) -> None:
    paths = []
    for history_tokens, direct_nll in ((64000, 1.1), (128000, 1.2)):
        path = (
            tmp_path
            / f"length{history_tokens}_x"
            / "token_results.csv"
        )
        write_rows(
            path,
            [
                {
                    "topic": "x",
                    "window": 0,
                    "target_index": 0,
                    "token_id": 10,
                    "method": "full_attention",
                    "nll": 1.0,
                },
                {
                    "topic": "x",
                    "window": 0,
                    "target_index": 0,
                    "token_id": 10,
                    "method": "direct_countcap",
                    "nll": direct_nll,
                },
            ],
        )
        paths.append(path)

    rows = paired_deltas(paths)

    assert len(rows) == 2
    assert {row["history_tokens"] for row in rows} == {64000, 128000}
