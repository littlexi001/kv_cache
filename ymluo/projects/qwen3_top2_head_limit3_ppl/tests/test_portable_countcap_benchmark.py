import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import portable_countcap_benchmark as portable  # noqa: E402


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def method_row(method: str, score: float, seconds: float) -> dict[str, object]:
    is_full = method == "full_kv"
    return {
        "task": "task",
        "sample_id": "0",
        "method": method,
        "score": score,
        "prompt_tokens": 100,
        "generated_tokens": 10,
        "configured_attention_fraction": 1.0 if is_full else 0.02,
        "configured_candidate_fraction": 1.0 if is_full else 0.06,
        "attention_link_ratio": 1.0 if is_full else 0.02,
        "exact_qk_ratio": 1.0 if is_full else 0.06,
        "temporal_reuse_rate": 0.0,
        "gpu_kv_storage_ratio": 1.0,
        "scan_dimension_fraction": 1.0 if is_full else 0.375,
        "online_seconds": seconds,
    }


def test_finalize_writes_paired_markdown_report(tmp_path):
    long_rows = [
        method_row("full_kv", 0.8, 2.0),
        method_row("countcap", 0.76, 1.0),
    ]
    ruler_rows = []
    for row in long_rows:
        current = dict(row)
        current.update(
            {
                "task": "niah_single_1_65536",
                "base_task": "niah_single_1",
                "requested_length": 65536,
            }
        )
        ruler_rows.append(current)

    write_rows(tmp_path / "long.csv", long_rows)
    write_rows(tmp_path / "ruler.csv", ruler_rows)
    output = tmp_path / "final"
    portable.finalize(
        argparse.Namespace(
            project_root=PROJECT_ROOT,
            longbench_glob=str(tmp_path / "long.csv"),
            ruler_glob=str(tmp_path / "ruler.csv"),
            output_dir=output,
        )
    )

    report = (output / "RESULTS.md").read_text(encoding="utf-8")
    assert "95.00%" in report
    assert "2.000x" in report
    assert (output / "results.json").is_file()
